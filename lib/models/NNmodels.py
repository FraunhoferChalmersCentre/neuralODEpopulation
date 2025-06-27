"""
Created on Fri Jun 27 10:09:53 2025

@author: Baaz
"""
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import torch.nn.functional as F

import os

hid_dim = 128
latent_dim = 2
dim_parameter_encoder = 2


# ---- ODE ----
class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim  + 2 + dim_parameter_encoder , hid_dim),  # latent_dim + dose + T + encoded_params
            nn.SELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, latent_dim),
        )

    def T(self, t, dose_times):
        t_scalar = t.item()
        relevant_doses = dose_times[dose_times <= t_scalar]
        return t_scalar if len(relevant_doses) == 0 else t_scalar - relevant_doses.max().item()

    def forward(self, t, x, dose, dose_times, encoded_params):
        if x.dim() == 1:
            x = x.unsqueeze(0)
    
        batch_size = x.size(0)
        T_val = self.T(t, dose_times)
        T = T_val * torch.ones(batch_size, 1, device=x.device)  # (batch_size,1)
    
        dose_exp = dose.unsqueeze(0).unsqueeze(1).expand(batch_size, 1)
        encoded_params_exp = encoded_params.unsqueeze(0).expand(batch_size, -1)
        t_val = t.item()
        t_tensor = torch.tensor([[t_val]], device=x.device)
        
        inp1 = torch.cat([T, x,dose_exp, encoded_params_exp], dim=1)

        dxdt = self.net(inp1)  # (batch, 1)
        return dxdt


class TrainableNoise(nn.Module):
    def __init__(self, size=1, init_add_std=0.1, init_prop_std=0.1):
        super().__init__()
        init_log_sigma_add = torch.log(torch.tensor(init_add_std))
        init_log_sigma_prop = torch.log(torch.tensor(init_prop_std))

        self.log_sigma_add = nn.Parameter(torch.full((size,), init_log_sigma_add))
        self.log_sigma_prop = nn.Parameter(torch.full((size,), init_log_sigma_prop))

    def forward(self, x_pred):
        # Return a single sample from the noise distribution
        return self.sample(x_pred, n_samples=1).squeeze(0)

    def nll(self, x_true, x_pred):
        sigma_add = torch.exp(self.log_sigma_add)
        sigma_prop = torch.exp(self.log_sigma_prop)
        sigma_total = torch.sqrt(sigma_add**2 + (sigma_prop * x_pred)**2)

        nll_elementwise = (
            0.5 * ((x_true - x_pred) / sigma_total) ** 2
            + torch.log(sigma_total)
            + 0.5 * np.log(2 * np.pi)
        )
        return torch.mean(nll_elementwise)

    def sample(self, x_pred, n_samples=1):
        """
        Samples from N(x_pred, sigma_total^2)

        Args:
            x_pred (Tensor): Predicted mean, shape (B,) or (B, ...)
            n_samples (int): Number of samples to draw

        Returns:
            Tensor of shape (n_samples, *x_pred.shape)
        """
        sigma_add = torch.exp(self.log_sigma_add)
        sigma_prop = torch.exp(self.log_sigma_prop)
        sigma_total = torch.sqrt(sigma_add**2 + (sigma_prop * x_pred)**2)

        # Expand sigma to match x_pred shape
        sigma_total = sigma_total.expand_as(x_pred)

        # Draw samples
        eps = torch.randn((n_samples,) + x_pred.shape, device=x_pred.device)
        return x_pred.unsqueeze(0) + sigma_total.unsqueeze(0) * eps


# ---- Global and Refiner ----
class GlobalLatent(nn.Module):
    def __init__(self, latent_dim=dim_parameter_encoder,initial_var=1):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(latent_dim))
        initial_logvar = torch.log(torch.full((latent_dim,), initial_var))
        self.logvar = nn.Parameter(initial_logvar)

        self.register_buffer('iteration', torch.tensor(1.0))

    def sample(self):
        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std)
        return self.mu + eps * std

    def update(self, z_all, gamma=None):
        if gamma is None:
            alpha = 0.6
            gamma = 1.0 / (self.iteration ** alpha)
        with torch.no_grad():
            batch_mu = z_all.mean(dim=0)
            batch_var = z_all.var(dim=0, unbiased=False).clamp(min=1e-6)
            batch_logvar = torch.log(batch_var)

            self.mu.data = (1 - gamma) * self.mu.data + gamma * batch_mu
            self.logvar.data = (1 - gamma) * self.logvar.data + gamma * batch_logvar
            self.iteration += 1.0


class IndividualRefiner(nn.Module):
    def __init__(self, input_dim=2 + dim_parameter_encoder, latent_dim=dim_parameter_encoder, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # Self-attention layers
        self.attn_query = nn.Linear(hidden_dim, hidden_dim)
        self.attn_key = nn.Linear(hidden_dim, hidden_dim)
        self.attn_value = nn.Linear(hidden_dim, hidden_dim)

        # Final projections for mean and log-variance of q(z|x)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, t, x, z_global):
        """
        Args:
            t: (seq_len,)
            x: (seq_len,)
            z_global: (latent_dim,)
        
        Returns:
            mu_q: (latent_dim,)
            logvar_q: (latent_dim,)
        """
        seq_len = t.size(0)
        z_expanded = z_global.unsqueeze(0).expand(seq_len, -1)  # (seq_len, latent_dim)
        inp = torch.cat([t.unsqueeze(1), x.unsqueeze(1), z_expanded], dim=1)  # (seq_len, input_dim)
        inp = inp.unsqueeze(0)  # (1, seq_len, input_dim)

        lstm_out, _ = self.lstm(inp)  # (1, seq_len, hidden_dim)

        # Self-attention mechanism
        Q = self.attn_query(lstm_out)
        K = self.attn_key(lstm_out)
        V = self.attn_value(lstm_out)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights, V)  # (1, seq_len, hidden_dim)

        # Pool across time
        pooled = context.mean(dim=1)  # (1, hidden_dim)

        mu_q = self.fc_mu(pooled).squeeze(0)         # (latent_dim,)
        logvar_q = self.fc_logvar(pooled).squeeze(0) # (latent_dim,)

        return mu_q, logvar_q


class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=latent_dim, hidden_dim=128):
        super(SimpleDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.relu = nn.ReLU()        # instantiate ReLU module here
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    def forward(self, z):
        x = self.relu(self.fc1(z))  # call the instance, not the class
        y=self.relu(self.fc2(x))
        concentration = self.fc3(y)
        #concentration = self.relu(z)

        return concentration.squeeze(-1)


class InitialConditionEncoder(nn.Module):
    def __init__(self, latent_dim=latent_dim, hidden_dim=32):
        super(InitialConditionEncoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.relu = nn.SELU()        # instantiate ReLU module here
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.relu(self.fc1(x))   # call the instance
        z0 = self.fc2(h)
        return z0
