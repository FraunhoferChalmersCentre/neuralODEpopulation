# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 14:39:09 2025

@author: Baaz
"""

# -*- coding: utf-8 -*-
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
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import os



class ODEWrapper(nn.Module):
    def __init__(self, func, dose_times_padded,dose, dose_mask):
        super().__init__()
        self.func = func
        self.dose_times = dose_times_padded  # shape: [batch_size, max_len]
        self.dose_mask = dose_mask            # shape: [batch_size, max_len]
        self.done_dose = dose
              # shape: [batch_size, latent_dim]

    def forward(self, t, x):
        # x shape: [batch_size, latent_dim + ...]
        return self.func(t, x, self.dose_times, self.done_dose, self.dose_mask)

# ---- ODE ----
class ODEFunc(nn.Module):
    def __init__(self, latent_dim, dim_parameter_encoder, hidden_dim):
        super().__init__()
        self.dim_parameter_encoder = dim_parameter_encoder  # <-- Save here
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2 + dim_parameter_encoder, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        self.skipnet = nn.Sequential(
            nn.Linear(latent_dim + 2 + dim_parameter_encoder , hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        
        self.ema_decay = 0.999
        self._ema_initialized = False
        self._ema_params = {
        name: param.detach().clone()
        for name, param in self.named_parameters()
        if param.requires_grad
    }

    def _initialize_ema(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.register_buffer(f"{name.replace('.', '_')}_ema", param.detach().clone())
        self._ema_initialized = True

    def update_ema(self, alpha=None):
        if alpha is None:
            alpha = self.ema_decay
    
        # If EMA params not registered, initialize them now
        if self._ema_params is None:
            self.register_ema()
    
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    ema_param = self._ema_params[name]
                    ema_param.mul_(alpha).add_(param, alpha=1 - alpha)


    def apply_ema_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                    
                    param.data.copy_(ema_param)
    def T(self, t, dose_times):
         t_scalar = t.item()
         relevant_doses = dose_times[dose_times <= t_scalar]
         return t_scalar if len(relevant_doses) == 0 else t_scalar - relevant_doses.max().item()

    def forward(self, t, x, dose_times, dose, dose_mask):
        """
        t: scalar time float
        x: tensor of shape [batch_size, state_dim]
        dose_times: tensor [batch_size, max_len]
        dose_mask: bool tensor [batch_size, max_len]
        encoded_params: tensor [batch_size, latent_dim]
        """
      #  print(f"At time {t}: x.shape = {x.shape}")

        batch_size = x.size(0)
        device = x.device

        # Compute T for each batch sample (vectorized)
        # t is a scalar, dose_times: [batch, max_len]
        # For each sample, find max dose time <= t
        t_scalar = t.item()
        # Mask dose_times > t_scalar, set invalid to a large negative number for max
        valid_doses = torch.where(dose_mask & (dose_times <= t_scalar), dose_times, torch.tensor(float('-inf'), device=device))
        max_dose_times, _ = valid_doses.max(dim=1)  # shape: [batch_size]

        # If no valid doses (all -inf), then max_dose_times = -inf, replace with 0
        max_dose_times = torch.where(max_dose_times == float('-inf'), torch.zeros_like(max_dose_times), max_dose_times)
        T = (t_scalar - max_dose_times).unsqueeze(1)  # shape: [batch_size, 1]
        dose_exp = dose  # shape should already be [batch_size, 1]



      #  encoded_params: [batch_size, latent_dim]
        # x: [batch_size, state_dim]
        
        inp = torch.cat([x, T, dose_exp], dim=1)  # shape: [batch_size, 1 + state_dim + latent_dim]

        dxdt = self.net(inp)  # shape: [batch_size, latent_dim-1]
        dxdt_skip = self.skipnet(inp)
        zero = torch.zeros(batch_size, self.dim_parameter_encoder, device=device)
        dxdt_concat = torch.cat([dxdt+dxdt_skip, zero], dim=1)  # shape: [batch_size, latent_dim]
        output=dxdt+dxdt_skip
        return dxdt_concat

   

    
    
    
    
    

class TrainableNoise(nn.Module):
    def __init__(self, conc_std, size=1, init_add_std=0.8, init_prop_std=0):
        super().__init__()
        init_log_sigma_add = torch.log(torch.tensor(init_add_std/conc_std))
        init_log_sigma_prop = torch.log(torch.tensor(init_prop_std))

        self.log_sigma_add = nn.Parameter(torch.full((size,), init_log_sigma_add))
        self.log_sigma_prop = nn.Parameter(torch.full((size,), init_log_sigma_prop))

    def forward(self, x_pred):
        # Return a single sample from the noise distribution
        return self.sample(x_pred, n_samples=1).squeeze(0)

    def nll(self, x_true, x_pred, mask=None):
        sigma_add = torch.exp(self.log_sigma_add)
        sigma_prop = torch.exp(self.log_sigma_prop)
        
        # If size>1, broadcast sigma params to match x_pred shape (assume last dims)
        sigma_add = sigma_add.view(*([1] * (x_pred.dim() - sigma_add.dim())), *sigma_add.shape)
        sigma_prop = sigma_prop.view(*([1] * (x_pred.dim() - sigma_prop.dim())), *sigma_prop.shape)
        
        sigma_total = torch.sqrt(sigma_add**2 + (sigma_prop * x_pred)**2)
    
        nll_elementwise = (
            0.5 * ((x_true - x_pred) / sigma_total) ** 2
            + torch.log(sigma_total)
            + 0.5 * np.log(2 * np.pi)
        )
        
        if mask is not None:
            # Expand mask dims to match nll_elementwise if needed
            if nll_elementwise.dim() > mask.dim():
                mask = mask.unsqueeze(-1)
            
            nll_elementwise = nll_elementwise * mask
            total_valid = mask.sum().clamp(min=1)  # avoid div by zero
            return nll_elementwise.sum() / total_valid
        
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



class IndividualRefiner_VAE(nn.Module):
    def __init__(self, input_dim , hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # Self-attention layers
        self.attn_query = nn.Linear(hidden_dim, hidden_dim)
        self.attn_key = nn.Linear(hidden_dim, hidden_dim)
        self.attn_value = nn.Linear(hidden_dim, hidden_dim)

        # Final projections for mean and log-variance of q(z|x)
        self.fc_mu = nn.Linear(hidden_dim, input_dim)
        self.fc_logvar = nn.Linear(hidden_dim, input_dim)
        
        for name, param in self.named_parameters():
           self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
        
        self.register_buffer("iteration", torch.tensor(1.0))
        
    def update_ema(self, alpha=0.1):
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                ema_param.mul_(1 - alpha).add_(alpha * param.data)
            self.iteration += 1.0

    def apply_ema_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                param.data.copy_(ema_param)


    def forward(self, t, x):
        """
        t: (batch_size, seq_len)
        x: (batch_size, seq_len)
        global_mu: (latent_dim,)
        global_logvar: (latent_dim,)
        """
    
        batch_size, seq_len = t.shape

    
        inp = torch.cat([
            t.unsqueeze(-1),  # (batch_size, seq_len, 1)
            x.unsqueeze(-1),  # (batch_size, seq_len, 1)
        ], dim=-1)  # (batch_size, seq_len, input_dim)
    
        lstm_out, _ = self.lstm(inp)  # (batch_size, seq_len, hidden_dim)
    
        # Attention (batch version)
        Q = self.attn_query(lstm_out)
        K = self.attn_key(lstm_out)
        V = self.attn_value(lstm_out)
    
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)  # (batch_size, seq_len, seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights, V)  # (batch_size, seq_len, hidden_dim)
    
        pooled = context.mean(dim=1)  # (batch_size, hidden_dim)
    
        mu_q = self.fc_mu(pooled)      # (batch_size, latent_dim)
        logvar_q = self.fc_logvar(pooled)  # (batch_size, latent_dim)
    
        return mu_q, logvar_q


    
class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=128):
        super(SimpleDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.relu = nn.ReLU()        # instantiate ReLU module here
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        for name, param in self.named_parameters():
           self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
           
        self.register_buffer("iteration", torch.tensor(1.0))
           
           
           

    def update_ema(self, alpha=0.1):
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                ema_param.mul_(1 - alpha).add_(alpha * param.data)
            self.iteration += 1.0

    def apply_ema_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                param.data.copy_(ema_param)
    def forward(self, z):
        x = self.relu(self.fc1(z))  # call the instance, not the class
        y=self.relu(self.fc2(x))
        concentration = self.fc3(y)
        #concentration = self.relu(z)

        return concentration.squeeze(-1)

class InitialConditionEncoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=32):
        super(InitialConditionEncoder, self).__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.relu = nn.SELU()        # instantiate ReLU module here
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        
        for name, param in self.named_parameters():
            self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
            
        self.register_buffer("iteration", torch.tensor(1.0))

    def forward(self, x):
        h = self.relu(self.fc1(x))   # call the instance
        z0 = self.fc2(h)
        return z0
    
    
    def update_ema(self, alpha=0.1):
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                ema_param.mul_(1 - alpha).add_(alpha * param.data)
            self.iteration += 1.0

    def apply_ema_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                param.data.copy_(ema_param)