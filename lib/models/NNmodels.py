# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 17:44:57 2025

@author: Baaz
"""

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
#import ast
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches

import torch
import torch.nn as nn
#from torch.utils.data import Dataset, DataLoader
#from torch.nn.utils.rnn import pad_sequence
#from torchdiffeq import odeint_adjoint as odeint


#from sklearn.decomposition import PCA
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import r2_score
import math
#import os

#import torch.nn.functional as F



class ODEFunc(nn.Module):
    """
    ODE function with optional EMA for parameters, dose handling, and skip connection.
    """
    def __init__(self, latent_dim, dim_parameter_encoder, hid_dim):
        super().__init__()
        self.dim_latent = latent_dim
        self.dim_parameter_encoder = dim_parameter_encoder

        # Deep network
        self.net = nn.Sequential(
            nn.Linear(latent_dim + dim_parameter_encoder + 1, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, latent_dim)
        )

        # Skip connection
        self.skip = nn.Sequential(
            nn.Linear(latent_dim + dim_parameter_encoder + 1, latent_dim)
        )

        # Noise parameter for dose pulses
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(0.05)))

        # EMA buffers
        for name, param in self.named_parameters():
            self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
        self.register_buffer("iteration", torch.tensor(1.0))

    # === EMA methods ===
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

    # === Sigma getter ===
    def get_sigma(self):
        return torch.exp(self.log_sigma)

    # === Dose pulse computation ===
    def dirac_pulse(self, t, dose_times, dose_amounts, dose_mask):
        """
        Compute Gaussian-smoothed dose pulses at given time points.
        """
        sigma = self.get_sigma()
        diff = t - dose_times  # [batch_size, num_doses]

        # Only past doses
        mask = (diff >= 0).float() * dose_mask.float()
        gauss = torch.exp(-0.5 * (diff / sigma) ** 2) * mask

        if dose_amounts.dim() == 3:
            dose_amounts = dose_amounts.squeeze(-1)

        weighted = gauss * dose_amounts
        dose_signal = weighted.sum(dim=1, keepdim=True)  # [batch_size, 1]
        # if t.item() == 0:
        #     dose_signal = torch.ones_like(dose_amounts)
        # else:
        #     dose_signal = torch.zeros_like(dose_amounts)
        return dose_signal

    # === Forward pass ===
    def forward(self, t, x, dose_times, dose_amounts, dose_mask, keep_mask=None):
        batch_size = x.size(0)
        device = x.device

        if keep_mask is None:
            keep_mask = torch.ones(batch_size, 1, device=device)

        # Compute dose input
        dose_input = self.dirac_pulse(t, dose_times, dose_amounts, dose_mask)

        # Concatenate latent state and dose
        inp = torch.cat([x, dose_input], dim=1)

        # Deep network + skip connection
        dxdt_deep = self.net(inp)
        dxdt_skip = self.skip(inp)
        dxdt = dxdt_deep + dxdt_skip

        # Concatenate zeros for parameter dimensions
        zeros_param = torch.zeros(batch_size, self.dim_parameter_encoder, device=device)
        dxdt_concat = torch.cat([dxdt, zeros_param], dim=1)

        return dxdt_concat





    

class TrainableNoise(nn.Module):
    """
    Estimates additive + optional proportional noise with EMA support.
    """
    def __init__(self, size=1, init_add_std=0.1, init_prop_std=0.0):
        super().__init__()

        # Additive noise
        self.log_sigma_add = nn.Parameter(torch.full((size,), torch.log(torch.tensor(init_add_std))))

        # Optional proportional noise
        if init_prop_std > 0:
            self.use_prop = True
            self.log_sigma_prop = nn.Parameter(torch.full((size,), torch.log(torch.tensor(init_prop_std))))
        else:
            self.use_prop = False
            self.log_sigma_prop = None

        # EMA buffers
        self.register_buffer("log_sigma_add_ema", self.log_sigma_add.data.clone())
        if self.use_prop:
            self.register_buffer("log_sigma_prop_ema", self.log_sigma_prop.data.clone())
        self.register_buffer("iteration", torch.tensor(1.0))

    @property
    def sigma_add(self):
        return torch.exp(self.log_sigma_add)

    @property
    def sigma_prop(self):
        if not self.use_prop:
            return torch.zeros_like(self.log_sigma_add)
        return torch.exp(self.log_sigma_prop)

    def _compute_sigma_total(self, x_pred):
        sigma_add = self.sigma_add
        if self.use_prop:
            sigma_prop_value = self.sigma_prop.view(*([1] * (x_pred.dim() - self.sigma_prop.dim())), *self.sigma_prop.shape)
            sigma_total = torch.sqrt(sigma_add**2 + (sigma_prop_value * x_pred)**2)
        else:
            sigma_total = sigma_add
        return torch.clamp(sigma_total, min=1e-6)

    def forward(self, x_pred):
        sigma_total = self._compute_sigma_total(x_pred)
        eps = torch.randn_like(x_pred)
        return x_pred + sigma_total * eps

    def nll(self, x_true, x_pred, replace_mask=None, mask=None):
        sigma_total = self._compute_sigma_total(x_pred)
        x_pred = torch.clamp(x_pred, min=-1e6, max=1e6)
        
        
        #noisy_mask = torch.ones_like(x_true, dtype=torch.bool)
        #noisy_mask[:, 0] = False  # first point is noiseless

        # Elementwise NLL (Gaussian)
        nll_elementwise = 0.5 * ((x_true - x_pred) / sigma_total) ** 2 + torch.log(sigma_total)
        mse_elementwise = (x_true - x_pred) ** 2
        
        # L1 for masked/dropped points
        if replace_mask is not None:
            dropout_mask = replace_mask.bool()
            if dropout_mask.any():
                L1_nll_elementwise = (2 ** 0.5) * (x_true - x_pred).abs() / sigma_total + torch.log(2 ** 0.5 * sigma_total)
                L1_elementwise = (x_true - x_pred).abs()
                nll_elementwise[dropout_mask] = L1_nll_elementwise[dropout_mask]
                mse_elementwise[dropout_mask] = L1_elementwise[dropout_mask]

        # Apply mask
        if mask is not None:
            nll_elementwise = nll_elementwise * mask
            mse_elementwise = mse_elementwise * mask

        # Per-individual sum
        nll_per_ind = nll_elementwise.sum(dim=1)
        mse_per_ind = mse_elementwise.sum(dim=1)

        return nll_per_ind.mean(), mse_per_ind.mean()

    def sample(self, x_pred, n_samples=1):
        sigma_total = self._compute_sigma_total(x_pred)
        if n_samples > 1:
            x_pred = x_pred.unsqueeze(0).expand(n_samples, *x_pred.shape)
            sigma_total = sigma_total.unsqueeze(0).expand(n_samples, *sigma_total.shape)
        eps = torch.randn_like(x_pred)
        return x_pred + sigma_total * eps

    # === EMA methods ===
    def update_ema(self, alpha=0.1):
        with torch.no_grad():
            self.log_sigma_add_ema.mul_(1 - alpha).add_(alpha * self.log_sigma_add)
            if self.use_prop:
                self.log_sigma_prop_ema.mul_(1 - alpha).add_(alpha * self.log_sigma_prop)
            self.iteration += 1.0

    def apply_ema_weights(self):
        with torch.no_grad():
            self.log_sigma_add.data.copy_(self.log_sigma_add_ema)
            if self.use_prop:
                self.log_sigma_prop.data.copy_(self.log_sigma_prop_ema)









class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x
                





class Encoder_Transformer(nn.Module):
    def __init__(self, latent_dim, input_dim=2, model_dim=64,
                 hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj1 = nn.Linear(input_dim, hidden_dim)
        self.input_proj2 = nn.Linear(hidden_dim, model_dim)
        self.selu = nn.SELU()
        self.latent_dim = latent_dim

        self.pos_encoder = PositionalEncoding(model_dim)
        self.raw_encoder = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_mu1 = nn.Linear(model_dim, hidden_dim)
        self.fc_mu2 = nn.Linear(hidden_dim, latent_dim)

        self.fc_logvar1 = nn.Linear(model_dim, hidden_dim)
        self.fc_logvar2 = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.fc_mu2.weight)
        nn.init.zeros_(self.fc_mu2.bias)
        nn.init.zeros_(self.fc_logvar2.weight)
        nn.init.zeros_(self.fc_logvar2.bias)

   


        # Register EMA buffers
        for name, param in self.named_parameters():
            self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())

        self.register_buffer("iteration", torch.tensor(1.0))

    def sample(self, mu_q, logvar_q, num_samples=10):
        """
        Sample multiple latent vectors from q(z|x).

        Args:
            mu_q: [B, latent_dim] posterior mean
            logvar_q: [B, latent_dim] posterior log variance
            num_samples: number of samples per batch element

        Returns:
            z_samples: [B, num_samples, latent_dim] latent samples
        """
        B, D = mu_q.shape

        mu_exp = mu_q.unsqueeze(1).expand(B, num_samples, D)
        logvar_exp = logvar_q.unsqueeze(1).expand(B, num_samples, D)
        std_exp = torch.exp(0.5 * logvar_exp)

        eps = torch.randn_like(std_exp)
        z_samples = mu_exp + std_exp * eps  # [B, num_samples, D]

        return z_samples
    
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


    def forward(self, t, x, mask=None):
        """
        t: [B, T]
        x: [B, T]
        mask: optional [B, T] bool mask for valid positions
        """
        inp = torch.cat([t.unsqueeze(-1), x.unsqueeze(-1)], dim=-1)  # [B, T, 2]
        h1 = self.selu(self.input_proj1(inp))
        h2 = self.input_proj2(h1)
        h = self.pos_encoder(h2)

        if mask is None:
            mask = torch.ones(t.shape[0], t.shape[1], dtype=torch.bool, device=t.device)

        all_masked = (mask.sum(dim=1) == 0)  # shape [B]

        h_enc = self.transformer_encoder(
            h, src_key_padding_mask=~mask if mask is not None else None
        )  # [B, T, model_dim]

        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (h_enc * mask.unsqueeze(-1)).sum(dim=1) / valid_counts
        pooled[all_masked] = 0.0

        # Skip connection data (not used for flows anymore, but could be used elsewhere)
        skip_data = self.raw_encoder(inp)
        skip_data_pooled = skip_data.mean(dim=1)  # [B, model_dim]

 
        mu_q =self.fc_mu2(self.selu(self.fc_mu1(pooled)))

        logvar_q = self.fc_logvar2(self.selu(self.fc_logvar1(pooled)))

        mu_q[all_masked] = 0.0
        logvar_q[all_masked] = 0.0

        std = torch.exp(0.5 * logvar_q)
        eps = torch.randn_like(std)
        z = mu_q + eps * std
        z = torch.clamp(z, -1e2, 1e2)

        return z, mu_q, logvar_q

    



class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        
        # Define the network
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1)        )
        
       


        # EMA buffers
        for name, param in self.named_parameters():
            self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
        self.register_buffer("iteration", torch.tensor(1.0))




    def forward(self, z):
        return self.net(z).squeeze(-1)

    def update_ema(self, alpha=0.1):
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                ema_param = ema_param.to(param.device)
                ema_param.mul_(1 - alpha).add_(alpha * param.data)
                setattr(self, f"{name.replace('.', '_')}_ema", ema_param)
            self.iteration += 1.0

    def apply_ema_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                param.data.copy_(ema_param)



class InitialConditionEncoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

 
        # === EMA buffers ===
        for name, param in self.named_parameters():
            self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
        self.register_buffer("iteration", torch.tensor(1.0))

    def forward(self, x):
        # Return same output three times (mu, logvar, something) if required
        out = self.net(x)
        return out, out, out

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


    
    



class InitialConditionVAEEncoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.SELU = nn.SELU()
        
        # Two separate layers for mean and log variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # EMA buffers
        for name, param in self.named_parameters():
            self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
        self.register_buffer("iteration", torch.tensor(1.0))

    def forward(self, x):
        h = self.SELU(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def update_ema(self, alpha=0.1):
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                ema_param = ema_param.to(param.device)
                ema_param.mul_(1 - alpha).add_(alpha * param.data)
                setattr(self, f"{name.replace('.', '_')}_ema", ema_param)
            self.iteration += 1.0

    def apply_ema_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                param.data.copy_(ema_param)
 