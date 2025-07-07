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
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import math
import os

import torch.nn.functional as F
from lib.utils.my_utils_parallel import *
from lib.utils.my_utils_parallel import destandardize_concentration


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
    def __init__(self, latent_dim, dim_parameter_encoder, hid_dim):
        super().__init__()
        self.dim_parameter_encoder = dim_parameter_encoder  
        self.dim_latent = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2+dim_parameter_encoder, hid_dim),
           nn.SELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, latent_dim ),
        )

        self.skipnet = nn.Sequential(
            nn.Linear(latent_dim + 2+dim_parameter_encoder , hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, latent_dim),
        )
        
        self.encodes = nn.Sequential(
            nn.Linear(dim_parameter_encoder , hid_dim),
           nn.SELU(),
            nn.Linear(hid_dim, latent_dim),
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
                    if ema_param.device != param.device:
                        ema_param = ema_param.to(param.device)
                        self._ema_params[name] = ema_param
                    ema_param.mul_(alpha).add_(param, alpha=1 - alpha)



    def apply_ema_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                    param.data.copy_(ema_param)

    def forward(self, t, x, dose_times,dose, dose_mask):
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

        # encoded_params: [batch_size, latent_dim]
        # x: [batch_size, state_dim]
        
        inp = torch.cat([x, T, dose], dim=1)  # shape: [batch_size, 1 + state_dim + latent_dim]
      #  if self.training:
        #inp = self.dropout(inp)

        dxdt_params=self.encodes(x[:,self.dim_latent:])
        dxdt = self.net(inp)  # shape: [batch_size, latent_dim-1]
        dxdt_skip = self.skipnet(inp)
        zero = torch.zeros(batch_size, self.dim_parameter_encoder, device=device)
        dxdt_concat = torch.cat([dxdt+dxdt_skip+dxdt_params, zero], dim=1)  # shape: [batch_size, latent_dim]
     
        return dxdt_concat

    def T(self, t, dose_times):
        t_scalar = t.item()
        relevant_doses = dose_times[dose_times <= t_scalar]
        return t_scalar if len(relevant_doses) == 0 else t_scalar - relevant_doses.max().item()


    
    
class TrainableNoise(nn.Module):
    def __init__(self, dataset, size=1, init_add_std=2, init_prop_std=0.001):
        super().__init__()
        init_log_sigma_add = torch.log(torch.tensor(init_add_std))
        init_log_sigma_prop = torch.log(torch.tensor(init_prop_std))
        self.conc_mean, self.conc_std = dataset.conc_mean, dataset.conc_std
        
        self.log_sigma_add = nn.Parameter(torch.full((size,), init_log_sigma_add))
        self.log_sigma_prop = nn.Parameter(torch.full((size,), init_log_sigma_prop))

    def forward(self, x_pred):
        # Return a single sample from the noise distribution
        return self.sample(x_pred, n_samples=1).squeeze(0)

    def nll(self, x_true, x_pred, mask=None):
        sigma_add = torch.exp(self.log_sigma_add)
        sigma_prop = torch.exp(self.log_sigma_prop)
        x_true=destandardize_concentration(x_true, self.conc_mean, self.conc_std)
        x_pred=destandardize_concentration(x_pred, self.conc_mean, self.conc_std)
        
        # If size>1, broadcast sigma params to match x_pred shape (assume last dims)
        sigma_add = sigma_add.view(*([1] * (x_pred.dim() - sigma_add.dim())), *sigma_add.shape)
        #sigma_prop = sigma_prop.view(*([1] * (x_pred.dim() - sigma_prop.dim())), *sigma_prop.shape)
        #+ (sigma_prop * x_pred)**2
        sigma_total = torch.sqrt(sigma_add**2 )
    
        nll_elementwise = (
            0.5 * ((x_true - x_pred) / sigma_total) ** 2
            + torch.log(sigma_total)

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


class ConditionalPlanarFlow(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Network to output flow params from conditioning vector c
        # It outputs u, w, b concatenated, so output dim = 2*latent_dim + 1
        self.param_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim + 1)
        )
    # Enforce invertibility constraint on u
    def u_hat(self, u, w):
        wT_u = (w * u).sum(dim=1, keepdim=True)
        m = -1 + torch.nn.functional.softplus(wT_u)
        return u + (m - wT_u) * w / (w.norm(p=2, dim=1, keepdim=True)**2 + 1e-8)

    def forward(self, z, c):
        """
        z: [B, latent_dim] latent variable
        c: [B, cond_dim] conditioning vector (e.g., individual data summary)
        """
        params = self.param_net(c)  # [B, 2*latent_dim + 1]
        
        u = params[:, :self.latent_dim]          # [B, latent_dim]
        w = params[:, self.latent_dim:2*self.latent_dim]  # [B, latent_dim]
        b = params[:, -1].unsqueeze(-1)          # [B, 1]
        u = self.u_hat(u,w)
        linear = (z * w).sum(dim=1) + b.squeeze(-1)

        h = torch.tanh(linear)  # [B]
    

        # Flow transform: z_new = z + u * h
        z_new = z + u * h.unsqueeze(-1)  # broadcasting h
      

        # Compute psi = h'(linear) * w
        psi = (1 - torch.tanh(linear)**2).unsqueeze(-1) * w  # [B, latent_dim]
     
        # det Jacobian: |1 + u^T psi| per batch element
        det_jacobian = torch.abs(1 + (psi * u).sum(dim=1))


        log_det = torch.log(det_jacobian + 1e-8)

        return z_new, log_det
    


class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_flow_dim,num_flows=2):
        super().__init__()
        self.flows = nn.ModuleList([ConditionalPlanarFlow(latent_dim, cond_dim, hidden_flow_dim) for _ in range(num_flows)])
    
    def forward(self, z0, c):
        log_det_sum = 0
        z = z0
        for flow in self.flows:
            z, log_det = flow(z, c)
            log_det_sum += log_det
        return z, log_det_sum


    
    


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
    


class Encoder_Transformer_NF(nn.Module):
    def __init__(self, latent_dim, input_dim=2, model_dim=64,hidden_dim=64, hidden_flow_dim=32, num_heads=4, num_layers=2, num_flow_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj1 = nn.Linear(input_dim, hidden_dim)
        self.input_proj2 = nn.Linear(hidden_dim, model_dim)
        self.selu = nn.SELU()        # instantiate ReLU module here
        #self.flow = NormalizingFlow(latent_dim, num_flows=4)
        self.flow = ConditionalNormalizingFlow(latent_dim,cond_dim=2*model_dim,hidden_flow_dim=32, num_flows=num_flow_layers)

        self.pos_encoder = PositionalEncoding(model_dim)
        self.raw_encoder = nn.Linear(input_dim,model_dim)
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


        for name, param in self.named_parameters():
            self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())

        self.register_buffer("iteration", torch.tensor(1.0))

    def update_ema(self, alpha=0.1):
         with torch.no_grad():
             for name, param in self.named_parameters():
                 ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                 ema_param = ema_param.to(param.device)  # ensure same device
                 ema_param.mul_(1 - alpha).add_(alpha * param.data)
                 # Re-assign back the buffer (optional, since inplace)
                 setattr(self, f"{name.replace('.', '_')}_ema", ema_param)
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
        h = self.selu(self.input_proj1(inp))                         # [B, T, hidden_dim]
        h = self.input_proj2(h)                                      # [B, T, model_dim]
        h = self.pos_encoder(h)
    
        h_enc = self.transformer_encoder(
            h, src_key_padding_mask=~mask if mask is not None else None
        )  # [B, T, model_dim]
    
        pooled = h_enc.mean(dim=1)  # [B, model_dim]
        skip_data=self.raw_encoder(inp)
        skip_data_pooled = skip_data.mean(dim=1)  # [B, model_dim]
        cond_vec = torch.cat([pooled, skip_data_pooled], dim=-1)  # [B, model_dim + raw_feat_dim]

        mu_q = self.fc_mu2(self.selu(self.fc_mu1(pooled)))
        logvar_q = self.fc_logvar2(self.selu(self.fc_logvar1(pooled)))
        
        std = torch.exp(0.5 * logvar_q)
        eps = torch.randn_like(std)
        z0 = mu_q + eps * std
    
        if self.training:
                z_k, log_det = self.flow(z0, cond_vec)  # pass both z0 and pooled conditioning vector
        else:
                z_k = mu_q
                log_det = torch.zeros(z_k.size(0), device=z_k.device)

    
        return z_k, mu_q, logvar_q, log_det

   
   
    
class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.relu = nn.ReLU()        # instantiate ReLU module here
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu2 = nn.ReLU()        # instantiate ReLU module here

        for name, param in self.named_parameters():
           self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
           
        self.register_buffer("iteration", torch.tensor(1.0))
           
           
           

    def update_ema(self, alpha=0.1):
          with torch.no_grad():
              for name, param in self.named_parameters():
                  ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                  ema_param = ema_param.to(param.device)  # ensure same device
                  ema_param.mul_(1 - alpha).add_(alpha * param.data)
                  # Re-assign back the buffer (optional, since inplace)
                  setattr(self, f"{name.replace('.', '_')}_ema", ema_param)
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
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
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
           def update_ema(self, alpha=0.1):
                with torch.no_grad():
                    for name, param in self.named_parameters():
                        ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                        ema_param = ema_param.to(param.device)  # ensure same device
                        ema_param.mul_(1 - alpha).add_(alpha * param.data)
                        # Re-assign back the buffer (optional, since inplace)
                        setattr(self, f"{name.replace('.', '_')}_ema", ema_param)
                    self.iteration += 1.0


    def apply_ema_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                param.data.copy_(ema_param)
                
                
                
class NormalizingFlow(nn.Module):
    def __init__(self, latent_dim, num_flows=4):
        super().__init__()
        self.flows = nn.ModuleList([ConditionalPlanarFlow(latent_dim) for _ in range(num_flows)])

    def forward(self, z0):
        """
        z0: initial latent vector (from encoder)
        Returns:
            z_k: transformed latent vector after flows
            sum_log_det: accumulated log-det-Jacobian (for modified ELBO)
        """
        log_det_sum = 0
        z = z0
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_sum += log_det
        return z, log_det_sum
class PlanarFlow(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.u = nn.Parameter(torch.randn(1, latent_dim))
        self.w = nn.Parameter(torch.randn(1, latent_dim))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        # Compute flow transformation
        linear = torch.matmul(z, self.w.t()) + self.b
        h = torch.tanh(linear)
        z_new = z + self.u * h

        # Compute log-determinant of Jacobian for loss
        psi = (1 - torch.tanh(linear) ** 2) * self.w  # [B, D]
        
        det_jacobian = torch.abs(1 + torch.matmul(psi, self.u.t()))
        log_det = torch.log(det_jacobian + 1e-8).squeeze(-1)

        return z_new, log_det
    
    

class Encoder_Transformer_VAE(nn.Module):
    def __init__(self, latent_dim, input_dim=2, model_dim=64,hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj1 = nn.Linear(input_dim, hidden_dim)
        self.input_proj2 = nn.Linear(hidden_dim, model_dim)
        self.selu = nn.SELU()        # instantiate ReLU module here

        self.pos_encoder = PositionalEncoding(model_dim)

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


        for name, param in self.named_parameters():
            self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())

        self.register_buffer("iteration", torch.tensor(1.0))

    def update_ema(self, alpha=0.1):
         with torch.no_grad():
             for name, param in self.named_parameters():
                 ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                 ema_param = ema_param.to(param.device)  # ensure same device
                 ema_param.mul_(1 - alpha).add_(alpha * param.data)
                 # Re-assign back the buffer (optional, since inplace)
                 setattr(self, f"{name.replace('.', '_')}_ema", ema_param)
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
        h = self.selu(self.input_proj1(inp))  # [B, T, model_dim]
        h = self.input_proj2(h)  # [B, T, model_dim]

        h = self.pos_encoder(h)   # [B, T, model_dim]

        h_enc = self.transformer_encoder(h, src_key_padding_mask=~mask if mask is not None else None)  # [B, T, model_dim]

        pooled = h_enc.mean(dim=1)  # [B, model_dim]
        mu_q = self.fc_mu2(self.selu(self.fc_mu1(pooled)))
        logvar_q = self.fc_logvar2(self.selu(self.fc_logvar1(pooled)))

        return mu_q, logvar_q