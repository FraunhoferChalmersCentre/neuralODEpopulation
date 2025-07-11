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


from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import math
import os

import torch.nn.functional as F
from lib.utils.my_utils_parallel import *


class ODEWrapper(nn.Module):
    def __init__(self, func, dose_times, dose_amounts, dose_mask):
        super().__init__()
        self.func = func
        self.dose_times = dose_times          # [batch, max_len]
        self.dose_amounts = dose_amounts      # [batch, max_len]
        self.dose_mask = dose_mask            # [batch, max_len]
  

    def forward(self, t, x):
        # Pass all dose info separately to ODEFunc
        
        return self.func(t, x, self.dose_times, self.dose_amounts, self.dose_mask)
   




class DoseAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
        
        # Learned default output when no valid doses
        self.default_output = nn.Parameter(torch.zeros(hidden_dim))
        #self.time_bias_scale = nn.Parameter(torch.tensor(-7.0))  # initialized to -1.0
        self.time_bias_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.SELU(),
            nn.Linear(16, 1)
        )
        nn.init.constant_(self.time_bias_net[-1].bias, -7.0)
        nn.init.zeros_(self.time_bias_net[-1].weight)  # optionally zero the weights

    def forward(self, gru_outputs, mask,T2_all, x):
        # gru_outputs: [batch, seq_len, hidden_dim]
        # mask: [batch, seq_len] bool tensor, True for valid dose times

        batch_size, seq_len, hidden_dim = gru_outputs.size()
        
        Q = self.query(gru_outputs)  # [B, T, H]
        K = self.key(gru_outputs)    # [B, T, H]
        V = self.value(gru_outputs)  # [B, T, H]

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [B, T, T]

        # Create mask for attention scores: expand dims
        attn_mask = mask.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, T]
        
        # Identify batch elements with all False masks along last dim (no valid doses)
        no_valid_mask = (~mask).all(dim=1)  # [B] True if all False
        
        # Mask invalid positions with -inf
        scores = scores.masked_fill(~attn_mask, float('-inf'))
        if T2_all is not None:
            # x: [B, dim_parameter_encoder]
            # expand to [B, T, dim_parameter_encoder]
            x_expanded = x.unsqueeze(1).expand(-1, T2_all.size(1), -1)
        
            # Pass only x_expanded (no T_expanded)
            time_bias = self.time_bias_net(x_expanded)  # [B, T, 1]
        
            # Transpose if needed for broadcasting in scores addition
            time_bias = time_bias.transpose(1, 2)  # [B, 1, T]
        
            scores = scores + time_bias



        
        attn_weights = torch.zeros_like(scores)

        # For batches with valid masks, compute softmax normally
        valid_batches = ~no_valid_mask
        if valid_batches.any():
            attn_weights[valid_batches] = F.softmax(scores[valid_batches], dim=-1)
        
        # For batches with no valid doses, attention weights remain zeros
        
        attn_output = torch.bmm(attn_weights, V)  # [B, T, H]

        # Average over sequence dim for output
        output = attn_output.mean(dim=1)  # [B, H]

        # Replace outputs for no_valid_mask batches with default learned embedding
        if no_valid_mask.any():
            output[no_valid_mask] = self.default_output.unsqueeze(0).expand(no_valid_mask.sum(), -1)
        
        return output, attn_weights



    
# ---- ODE ----
class ODEFunc(nn.Module):
    def __init__(self, latent_dim, dim_parameter_encoder, hid_dim):
        super().__init__()
        self.dim_parameter_encoder = dim_parameter_encoder  
        self.dim_latent = latent_dim
        self.net = nn.Sequential(
            nn.Linear((latent_dim+dim_parameter_encoder +3), hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, latent_dim ))
        
        self.skip = nn.Sequential(
            nn.Linear((latent_dim+dim_parameter_encoder +3), hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, latent_dim ))
        
        self.dose = nn.Sequential(
            nn.Linear(3, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, latent_dim ))

        self.attention = DoseAttention(hidden_dim=2)

        self.gru_output_proj = nn.Sequential(
                    nn.Linear(8, 4),
              nn.SELU(),
              nn.Linear(4, 2),
                      )
        self.gru = nn.GRU(input_size=1, hidden_size=8, batch_first=True)
 
        self.dose_proj = nn.Linear(8, 2)  # project to latent_dim
    

        self.linear1 = nn.Sequential(nn.Linear(1, 4), nn.SELU(), nn.Linear(4, 1))
        self.linear2 = nn.Sequential(nn.Linear(1, 4), nn.SELU(), nn.Linear(4, 1))
   

   
    def compute_T_all(self, t, dose_times, dose_mask):
        t_scalar = t.item()
        
        # Squeeze last dim if it's singleton
        if dose_times.dim() == 3 and dose_times.size(-1) == 1:
            dose_times = dose_times.squeeze(-1)  # shape becomes [10,4]
        
        delta_t = t_scalar - dose_times  # [10,4]
        T_all = torch.where(dose_mask & (dose_times <= t_scalar), delta_t, torch.zeros_like(delta_t))
        return T_all  # shape: [10,4]
    
    def T(self, t, dose_times):
        t_scalar = t.item()
        relevant_doses = dose_times[dose_times <= t_scalar]
        if len(relevant_doses) == 0:
            return t_scalar
        else:
            return t_scalar - relevant_doses.max().item()


  

    def forward(self, t, x, dose_times,dose_amounts, dose_mask):
        batch_size = x.size(0)
        device = x.device
    
        t_scalar = t.item()
     
        T_all = self.T(t, dose_times)  # time since each dose
        
        T2_all=self.compute_T_all(t, dose_times,dose_mask) 

        new_dose_mask = (T2_all > 0)

     
        dose_amounts_squeezed = dose_amounts.squeeze(-1)  # [10, 4]
     
        gru_input = T2_all.unsqueeze(-1) # , dose_amounts_squeezed], dim=-1)  # [10, 4, 2]

        gru_output, _ = self.gru(gru_input)  # [B, T, H]
 
        projected = self.gru_output_proj(gru_output)  # [B, T, 2]
        
        x_param = x[:, self.dim_latent+2:]  # [B, dim_parameter_encoder]


        attn_output, attn_weights = self.attention(projected, new_dose_mask,T2_all,x_param)
        
        agg = attn_output  # [B, 2], dose embedding weighted by attention


        out1 = self.linear1(agg[:, 0].unsqueeze(-1))  # [batch, 1]
        out2 = self.linear2(agg[:, 1].unsqueeze(-1))  # [batch, 1]

        final = torch.cat([out1, out2], dim=1)  # [batch, 2]
     
        t_tensor = torch.full((batch_size, 1), t.item(), device=x.device)

        dose_single=dose_amounts_squeezed[:, 0]
        
        T_tensor = torch.full((batch_size, 1), T_all, device=x.device)
        

        inp = torch.cat([x,dose_single.unsqueeze(1),  final], dim=1)
        inp2 = torch.cat([dose_single.unsqueeze(1),  final], dim=1)

        dxdt = self.net(inp)
        dxdt_dose= self.dose(inp2)
        dxdt_skip=self.skip(inp)
        zero = torch.zeros(batch_size, self.dim_parameter_encoder, device=device)
        dxdt_concat = torch.cat([dxdt+dxdt_skip, zero], dim=1)  # shape: [batch_size, latent_dim]
     #   print(t_scalar)
     #   print(new_dose_mask)
     
     #   print(dose_single.unsqueeze(1))
     #   print(attn_output)
     #   print(attn_weights)
        return dxdt_concat

   




    
    
class TrainableNoise(nn.Module):
    def __init__(self, dataset, size=1, init_add_std=0.1, init_prop_std=0.001):
        super().__init__()
        init_log_sigma_add = torch.log(torch.tensor(init_add_std))
        init_log_sigma_prop = torch.log(torch.tensor(init_prop_std))
        self.conc_mean, self.conc_std = dataset.conc_mean, dataset.conc_std
        
        self.log_sigma_add = nn.Parameter(torch.full((size,), init_log_sigma_add))
       # self.log_sigma_prop = nn.Parameter(torch.full((size,), init_log_sigma_prop))
        self.register_buffer('log_sigma_prop', torch.full((size,), init_log_sigma_prop))  # Now non-trainable

    def forward(self, x_pred):
        # Return a single sample from the noise distribution
        return self.sample(x_pred, n_samples=1).squeeze(0)

    def nll(self, x_true, x_pred, mask=None):
        sigma_add = torch.exp(self.log_sigma_add)
        sigma_prop = torch.exp(self.log_sigma_prop)
    
        sigma_add = sigma_add.view(*([1] * (x_pred.dim() - sigma_add.dim())), *sigma_add.shape)
        sigma_prop = sigma_prop.view(*([1] * (x_pred.dim() - sigma_prop.dim())), *sigma_prop.shape)
    
        sigma_total = torch.sqrt(sigma_add ** 2 + (sigma_prop * x_pred) ** 2 + 1e-6)
    
        nll_elementwise = 0.5 * ((x_true - x_pred) / sigma_total) ** 2 + torch.log(sigma_total)
        mse_elementwise = (x_true - x_pred) ** 2
    
        if mask is not None:
            if nll_elementwise.dim() > mask.dim():
                mask = mask.unsqueeze(-1)
            nll_elementwise = nll_elementwise * mask
            mse_elementwise = mse_elementwise * mask
            total_valid = mask.sum().clamp(min=1)
            nll_value = nll_elementwise.sum() / total_valid
            mse_value = mse_elementwise.sum() / total_valid
        else:
            nll_value = nll_elementwise.mean()
            mse_value = mse_elementwise.mean()
    
        return nll_value, mse_value


    
            

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
                
class NormalizingFlow(nn.Module):
    def __init__(self, latent_dim, num_flows=2):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(num_flows)])

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
        


class Encoder_Transformer_NF(nn.Module):
    def __init__(self, latent_dim, input_dim=2, model_dim=64,hidden_dim=64, hidden_flow_dim=32, num_heads=4, num_layers=2, num_flow_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj1 = nn.Linear(input_dim, hidden_dim)
        self.input_proj2 = nn.Linear(hidden_dim, model_dim)
        self.selu = nn.SELU()        # instantiate ReLU module here
        self.flow = NormalizingFlow(latent_dim, num_flows=2)
        #self.flow = ConditionalNormalizingFlow(latent_dim,cond_dim=2*model_dim,hidden_flow_dim=32, num_flows=num_flow_layers)

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
       # skip_data=self.raw_encoder(inp)
        #skip_data_pooled = skip_data.mean(dim=1)  # [B, model_dim]
       # cond_vec = torch.cat([pooled, skip_data_pooled], dim=-1)  # [B, model_dim + raw_feat_dim]

        mu_q = self.fc_mu2(self.selu(self.fc_mu1(pooled)))
        logvar_q = self.fc_logvar2(self.selu(self.fc_logvar1(pooled)))
        
        std = torch.exp(0.5 * logvar_q)
        eps = torch.randn_like(std)
        z0 = mu_q + eps * std
    
        if self.training:
               # z_k, log_det = self.flow(z0, cond_vec)  # pass both z0 and pooled conditioning vector
                z_k, log_det = self.flow(z0)  # pass both z0 and pooled conditioning vector

        else:
                z_k = mu_q
                log_det = torch.zeros(z_k.size(0), device=z_k.device)

    
        return z_k, mu_q, logvar_q, log_det

   
   
    
    
class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.relu = nn.SELU()        # instantiate ReLU module here
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu2 = nn.ReLU()        # instantiate ReLU module here
       

        nn.init.zeros_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, -1)

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
        self.SELU = nn.SELU()        # instantiate ReLU module here
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        
        for name, param in self.named_parameters():
            self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
            
        self.register_buffer("iteration", torch.tensor(1.0))

    def forward(self, x):
        h = self.SELU(self.fc1(x))   # call the instance
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