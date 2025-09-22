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
#from lib.utils.my_utils import *






class ODEFunc(nn.Module):
    def __init__(self, latent_dim, dim_parameter_encoder, hid_dim):
        super().__init__()
        self.dim_parameter_encoder = dim_parameter_encoder  
        self.dim_latent = latent_dim
        
        self.net = nn.Sequential(
            nn.Linear((latent_dim+dim_parameter_encoder+1), hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, latent_dim ))
        
    
        self.skip = nn.Sequential(
            nn.Linear((latent_dim+dim_parameter_encoder+1), latent_dim))
   
   


        self.log_sigma = nn.Parameter(torch.log(torch.tensor(0.05)))

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


    def get_sigma(self):
        # Ensure positivity with exp()
       return torch.exp(self.log_sigma)  
    #  return torch.tensor(0.01)  # or any fixed positive value
    def dirac_pulse(self, t, dose_times, dose_amounts, dose_mask):
        sigma = self.get_sigma()
        diff = t - dose_times  # [batch_size, num_doses]
        
       # dose_mask = dose_mask.unsqueeze(-1)  # [1920, 4, 1]

        # Only consider doses in the past 
        mask = (diff >= 0).float() * dose_mask.float()
        
        gauss  = torch.exp(-0.5 * (diff / sigma) ** 2) #torch.exp(-0.5 * (diff / sigma) ** 2) / (sigma * (2 * 3.1415) ** 0.5)
        gauss = gauss * mask
   
    
        dose_amounts = dose_amounts.squeeze(-1) if dose_amounts.dim() == 3 else dose_amounts  # [batch, num_doses]
        
        weighted = gauss * dose_amounts
         
        dose_signal = weighted.sum(dim=1, keepdim=True)

        return dose_signal


    def forward(self, t, x, dose_times,dose_amounts, dose_mask, keep_mask=None):
        batch_size = x.size(0)
        device = x.device
        if keep_mask is None:
            keep_mask = torch.ones(batch_size, 1, device=device)  # All rows are "kept"
    
   
         
     
        
        
        dose_input = self.dirac_pulse(t, dose_times, dose_amounts, dose_mask)  # [batch_size, 1]
        dose_amounts = dose_amounts.squeeze(-1)
        
      
 
        inp_dose2= torch.cat([x, dose_input* 10],dim=1)



        dxdt_deep = self.net(inp_dose2)
        dxdt_skip=self.skip(inp_dose2)

   
        #
        dxdt=dxdt_deep      +  dxdt_skip 
        
        dxdt = dxdt
        
        zero = torch.zeros(batch_size, self.dim_parameter_encoder, device=device)
        dxdt_concat = torch.cat([dxdt, zero], dim=1)
    
           
        return dxdt_concat




    
 

class TrainableNoise(nn.Module):
    """
    Estimates additive + optional proportional noise.
    If init_prop_std=0, proportional noise is disabled (no parameter, no gradients).
    """
    def __init__(self, size=1, init_add_std=0.1, init_prop_std=0.0):
        super().__init__()

        # always have additive noise
        self.log_sigma_add = nn.Parameter(
            torch.full((size,), torch.log(torch.tensor(init_add_std)))
        )

        # optional proportional noise
        if init_prop_std > 0:
            self.use_prop = True
            self.log_sigma_prop = nn.Parameter(
                torch.full((size,), torch.log(torch.tensor(init_prop_std)))
            )
        else:
            self.use_prop = False
            self.log_sigma_prop = None

    @property
    def sigma_add(self):
        return torch.exp(self.log_sigma_add)

    @property
    
    def sigma_prop(self):
        if self.log_sigma_prop is None:
            return torch.zeros_like(self.log_sigma_add)  # -> tensor([0.])
        return torch.exp(self.log_sigma_prop)


    def forward(self, x_pred):
        sigma_add = self.sigma_add
        if self.use_prop:
            sigma_prop_value = self.sigma_prop.view(
                *([1] * (x_pred.dim() - self.sigma_prop.dim())), *self.sigma_prop.shape
            )
            sigma_total = torch.sqrt(sigma_add**2 + (sigma_prop_value * x_pred)**2)
        else:
            sigma_total = sigma_add
    
        sigma_total = torch.clamp(sigma_total, min=1e-6)
    
        # Reparameterization trick: eps is independent, gradient flows through sigma_total
        eps = torch.randn_like(x_pred)
        return x_pred + sigma_total * eps  # gradients flow through sigma_total

    def nll(self, x_true, x_pred, mask=None, auc_tensor=None):
        sigma_add = self.sigma_add
        x_pred = torch.clamp(x_pred, min=-1e6, max=1e6)
      #  print(x_pred)
        if self.use_prop:
            sigma_prop_value = self.sigma_prop.view(
                *([1] * (x_pred.dim() - self.sigma_prop.dim())), *self.sigma_prop.shape
            )
            sigma_total = torch.sqrt(sigma_add**2 + (sigma_prop_value * x_pred)**2)
        else:
            sigma_total = sigma_add
    
        sigma_total = torch.clamp(sigma_total, min=1e-6)  # avoid div by zero
        
    
   
      #  x_true_log = torch.log(x_true+1e-8)
       # x_pred_log = torch.log(x_pred+1e-8)
            
      

       # nll_elementwise = (x_true_log - x_pred_log) ** 2 #0.5 * ((x_true_log - x_pred_log) / sigma_add) ** 2 + torch.log(sigma_add)
      #  mse_elementwise = (x_true_log - x_pred_log) ** 2
        
        
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
            nll_value = nll_elementwise.sum()
            mse_value = mse_elementwise.sum()
    
        # === Scale only MSE with AUC if provided ===
 
    
        return nll_value, mse_value


    def sample(self, x_pred, n_samples=1):
        sigma_add = self.sigma_add
    
        if self.use_prop:
            sigma_prop_value = self.sigma_prop.view(
                *([1] * (x_pred.dim() - self.sigma_prop.dim())), *self.sigma_prop.shape
            )
            sigma_total = torch.sqrt(sigma_add**2 + (sigma_prop_value * x_pred)**2)
        else:
            sigma_total = sigma_add
    
        if n_samples > 1:
            x_pred = x_pred.unsqueeze(0).expand(n_samples, *x_pred.shape)
            sigma_total = sigma_total.unsqueeze(0).expand(n_samples, *sigma_total.shape)
    
        eps = torch.randn_like(x_pred)
        return x_pred + sigma_total * eps








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
        self.flow = NormalizingFlow(latent_dim, num_flows=num_flow_layers)
      #  self.flow = ConditionalNormalizingFlow(latent_dim,cond_dim=2*model_dim,hidden_flow_dim=32, num_flows=num_flow_layers)
        self.latent_dim = latent_dim

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
        # Register EMA buffers
        for name, param in self.named_parameters():
            self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
        
        self.register_buffer("iteration", torch.tensor(1.0))
      #  self.logvar_scale = nn.Parameter(torch.tensor(0.0))  


     
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
    @torch.no_grad()
    def sample_prior(self, num_samples=10, device="cpu"):
        z0 = torch.randn(num_samples, self.latent_dim, device=device)
        z_k, _ = self.flow(z0)
        return z_k

                
    def sample(self, mu_q, logvar_q, num_samples=10):
        """
        Sample multiple latent vectors from q(z|x), transform with flow.
    
        Args:
            mu_q: [B, latent_dim] posterior mean
            logvar_q: [B, latent_dim] posterior log variance
            num_samples: number of samples per batch element
    
        Returns:
            z_k_samples: [B, num_samples, latent_dim] transformed latent samples
            log_det_samples: [B, num_samples] flow log determinants for each sample
        """
        B, D = mu_q.shape
    
        mu_exp = mu_q.unsqueeze(1).expand(B, num_samples, D)
        logvar_exp = logvar_q.unsqueeze(1).expand(B, num_samples, D)
        std_exp = torch.exp(0.5 * logvar_exp)
    
        eps = torch.randn_like(std_exp)
        z0_samples = mu_exp + std_exp * eps  # [B, num_samples, D]
    
        # Flatten batch and samples dims for flow input
        z0_flat = z0_samples.view(B * num_samples, D)
    
        # Apply flow to all samples
        z_k_flat, log_det_flat = self.flow(z0_flat)  # assume flow returns log_det per sample
    
        z_k_samples = z_k_flat.view(B, num_samples, D)
        log_det_samples = log_det_flat.view(B, num_samples)
    
        return z_k_samples, log_det_samples



    def forward(self,t, x, mask=None):
        """
        t: [B, T]
        x: [B, T]
        mask: optional [B, T] bool mask for valid positions
        """
        inp = torch.cat([t.unsqueeze(-1), x.unsqueeze(-1)], dim=-1)  # [B, T, 2]
      #  print("inp min/max/mean:", inp.min().item(), inp.max().item(), inp.mean().item())
        h1 = self.selu(self.input_proj1(inp))
      #  print("h1 min/max/mean:", h1.min().item(), h1.max().item(), h1.mean().item())
        h2 = self.input_proj2(h1)
      #  print("h2 min/max/mean:", h2.min().item(), h2.max().item(), h2.mean().item())
        h = self.pos_encoder(h2)
    #    print("h after pos_encoder min/max/mean:", h.min().item(), h.max().item(), h.mean().item())

        if mask is None:
       # All positions are valid if mask not provided
            mask = torch.ones(t.shape[0], t.shape[1], dtype=torch.bool, device=t.device)
            
        all_masked = (mask.sum(dim=1) == 0)  # shape [B], True if all time steps are masked

        h_enc = self.transformer_encoder(
               h, src_key_padding_mask=~mask if mask is not None else None
           )  # [B, T, model_dim]
        
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # avoid divide by zero
        pooled = (h_enc * mask.unsqueeze(-1)).sum(dim=1) / valid_counts
        pooled[all_masked] = 0.0  # ensure zero input for fully masked


        skip_data=self.raw_encoder(inp)
        skip_data_pooled = skip_data.mean(dim=1)  # [B, model_dim]
        cond_vec = torch.cat([pooled, skip_data_pooled], dim=-1)  # [B, model_dim + raw_feat_dim]
        
      
        
        mu_q = torch.clamp(self.fc_mu2(self.selu(self.fc_mu1(pooled))), -1e2, 1e2)
        logvar_q = torch.clamp(self.fc_logvar2(self.selu(self.fc_logvar1(pooled))), -10, 10)

        
        mu_q[all_masked] = 0.0
        logvar_q[all_masked] = 0.0  # logvar=0 → std=1
        
        logvar_q = torch.clamp(logvar_q, -10, 10)  # choose bounds to keep std reasonable

        std = torch.exp(0.5 * logvar_q)
        eps = torch.randn_like(std)
        z0 = mu_q + eps * std
        z0 = torch.clamp(z0, -1e2, 1e2)

        if self.training:
               # z_k, log_det = self.flow(z0, cond_vec)  # pass both z0 and pooled conditioning vector
                z_k, log_det = self.flow(z0)  # pass both z0 and pooled conditioning vector
                z_k_0, _ =self.flow(z0)
        else:
                z_k = mu_q
                log_det = torch.zeros(z_k.size(0), device=z_k.device)
                z_k_0 = torch.zeros_like(z0)
        
    
        return z_k_0, z_k, mu_q, logvar_q, log_det

   
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
        
        # Small weights, close to zero => small transformation
        self.u = nn.Parameter(torch.randn(1, latent_dim) * 0.01)
        self.w = nn.Parameter(torch.randn(1, latent_dim) * 0.01)

        self.b = nn.Parameter(torch.zeros(1))

        

    def forward(self, z):
        w = self.w  # shape [1, D]
        u = self.u
        b = self.b
    
        # 1. Enforce invertibility: u_hat
        wu = torch.matmul(w, u.t())  # shape [1, 1]
        m = -1 + F.softplus(wu)
        u_hat = u + (m - wu) * w / (torch.norm(w, p=2) ** 2 + 1e-8)
    
        # 2. Flow transform
        linear = torch.matmul(z, w.t()) + b  # shape [B, 1]
        h = torch.tanh(linear)               # shape [B, 1]
        z_new = z + u_hat * h                # shape [B, D]
    
        # 3. Compute log-det-Jacobian
        psi = (1 - h ** 2) * w               # [B, D]
        det_jacobian = 1 + torch.matmul(psi, u_hat.t())  # [B, 1]
        log_det = torch.log(torch.abs(det_jacobian) + 1e-8).squeeze(-1)  # [B]
    
        return z_new, log_det
    
class PlanarFlow2(nn.Module):
   def __init__(self, latent_dim):
       super().__init__()
       self.latent_dim = latent_dim

       #epsilon = 1e-20
       # Reasonable small stddev for stable flow init
       std = 0.01
       self.u = nn.Parameter(torch.randn(1, latent_dim) * std)
       self.w = nn.Parameter(torch.randn(1, latent_dim) * std)
       self.b = nn.Parameter(torch.zeros(1))


       # Learnable raw mixture weight parameter (unconstrained)
       self.mix_weight_raw = nn.Parameter(torch.tensor(1.0))  # initialized to 0 => sigmoid(0)=0.5

   def forward(self, z):
       w = self.w  # [1, D]
       u = self.u
       b = self.b

       # Enforce invertibility (u_hat)
       wu = torch.matmul(w, u.t())  # [1,1]
       m = -1 + F.softplus(wu)
       u_hat = u + (m - wu) * w / (torch.norm(w, p=2) ** 2 + 1e-8)

       linear = torch.matmul(z, w.t()) + b  # [B, 1]

       # Learnable mixture weight alpha in [0,1]
       alpha = torch.sigmoid(self.mix_weight_raw)

       h_softplus = F.softplus(linear)
       h_tanh = torch.tanh(linear)
       h = alpha * h_softplus + (1 - alpha) * h_tanh  # [B, 1]

       z_new = z + u_hat * h  # [B, D]

       # Derivative dh/dlinear
       d_softplus = torch.sigmoid(linear)
       d_tanh = 1 - h_tanh ** 2
       dh = alpha * d_softplus + (1 - alpha) * d_tanh  # [B, 1]

       psi = dh * w  # [B, D]

       det_jacobian = 1 + torch.matmul(psi, u_hat.t())  # [B, 1]
       log_det = torch.log(torch.abs(det_jacobian) + 1e-8).squeeze(-1)  # [B]

       return z_new, log_det  
   
    

class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        
        # Define the entire network as a single sequential
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1)#,
            #nn.ReLU() # ensures positive output
        )

        # Initialize the last layer
      #  nn.init.zeros_(self.net[-2].weight)  # second-to-last layer is last Linear
        #nn.init.constant_(self.net[-2].bias, -1)
        last_linear = self.net[4]  # last Linear layer before Softplus
        nn.init.zeros_(last_linear.weight)  # small weights -> output ~0
        nn.init.constant_(last_linear.bias, 1.0)  # bias ~0
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
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU() # ensures positive output
        )


    def forward(self, x):
        z0 = self.net(x)
        return z0
    
    



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
 