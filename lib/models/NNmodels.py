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

import torch
import torch.nn as nn

import math

import torch.nn.functional as F


class ODEFunc(nn.Module):
    """
    ODE function with optional EMA for parameters, dose handling, and skip connection.
    Supports multiple event types via `evid` (bolus=1, infusion=2, observation=0).
    """
    def __init__(self, latent_dim, dim_parameter_encoder, hid_dim, drug_dim):
        super().__init__()
        self.drug_dim=drug_dim
        self.dim_latent = latent_dim
        self.dim_parameter_encoder = dim_parameter_encoder
        self.fc1 = nn.Linear(1, hid_dim)
        self.SELU = nn.SELU()
        
        # Two separate layers for mean and log variance
        self.fc_mu = nn.Linear(hid_dim, latent_dim)
        self.fc_logvar = nn.Linear(hid_dim, latent_dim)
        #self.prior_mu = nn.Parameter(torch.zeros(latent_dim))
        
        
        
        # Deep network
        self.net = nn.Sequential(
            nn.Linear(latent_dim + dim_parameter_encoder+drug_dim, hid_dim),
             nn.SELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SELU(),
           nn.Linear(hid_dim, latent_dim)
        )
        
        self.drug = nn.Sequential(
            nn.Linear(drug_dim, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, drug_dim),
      
        )


        # Skip connection
        self.skip = nn.Sequential(
            nn.Linear(latent_dim + dim_parameter_encoder+drug_dim, latent_dim)
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

    def get_sigma(self):
        return torch.exp(self.log_sigma)
    

    def make_drug_masks(self, evid, n_drugs):
        """
        Create one mask per drug based on EVID.
        """
        n_drugs = int(n_drugs)
        masks = []
        for drug_id in range(1, n_drugs + 1):
            mask = (evid.long() == drug_id).float()
            masks.append(mask)
        return masks

    # def bolus_pulse(self, t, dose_times, dose_amounts, dose_mask, evid, n_drugs):
    #     """
    #     Compute dose signal per drug using Gaussian pulses.
    #     Zero contribution if no doses exist for that individual/drug.
    #     Output: [batch, n_drugs]
    #     """
    #     sigma = self.get_sigma()
    #     batch_size = evid.size(0)
    #     n_times = t.size(1) if t.dim() > 1 else 1
    #     device = t.device
    #     dose_signal = torch.zeros(batch_size, n_drugs, device=device)
    
    #     # Ensure t has shape [batch, num_times]
    #     if t.dim() == 1:
    #         t = t.unsqueeze(0).repeat(batch_size, 1)  # [batch, num_times]
    
    #     for drug_id in range(1, n_drugs + 1):
    #         # Mask doses for this drug
    #         mask = (evid == drug_id)  # [batch, num_doses]
    #         masked_times = dose_times * mask.float()
    #         masked_amounts = dose_amounts * mask.float()
    
    #         # Identify which batch elements actually have doses
    #         has_dose = mask.any(dim=1)  # [batch]
    
    #         if not has_dose.any():
    #             continue
    
    #         # Compute Gaussian pulse
    #         diff = t.unsqueeze(-1) - masked_times.unsqueeze(1)  # [batch, num_times, num_doses]
    #         gauss = torch.exp(-(diff / sigma)**2) * masked_amounts.unsqueeze(1)
    #         mask_nonnegative = diff >= 0
    #         pulse = gauss * mask_nonnegative.float()
    
    #         # Zero out contributions for batches with no doses
    #         pulse = pulse * has_dose.unsqueeze(-1).unsqueeze(-1).float()
    
    #         # Sum over doses
    #         dose_signal[:, drug_id - 1] = pulse.sum(dim=-1).squeeze()
         
    #     return dose_signal


        
    def bolus_pulse(self, t, dose_times, dose_amounts, dose_mask, evid, n_drugs):
        """
        Compute dose signal per drug using interleaved dose arrays and EVID codes.
        """
        sigma=self.get_sigma()
        batch_size = evid.size(0)
        device = t.device
       # dose_signal = torch.zeros(batch_size, n_drugs, device=device)
        
        diff = t.unsqueeze(-1) - dose_times  # [batch, num_times, num_doses]
        mask_nonnegative = diff >= 0

        gauss = torch.exp(-(diff / sigma)**2) * dose_amounts
   
        
        
     
        dose_signal=gauss*mask_nonnegative
        #print(dose_signal.sum(1),t )
      
        return dose_signal.sum(1)       
                
                    
            
        


    # === Forward pass ===
    def forward(self, t, x, dose_times, dose_amounts, evid, dose_mask, keep_mask=None):
        batch_size = x.size(0)
        device = x.device
        
     
        
        if keep_mask is None:
            keep_mask = torch.ones(batch_size, 1, device=device)

        # Compute dose input (bolus + infusion)
        bolus_signal = self.bolus_pulse(t, dose_times, dose_amounts, dose_mask, evid,self.drug_dim)
        dose_input = bolus_signal
        
        # Concatenate latent state and dose
      
        if dose_input.dim() == 1:
         dose_input = dose_input.unsqueeze(1)
       
        t=t+torch.zeros(batch_size, 1, device=device)
        x[:, -1] = 0
      
        inp1 = torch.cat([x, self.drug (dose_input)], dim=1)
      #  print(x)
    #    inp1 = torch.cat([x, bolus_signal], dim=1)
        #inp1 = x
      #  inp2 = torch.cat([x, dose_input], dim=1)
        
        # Deep network + skip connection
        dxdt_deep = self.net(inp1)
       # dxdt_skip = self.skip(inp1)
        dxdt = dxdt_deep
       
        # Concatenate zeros for parameter dimensions
        zeros_param = torch.zeros(batch_size, self.dim_parameter_encoder, device=device)
        dxdt_concat = torch.cat([dxdt, zeros_param], dim=1)
      #  print(dxdt_concat)
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
        # weights = torch.ones_like(nll_elementwise)
        
        # weights[:, 0] = nll_elementwise.size(1) - 1
        
        # orig_row_sum = nll_elementwise.sum(dim=1) 
        # weighted_row_sum = (weights * nll_elementwise).sum(dim=1)  # [100]
        # eps = 1e-12
        # scale = orig_row_sum / (weighted_row_sum + eps)  # [100]
     
        # nll_elementwise=nll_elementwise*weights
     
        
        #L1 for masked/dropped points
        if replace_mask is not None:
            dropout_mask = replace_mask.bool()
            if dropout_mask.any():
                L1_nll_elementwise = (2 ** 0.5) * (x_true - x_pred).abs() / sigma_total + torch.log(2 ** 0.5 * sigma_total)
                L1_elementwise = (x_true - x_pred).abs()
                orig_row_sum = nll_elementwise.sum(dim=1) 
                # weighted_row_sum = (weights * nll_elementwise).sum(dim=1)  # [100]
                # eps = 1e-12
                # scale = orig_row_sum / (weighted_row_sum + eps)  # [100]
             
                # nll_elementwise=nll_elementwise*weights
                
                
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
                


import torch.nn.init as init


class Encoder_Transformer_Full(nn.Module):
    def __init__(self, latent_dim, parameter_dim, input_dim=2, model_dim=64,
                 hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1,
                 cov_diag_epsilon=1e-5, learn_prior_mean=True, learn_prior_covariance=True,
                 diagonal_only=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.parameter_dim = parameter_dim
        total_parameters=parameter_dim
        self.total_parameters=total_parameters
        self.cov_diag_epsilon = cov_diag_epsilon
        self.diagonal_only = diagonal_only
        self.selu = nn.SELU()
        self.learn_prior_covariance = learn_prior_covariance
        self.learn_prior_mean = learn_prior_mean

        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, model_dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads,
            dim_feedforward=model_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Posterior mean
        self.fc_mu1 = nn.Linear(model_dim, hidden_dim)
        self.fc_mu2 = nn.Linear(hidden_dim, total_parameters//2)
        
        self.fc_mu11 = nn.Linear(model_dim, hidden_dim)
        self.fc_mu22 = nn.Linear(hidden_dim, total_parameters//2)
        
                # initialization
        # init.normal_(self.fc_mu1.weight, mean=0.0, std=0.05)
        # init.normal_(self.fc_mu2.weight, mean=0.0, std=0.05)
        # init.normal_(self.fc_mu11.weight, mean=0.0, std=0.05)
        # init.normal_(self.fc_mu22.weight, mean=0.0, std=0.05)
        
        # self.fc_mu1.bias.data.zero_()
        # self.fc_mu2.bias.data.zero_()
        # self.fc_mu22.bias.data.zero_()
        # self.fc_mu11.bias.data.zero_()
        
        # Posterior covariance
        self.fc_A1 = nn.Linear(model_dim, hidden_dim)
        self.fc_A2 = nn.Linear(hidden_dim, total_parameters * total_parameters)
        nn.init.zeros_(self.fc_A2.weight)
        with torch.no_grad():
            self.fc_A2.bias.copy_(torch.eye(total_parameters).flatten())
        
        nn.init.normal_(self.fc_A2.weight, std=0.1)
     
        # NODE initial condition
        self.z0_mu = nn.Parameter(torch.zeros(latent_dim))
       # self.z0_mu = nn.Linear(1, latent_dim)


        
        self.z0_iiv = nn.Sequential(
            nn.Linear(parameter_dim+latent_dim, latent_dim)
        )
        
 
        self.fc_first1 = nn.Linear(2, 1)
     #   self.fc_first2 = nn.Linear(model_dim, 1)

        # Learnable prior parameters
        if learn_prior_mean:
           self.mu_p = nn.Parameter(torch.zeros(total_parameters))
        else:
           self.register_buffer("mu_p", torch.zeros(total_parameters))

        if learn_prior_covariance:
           init_std = 0.01
           self.prior_A = nn.Parameter(
               torch.eye(total_parameters) + torch.randn(total_parameters, total_parameters) * init_std
           )
        else:
           self.register_buffer("prior_A", torch.zeros(total_parameters, total_parameters))

       # Track if priors are currently frozen
        self._prior_frozen = False
       
        for name, param in self.named_parameters():
            self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
        self.register_buffer("iteration", torch.tensor(1.0))
        
    def update_ema(self, alpha=0.1):
        """Update EMA of all parameters."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                ema_param = ema_param.to(param.device)
                ema_param.mul_(1 - alpha).add_(alpha * param.data)
                setattr(self, f"{name.replace('.', '_')}_ema", ema_param)
            self.iteration += 1.0

    def apply_ema_weights(self):
        """Replace model weights with their EMA values."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                param.data.copy_(ema_param)
        
        
    def get_prior(self, batch_size=None):
        D = self.total_parameters
        device = self.mu_p.device
    
        if self.learn_prior_covariance:
            # Full or learned covariance
            Sigma_p = self.prior_A @ self.prior_A.T + self.cov_diag_epsilon * torch.eye(D, device=device)
        else:
            # Fixed covariance (identity)
            if self.diagonal_only:
                # Independent Gaussian prior
                Sigma_p = torch.eye(D, device=device)
            else:
                # Full (but fixed) covariance
                Sigma_p = torch.eye(D, device=device)
    
        L_p = torch.linalg.cholesky(Sigma_p)
    
        mu_p = self.mu_p
        if batch_size is not None:
            mu_p = mu_p.unsqueeze(0).expand(batch_size, -1)
            L_p = L_p.unsqueeze(0).expand(batch_size, -1, -1)
    
        return mu_p, L_p

    def freeze_prior_until(self, current_epoch, freeze_epochs=10):
        """Freeze the prior for the first `freeze_epochs` epochs."""
        should_freeze = current_epoch < freeze_epochs

        if should_freeze and not self._prior_frozen:
            if hasattr(self, "mu_p") and isinstance(self.mu_p, nn.Parameter):
                self.mu_p.requires_grad_(False)
            if hasattr(self, "prior_A") and isinstance(self.prior_A, nn.Parameter):
                self.prior_A.requires_grad_(False)
            self._prior_frozen = True

        elif not should_freeze and self._prior_frozen:
            if hasattr(self, "mu_p") and isinstance(self.mu_p, nn.Parameter):
                self.mu_p.requires_grad_(True)
            if hasattr(self, "prior_A") and isinstance(self.prior_A, nn.Parameter):
                self.prior_A.requires_grad_(True)
            self._prior_frozen = False
            
    def forward(self, t, x, dose, only_median=False, mask=None):
        B, T = t.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=t.device)
            
          # ===== Split the sequence =====
         # ===== Split inputs =====
        t_first, x_first = t[:, :1], x[:, :1]
        t_rest,  x_rest  = t[:, 1:], x[:, 1:]
        mask_rest = mask[:, 1:]
    
        # ===== Encode first timepoint =====
        inp_first = torch.stack([t_first, x_first], dim=-1).squeeze(1)  # [B, 2]
        #h_first = self.selu(self.fc_first1(inp_first))
        
      
        
        mu_q_1 = self.fc_first1(inp_first)
        
  
        
        # ===== Encode rest of sequence =====
        inp_rest = torch.stack([t_rest, x_rest], dim=-1)      # [B, T-1, 2]
        h_rest = self.input_proj(inp_rest)
        h_rest = self.pos_encoder(h_rest)
        h_rest = self.transformer(h_rest, src_key_padding_mask=~mask_rest)
        valid_counts = mask_rest.sum(dim=1, keepdim=True).clamp(min=1)
        pooled_rest = (h_rest * mask_rest.unsqueeze(-1)).sum(dim=1) / valid_counts
        pooled_rest += torch.randn_like(pooled_rest) * 1e-5
    
        # ===== Encode all timepoints (for A_q) =====
        inp_all = torch.stack([t, x], dim=-1)                 # [B, T, 2]
        h_all = self.input_proj(inp_all)
        h_all = self.pos_encoder(h_all)
        h_all = self.transformer(h_all, src_key_padding_mask=~mask)
        valid_counts_all = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled_all = (h_all * mask.unsqueeze(-1)).sum(dim=1) / valid_counts_all
    
        # ===== Posterior means (two independent paths) =====
       # mu_q_1 = self.fc_mu2(self.selu(self.fc_mu1(pooled_first)))  # [B, D]
        mu_q_2 = self.fc_mu22(self.selu(self.fc_mu11(pooled_rest)))   # [B, D]
        mu_q = torch.cat([mu_q_1, mu_q_2], dim=-1)                  # [B, 2*D]
        
    
        # ===== Posterior covariance (from full pooled encoding) =====
        A_q_flat = self.fc_A2(self.selu(self.fc_A1(pooled_all)))    # [B, D_flat]
        A_q = A_q_flat.view(B, self.total_parameters, self.total_parameters)

        if self.diagonal_only:
            # Keep only diagonal elements
            A_q = torch.diag_embed(torch.diagonal(A_q, dim1=-2, dim2=-1))

        Sigma_q = torch.matmul(A_q, A_q.transpose(-1, -2)) + self.cov_diag_epsilon * torch.eye(self.total_parameters, device=A_q.device)
        L_q = torch.linalg.cholesky(Sigma_q)

        # Sample latent
        eps = torch.randn(B, self.total_parameters, device=mu_q.device)
        k_params = mu_q + torch.einsum("bij,bj->bi", L_q, eps)

        # NODE initial condition
        # if dose is not None and dose.shape[1] > 0:
        #     z0_median = self.z0_mu(dose[:, :1])  # take first column
        # else:
        #     z0_median = self.z0_mu(torch.zeros(B, 1, device=t.device))  # fallback to zeros
        z0_median = self.z0_mu.unsqueeze(0).expand(B, -1)  # shape: (B, latent_dim)

        if only_median:
            k_params=k_params*0
       
        inp1 = torch.cat([z0_median, k_params], dim=1)
        inp1[:, -2] = 0
        z0=self.z0_iiv(inp1)

        # Prior
        mu_p, L_p = self.get_prior(batch_size=B)

        return k_params, z0, mu_q, L_q, mu_p, L_p












    




class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.latent_dim=latent_dim
        
        # Define the network
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1))#,
             # nn.SELU(),
             # nn.Linear(hidden_dim, hidden_dim),
             # nn.SELU(),
             # nn.Linear(hidden_dim, 1)        )
             
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.5)
                nn.init.constant_(m.bias, 0.0)     

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

