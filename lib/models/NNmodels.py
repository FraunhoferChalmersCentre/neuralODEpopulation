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
       
      #  mask = torch.ones_like(x)
       # mask[:, 2] = 0
       # print(mask)
      #  x=x*mask
     #   print(x)
        t = torch.full((batch_size, 1), t, device=x.device)
      

        inp1 = torch.cat([x, self.drug (dose_input)], dim=1)
    #    inp1 = torch.cat([x, bolus_signal], dim=1)
        #inp1 = x
      #  inp2 = torch.cat([x, dose_input], dim=1)
        
        # Deep network + skip connection
        dxdt_deep = self.net(inp1)
        dxdt_skip = self.skip(inp1)
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
                





class Encoder_Transformer_Full(nn.Module):
    def __init__(self, latent_dim, parameter_dim, input_dim=2, model_dim=64,
                 hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1,
                 cov_diag_epsilon=1e-5, learn_prior_mean=True, learn_prior_covariance=True,
                 diagonal_only=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.parameter_dim = parameter_dim
        total_parameters=parameter_dim + latent_dim
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
        self.fc_mu2 = nn.Linear(hidden_dim, total_parameters)
   
        # Posterior covariance
        self.fc_A1 = nn.Linear(model_dim, hidden_dim)
        self.fc_A2 = nn.Linear(hidden_dim, total_parameters * total_parameters)
        nn.init.zeros_(self.fc_A2.weight)
        with torch.no_grad():
            self.fc_A2.bias.copy_(torch.eye(total_parameters).flatten())
        
               
        # NODE initial condition
        self.z0_mu =nn.Linear(1, latent_dim) 
        nn.init.constant_(self.z0_mu.weight, 1)
        nn.init.constant_(self.z0_mu.bias, 0.0)
        
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


    def forward(self, t, x, dose, mask=None):
        B, T = t.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=t.device)

        # Encode
        inp = torch.stack([t, x], dim=-1)
        h = self.input_proj(inp)
        h = self.pos_encoder(h)
        h_enc = self.transformer(h, src_key_padding_mask=~mask)

        # Masked pooling
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (h_enc * mask.unsqueeze(-1)).sum(dim=1) / valid_counts
        pooled += torch.randn_like(pooled)  # tiny noise to break symmetry

        # Posterior mean
        mu_q = self.fc_mu2(self.selu(self.fc_mu1(pooled)))

        # Posterior covariance
        A_q_flat = self.fc_A2(self.selu(self.fc_A1(pooled)))
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
        #z0 = self.z0_mu.unsqueeze(0).expand(B, -1)
      #  z0 = self.z0_mu(dose[:, :1])

        # Prior
        mu_p, L_p = self.get_prior(batch_size=B)

        return k_params, mu_q, mu_q, L_q, mu_p, L_p















# class Encoder_Transformer_Full(nn.Module):
#     def __init__(self, latent_dim, parameter_dim, input_dim=2, model_dim=64,
#                  hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1,
#                  cov_diag_epsilon=1e-5, learn_prior_mean=True,
#                  learn_prior_covariance=True):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.cov_diag_epsilon = cov_diag_epsilon
#         self.selu = nn.SELU()

#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(model_dim)

#         # Input projection
#         self.input_proj = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.SELU(),
#             nn.Linear(hidden_dim, model_dim)
#         )

#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=model_dim, nhead=num_heads,
#             dim_feedforward=model_dim * 4, dropout=dropout, batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         # Output layers for posterior mean
#         self.fc_mu1 = nn.Linear(model_dim, hidden_dim)
#         self.fc_mu2 = nn.Linear(hidden_dim, latent_dim)
#         nn.init.zeros_(self.fc_mu2.weight)
#         nn.init.zeros_(self.fc_mu2.bias)

        # Output layers for posterior full covariance (A_q)
        # self.fc_A1 = nn.Linear(model_dim, hidden_dim)
        # self.fc_A2 = nn.Linear(hidden_dim, latent_dim * latent_dim)
        # nn.init.zeros_(self.fc_A2.weight)
        # nn.init.zeros_(self.fc_A2.bias)
        # eye = torch.eye(latent_dim).flatten()  # shape: [latent_dim*latent_dim]

        # with torch.no_grad():
        #     nn.init.zeros_(self.fc_A2.weight)
        #     self.fc_A2.bias.copy_(torch.eye(latent_dim).flatten())

#         # ---- Learnable prior parameters ----
#         if learn_prior_mean:
#             self.mu_p = nn.Parameter(torch.zeros(latent_dim))
#         else:
#             self.register_buffer("mu_p", torch.zeros(latent_dim))

#         if learn_prior_covariance:
#             # Prior A matrix: initialize close to identity
#             init_std = 0.01
#             self.prior_A = nn.Parameter(
#                 torch.eye(latent_dim) + torch.randn(latent_dim, latent_dim) * init_std
#             )
#         else:
#             self.register_buffer("prior_A", torch.zeros(latent_dim, latent_dim))

#     # ----- Prior computation -----
#     def get_prior(self, batch_size=None):
#         """
#         Build full prior covariance using Sigma_p = A A^T + eps * I
#         """
#         D = self.latent_dim
#         device = self.prior_A.device

#         Sigma_p = self.prior_A @ self.prior_A.T + self.cov_diag_epsilon * torch.eye(D, device=device)
#         L_p = torch.linalg.cholesky(Sigma_p)

#         if batch_size is not None:
#             L_p = L_p.unsqueeze(0).expand(batch_size, -1, -1)
#             mu_p = self.mu_p.unsqueeze(0).expand(batch_size, -1)
#         else:
#             mu_p = self.mu_p

#         return mu_p, L_p

#     # ----- Forward -----
#     def forward(self, t, x, mask=None):
#         B, T = t.shape
#         if mask is None:
#             mask = torch.ones(B, T, dtype=torch.bool, device=t.device)

#         # Combine inputs
#         inp = torch.stack([t, x], dim=-1)  # [B, T, 2]
#         h = self.input_proj(inp)
#         h = self.pos_encoder(h)

#         # Transformer encoding with masking
#         h_enc = self.transformer(h, src_key_padding_mask=~mask)

#         # Pool over time (masked mean)
#         valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
#         pooled = (h_enc * mask.unsqueeze(-1)).sum(dim=1) / valid_counts

#         # Posterior mean
#         mu_q = self.fc_mu2(self.selu(self.fc_mu1(pooled)))

#         # Posterior covariance: Sigma_q = A_q A_q^T + eps * I
#         A_q_flat = self.fc_A2(self.selu(self.fc_A1(pooled)))  # [B, D*D]
#         A_q = A_q_flat.view(B, self.latent_dim, self.latent_dim)
#         Sigma_q = torch.matmul(A_q, A_q.transpose(-1, -2)) + self.cov_diag_epsilon * torch.eye(self.latent_dim, device=A_q.device)
#         L_q = torch.linalg.cholesky(Sigma_q)

#         # Sample latent
#         eps = torch.randn(B, self.latent_dim, device=mu_q.device)
#         z = mu_q + torch.einsum("bij,bj->bi", L_q, eps)

#         # Prior
#         mu_p, L_p = self.get_prior(batch_size=B)

#         return z, mu_q, L_q, mu_p, L_p




# class Encoder_Transformer_Full(nn.Module):
#     def __init__(self, latent_dim, input_dim=2, model_dim=64,
#                  hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1,
#                  cov_diag_epsilon=1e-5, learn_prior_mean=True,
#                  learn_prior_variance=True, learn_prior_covariance=True):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.cov_diag_epsilon = cov_diag_epsilon
#         self.selu = nn.SELU()

#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(model_dim)

#         # Input projection
#         self.input_proj = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.SELU(),
#             nn.Linear(hidden_dim, model_dim)
#         )

#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=model_dim, nhead=num_heads,
#             dim_feedforward=model_dim * 4, dropout=dropout, batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         # Output layers for mean
#         self.fc_mu1 = nn.Linear(model_dim, hidden_dim)
#         self.fc_mu2 = nn.Linear(hidden_dim, latent_dim)

#         # Output layers for full covariance
#         n_tril = latent_dim * (latent_dim + 1) // 2
#         self.fc_cov1 = nn.Linear(model_dim, hidden_dim)
#         self.fc_cov2 = nn.Linear(hidden_dim, n_tril)

#         # Learnable priors
#         if learn_prior_mean:
#             self.mu_p = nn.Parameter(torch.zeros(latent_dim))
#         else:
#             self.register_buffer("mu_p", torch.zeros(latent_dim))

#         if learn_prior_variance:
#             self.prior_diag_unconstrained = nn.Parameter(torch.zeros(latent_dim))
#         else:
#             self.register_buffer(
#                 "prior_diag_unconstrained",
#                 torch.full((latent_dim,), torch.log(torch.exp(torch.tensor(1.0)) - 1.0))
#             )

#         if learn_prior_covariance:
#             self.prior_subdiag = nn.Parameter(torch.zeros(latent_dim * (latent_dim - 1) // 2))
#         else:
#             self.register_buffer("prior_subdiag", torch.zeros(latent_dim * (latent_dim - 1) // 2))

#     # ----- Cholesky helper -----
#     def _build_cholesky_from_tril(self, tril_flat, latent_dim):
#         B = tril_flat.shape[0]
#         L = torch.zeros(B, latent_dim, latent_dim, device=tril_flat.device)
#         idx = 0
#         for i in range(latent_dim):
#             for j in range(i + 1):
#                 val = tril_flat[:, idx]
#                 if i == j:
#                     val = F.softplus(val) + self.cov_diag_epsilon
#                 L[:, i, j] = val
#                 idx += 1
#         return L

#     # ----- Prior -----
#     def get_prior_cholesky(self):
#         D = self.latent_dim
#         L_p = torch.zeros(D, D, device=self.prior_subdiag.device)
#         tril_idx = torch.tril_indices(D, D, offset=-1)
#         L_p[tril_idx[0], tril_idx[1]] = self.prior_subdiag
#         diag = F.softplus(self.prior_diag_unconstrained) + self.cov_diag_epsilon
#         L_p = L_p + torch.diag(diag)
#         return L_p.unsqueeze(0)

#     def get_prior(self, batch_size=None):
#         L_p = self.get_prior_cholesky()
#         mu_p = self.mu_p.unsqueeze(0)
#         if batch_size is not None:
#             L_p = L_p.expand(batch_size, -1, -1)
#             mu_p = mu_p.expand(batch_size, -1)
#         return mu_p, L_p

#     # ----- Forward -----
#     def forward(self, t, x, mask=None):
#         B, T = t.shape
#         if mask is None:
#             mask = torch.ones(B, T, dtype=torch.bool, device=t.device)

#         # Combine inputs
#         inp = torch.stack([t, x], dim=-1)  # [B, T, 2]
#         h = self.input_proj(inp)
#         h = self.pos_encoder(h)

#         # Transformer encoding with masking
#         h_enc = self.transformer(h, src_key_padding_mask=~mask)

#         # Pool over time (masked mean)
#         valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
#         pooled = (h_enc * mask.unsqueeze(-1)).sum(dim=1) / valid_counts

#         # Mean
#         mu_q = self.fc_mu2(self.selu(self.fc_mu1(pooled)))

#         # Covariance
#         tril_flat = self.fc_cov2(self.selu(self.fc_cov1(pooled)))
#         L_q = self._build_cholesky_from_tril(tril_flat, self.latent_dim)

#         # Sample latent
#         eps = torch.randn(B, self.latent_dim, device=mu_q.device)
#         z = mu_q + torch.einsum("bij,bj->bi", L_q, eps)

#         # Prior
#         mu_p, L_p = self.get_prior(batch_size=B)
#         return z, mu_q, L_q, mu_p, L_p


class Encoder_Transformer(nn.Module):
    def __init__(self, latent_dim, input_dim=2, model_dim=64,
                 hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1,
                 cov_diag_epsilon=1e-5,
                 learn_prior_mean=True,
                 learn_prior_variance=True,
                 learn_prior_covariance=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.cov_diag_epsilon = cov_diag_epsilon
        self.selu = nn.SELU()
        self.pos_encoder = PositionalEncoding(model_dim)

        # ---------- REST OF SEQUENCE ENCODER ----------
        self.input_proj1_rest = nn.Linear(input_dim, hidden_dim)
        self.input_proj2_rest = nn.Linear(hidden_dim, model_dim)
        encoder_layer_rest = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads,
            dim_feedforward=model_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer_rest = nn.TransformerEncoder(encoder_layer_rest, num_layers=num_layers)
        self.fc_mu1_rest = nn.Linear(model_dim, hidden_dim)
        self.fc_mu2_rest = nn.Linear(hidden_dim, latent_dim - 2)
        self.fc_cov1_rest = nn.Linear(model_dim, hidden_dim)
        n_tril_rest = (latent_dim - 2) * (latent_dim - 1) // 2
        self.fc_cov2_rest = nn.Linear(hidden_dim, n_tril_rest)

        # ---------- SIMPLIFIED FIRST OBSERVATION ENCODER ----------
        self.fc_first_mu1 = nn.Linear(input_dim, hidden_dim)
        self.fc_first_mu2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_first_mu_out = nn.Linear(hidden_dim, 2)

        self.fc_first_cov1 = nn.Linear(input_dim, hidden_dim)
        self.fc_first_cov2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_first_cov_out = nn.Linear(hidden_dim, 3)

        # ---------- Learnable priors ----------
        # Mean
        # ---------- Learnable priors (mean, variance, covariance) ----------
        # ---------- Learnable priors (mean, variance, covariance) ----------
        if learn_prior_mean:
            self.mu_p = nn.Parameter(torch.zeros(latent_dim))
        else:
            self.register_buffer("mu_p", torch.zeros(latent_dim))
        
        if learn_prior_variance:
            self.prior_diag_unconstrained = nn.Parameter(torch.zeros(latent_dim))
            nn.init.normal_(self.prior_diag_unconstrained, mean=0.0, std=1e-3)
        else:
            # value so that softplus(x) ≈ 1 → var = 1
            self.register_buffer(
                "prior_diag_unconstrained",
                torch.full((latent_dim,), torch.log(torch.exp(torch.tensor(1.0)) - 1.0))
            )
        
        if learn_prior_covariance:
            self.prior_subdiag = nn.Parameter(torch.zeros(latent_dim * (latent_dim - 1) // 2))
            nn.init.normal_(self.prior_subdiag, mean=0.0, std=1e-3)
        else:
            self.register_buffer("prior_subdiag", torch.zeros(latent_dim * (latent_dim - 1) // 2))



    # ---------- Helper functions (same as before) ----------
    def get_prior_cholesky(self):
        D = self.latent_dim
        L_p = torch.zeros(D, D, device=self.prior_subdiag.device)
        tril_idx = torch.tril_indices(D, D, offset=-1)
        L_p[tril_idx[0], tril_idx[1]] = self.prior_subdiag
        diag = F.softplus(self.prior_diag_unconstrained) + self.cov_diag_epsilon
        L_p = L_p + torch.diag(diag)
        return L_p.unsqueeze(0)

    def get_prior(self, batch_size=None):
        L_p = self.get_prior_cholesky()
        mu_p = self.mu_p.unsqueeze(0)
        if batch_size is not None:
            L_p = L_p.expand(batch_size, -1, -1)
            mu_p = mu_p.expand(batch_size, -1)
        return mu_p, L_p


    def _build_cholesky_from_tril(self, tril_flat, latent_dim):
        B = tril_flat.shape[0]
        L = torch.zeros(B, latent_dim, latent_dim, device=tril_flat.device)
        idx = 0
        for i in range(latent_dim):
            for j in range(i + 1):
                val = tril_flat[:, idx]
                if i == j:
                    val = F.softplus(val) + self.cov_diag_epsilon
                L[:, i, j] = val
                idx += 1
        return L

    # ---------- Forward ----------
    def forward(self, t, x, mask=None):
        B, T = t.shape
        assert T > 1, "Need at least one 'first' and one 'rest' observation"

        t_first, x_first = t[:, 0:1], x[:, 0:1]
        t_rest, x_rest = t[:, 1:], x[:, 1:]

        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=t.device)
        mask_first = mask[:, 0:1]
        mask_rest = mask[:, 1:]

        # ----- FIRST OBSERVATION -----
        # Prepare input
        inp_first = torch.cat([t_first, x_first], dim=-1).view(B, -1)  # flatten [B, input_dim]
        
        # ----- MU -----
        h_mu = self.selu(self.fc_first_mu1(inp_first))
        h_mu = self.selu(self.fc_first_mu2(h_mu))
        mu_first = self.fc_first_mu_out(h_mu)  # [B,2]
        
        # ----- COVARIANCE (Cholesky) -----
        h_cov = self.selu(self.fc_first_cov1(inp_first))
        h_cov = self.selu(self.fc_first_cov2(h_cov))
        tril_first = self.fc_first_cov_out(h_cov)  # [B,3]
        L_first = self._build_cholesky_from_tril(tril_first, 2)

        # ----- REST OF SEQUENCE -----
        inp_rest = torch.cat([t_rest.unsqueeze(-1), x_rest.unsqueeze(-1)], dim=-1)
        h_rest = self.selu(self.input_proj1_rest(inp_rest))
        h_rest = self.input_proj2_rest(h_rest)
        h_rest = self.pos_encoder(h_rest)
        h_enc_rest = self.transformer_rest(h_rest, src_key_padding_mask=~mask_rest)
        valid_counts = mask_rest.sum(dim=1, keepdim=True).clamp(min=1)
        pooled_rest = (h_enc_rest * mask_rest.unsqueeze(-1)).sum(dim=1) / valid_counts
        mu_rest = self.fc_mu2_rest(self.selu(self.fc_mu1_rest(pooled_rest)))
        tril_rest = self.fc_cov2_rest(self.selu(self.fc_cov1_rest(pooled_rest)))
        L_rest = self._build_cholesky_from_tril(tril_rest, self.latent_dim - 2)

        # ----- Combine -----
        mu_q = torch.cat([mu_first, mu_rest], dim=-1)
        L_q = torch.zeros(B, self.latent_dim, self.latent_dim, device=t.device)
        L_q[:, :self.latent_dim - 2, :self.latent_dim - 2] = L_first
        L_q[:, -2:, -2:] =  L_rest

        eps = torch.randn(B, self.latent_dim, device=mu_q.device)
        z = mu_q + torch.einsum("bij,bj->bi", L_q, eps)
        z = torch.clamp(z, -1e2, 1e2)

        mu_p, L_p = self.get_prior(batch_size=B)
        return z, mu_q, L_q, mu_p, L_p


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder_Transformer2(nn.Module):
    def __init__(self, latent_dim=2, parameter_dim=2, input_dim=2,
                 model_dim=64, hidden_dim=64, num_heads=4, num_layers=2,
                 dropout=0.1, cov_diag_epsilon=1e-5, ema_alpha=0.1,
                 learn_prior=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.parameter_dim=parameter_dim
        

        self.cov_diag_epsilon = cov_diag_epsilon
        self.selu = nn.SELU()
        self.ema_alpha = ema_alpha
        self.pos_encoder = PositionalEncoding(model_dim)
        self.learn_prior = learn_prior

        # ---------- FIRST OBSERVATION ENCODER ----------
        self.fc_first = nn.Sequential(
            nn.Linear(input_dim , hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fc_first_mu = nn.Linear(hidden_dim, latent_dim)
        
        n_tril_first = latent_dim * (latent_dim + 1) // 2
        self.fc_first_cov = nn.Linear(hidden_dim, n_tril_first)

        # ---------- REST OF SEQUENCE ENCODER ----------
        self.input_proj1_rest = nn.Linear(input_dim, hidden_dim)
        self.input_proj2_rest = nn.Linear(hidden_dim, model_dim)
        encoder_layer_rest = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads,
            dim_feedforward=model_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer_rest = nn.TransformerEncoder(encoder_layer_rest, num_layers=num_layers)
        self.fc_rest_mu1 = nn.Linear(model_dim, hidden_dim)
        self.fc_rest_mu2 = nn.Linear(hidden_dim, parameter_dim)
        n_tril_rest = parameter_dim * (parameter_dim + 1) // 2
        self.fc_rest_cov1 = nn.Linear(model_dim, hidden_dim)
        self.fc_rest_cov2 = nn.Linear(hidden_dim, n_tril_rest)

        # ---------- Learnable prior ----------
        if learn_prior:
            # Mean stays the same
            total_dim = self.latent_dim + self.parameter_dim

            self.mu_p = nn.Parameter(torch.zeros(total_dim))
        
            # Learnable Cholesky for each block
            n_tril_first = latent_dim * (latent_dim + 1) // 2
            n_tril_rest = parameter_dim * (parameter_dim + 1) // 2
        
            self.prior_tril_first = nn.Parameter(torch.zeros(n_tril_first))
            self.prior_tril_rest = nn.Parameter(torch.zeros(n_tril_rest))
        else:
            self.register_buffer("mu_p", torch.zeros(self.latent_dim))
            self.register_buffer("prior_tril_first", torch.zeros(latent_dim * (latent_dim + 1) // 2))
            self.register_buffer("prior_tril_rest", torch.zeros(parameter_dim * (parameter_dim + 1) // 2))


        # ---------- EMA ----------
        for name, param in self.named_parameters():
            self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
        self.register_buffer("iteration", torch.tensor(1.0))

    # ---------- EMA ----------
    def update_ema(self, alpha=None):
        alpha = alpha if alpha is not None else self.ema_alpha
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                ema_param.mul_(1 - alpha).add_(alpha * param.data)
                setattr(self, f"{name.replace('.', '_')}_ema", ema_param)
            self.iteration += 1.0

    def apply_ema_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                param.data.copy_(ema_param)

    # ---------- Utility ----------
    def _build_cholesky_from_tril(self, tril_flat, latent_dim):
        B = tril_flat.shape[0]
        L = torch.zeros(B, latent_dim, latent_dim, device=tril_flat.device)
        idx = 0
        for i in range(latent_dim):
            for j in range(i + 1):
                val = tril_flat[:, idx]
                if i == j:
                    val = F.softplus(val) + self.cov_diag_epsilon
                L[:, i, j] = val
                idx += 1
        return L

    # ---------- Prior ----------
    def get_prior(self, batch_size=None):
        L_first = self._build_cholesky_from_tril(
            self.prior_tril_first.unsqueeze(0), self.latent_dim
        ).squeeze(0)
        L_rest = self._build_cholesky_from_tril(
            self.prior_tril_rest.unsqueeze(0), self.parameter_dim
        ).squeeze(0)
        total_dim = self.latent_dim + self.parameter_dim

        # Construct block-diagonal L_p
        L_p = torch.zeros(total_dim, total_dim, device=L_first.device)  # square
        L_p[:self.latent_dim, :self.latent_dim] = L_first                 # top-left block
        L_p[self.latent_dim:, self.latent_dim:] = L_rest                 # bottom-right block
    
        mu_p = self.mu_p.unsqueeze(0)
        if batch_size is not None:
            L_p = L_p.unsqueeze(0).expand(batch_size, -1, -1)
            mu_p = mu_p.expand(batch_size, -1)
    
        return mu_p, L_p


    # ---------- Forward ----------
    def forward(self, t, x, mask=None):
        B, T = t.shape
        assert T > 1, "Need at least one first and one rest observation"

        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=t.device)

        # ---------- FIRST OBS ----------
        t_first, x_first = t[:, 0:1], x[:, 0:1]
        inp_first = torch.cat([t_first, x_first], dim=-1)  # [B, input_dim+1]
        h_first = self.fc_first(inp_first)  # pass through fully connected SELU MLP
        mu_first = self.fc_first_mu(h_first)
        tril_first = self.fc_first_cov(h_first)
        L_first = self._build_cholesky_from_tril(tril_first, self.latent_dim)

        # ---------- REST ----------
        t_rest, x_rest = t[:, 1:], x[:, 1:]
        mask_rest = mask[:, 1:]
        inp_rest = torch.cat([t_rest.unsqueeze(-1), x_rest.unsqueeze(-1)], dim=-1)  # [B, T-1, input_dim+1]
        h_rest = self.selu(self.input_proj1_rest(inp_rest))
        h_rest = self.selu(self.input_proj2_rest(h_rest))
        h_rest = self.pos_encoder(h_rest)
        h_enc_rest = self.transformer_rest(h_rest, src_key_padding_mask=~mask_rest)
        valid_counts = mask_rest.sum(dim=1, keepdim=True).clamp(min=1)
        pooled_rest = (h_enc_rest * mask_rest.unsqueeze(-1)).sum(dim=1) / valid_counts
        mu_rest = self.fc_rest_mu2(self.selu(self.fc_rest_mu1(pooled_rest)))
        tril_rest = self.fc_rest_cov2(self.selu(self.fc_rest_cov1(pooled_rest)))
        L_rest = self._build_cholesky_from_tril(tril_rest, self.parameter_dim)

        # ---------- Concatenate ----------
        total_dim = self.latent_dim + self.parameter_dim
        mu_q = torch.cat([mu_first, mu_rest], dim=-1)  # [B, total_dim]
        
        
        L_q = torch.zeros(B, total_dim, total_dim, device=t.device)
        L_q[:, :self.latent_dim, :self.latent_dim] = L_first        # top-left block
        L_q[:, self.latent_dim:, self.latent_dim:] = L_rest         # bottom-right block

        # ---------- Sample z ----------
        eps = torch.randn(B, total_dim, device=mu_q.device)
        z = mu_q + torch.einsum("bij,bj->bi", L_q, eps)

        # ---------- Prior ----------
        mu_p, L_p = self.get_prior(batch_size=B)

        return z, mu_q, L_q, mu_p, L_p


class Encoder_Transformer3(nn.Module):
    def __init__(self, latent_dim=2, parameter_dim=2, input_dim=2,
                 model_dim=64, hidden_dim=64, num_heads=4, num_layers=2,
                 dropout=0.1, cov_diag_epsilon=1e-5, ema_alpha=0.1,
                 learn_prior=True, prior_rank=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.selu = nn.SELU()

        self.parameter_dim = parameter_dim
        self.total_dim = latent_dim + parameter_dim
        self.cov_diag_epsilon = cov_diag_epsilon
        self.ema_alpha = ema_alpha
        self.pos_encoder = PositionalEncoding(model_dim)
        self.learn_prior = learn_prior
        self.prior_rank = prior_rank

        # ---------- FIRST OBSERVATION ENCODER ----------
        self.fc_first = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fc_first_mu = nn.Linear(hidden_dim, latent_dim)
        n_tril_first = latent_dim * (latent_dim + 1) // 2
        self.fc_first_cov = nn.Linear(hidden_dim, n_tril_first)

        # ---------- REST OF SEQUENCE ENCODER ----------
        self.input_proj1_rest = nn.Linear(input_dim, hidden_dim)
        self.input_proj2_rest = nn.Linear(hidden_dim, model_dim)
        encoder_layer_rest = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads,
            dim_feedforward=model_dim*4, dropout=dropout, batch_first=True
        )
        self.transformer_rest = nn.TransformerEncoder(encoder_layer_rest, num_layers=num_layers)
        self.fc_rest_mu1 = nn.Linear(model_dim, hidden_dim)
        self.fc_rest_mu2 = nn.Linear(hidden_dim, parameter_dim)
        n_tril_rest = parameter_dim * (parameter_dim + 1) // 2
        self.fc_rest_cov1 = nn.Linear(model_dim, hidden_dim)
        self.fc_rest_cov2 = nn.Linear(hidden_dim, n_tril_rest)

        # ---------- Learnable low-rank prior ----------
        if learn_prior:
            # Prior mean
            self.mu_p = nn.Parameter(torch.zeros(self.total_dim))
            # Diagonal for stability (param in unconstrained space -> softplus)
            self.prior_diag = nn.Parameter(torch.log(torch.exp(torch.ones(self.total_dim) * 0.1) - 1.0))
            # Low-rank factor (D x r)
            self.prior_U = nn.Parameter(torch.zeros(self.total_dim, prior_rank))
        else:
            self.register_buffer("mu_p", torch.zeros(self.total_dim))
            self.register_buffer("prior_diag", torch.log(torch.exp(torch.ones(self.total_dim) * 0.1) - 1.0))
            self.register_buffer("prior_U", torch.zeros(self.total_dim, prior_rank))

        # ---------- EMA ----------
        for name, param in self.named_parameters():
            self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
        self.register_buffer("iteration", torch.tensor(1.0))

    # ---------- EMA ----------
    def update_ema(self, alpha=None):
        alpha = alpha if alpha is not None else self.ema_alpha
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                ema_param.mul_(1 - alpha).add_(alpha * param.data)
                setattr(self, f"{name.replace('.', '_')}_ema", ema_param)
            self.iteration += 1.0

    def apply_ema_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                param.data.copy_(ema_param)

    # ---------- Utility ----------
    def _build_cholesky_from_tril(self, tril_flat, latent_dim):
        # tril_flat: [B, n_tril]
        B = tril_flat.shape[0]
        L = torch.zeros(B, latent_dim, latent_dim, device=tril_flat.device)
        idx = 0
        for i in range(latent_dim):
            for j in range(i + 1):
                val = tril_flat[:, idx]
                if i == j:
                    val = F.softplus(val) + self.cov_diag_epsilon
                L[:, i, j] = val
                idx += 1
        return L

    # ---------- Low-rank prior ----------
    def get_prior(self, batch_size=None):
        """
        Returns:
            mu_p: [D] or [B, D]
            L_p:  [D, D] or [B, D, D]  (Cholesky lower-triangular)
        Uses Sigma_p = diag(d) + U U^T, with d = softplus(prior_diag)+eps
        """
        device = self.mu_p.device
        D = self.total_dim

        # diag with positivity
        diag = F.softplus(self.prior_diag.to(device)) + self.cov_diag_epsilon  # [D]
        U = self.prior_U.to(device)  # [D, r]

        # Build Sigma_p (D, D)
        Sigma_p = torch.diag(diag) + (U @ U.t())  # [D, D]

        # Cholesky (lower-triangular)
        # add tiny jitter for extra stability if needed
        jitter = self.cov_diag_epsilon * torch.eye(D, device=device)
        try:
            L_p = torch.linalg.cholesky(Sigma_p + jitter)
        except RuntimeError:
            # fallback: add larger jitter
            L_p = torch.linalg.cholesky(Sigma_p + (self.cov_diag_epsilon * 10.0) * torch.eye(D, device=device))

        mu_p = self.mu_p.to(device)  # [D]

        if batch_size is not None:
            # Use repeat so gradients flow back to prior parameters
            mu_p = mu_p.unsqueeze(0).repeat(batch_size, 1)    # [B, D]
            L_p = L_p.unsqueeze(0).repeat(batch_size, 1, 1)   # [B, D, D]

        return mu_p, L_p

    # ---------- Forward ----------
    def forward(self, t, x, mask=None):
        B, T = t.shape
        assert T > 1, "Need at least one first and one rest observation"
        device = t.device

        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=device)

        # ---------- FIRST OBS ----------
        t_first, x_first = t[:, 0:1], x[:, 0:1]
        inp_first = torch.cat([t_first, x_first], dim=-1)
        h_first = self.fc_first(inp_first)
        mu_first = self.fc_first_mu(h_first)
        tril_first = self.fc_first_cov(h_first)
        L_first = self._build_cholesky_from_tril(tril_first, self.latent_dim)

        # ---------- REST ----------
        t_rest, x_rest = t[:, 1:], x[:, 1:]
        mask_rest = mask[:, 1:]
        inp_rest = torch.cat([t_rest.unsqueeze(-1), x_rest.unsqueeze(-1)], dim=-1)
        h_rest = self.selu(self.input_proj1_rest(inp_rest))
        h_rest = self.selu(self.input_proj2_rest(h_rest))
        h_rest = self.pos_encoder(h_rest)
        h_enc_rest = self.transformer_rest(h_rest, src_key_padding_mask=~mask_rest)
        valid_counts = mask_rest.sum(dim=1, keepdim=True).clamp(min=1)
        pooled_rest = (h_enc_rest * mask_rest.unsqueeze(-1)).sum(dim=1) / valid_counts
        mu_rest = self.fc_rest_mu2(self.selu(self.fc_rest_mu1(pooled_rest)))
        tril_rest = self.fc_rest_cov2(self.selu(self.fc_rest_cov1(pooled_rest)))
        L_rest = self._build_cholesky_from_tril(tril_rest, self.parameter_dim)

        # ---------- Concatenate ----------
        mu_q = torch.cat([mu_first, mu_rest], dim=-1)  # [B, total_dim]
        L_q = torch.zeros(B, self.total_dim, self.total_dim, device=device)
        L_q[:, :self.latent_dim, :self.latent_dim] = L_first
        L_q[:, self.latent_dim:, self.latent_dim:] = L_rest

        # ---------- Sample z ----------
        eps = torch.randn(B, self.total_dim, device=device)
        z = mu_q + torch.einsum("bij,bj->bi", L_q, eps)

        # ---------- Prior ----------
        mu_p, L_p = self.get_prior(batch_size=B)

        return z, mu_q, L_q, mu_p, L_p



    
    




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

