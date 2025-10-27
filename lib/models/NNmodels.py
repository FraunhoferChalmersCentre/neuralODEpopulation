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
import numpy as np

import math

import torch.nn.functional as F


class ODEFunc(nn.Module):
    """
    ODE function with optional EMA for parameters, dose handling, and skip connection.
    Supports multiple event types via `evid` (bolus=1, infusion=2, observation=0).
    """
    def __init__(self, dim_latent, dim_parameters_IC, dim_parameters_dynamic, hid_dim, drug_dim):
        super().__init__()
        self.drug_dim=drug_dim
        self.dim_latent = dim_latent
        self.dim_parameters_IC=dim_parameters_IC
        self.dim_parameters_dynamic = dim_parameters_dynamic
        self.fc1 = nn.Linear(1, hid_dim)
        self.SELU = nn.SELU()
        
        # Two separate layers for mean and log variance
        self.fc_mu = nn.Linear(hid_dim, dim_latent)
        self.fc_logvar = nn.Linear(hid_dim, dim_latent)
        
        
        
        # Deep network
        self.net = nn.Sequential(
            nn.Linear(dim_latent + dim_parameters_dynamic+drug_dim+1, hid_dim),
             nn.SELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SELU(),
           nn.Linear(hid_dim, dim_latent)
        )
        
        self.drug = nn.Sequential(
            nn.Linear(drug_dim, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, drug_dim),
      
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
        
        diff = t.unsqueeze(-1) - dose_times  # [batch, num_times, num_doses]
        mask_nonnegative = diff >= 0

        gauss = torch.exp(-(diff / sigma)**2) * dose_amounts
 
     
        dose_signal=gauss*mask_nonnegative

      
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
         
        if t.dim() == 0:
            t = t.expand(batch_size, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1).expand(batch_size, 1)
        
        inp1 = torch.cat([x, self.drug(dose_input),t], dim=1)

        dxdt_deep = self.net(inp1)
        dxdt = dxdt_deep
       
        # Concatenate zeros for parameter dimensions
        zeros_param = torch.zeros(batch_size, self.dim_parameters_dynamic, device=device)
        dxdt_concat = torch.cat([dxdt, zeros_param], dim=1)
        return dxdt_concat





    

class TrainableNoise(nn.Module):
    """
    Estimates additive + optional proportional noise with EMA support.
    """
    def __init__(self, init_add_std=0.1, init_prop_std=0.0):
        super().__init__()
        size=1
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
        
        
       
        
        nll_elementwise = 0.5 * ((x_true - x_pred) / sigma_total) ** 2 + torch.log(sigma_total)
        mse_elementwise = (x_true - x_pred) ** 2
  
        
        #L1 for masked/dropped points
        if replace_mask is not None:
            dropout_mask = replace_mask.bool()
            dropout_mask = dropout_mask.expand_as(nll_elementwise)

            if dropout_mask.any():
               # L1_nll_elementwise = (2 ** 0.5) * (x_true - x_pred).abs() / sigma_total + torch.log(2 ** 0.5 * sigma_total)
                L1_nll_elementwise =  (x_true - x_pred).abs()

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









# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=500):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
#         position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

#         pe[:, 0::2] = torch.sin(position * div_term)  # even indices
#         pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

#         pe = pe.unsqueeze(0)  # [1, max_len, d_model]
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # x: [batch_size, seq_len, d_model]
#         x = x + self.pe[:, :x.size(1), :]
#         return x
                
# class MultiHeadAttentionPooling(nn.Module):
#     def __init__(self, d_model, num_heads=4):
#         super().__init__()
#         self.num_heads = num_heads
#         # One learnable query per head
#         self.query = nn.Parameter(torch.randn(num_heads, d_model))
#         self.scale = math.sqrt(d_model)

#     def forward(self, h, mask):
#         """
#         h: [B, T, d_model]
#         mask: [B, T] boolean
#         """
#         B, T, D = h.shape
#         H = self.num_heads

#         # Compute scores: [B, T, H] 
#         # einsum: b t d , h d -> b t h
#         scores = torch.einsum("btd,hd->bth", h, self.query) / self.scale

#         # Mask invalid positions
#         scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))

#         # Attention weights
#         attn = torch.softmax(scores, dim=1)  # [B, T, H]

#         # Weighted sum per head: [B, H, D]
#         pooled = torch.einsum("btd,bth->bhd", h, attn)

#         # Flatten heads: [B, H*D]
#         pooled = pooled.reshape(B, H * D)
#         return pooled            


# import torch.nn.init as init

# class Encoder_Transformer_Full(nn.Module):
#     def __init__(self, dim_latent, dim_parameter, model_dim=64,
#                  hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1,
#                  cov_diag_epsilon=1e-5, learn_prior_mean=True, learn_prior_covariance=True,
#                  diagonal_only=False):
#         super().__init__()
#         input_dim=2 ## Time and DV
#         # replace mean pooling
#         self.pooler = MultiHeadAttentionPooling(model_dim, num_heads=num_heads)
#         self.pool_norm = nn.LayerNorm(model_dim * num_heads)  # adjust for num_heads

#         self.dim_latent = dim_latent
#         self.dim_parameter = dim_parameter
#         total_parameters=dim_parameter
#         self.total_parameters=total_parameters
#         self.cov_diag_epsilon = cov_diag_epsilon
#         self.diagonal_only = diagonal_only
#         self.selu = nn.SELU()
#         self.learn_prior_covariance = learn_prior_covariance
#         self.learn_prior_mean = learn_prior_mean
#         self._prior_frozen = False

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

#         # Posterior mean
#         self.fc_mu1 = nn.Linear(model_dim, hidden_dim)
#         self.fc_mu2 = nn.Linear(hidden_dim, total_parameters)
        
        
#         self.fc_mu2 = nn.Sequential(
#                 nn.Linear(model_dim * num_heads, 32),
#                 nn.ReLU(),
#                 nn.Linear(32, 1),
#             )
        
#         self.fc_mu1 = nn.Sequential(
#                 nn.Linear(model_dim * num_heads, 128),
#                 nn.GELU(),
#                 nn.Linear(128, 128),
#                 nn.GELU(),
#                 nn.Linear(128, 1),
#             )

#      #   nn.init.orthogonal_(self.fc_mu[-1].weight)
#       #  nn.init.normal_(self.fc_mu[-1].bias, 0.0, 1e-1)
#         # Posterior covariance
#         self.fc_A1 = nn.Linear(model_dim * num_heads, hidden_dim)
#         self.fc_A2 = nn.Linear(hidden_dim, total_parameters * total_parameters)

#         nn.init.zeros_(self.fc_A2.weight)
#         with torch.no_grad():
#             self.fc_A2.bias.copy_(torch.eye(total_parameters).flatten())
        
               
#         # NODE initial condition
        
#         self.z0_mu = nn.Parameter(torch.zeros(dim_latent))

        
#         # Learnable prior parameters
#         if learn_prior_mean:
#             self.mu_p = nn.Parameter(torch.zeros(total_parameters))
#         else:
#             self.register_buffer("mu_p", torch.zeros(total_parameters))

#         if learn_prior_covariance:
#             init_std = 0.01
#             self.prior_A = nn.Parameter(
#                 torch.eye(total_parameters) + torch.randn(total_parameters, total_parameters) * init_std
#             )
#         else:
#             self.register_buffer("prior_A", torch.zeros(total_parameters, total_parameters))
#         for name, param in self.named_parameters():
#             self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
#         self.register_buffer("iteration", torch.tensor(1.0))
        
#     def update_ema(self, alpha=0.1):
#         """Update EMA of all parameters."""
#         with torch.no_grad():
#             for name, param in self.named_parameters():
#                 ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
#                 ema_param = ema_param.to(param.device)
#                 ema_param.mul_(1 - alpha).add_(alpha * param.data)
#                 setattr(self, f"{name.replace('.', '_')}_ema", ema_param)
#             self.iteration += 1.0

#     def apply_ema_weights(self):
#         """Replace model weights with their EMA values."""
#         with torch.no_grad():
#             for name, param in self.named_parameters():
#                 ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
#                 param.data.copy_(ema_param)
        
        
#     def get_prior(self, batch_size=None):
#         D = self.total_parameters
#         device = self.mu_p.device
    
#         if self.learn_prior_covariance:
#             # Full or learned covariance
#             Sigma_p = self.prior_A @ self.prior_A.T + self.cov_diag_epsilon * torch.eye(D, device=device)
#         else:
#             # Fixed covariance (identity)
#             if self.diagonal_only:
#                 # Independent Gaussian prior
#                 Sigma_p = torch.eye(D, device=device)
#             else:
#                 # Full (but fixed) covariance
#                 Sigma_p = torch.eye(D, device=device)
    
#         L_p = torch.linalg.cholesky(Sigma_p)
    
#         mu_p = self.mu_p
#         if batch_size is not None:
#             mu_p = mu_p.unsqueeze(0).expand(batch_size, -1)
#             L_p = L_p.unsqueeze(0).expand(batch_size, -1, -1)
    
#         return mu_p, L_p
    
    
#     def freeze_prior_until(self, current_epoch, freeze_epochs=10):
#         """Freeze the prior for the first `freeze_epochs` epochs."""
#         should_freeze = current_epoch < freeze_epochs

#         if should_freeze and not self._prior_frozen:
#             if hasattr(self, "mu_p") and isinstance(self.mu_p, nn.Parameter):
#                 self.mu_p.requires_grad_(False)
#             if hasattr(self, "prior_A") and isinstance(self.prior_A, nn.Parameter):
#                 self.prior_A.requires_grad_(False)
#             self._prior_frozen = True

#         elif not should_freeze and self._prior_frozen:
#             if hasattr(self, "mu_p") and isinstance(self.mu_p, nn.Parameter):
#                 self.mu_p.requires_grad_(True)
#             if hasattr(self, "prior_A") and isinstance(self.prior_A, nn.Parameter):
#                 self.prior_A.requires_grad_(True)
#             self._prior_frozen = False


#     def forward(self, t, x, dose, only_median=False, mask=None):
#         B, T = t.shape
#         if mask is None:
#             mask = torch.ones(B, T, dtype=torch.bool, device=t.device)

#         # Encode
#         inp = torch.stack([t, x], dim=-1)
#         h = self.input_proj(inp)
#         h = self.pos_encoder(h)
#         h_enc = self.transformer(h, src_key_padding_mask=~mask)

#         # Masked pooling
#       #  valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
#        # pooled = (h_enc * mask.unsqueeze(-1)).sum(dim=1) / valid_counts
#         pooled = self.pooler(h_enc, mask)
#         pooled = self.pool_norm(pooled)
#         pooled1 = F.dropout(pooled, p=0.1, training=self.training)
#        # pooled1 = pooled + torch.randn_like(pooled) * 1e-1

#      #   pooled += torch.randn_like(pooled)  # tiny noise to break symmetry
#        # print("pooled mean/std:", pooled.mean().item(), pooled.std().item())
#         #print("pooled per-dim std:", pooled.std(dim=0).mean().item())  # check per-feature spread
#         # Posterior mean
        
#         # first_mask = torch.zeros_like(mask)
#         # first_mask[:, 0] = True
        
#         # pooled2 = self.pooler(h_enc, first_mask)
        
#         # mu_q1 = self.fc_mu1(pooled1)
#         # mu_q2 = self.fc_mu2(pooled2)
#         # mu_q = torch.cat([mu_q2, mu_q1], dim=-1)


#         mu_q = self.fc_mu1(pooled1)

        
        
#         #self.fc_mu2(self.selu(self.fc_mu1(pooled)))
#      #   print("mu_q mean/std:", mu_q.mean().item(), mu_q.std().item())
#         # Posterior covariance
#         A_q_flat = self.fc_A2(self.selu(self.fc_A1(pooled)))
#         A_q = A_q_flat.view(B, self.total_parameters, self.total_parameters)

#         if self.diagonal_only:
#             # Keep only diagonal elements
#             A_q = torch.diag_embed(torch.diagonal(A_q, dim1=-2, dim2=-1))

#         Sigma_q = torch.matmul(A_q, A_q.transpose(-1, -2)) + self.cov_diag_epsilon * torch.eye(self.total_parameters, device=A_q.device)
#         L_q = torch.linalg.cholesky(Sigma_q)

#         # Sample latent
#         eps = torch.randn(B, self.total_parameters, device=mu_q.device)
#         k_params = mu_q + torch.einsum("bij,bj->bi", L_q, eps)
#         #print(k_params)
#         if only_median:
#             k_params=k_params*0
        
      
#         z0 = self.z0_mu.unsqueeze(0).expand(B, -1)+k_params[:, 0].unsqueeze(1)
#        # print(k_params[:, 0].unsqueeze(1))

#         mu_p, L_p = self.get_prior(batch_size=B)

#         return k_params, z0, mu_q, L_q, mu_p, L_p











import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling layer with learnable queries."""
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.randn(num_heads, d_model))
        self.scale = math.sqrt(d_model)

    def forward(self, h, mask):
        """
        h: [B, T, D]
        mask: [B, T] boolean (True = valid)
        Returns: [B, H*D]
        """
        B, T, D = h.shape
        H = self.num_heads
        scores = torch.einsum("btd,hd->bth", h, self.query) / self.scale
        scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        attn = torch.softmax(scores, dim=1)  # [B, T, H]
        pooled = torch.einsum("btd,bth->bhd", h, attn)
        return pooled.reshape(B, H * D)


class MHAEncoderLayer(nn.Module):
    """Transformer encoder layer (batch_first=True)."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation=nn.SELU()):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                         dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout2(ffn_out))


class Encoder_Transformer(nn.Module):
    """
    Transformer-based VAE encoder with full-covariance posterior and prior.
    Supports learnable prior mean and covariance.
    """
    def __init__(
        self,
        dim_latent,
        dim_parameters_IC,
        dim_parameters_dynamic,
        model_dim=64,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        cov_diag_epsilon=1e-5,
        learn_prior_mean=True,
        learn_prior_covariance=False,
        diagonal_only=False,
        IC_dose_dependent=False,
        dropout_param=0.1
    ):
        super().__init__()
        input_dim = 2  # (t, x)
        self.dim_latent = dim_latent
        self.dim_parameter_IC = dim_parameters_IC
        self.dim_parameter_dynamic = dim_parameters_dynamic
        self.IC_dose_dependent=IC_dose_dependent
        self.total_parameters = dim_parameters_IC+dim_parameters_dynamic
        self.dropout_param=dropout_param
        
        
        self.cov_diag_epsilon = cov_diag_epsilon
        self.diagonal_only = diagonal_only
        self.selu = nn.SELU()
        self.learn_prior_mean = learn_prior_mean
        self.learn_prior_covariance = learn_prior_covariance

        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, model_dim),
        )

        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads,
            dim_feedforward=model_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj1 = nn.Linear(input_dim, hidden_dim)
        self.input_proj2 = nn.Linear(hidden_dim, model_dim)

        # Pooling
        self.pooler = MultiHeadAttentionPooling(model_dim, num_heads=num_heads)
        self.pool_norm = nn.LayerNorm(model_dim * num_heads)

        # Posterior mean
        self.fc_mu1 = nn.Linear(model_dim * num_heads, hidden_dim)
        self.fc_mu2 = nn.Linear(hidden_dim, self.total_parameters)

        # Posterior covariance
        self.fc_A1 = nn.Linear(model_dim * num_heads, hidden_dim)
        self.fc_A2 = nn.Linear(hidden_dim, self.total_parameters * self.total_parameters)
        nn.init.zeros_(self.fc_A2.weight)
        with torch.no_grad():
            self.fc_A2.bias.copy_(torch.eye(self.total_parameters).flatten())
        

        self.z0_transformed=nn.Sequential(
             nn.Linear(dim_latent+dim_parameters_IC, 512),
             nn.SELU(),
             nn.Linear(512, 512),
             nn.SELU(),
       nn.Linear(512, dim_latent),
         )
        
        # NODE initial condition
        if IC_dose_dependent:
            self.z0_mu=nn.Sequential(
                 nn.Linear(1, 128),
                 nn.SELU(),
                 nn.Linear(128, 128),
                 nn.SELU(),
           nn.Linear(128, dim_latent),
             )
            
            
            

            
        else: 

            self.z0_mu = nn.Parameter(torch.zeros(dim_latent))

        # Learnable prior parameters
        if learn_prior_mean:
            self.mu_p = nn.Parameter(torch.zeros(self.total_parameters))
        else:
            self.register_buffer("mu_p", torch.zeros(self.total_parameters))

        if learn_prior_covariance:
            init_std = 0.01
            self.prior_A = nn.Parameter(
                torch.eye(self.total_parameters) + torch.randn(self.total_parameters, self.total_parameters) * init_std
            )
        else:
            self.register_buffer("prior_A", torch.eye(self.total_parameters))

        # EMA buffers
        for name, param in self.named_parameters():
            self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
        self.register_buffer("iteration", torch.tensor(1.0))

    # ---------------- EMA helpers ----------------
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

    # ---------------- Prior ----------------
    def get_prior(self, batch_size=None):
        D = self.total_parameters
        device = self.mu_p.device

        if self.learn_prior_covariance:
            Sigma_p = self.prior_A @ self.prior_A.T + self.cov_diag_epsilon * torch.eye(D, device=device)
        else:
            Sigma_p = torch.eye(D, device=device)

        L_p = torch.linalg.cholesky(Sigma_p)

        mu_p = self.mu_p
        if batch_size is not None:
            mu_p = mu_p.unsqueeze(0).expand(batch_size, -1)
            L_p = L_p.unsqueeze(0).expand(batch_size, -1, -1)

        return mu_p, L_p
    def _sample_posterior(self, mu_q, L_q, num_samples=1):
        B, D = mu_q.shape
        device = mu_q.device
    
        if num_samples > 1:
            eps = torch.randn(B, D, num_samples, device=device)  # [B, D, S]
            samples = mu_q.unsqueeze(1) + torch.einsum("bij,bjs->bis", L_q, eps).permute(0, 2, 1)  # [B, S, D]
            return samples.reshape(B * num_samples, D)  # flatten into [B*S, D]
        else:
            eps = torch.randn(B, D, device=device)  # [B, D]
            return mu_q + torch.einsum("bij,bj->bi", L_q, eps)  # [B, D]



    # def compute_cholesky(self, A_q_flat, B, D, eps=1e-5):

    #     """
    #     Convert flat outputs to a lower-triangular Cholesky factor.
    #     """
    #     # Reshape flat vector to [B, D, D]
    #     A_q = A_q_flat.view(B, D, D)
        
    #     # Make strictly lower-triangular + positive diagonal
    #     tril_mask = torch.tril(torch.ones(D, D, device=A_q.device), diagonal=-1)
    #     L_q = A_q * tril_mask
        
    #     # Diagonal (softplus to ensure positive)
    #     diag = F.softplus(torch.diagonal(A_q, dim1=-2, dim2=-1)) + eps
    #     L_q = L_q + torch.diag_embed(diag)
        
    #     return L_q
   
    # ---------------- Forward ----------------
    def forward(self, t, x, dose_tensor, mask=None, only_median=False,enable_ae=False, num_samples=1, min_batch_size=None, augment=False,sample_posterior=True):

        B, T = t.shape
        device = t.device
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=device)
        key_padding_mask = ~mask
    
        # ------------------ Posterior ------------------
        inp = torch.stack([t, x], dim=-1)
        h = self.input_proj(inp)
        h = self.pos_encoder(h)
        h_enc = self.transformer(h, src_key_padding_mask=~mask)
     
        # Masked pooling
        pooled = self.pooler(h_enc, mask)
        pooled = self.pool_norm(pooled)
        pooled = F.dropout(pooled, p=0.1, training=self.training)
     
        # Posterior mean
        mu_q = self.fc_mu2 (self.fc_mu1(pooled))
     
        # Posterior covariance
        # A_q_flat = self.fc_A2(self.selu(self.fc_A1(pooled)))
        # A_q = A_q_flat.view(B, self.total_parameters, self.total_parameters)
        A_q_flat = self.fc_A2(self.selu(self.fc_A1(pooled)))  # [B, D*D]
        #L_q = self.compute_cholesky(A_q_flat, B, self.total_parameters)  # [B, D, D]
        A_q = A_q_flat.view(B, self.total_parameters, self.total_parameters)                  # [B, D, D]
        Sigma_q = A_q @ A_q.transpose(-1, -2) + self.cov_diag_epsilon * torch.eye(self.total_parameters, device=A_q.device)
        L_q = torch.linalg.cholesky(Sigma_q)          # Cholesky factor

        # if self.diagonal_only:
        #     A_q = torch.diag_embed(torch.diagonal(A_q, dim1=-2, dim2=-1))
     
        # Sigma_q = A_q @ A_q.transpose(-1, -2) + self.cov_diag_epsilon * torch.eye(self.total_parameters, device=A_q.device)
        # L_q = torch.linalg.cholesky(Sigma_q)

    
        # ------------------ Prior ------------------
        
        # ------------------ Median-only ------------------
        if only_median:
            k_params = torch.zeros(B, self.total_parameters, device=device)
            mu_p = torch.zeros_like(mu_q)
            mu_q=mu_p
            L_p = torch.eye(self.total_parameters, device=device).unsqueeze(0).expand(B, self.total_parameters, self.total_parameters)
            L_q=L_p
            
            mask = torch.ones(B, 1, device=device)

     
        else:
        # ------------------ Autoencoder ------------------
            if enable_ae: 
               
                k_params=mu_q
                mu_p = torch.zeros_like(mu_q)
               # mu_q=mu_p
    
                
                L_p = torch.eye(self.total_parameters, device=device).unsqueeze(0).expand(B, self.total_parameters, self.total_parameters)
                L_q=L_p
                
                mask = (torch.rand(B, 1, device=device) < self.dropout_param)  # BOOL mask, not float
                mask2 = mask.expand(-1, self.total_parameters)  # [B, D] bool tensor
                
                # Blend posterior sample with prior mean
              #  k_params = torch.where(mask2, mu_p.expand_as(k_params), k_params)
                
     
            # ------------------ Variational Autoencoder ------------------    
            else:   
                mu_p, L_p = self.get_prior(batch_size=B)
            
                if sample_posterior: # Sample from posterior (during training, individual predictions)
                    k_params=self._sample_posterior(mu_q, L_q, num_samples=num_samples)
                    mask = (torch.rand(B, 1, device=device) < self.dropout_param)  # BOOL mask, not float
                    mask2 = mask.expand(-1, self.total_parameters)  # [B, D] bool tensor
                    
                    # Blend posterior sample with prior mean
                    k_params = torch.where(mask2, mu_p.expand_as(k_params), k_params)
                     
                else:                # Sample from the prior (in VPC e.g.,)
                    k_params=self._sample_posterior(mu_p, L_p, num_samples=num_samples)
           
                    mask = (torch.rand(B, 1, device=device) < self.dropout_param)  # BOOL mask, not float

                    
                   
         

        # ------------------ Split k into dynamic and IC -----------------
        dyn_start = self.dim_parameter_IC
        dyn_end = self.total_parameters

        k_params_IC =k_params[:, 0:dyn_start]  
        k_params_K = k_params[:, dyn_start:dyn_end] 

        # ------------------ Dose-dependent IC ------------------
        if self.IC_dose_dependent:
            dose_input = dose_tensor[:, 0:1]  # shape [B, 1]
           
            # Compute z0 from dose
            z0_base = self.z0_mu(dose_input)  # [B, dim_latent]
         
      
         # ------------------ Dose-independent IC ------------------    
        else: 
            z0_base = self.z0_mu  # [B, dim_latent]
            z0_base = z0_base.repeat(B, 1)   # shape: [B, dim_latent]
 
                
 
    
        # ------------------ Multiple Monte Carlo samples ------------------
        if num_samples > 1:
            z0 = z0_base.unsqueeze(1).expand(-1, num_samples, -1).reshape(B * num_samples, -1) 
        else:
            z0 = z0_base
            

        # ------------------ Variability on IC -----------------
        if self.dim_parameter_IC > 0:
            z0 = self.z0_transformed(torch.cat([z0, k_params_IC],dim=-1))
                  
   
        k_params=torch.cat([z0, k_params_K], dim=-1)
        return k_params, z0, mu_q, L_q, mu_p, L_p, mask



    

    




class SimpleDecoder(nn.Module):
    def __init__(self, dim_latent, hidden_dim=16):
        super().__init__()
        self.dim_latent=dim_latent
        
        # Define the network
        self.net = nn.Sequential(
            nn.Linear(dim_latent, 1))
        
   

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

