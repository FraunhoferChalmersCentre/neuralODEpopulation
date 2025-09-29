# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 22:01:56 2025

@author: Baaz
"""

import copy
import gc
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchdiffeq import odeint as odeint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm

import pandas as pd
from lib.utils.Theophylline.utils_preprocess_theo import standardize_concentration, destandardize_concentration

from torch.nn.utils.rnn import pad_sequence


def torch_linear_interpolate2(x_dense, y_dense, x_target):
    """
    Differentiable linear interpolation in PyTorch.
    Assumes x_dense is sorted and x_target lies within x_dense range.
    """
    idx = torch.searchsorted(x_dense, x_target, right=True)
    idx = torch.clamp(idx, 1, len(x_dense) - 1)

    x0 = x_dense[idx - 1]
    x1 = x_dense[idx]
    y0 = y_dense[idx - 1]
    y1 = y_dense[idx]

    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x_target - x0)

def linear_interpolate_with_extrapolation(t_obs, x_obs, t_dense):
    import numpy as np
    
    t_obs_np = t_obs.cpu().numpy()
    x_obs_np = x_obs.cpu().numpy()
    t_dense_np = t_dense.cpu().numpy()
    
    x_interp_np = np.interp(t_dense_np, t_obs_np, x_obs_np)
    
    # Fill values before first t_obs with first observation
    x_interp_np[t_dense_np < t_obs_np[0]] = x_obs_np[0]
    # Fill values after last t_obs with last observation
    x_interp_np[t_dense_np > t_obs_np[-1]] = x_obs_np[-1]
    
    return torch.tensor(x_interp_np, device=t_dense.device, dtype=x_obs.dtype)





def batch_linear_interpolate_1d(y, t_src, t_target):
    """
    y: [batch, N] tensor - values at source times
    t_src: [batch, N] tensor - source time points (must be sorted ascending)
    t_target: [batch, M] tensor - target time points for interpolation
    
    Returns:
    y_interp: [batch, M] tensor - interpolated values at t_target
    """
    device = y.device
    batch_size, N = y.shape
    t_src = t_src.to(device)
    t_target = t_target.to(device)
    M = t_target.shape[1]

    y_interp = torch.zeros(batch_size, M, device=y.device, dtype=y.dtype)

    for i in range(batch_size):
        # get source and target for batch i
        t_src_i = t_src[i]      # shape: [N]
        y_i = y[i]              # shape: [N]
        t_target_i = t_target[i]  # shape: [M]

        # Find indices k such that t_src_i[k] <= t_target_i < t_src_i[k+1]
        k = torch.searchsorted(t_src_i, t_target_i, right=True) - 1
        k = torch.clamp(k, 0, N - 2)

        t0 = t_src_i[k]         # [M]
        t1 = t_src_i[k + 1]     # [M]
        y0 = y_i[k]             # [M]
        y1 = y_i[k + 1]         # [M]

        # Linear interpolation weights
        denom = (t1 - t0)
        denom[denom == 0] = 1e-8  # avoid div by zero
        alpha = (t_target_i - t0) / denom  # [M]

        y_interp[i] = y0 + alpha * (y1 - y0)

    return y_interp



def truncate_time_series(t_batch, x_batch, truncation_time):
    """
    Truncate a batch of time series to a specified cutoff time, and also return
    the values that were removed.

    Args:
        t_batch (list[torch.Tensor]): List of time tensors, each shaped [T_i].
        x_batch (list[torch.Tensor]): List of value tensors corresponding to t_batch, each shaped [T_i, ...].
        truncation_time (float): Time cutoff. All entries with time > cutoff are removed.

    Returns:
        tuple:
            truncated_t (list[torch.Tensor]): Truncated time tensors.
            truncated_x (list[torch.Tensor]): Truncated value tensors.
            masks (list[torch.BoolTensor]): Boolean masks indicating kept indices.
            removed_t (list[torch.Tensor]): Time tensors that were removed.
            removed_x (list[torch.Tensor]): Value tensors that were removed.
    """
    truncated_t, truncated_x, masks = [], [], []
    removed_t, removed_x = [], []

    for t_i, x_i in zip(t_batch, x_batch):
        # Boolean mask: True where time ≤ cutoff
        mask = t_i <= truncation_time

        # Keep only values within cutoff
        truncated_t.append(t_i[mask])
        truncated_x.append(x_i[mask])
        masks.append(mask)

        # Keep values that were removed (inverse mask)
        inv_mask = ~mask
        removed_t.append(t_i[inv_mask])
        removed_x.append(x_i[inv_mask])

    return truncated_t, truncated_x, masks, removed_t, removed_x




class ODEWrapper(nn.Module):
    def __init__(self, func, dose_times, transfusion_times, dose_amounts, dose_mask, keep_mask=None):
        super().__init__()
        self.func = func
        self.transfusion_times = transfusion_times          # [batch, max_len]
        self.dose_times = dose_times          # [batch, max_len]
        self.dose_amounts = dose_amounts      # [batch, max_len]
        self.dose_mask = dose_mask            # [batch, max_len]
        self.keep_mask = keep_mask            # [batch, 1] — binary mask (optional)

    def forward(self, t, x):
        # Pass keep_mask if provided
        if self.keep_mask is not None:
            return self.func(t, x, self.dose_times,self.transfusion_times,  self.dose_amounts, self.dose_mask, self.keep_mask)
        else:
            return self.func(t, x, self.dose_times,self.transfusion_times,  self.dose_amounts, self.dose_mask)

 
def preprocess_batch(batch, device, truncation=1, dose_pad_value=-1.0):
    """
    Preprocess a batch from TrajectoryDataset, truncating time series if needed.

    Args:
        batch: output from collate_fn
        device: torch device
        truncation: optional truncation for encoder
        dose_pad_value: padding value for doses (unused here but kept for compatibility)

    Returns:
        occ_list       : list of subject IDs
        t_padded       : padded time sequences [B, T]
        x_padded       : padded DV sequences [B, T]
        t_encoder      : encoder time sequences [B, T_max]
        x_encoder      : encoder DV sequences [B, T_max]
        t_cut          : truncated times that were removed [B, T_cut]
        x_cut          : truncated values that were removed [B, T_cut]
        masks          : mask for valid encoder points [B, T_max]
        dose_tensor    : per-subject dose amounts
        dose_times_list: per-subject dose times
    """

    # Unpack batch
    occ_list, t_padded, x_padded, mask, dose_tensor, dose_times_list = batch


    # Move tensors to device
    t_padded = t_padded.to(device)
    x_padded = x_padded.to(device)

    mask = mask.to(device)
    dose_tensor = dose_tensor.to(device)
    dose_times_list = [dt.to(device) for dt in dose_times_list]


    if truncation > 0:
        # Truncate time series
        t_encoder_list, x_encoder_list, masks_list, t_cut_list, x_cut_list = truncate_time_series(
            t_padded, x_padded, truncation
        )

        # Pad sequences to max length in batch
        t_encoder = pad_sequence(t_encoder_list, batch_first=True)       # [B, T_max]
        x_encoder = pad_sequence(x_encoder_list, batch_first=True)       # [B, T_max]
        # masks = pad_sequence(masks_list, batch_first=True)               # [B, T_max], bool

        # Also pad the removed/cut values for consistency
        t_cut = pad_sequence(t_cut_list, batch_first=True)               # [B, T_cut_max]
        x_cut = pad_sequence(x_cut_list, batch_first=True)               # [B, T_cut_max]

    else:
        t_encoder = t_padded
        x_encoder = x_padded
        t_cut, x_cut = t_padded, x_padded
        # masks = mask

    return (
        occ_list,
        t_padded,
        x_padded,
        t_encoder,
        x_encoder,
        t_cut,
        x_cut,
        mask,
        dose_tensor,
        dose_times_list,
        masks_list

    )




   

def encode_latent(encoder, t_encoder, x_normalized, 
                  enable_nf, enable_ae, enable_onlymedian):
    """
    Encodes input sequences into latent space, handling NF, AE, median-only, or VAE cases.

    Args:
        encoder: torch module
        t_encoder: padded time tensor [batch, seq_len]
        x_normalized: normalized concentration [batch, seq_len]
        enable_nf: whether to use normalizing flow
        enable_ae: whether AE mode
        enable_onlymedian: zero latent

    Returns:
        z_refined, mu_q, logvar_q, log_det (log_det is 0 if not NF)
    """
    # Remove first 4 measurements
    t_encoder_trimmed = t_encoder #[:, 4:]
    x_normalized_trimmed = x_normalized #[:, 4:]

    if enable_nf:
        _, z_refined, mu_q, logvar_q, log_det = encoder(t_encoder_trimmed, x_normalized_trimmed)
    elif enable_ae:
        _, _, mu_q, logvar_q, _ = encoder(t_encoder_trimmed, x_normalized_trimmed)
        z_refined = mu_q
        log_det = 0
    elif enable_onlymedian:
        _, _, mu_q, logvar_q, _ = encoder(t_encoder_trimmed, x_normalized_trimmed)
        z_refined = torch.zeros_like(mu_q)
        log_det = 0
    else:
        _, _, mu_q, logvar_q, _ = encoder(t_encoder_trimmed, x_normalized_trimmed)
        std_q = torch.exp(0.5 * logvar_q)
        z_refined = mu_q + std_q * torch.randn_like(mu_q)
        log_det = 0

    return z_refined, mu_q, logvar_q, log_det





def prepare_ode_input(
    initial_encoder,
    x_padded,
    z_refined,
    func,
    dose_tensor,
    dose_times_list,
    enable_vae
):
    """
    Prepare initial state and ODE function for the new TrajectoryDataset.

    Args:
        initial_encoder : nn.Module, encoder for initial state
        x_padded        : (batch, max_len) DV sequences
        z_refined       : latent refinement tensor
        func            : ODE function module
        dose_tensor     : (batch,) per-subject doses
        dose_times_list : list of per-subject dose times (tensors)

    Returns:
        x0       : initial latent state
        ode_func : ODEWrapper instance
    """

    # Encode initial state
#    print(x_padded[:, 0])
    z0, mu, logvar = initial_encoder(x_padded[:, 0].unsqueeze(1))
    

    if enable_vae:
        x0 = torch.cat([
               z0 ,
               z_refined
           ], dim=1)
    else:
        x0 = torch.cat([
               mu + 0.01 * torch.randn_like(mu) ,
               z_refined + 0.01 * torch.randn_like(z_refined) 
           ], dim=1)
        

    dose_mask = (dose_tensor != 0).float().unsqueeze(-1)  # [batch_size, 1]

    # Pad dose_times to a tensor
    max_doses = max([len(dt) for dt in dose_times_list])
    dose_times_padded = torch.zeros((len(dose_times_list), max_doses), device=x_padded.device)
    dose_times_mask = torch.zeros_like(dose_times_padded)
    for i, dt in enumerate(dose_times_list):
        if len(dt) > 0:
            dose_times_padded[i, :len(dt)] = dt
            dose_times_mask[i, :len(dt)] = 1

    # Expand dose_tensor to match dose_times
    dose_tensor_expanded = dose_tensor.unsqueeze(1).repeat(1, max_doses).unsqueeze(-1)
    


    # Create ODEWrapper with dose info
    ode_func = ODEWrapper(
        func,
        dose_times_padded,
        dose_tensor_expanded,
        dose_mask,
        dose_times_mask
    )

    return x0, ode_func, mu, logvar


def prepare_ode_input_eval(
    initial_encoder,
    x_padded,
    z_refined,
    func,
    dose_tensor,
    dose_times_list,

):
    """
    Prepare initial state and ODE function for the new TrajectoryDataset.

    Args:
        initial_encoder : nn.Module, encoder for initial state
        x_padded        : (batch, max_len) DV sequences
        z_refined       : latent refinement tensor
        func            : ODE function module
        dose_tensor     : (batch,) per-subject doses
        dose_times_list : list of per-subject dose times (tensors)

    Returns:
        x0       : initial latent state
        ode_func : ODEWrapper instance
    """

    # Encode initial state
#    print(x_padded[:, 0])
    z0, mu, logvar = initial_encoder(x_padded[:, 0].unsqueeze(1))
    

  
    x0 = torch.cat([mu ,z_refined], dim=1)
        

    dose_mask = (dose_tensor != 0).float().unsqueeze(-1)  # [batch_size, 1]

    # Pad dose_times to a tensor
    max_doses = max([len(dt) for dt in dose_times_list])
    dose_times_padded = torch.zeros((len(dose_times_list), max_doses), device=x_padded.device)
    dose_times_mask = torch.zeros_like(dose_times_padded)
    for i, dt in enumerate(dose_times_list):
        if len(dt) > 0:
            dose_times_padded[i, :len(dt)] = dt
            dose_times_mask[i, :len(dt)] = 1

    # Expand dose_tensor to match dose_times
    dose_tensor_expanded = dose_tensor.unsqueeze(1).repeat(1, max_doses).unsqueeze(-1)
    


    # Create ODEWrapper with dose info
    ode_func = ODEWrapper(
        func,
        dose_times_padded,
        dose_tensor_expanded,
        dose_mask,
        dose_times_mask
    )

    return x0, ode_func, mu, logvar



def make_predictions(t_padded, t_dense, x0, ode_func, reducer, latent_dim, global_mean, global_std):
    """
    Runs ODE forward and interpolates predictions at given time points.
    Returns:
        pred_interp: interpolated predictions at t_padded
        pred_batch: full ODE trajectory at dense time points
    """
    batch_size = t_padded.size(0)

    # Solve ODE
    pred = odeint(ode_func, x0, t_dense, method="rk4")
    
    pred_batch = pred.permute(1, 0, 2)  # [batch, time, latent_dim]
    
    
    
    # Reduce dimension and interpolate back to t_padded
    t_dense_exp = t_dense.unsqueeze(0).repeat(batch_size, 1)
    reduced = reducer(pred_batch[:, :, :latent_dim])
    
    reduced= destandardize_concentration(reduced, global_mean, global_std)
    
    pred_interp = batch_linear_interpolate_1d(reduced, t_dense_exp, t_padded)
    
    return pred_interp, reduced
