# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 00:16:56 2025

@author: Baaz
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 07:10:20 2025

@author: Baaz
"""

import torch



import matplotlib.pyplot as plt



from torch.nn.utils.rnn import pad_sequence


import torch.nn as nn
from torchdiffeq import odeint as odeint

from lib.utils.utils_preprocess import standardize_concentration, destandardize_concentration, collate_fn



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
    occ_list, t_padded, x_padded, mask, dose_tensor, dose_times_list, x_dose_padded, auc_tensor = batch


    # Move tensors to device
    t_padded = t_padded.to(device)
    x_padded = x_padded.to(device)
    x_dose_padded=x_dose_padded.to(device)
    mask = mask.to(device)
    dose_tensor = dose_tensor.to(device)
    dose_times_list = [dt.to(device) for dt in dose_times_list]
    auc_tensor = auc_tensor.to(device)

    # if truncation > 0:
    #     # Truncate time series
    #     t_encoder_list, x_encoder_list, masks_list, t_cut_list, x_cut_list = truncate_time_series(
    #         t_padded, x_dose_padded, truncation
    #     )

    #     # Pad sequences to max length in batch
    #     t_encoder = pad_sequence(t_encoder_list, batch_first=True)       # [B, T_max]
    #     x_encoder = pad_sequence(x_encoder_list, batch_first=True)       # [B, T_max]
    #    # masks = pad_sequence(masks_list, batch_first=True)               # [B, T_max], bool

    #     # Also pad the removed/cut values for consistency
    #     t_cut = pad_sequence(t_cut_list, batch_first=True)               # [B, T_cut_max]
    #     x_cut = pad_sequence(x_cut_list, batch_first=True)               # [B, T_cut_max]

   # else:
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
        auc_tensor

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
        k_param, mu_q, logvar_q, log_det (log_det is 0 if not NF)
    """
    # Remove first 4 measurements
    t_encoder_trimmed = t_encoder
    x_normalized_trimmed = x_normalized

    if enable_nf:
        _, k_param, mu_q, logvar_q, log_det = encoder(t_encoder_trimmed, x_normalized_trimmed)
    elif enable_ae:
        _, _, mu_q, logvar_q, _ = encoder(t_encoder_trimmed, x_normalized_trimmed)
        k_param = mu_q
        log_det = 0
        
    elif enable_onlymedian:
        _, _, mu_q, logvar_q, _ = encoder(t_encoder_trimmed, x_normalized_trimmed)
        k_param = torch.zeros_like(mu_q)
        mu_q=torch.zeros_like(mu_q)
        logvar_q= torch.zeros_like(logvar_q)
        log_det = 0
    else:
        _, _, mu_q, logvar_q, _ = encoder(t_encoder_trimmed, x_normalized_trimmed)
        std_q = torch.exp(0.5 * logvar_q)
        k_param = mu_q + std_q * torch.randn_like(mu_q)
        log_det = 0
        
    return k_param, mu_q, logvar_q, log_det


def prepare_ode_input(
    initial_encoder,
    x_padded,
    k_param,
    func,
    dose_tensor,
    dose_times_list,
    enable_ae
):
    """
    Prepare initial state and ODE function for the new TrajectoryDataset.

    Args:
        initial_encoder : nn.Module, encoder for initial state
        x_padded        : (batch, max_len) DV sequences
        k_param       : latent refinement tensor
        func            : ODE function module
        dose_tensor     : (batch,) per-subject doses
        dose_times_list : list of per-subject dose times (tensors)

    Returns:
        x0       : initial latent state
        ode_func : ODEWrapper instance
    """

    # Encode initial state
    x0_1 = initial_encoder(x_padded[:, 0].unsqueeze(1))
 
    
    
    if enable_ae:
        x0 = torch.cat([
            x0_1 + 0.01 * torch.randn_like(x0_1),
            k_param + 0.01 * torch.randn_like(k_param) 
        ], dim=1)
    else:    
        x0 = torch.cat([
            x0_1,
            k_param 
        ], dim=1)
        
    # Dose mask: 1 where dose exists, 0 otherwise
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

    return x0, ode_func






def make_predictions(t_padded, t_dense, x0, ode_func, reducer, latent_dim, global_mean, global_std):
    """
    Runs ODE forward and interpolates predictions at given time points.
    Scales extreme predictions to keep gradients alive.
    """
    batch_size = t_padded.size(0)

    # Solve ODE
    pred = odeint(ode_func, x0, t_dense, method="rk4")

    # Replace NaN/Inf with zero temporarily (or any finite number) to avoid breaking graph
    pred = torch.where(torch.isfinite(pred), pred, torch.zeros_like(pred))

    # Smoothly scale down large values while preserving gradients

    pred_batch = pred.permute(1, 0, 2)  # [batch, time, latent_dim]

    # Reduce dimension
    reduced = reducer(pred_batch[:, :, :latent_dim])

    # Destandardize
    reduced = destandardize_concentration(reduced, global_mean, global_std)

    # Interpolate back to padded time points
    t_dense_exp = t_dense.unsqueeze(0).repeat(batch_size, 1)
    pred_interp = batch_linear_interpolate_1d(reduced, t_dense_exp, t_padded)

    return pred_interp, reduced


