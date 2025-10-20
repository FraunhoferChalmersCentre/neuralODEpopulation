# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 00:16:56 2025

@author: Baaz
"""


import torch



import numpy as np

from contextlib import contextmanager


import torch.nn as nn
from torchdiffeq import odeint as odeint

from lib.utils.utils_preprocess import destandardize_concentration
from torch.nn.utils.rnn import pad_sequence




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




def randomly_truncate_time_series(t_batch, x_batch, drop_fraction):
    """
    Randomly truncate (remove) a percentage of points from each time series.

    Args:
        t_batch (list[torch.Tensor]): List of time tensors, each shaped [T_i].
        x_batch (list[torch.Tensor]): List of value tensors corresponding to t_batch, each shaped [T_i, ...].
        drop_fraction (float): Fraction of points to drop, e.g. 0.2 = remove 20%.

    Returns:
        tuple:
            kept_t (list[torch.Tensor]): Time tensors with points kept.
            kept_x (list[torch.Tensor]): Value tensors with points kept.
            masks (list[torch.BoolTensor]): Boolean masks indicating kept indices.
            removed_t (list[torch.Tensor]): Time tensors that were removed.
            removed_x (list[torch.Tensor]): Value tensors that were removed.
    """
    kept_t, kept_x, masks = [], [], []
    removed_t, removed_x = [], []

    for t_i, x_i in zip(t_batch, x_batch):
        n_points = t_i.shape[0]
        n_remove = int(n_points * drop_fraction)

        # Randomly choose indices to remove
        remove_indices = torch.randperm(n_points)[:n_remove]
        mask = torch.ones(n_points, dtype=torch.bool, device=t_i.device)
        mask[remove_indices] = False  # False = removed

        # Apply mask
        kept_t.append(t_i[mask])
        kept_x.append(x_i[mask])
        masks.append(mask)

        removed_t.append(t_i[~mask])
        removed_x.append(x_i[~mask])

    return kept_t, kept_x, masks, removed_t, removed_x


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
    def __init__(self, func, dose_times, dose_amounts, evid, dose_mask, keep_mask=None):
        super().__init__()
        self.func = func
        self.dose_times = dose_times      # [batch, max_doses]
        self.dose_amounts = dose_amounts  # [batch, max_doses]
        self.evid = evid                  # [batch, max_doses] (event type, e.g. 1=bolus, 2=transfusion)
        self.dose_mask = dose_mask        # [batch, max_doses]
        self.keep_mask = keep_mask        # [batch, 1] — optional binary mask

    def forward(self, t, x):
        if self.keep_mask is not None:
            return self.func(
                t, x,
                self.dose_times,
                self.dose_amounts,
                self.evid,
                self.dose_mask,
                self.keep_mask
            )
        else:
            return self.func(
                t, x,
                self.dose_times,
                self.dose_amounts,
                self.evid,
                self.dose_mask
            )


def preprocess_batch(batch, device, truncation=-1.0, skip_initial=0):
    """
    Preprocess a batch with optional truncation and removal of first `skip_initial` points.

    Args:
        batch: output from collate_fn
        device: torch device
        truncation: optional truncation for encoder
        skip_initial: number of initial time points to remove

    Returns:
        occ_list       : list of subject IDs
        treatment_list : list of treatment IDs
        t_padded       : padded time sequences [B, T]
        x_padded       : padded DV sequences [B, T]
        t_encoder      : encoder time sequences [B, T_max-skip_initial]
        x_encoder      : encoder DV sequences [B, T_max-skip_initial]
        t_cut          : truncated times that were removed [B, T_cut]
        x_cut          : truncated values that were removed [B, T_cut]
        masks          : mask for valid encoder points [B, T_max-skip_initial]
        dose_tensor    : per-subject dose amounts
        dose_times_list: per-subject dose times
        evid           : per-subject EVID tensor
    """

    # Unpack batch
    occ_list, treatment_list, t_batch, x_batch, mask, dose_tensor, dose_times_list, evid = batch

    # Move to device
    t_batch = [t.to(device) for t in t_batch]
    x_batch = [x.to(device) for x in x_batch]
    mask = mask.to(device)
    dose_tensor = dose_tensor.to(device)
    dose_times_list = [dt.to(device) for dt in dose_times_list]
    evid = evid.to(device)

    # Truncate if needed
    if truncation > 0:
        t_enc_list, x_enc_list, masks_list, t_cut_list, x_cut_list = truncate_time_series(
            t_batch, x_batch, truncation
        )
    else:
        t_enc_list, x_enc_list, masks_list = t_batch, x_batch, [torch.ones_like(xi, dtype=torch.bool) for xi in x_batch]
        t_cut_list, x_cut_list = [torch.zeros(0, device=device) for _ in t_batch], [torch.zeros(0, device=device) for _ in x_batch]

        # # Remove first `skip_initial` points from encoder sequences
        # t_enc_list = [ti[skip_initial:] for ti in t_enc_list]
        # x_enc_list = [xi[skip_initial:] for xi in x_enc_list]
        # masks_list = [mask_i[skip_initial:] for mask_i in masks_list]

    # Pad sequences
    t_encoder = pad_sequence(t_enc_list, batch_first=True)
    x_encoder = pad_sequence(x_enc_list, batch_first=True)
    masks = pad_sequence(masks_list, batch_first=True)

    t_padded = pad_sequence(t_batch, batch_first=True)
    x_padded = pad_sequence(x_batch, batch_first=True)
    t_cut = pad_sequence(t_cut_list, batch_first=True)
    x_cut = pad_sequence(x_cut_list, batch_first=True)

    return (
        occ_list,
        treatment_list,
        t_padded,
        x_padded,
        t_encoder,
        x_encoder,
        t_cut,
        x_cut,
        masks,
        dose_tensor,
        dose_times_list,
        evid
    )











def encode_latent(
    encoder,
    t_encoder,
    x_normalized,
    dose_tensor,
    enable_vae,
    enable_ae,
    enable_onlymedian,
    warmup_epochs_iiv=0,
    epoch=0,
    min_batch_size=50,
    augment=False
):
    """
    Encodes input sequences into latent space, handling AE, median-only, or VAE cases.
    For VAE, ensures the total number of latent samples (k_param) is at least min_batch_size.
    If below threshold, repeats each individual the same number of times.

    Args:
        encoder: torch module
        t_encoder: [batch, seq_len] time tensor
        x_normalized: [batch, seq_len] normalized concentration tensor
        enable_vae: use VAE mode
        enable_ae: use AE mode
        enable_onlymedian: zero latent
        warmup_epochs_iiv: warmup epochs for KL annealing
        epoch: current epoch
        min_batch_size: minimum number of latent samples

    Returns:
        k_param: latent samples [batch_repeated, latent_dim]
        mu_q: mean of latent [batch_repeated, latent_dim]
        logvar_q: log variance [batch_repeated, latent_dim]
        repeat_factor: how many times each individual was repeated
    """

    B, T = t_encoder.size()
    repeat_factor = 1  # default

    # Remove first time step
    t_encoder_trimmed = t_encoder
    x_normalized_trimmed = x_normalized

    # Handle empty sequence
    # if t_encoder_trimmed.size(1) == 0:
    #     latent_dim = getattr(encoder, "latent_dim", 2)
    #     k_param = torch.zeros(B, latent_dim, device=t_encoder.device)
    #     mu_q = torch.zeros(B, latent_dim, device=t_encoder.device)
    #     logvar_q = torch.zeros(B, latent_dim, device=t_encoder.device)
    #     return k_param, mu_q, L_p, repeat_factor

    # ----------------------------
    # AE and median-only cases
    # ----------------------------
    if enable_ae:
        k_param, z0, mu_q, L_q, mu_p, L_p = encoder(t_encoder_trimmed, x_normalized_trimmed,dose_tensor)
        k_param = mu_q
      #  k_param=torch.cat([ z0, k_param], dim=-1)
        


    elif enable_onlymedian:
        
        k_param, z0, mu_q, L_q, mu_p, L_p = encoder(t_encoder_trimmed, x_normalized_trimmed,dose_tensor)
        
#        k_param=torch.cat([z0,0*k_param], dim=-1)
        k_param =0* k_param
        mu_q = mu_p
        L_q = L_p
      #  print(k_param)
       # print(k_param)
    # ----------------------------
    # VAE case
    # ----------------------------
    # ----------------------------
    elif enable_vae:
        # KL weight warmup
        kl_weight = 1.0 if epoch + 1 >= warmup_epochs_iiv else min(1.0, epoch / warmup_epochs_iiv)
       # print(t_encoder_trimmed)
        # Encode original batch
        k_param, z0, mu_q, L_q, mu_p, L_p  = encoder(t_encoder_trimmed, x_normalized_trimmed,dose_tensor)  # L_q: [B, D, D]
    
        B_current = mu_q.size(0)
        D = mu_q.size(1)
    
        repeat_factor = 1  # default in case batch is large enough
    
        # Repeat if batch too small
        if B_current < min_batch_size and augment:
            repeat_factor = int(np.ceil(min_batch_size / B_current))
            mu_q = mu_q.repeat_interleave(repeat_factor, dim=0)  # [B*rf, D]
            mu_p = mu_p.repeat_interleave(repeat_factor, dim=0)  # [B*rf, D]
            z0=z0.repeat_interleave(repeat_factor, dim=0)  # [B*rf, D]
            
            
            L_p = L_p.repeat(repeat_factor, 1, 1)  
            L_q = L_q.repeat(repeat_factor, 1, 1)                # [B*rf, D, D]
          #  L_q = L_q.repeat_interleave(repeat_factor, dim=0)  # [B*rf, D]

        B, D = mu_q.shape
        eps = torch.randn(B, D, device=mu_q.device)  # standard normal
        k_param = mu_q + torch.einsum("bij,bj->bi", L_q, eps)  # [B, D]
     #   k_param=torch.cat([z0, k_param ], dim=-1)   
     #   print(k_param)    
  
   # print(k_param)
   # print(k_param)   
    return k_param, mu_q, L_q, mu_p, L_p, repeat_factor
    





@contextmanager
def use_ema(model):
    """
    Context manager to temporarily replace a model's parameters with EMA parameters.
    Falls back to current parameters if EMA has never been updated.

    Usage:
        with use_ema(model):
            y_pred = model(x)
    """
    # Check if EMA is available and has been updated
    has_ema = hasattr(model, "iteration") and model.iteration.item() > 1

    if not has_ema:
        # Fail-safe: just use the current parameters
        yield
        return

    ema_attrs = [
        name for name, _ in model.named_parameters()
        if hasattr(model, f"{name.replace('.', '_')}_ema")
    ]
    backup = {}

    if ema_attrs:
        # Backup and swap
        for name, param in model.named_parameters():
            backup[name] = param.data.clone()
            ema_param = getattr(model, f"{name.replace('.', '_')}_ema", None)
            if ema_param is not None:
                # Ensure correct device
                param.data.copy_(ema_param.to(param.device))
    try:
        yield
    finally:
        # Restore original params
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])





def normalize_encoder_input(
    x_encoder, t_encoder, x_padded, dose_tensor, dose_times_list, evid,
    encoder_med, func_med, reducer_med,
    t_dense, global_mean, global_std
):
    """
    Normalize encoder input using the median model (EMA weights).

    Handles broadcasting of dose_times and evid to match the time dimension.
    Works for variable batch sizes and number of doses.
    """

    batch_size, n_timepoints = x_encoder.shape

    # --- Encode latent with median encoder ---
    with use_ema(encoder_med):
        k_param, _, _, _ ,_,_= encode_latent(
            encoder_med,
            t_encoder,
            x_encoder,
            dose_tensor,
            enable_vae=False,
            enable_ae=False,
            enable_onlymedian=True
        )

    # --- Prepare ODE input with median models ---

        with use_ema(func_med):
            ode_func_med= prepare_ode_input(
    
                x_padded,
                k_param,
                func_med,
                dose_tensor,         # shape: [batch, T] or [batch, n_features]
                dose_times_list, # list of 1D tensors per individual
                evid,            # list of 1D tensors per individual
                enable_ae=False,
                enable_onlymedian=True
            )

        with use_ema(reducer_med):
                pred_interp, pred_batch = make_predictions(
                    t_encoder, t_dense, k_param, ode_func_med, reducer_med,
                     global_mean, global_std
                )

    # --- Normalize encoder input ---
    x_encoder_norm = destandardize_concentration(x_encoder, global_mean, global_std) / pred_interp
    return x_encoder_norm.detach()




def prepare_ode_input(
    x_padded,
    k_param,
    func,
    dose_tensor,
    dose_times_list,
    evid,
    enable_ae,enable_onlymedian=False, repeat_factor=1
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
    
    
    if repeat_factor > 1:
        x_padded_repeated = x_padded.repeat_interleave(repeat_factor, dim=0)
    else:
        x_padded_repeated = x_padded
        
        
        
 
    evid= evid.repeat_interleave(repeat_factor, dim=0)
   

   
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
    # Ensure dose_tensor is 2D: [batch, max_doses]
    # Expand dose_tensor to match dose_times
    # Ensure dose_tensor is 2D: [batch, 1] first
    if dose_tensor.dim() == 0:  # scalar
        dose_tensor_expanded = dose_tensor.unsqueeze(0).unsqueeze(1)  # [1,1]
    elif dose_tensor.dim() == 1:  # [batch]
        dose_tensor_expanded = dose_tensor.unsqueeze(1)  # [batch, 1]
    else:  # already 2D
        dose_tensor_expanded = dose_tensor
    
    # Repeat along columns to match max_doses
    dose_tensor_expanded = dose_tensor_expanded[:, :1].repeat(1, max_doses)  # [batch, max_doses]
    
      

    if repeat_factor > 1:
        # Repeat batch-dependent tensors along dim=0
        dose_mask = dose_mask.repeat_interleave(repeat_factor, dim=0)
        dose_times_padded = dose_times_padded.repeat_interleave(repeat_factor, dim=0)
        dose_times_mask = dose_times_mask.repeat_interleave(repeat_factor, dim=0)
        dose_tensor_expanded = dose_tensor_expanded.repeat_interleave(repeat_factor, dim=0)
    
   

    # Create ODEWrapper with dose info
    ode_func = ODEWrapper(
    func,
    dose_times_padded,
    dose_tensor_expanded,
    evid,
    dose_mask,
    dose_times_mask
)

    return ode_func






def make_predictions(t_padded, t_dense, k_param, ode_func, reducer, global_mean, global_std, repeat_factor=1):
    """
    Runs ODE forward and interpolates predictions at given time points.
    Scales extreme predictions to keep gradients alive.
    """
    # Expand t_padded if batch was repeated
    if repeat_factor > 1:
        t_padded = t_padded.repeat_interleave(repeat_factor, dim=0)

    batch_size = t_padded.size(0)
    latent_dim=reducer.latent_dim
    # Solve ODE
   # print(k_param)
  #  print(k_param)
    pred = odeint(ode_func, k_param, t_dense, method="rk4")  # [time, batch, latent_dim_total]
    
    # Smoothly scale down large values while preserving gradients
    pred_batch = pred.permute(1, 0, 2)  # [batch, time, latent_dim_total]
  #  print(pred_batch)
    # Reduce dimension
    preds=pred_batch[:, :, :latent_dim]
  #  B, T, D = pred_batch.shape
   # mask = torch.ones_like(preds)
   # mask[:, 1:, 2] = 0

   #masked_pred_batch = preds * mask
    reduced = reducer(preds)
    
    # Destandardize
    reduced = destandardize_concentration(reduced, global_mean, global_std)
    # Interpolate back to padded time points
    t_dense_exp = t_dense.unsqueeze(0).repeat(batch_size, 1)
    pred_interp = batch_linear_interpolate_1d(reduced, t_dense_exp, t_padded)

    return pred_interp, reduced



