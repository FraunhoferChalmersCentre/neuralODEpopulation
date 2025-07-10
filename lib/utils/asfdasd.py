# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 18:00:43 2025

@author: Baaz
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 17:44:37 2025

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
#from torchdiffeq import odeint as odeint
from torchdiffeq import odeint as odeint

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.integrate import solve_ivp

import os

# ---- Settings ----
compartment = 'C2'

from lib.models.NNmodels_parallel import *


def approx_dirac_delta_vectorized(t, dose_times, dt, scaling=0.02,normalize=1):
    epsilon = scaling
    return np.sum(np.exp(-((t - dose_times) / epsilon)**2) / (epsilon * np.sqrt(np.pi)))/normalize




def pk_2cpt_step(y, ka, cl, v, dose_amount, dose_times, t, dt):
    # y: shape (N_samples, 2) for C1, C2 at time t
    C1 = y[:, 0]
    C2 = y[:, 1]
    ka = np.asarray(ka)
    cl = np.asarray(cl)
    v = float(v)  # scalar
    dose_amount = float(dose_amount)  # scalar
    dose_times = np.asarray(dose_times)
    y = np.asarray(y)

    # Vectorized dose input (same for all samples since dose times & amount same)
    dose_input = dose_amount * approx_dirac_delta_vectorized(t, dose_times,dt)

    dC1dt = -ka * C1 + dose_input
    dC2dt = ka * C1 - (cl / v) * C2

    dy = np.stack([dC1dt, dC2dt], axis=1)
    y_next = y + dy * dt
    return y_next

def solve_individual_vectorized(ka, cl, v, dose_amount, dose_times, t_eval):
    # ka, cl shape: (N_samples,)
    N_samples = ka.shape[0]
    y = np.zeros((N_samples, 2))
    y[:, 0] = 0  # initial C1
    y[:, 1] = 0            # initial C2

    y_out = np.zeros((N_samples, len(t_eval)))
    y_out[:, 0] = y[:, 1]

    for i in range(1, len(t_eval)):
        dt = t_eval[i] - t_eval[i-1]
        y = pk_2cpt_step(y, ka, cl, v, dose_amount, dose_times, t_eval[i-1], dt)
        y_out[:, i] = y[:,1]  # store C2

    return y_out


def log_prior(phi, mu_prior, sigma_prior):
    # phi: vector of parameters [log ka, log cl, log v]
    # Assuming independent normals
    return -0.5 * np.sum(((phi - mu_prior) / sigma_prior)**2)

def log_likelihood(phi, t_np,t_data, y_obs, dose, dose_times, add_error, prop_error):
    """
    phi: array-like, log parameters [log(ka), log(cl)]
    t_np: 1D np.array of time points to evaluate
    y_obs: observed data at t_np
    dose: scalar dose amount
    dose_times: array-like of dose administration times
    add_error: additive error std dev
    prop_error: proportional error coefficient
    """

    ka, cl = np.exp(phi)
    v = np.log(5.0)  # fixed

    # Convert to 1D arrays if scalar (batch size = 1)
    ka_arr = np.atleast_1d(ka)
    cl_arr = np.atleast_1d(cl)

    # Solve ODE with batch size = 1
    sol = solve_individual_vectorized(
        ka_arr, cl_arr, v,
        dose,
        dose_times,
        t_np
    )
    # sol shape: (batch_size, len(t_np)) = (1, T)
    y_pred_dense = torch.tensor(sol, dtype=torch.float32)      # [batch=1, T_dense]
    t_dense = torch.tensor(t_np, dtype=torch.float32).unsqueeze(0)  # [1, T_dense]
    t_target = torch.tensor(t_data, dtype=torch.float32).unsqueeze(0)         # [1, T_obs]
   


     # Interpolate predictions at observed times
    y_pred_interp = torch_linear_interpolate(t_dense,y_pred_dense, t_target)
    y_pred_interp_np = y_pred_interp[0].numpy()  # back to numpy 1D array
    
     # Calculate residuals and std dev
    sigma = np.sqrt(add_error**2 + (prop_error * y_pred_interp_np)**2)
    residuals = y_obs - y_pred_interp_np
    log_like = -0.5 * np.sum((residuals / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))

    return log_like



def metropolis_hastings_sampling(
    y_obs, t_np,t_data, dose, dose_times, add_error, prop_error,
    mu_prior, sigma_prior,
    n_samples, burn_in, thinning
):
    

    # Initial guess at prior mean
    phi_curr = mu_prior.copy()
    log_post_curr = log_prior(phi_curr, mu_prior, sigma_prior) + log_likelihood(phi_curr, t_np,t_data, y_obs, dose, dose_times, add_error, prop_error)

    samples = []
    proposal_std = 0.1  # Tune this for acceptance rate ~0.3

    for i in range(n_samples * thinning + burn_in):
        phi_prop = phi_curr + np.random.normal(0, proposal_std, size=phi_curr.shape)
        log_post_prop = log_prior(phi_prop, mu_prior, sigma_prior) + log_likelihood(phi_prop, t_np,t_data, y_obs, dose, dose_times, add_error, prop_error)
        
        accept_ratio = np.exp(log_post_prop - log_post_curr)
        if np.random.rand() < accept_ratio:
            phi_curr = phi_prop
            log_post_curr = log_post_prop
        
        if i >= burn_in and (i - burn_in) % thinning == 0:
            samples.append(phi_curr.copy())

    return np.array(samples)

def pad_dose_times(dose_times_list, pad_value=-1.0):
    batch_size = len(dose_times_list)
    max_len = max([dt.size(0) for dt in dose_times_list])
    
    dose_times_padded = torch.full((batch_size, max_len), pad_value, dtype=dose_times_list[0].dtype, device=dose_times_list[0].device)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=dose_times_list[0].device)
    
    for i, dt in enumerate(dose_times_list):
        length = dt.size(0)
        dose_times_padded[i, :length] = dt
        mask[i, :length] = 1
    
    return dose_times_padded, mask

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

    
def torch_linear_interpolate(x_dense, y_dense, x_target):
    batch_size, N = x_dense.shape
    M = x_target.shape[1]

    # Find indices for interpolation
    idx = torch.searchsorted(x_dense, x_target, right=True)
    idx = torch.clamp(idx, 1, N - 1)  # shape: [batch, M]

    # Batch indices for advanced indexing
    batch_indices = torch.arange(batch_size, device=x_dense.device).unsqueeze(1).expand(-1, M)

    x0 = x_dense[batch_indices, idx - 1]
    x1 = x_dense[batch_indices, idx]
    y0 = y_dense[batch_indices, idx - 1]
    y1 = y_dense[batch_indices, idx]

    denom = x1 - x0
    denom[denom == 0] = 1e-8
    slope = (y1 - y0) / denom

    return y0 + slope * (x_target - x0)

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





def estimate_max_dose(df, dose_col='Dose'):
    """Estimate max dose from dataframe but limit to max_allowed."""
    max_dose = df[dose_col].max()
    return max_dose

def estimate_max_time(df, time_col='Time'):
    """Estimate max time from dataframe but limit to max_allowed."""
    max_time = df[time_col].max()
    return max_time

def truncate_time_series2(t_batch, x_batch, max_points):
    truncated_t, truncated_x = [], []
    for t_i, x_i in zip(t_batch, x_batch):
        num_points = min(max_points, t_i.size(0))
        truncated_t.append(t_i[:num_points])
        truncated_x.append(x_i[:num_points])
    return truncated_t, truncated_x

def truncate_time_series(t_batch, x_batch, truncation_time):
    truncated_t, truncated_x = [], []
    for t_i, x_i in zip(t_batch, x_batch):
        mask = t_i <= truncation_time
        truncated_t.append(t_i[mask])
        truncated_x.append(x_i[mask])
    return truncated_t, truncated_x


def standardize_concentration(conc, mean, std): return (conc - mean) / std
def destandardize_concentration(norm_conc, mean, std): return norm_conc * std + mean



    
def plot_from_training_records(batch_size, device, df, dataset, latent_dim,
                               records,
                               func,
                               reducer,
                               initial_encoder,
                               ODEWrapper,
                               t_dense,
                               max_plots=6, 
                               nr_row=2,
                               nr_col=3,
                               compartment="central",
                              ):
    MAX_TIME = estimate_max_time(df)
    MAX_DOSE = estimate_max_dose(df)
    conc_mean, conc_std = dataset.conc_mean, dataset.conc_std
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    dose_times_lists = df['Dose times'].apply(ast.literal_eval).tolist()
    all_times = np.concatenate(dose_times_lists)
    
    unique_times_np = np.unique(all_times)
    unique_times = torch.from_numpy(unique_times_np).to(dtype=torch.float32, device=device)
    dose_times_tensor = unique_times / MAX_TIME
    
    # Combine unique dose times and dense times, ensure sorted & unique
    t_dense = torch.unique(torch.cat([t_dense.to(device), dose_times_tensor]))
    
    device = next(func.parameters()).device  # get device from model (usually cuda)
    n = min(len(records), max_plots)

    # Prepare batch data containers
    subject_ids = []
    ts = []
    xs = []
    doses = []
    dose_times_list = []
    z_refined_list = []

    for i in range(n):
        subject_id, t_real, x_real, dose, dose_times, z_refined = records[i]
        subject_ids.append(subject_id)
        ts.append(t_real)
        xs.append(x_real)
        # Make sure doses are tensors with shape [num_doses] or [1]
        dose_tensor = dose if isinstance(dose, torch.Tensor) else torch.tensor(dose)
        dose_tensor = dose_tensor.unsqueeze(-1) if dose_tensor.dim() == 0 else dose_tensor
        doses.append(dose_tensor)
        
        dose_times_tensor = dose_times if isinstance(dose_times, torch.Tensor) else torch.tensor(dose_times)
        dose_times_list.append(dose_times_tensor)
        
        z_refined_tensor = z_refined if isinstance(z_refined, torch.Tensor) else torch.tensor(z_refined)
        z_refined_list.append(z_refined_tensor)

    # Pad dose times & doses to max_len for batch processing
    dose_times_padded = torch.nn.utils.rnn.pad_sequence(dose_times_list, batch_first=True, padding_value=0)  # [batch_size, max_len]
    doses_padded = torch.nn.utils.rnn.pad_sequence(doses, batch_first=True, padding_value=0)  # [batch_size, max_len] or [batch_size, max_len, 1]

    # Important fix: Make sure doses_padded is 2D [batch_size, max_len]
    if doses_padded.dim() == 3:
        doses_padded = doses_padded.squeeze(-1)  # remove last dim if it's 1

    # Create dose mask where dose_times != 0 (or some other logic depending on your data)
    dose_mask = (dose_times_padded != 0)  # bool tensor [batch_size, max_len]

    z_refined = torch.stack(z_refined_list)  # [batch_size, latent_dim]

    # Encode initial states
    x0_list = []
    for x in xs:
        out = initial_encoder(x[0].unsqueeze(0))  # expecting [1, feature_dim]
        if out.dim() == 1:
            out = out.unsqueeze(0)
        x0_list.append(out)
    x0_tensor = torch.cat(x0_list, dim=0)  # [batch_size, feature_dim]

    # Concatenate encoded latent states with refined z
    x0 = torch.cat([x0_tensor, z_refined], dim=1)  # [batch_size, encoded_dim + latent_dim]

    # Move all to device
    doses_padded = doses_padded.to(device)
    dose_times_padded = dose_times_padded.to(device)
    dose_mask = dose_mask.to(device)
    z_refined = z_refined.to(device)
    x0 = x0.to(device)
    t_dense = t_dense.to(device)

    # --- FIX START ---
    # doses_padded: [batch_size, max_len]
    # dose_times_padded: [batch_size, max_len]

    # Make sure dose amounts and times have the same shape for stacking
    # Unsqueeze last dim to make them [batch_size, max_len, 1]
    dose_times_expanded = dose_times_padded.unsqueeze(-1)  # [batch_size, max_len, 1]
    doses_expanded = doses_padded.unsqueeze(-1)           # [batch_size, max_len, 1]
    
  

    # Make sure doses_padded has shape [batch_size, 1]
    doses_padded = doses_padded.to(device)
    
    # Repeat doses to match dose_times length along dim=1
    dose_amounts_expanded = doses_padded.repeat(1, dose_times_padded.size(1))  # [5, 4]
    dose_amounts_expanded = dose_amounts_expanded.unsqueeze(-1)  # [5, 4, 1]
    
    dose_times_expanded = dose_times_padded.unsqueeze(-1)  # [5, 4, 1]
    dose_mask = dose_mask.to(device)
   
    
    ode_func = ODEWrapper(func, dose_times_expanded, dose_amounts_expanded, dose_mask)

    # --- FIX END ---

    pred = odeint(ode_func, x0, t_dense, method='dopri5')

    # Apply reducer on latent dims only (exclude dose dims)
    x_pred = reducer(pred[:, :, :latent_dim])  # [time_steps, batch_size, output_dim]

    # Plotting code unchanged
    fig, axs = plt.subplots(nr_row, nr_col, figsize=(nr_col * 6, nr_row * 5), sharex=True)
    axs = axs.flatten()

    for i in range(n):
        ax = axs[i]

        x_interp = torch.tensor(
            np.interp(
                t_dense.cpu().numpy(),
                ts[i].cpu().numpy(),
                xs[i].cpu().numpy()
            ),
            dtype=torch.float32,
            device=ts[i].device,
        )

        ax.plot(ts[i].cpu().numpy() * MAX_TIME, (xs[i].cpu().numpy() * conc_std) + conc_mean, 'o-', label='Actual')
        ax.plot(t_dense.cpu().numpy() * MAX_TIME, (x_pred[:, i].detach().cpu().numpy() * conc_std) + conc_mean, '-', label='Predicted')
        ax.set_title(f"Individual {subject_ids[i]} - Dose: {doses_padded[i].sum().item() * MAX_DOSE:.0f} mg")
        ax.set_ylabel(f"Concentration ({compartment})")
        ax.grid(True)
        ax.legend()

    axs[-1].set_xlabel("Time (hours)")
    plt.tight_layout()
    plt.show()
    plt.close()



def save_models(models: dict, save_dir: str, model_name: str):
    """
    Save multiple models to a given directory.

    Args:
        models (dict): Dictionary with model names as keys and model instances as values.
        save_dir (str): Directory to save the models.
        model_name (str): Base name to prepend to each saved model file.
    """
    os.makedirs(save_dir, exist_ok=True)
    for key, model in models.items():
        path = os.path.join(save_dir, f"{model_name}_{key}.pt")
        torch.save(model.state_dict(), path)
    print(f"Saved {len(models)} models to {save_dir}")

def load_models(models: dict, load_dir: str, model_name: str, device=torch.device("cpu")):
    """
    Load model states into existing model instances from a directory.

    Args:
        models (dict): Dictionary with model names as keys and initialized model instances as values.
        load_dir (str): Directory where models are saved.
        model_name (str): Base name prepended to each saved model file.
        device (torch.device): Device to map the loaded model parameters.

    Returns:
        dict: Dictionary with loaded model instances.
    """
    for key, model in models.items():
        path = os.path.join(load_dir, f"{model_name}_{key}.pt")
        if os.path.isfile(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            print(f"Loaded {key} from {path}")
        else:
            print(f"Warning: Model file {path} not found. Skipping load for {key}.")
    
   # models=models.to(device)
    return models    


import torch

def dropout_rows(z, p,device):
    # z: tensor of shape [num_rows, dim]
    # p: proportion of rows to zero out
    assert 0 <= p <= 1, "p must be between 0 and 1"
    
    num_rows = z.size(0)
    keep_mask = (torch.rand(num_rows, device=device) > p).float().unsqueeze(1)  # Shape: [num_rows, 1]
    output=z * keep_mask
    output=output.to(device)
    return output, keep_mask


def train_model(dose_encoder, p,models, optimizer,scheduler,dim_parameter_encoder, latent_dim, func, reducer, initial_encoder,
    encoder1, noise, device, t_dense,
     n_epochs, warmup_epochs_noise,warmup_epochs_iiv,
    smoothing_start_epoch,
    remove_encoder, ae,nf,onlymedian,plot_from_training_records_enable, free_bits,batch_size,df, dataset, max_points_visible, lr, print_epoch=1,
    plot_epoch=1,
    max_plots=4,
    nr_col=1,
    nr_row=5):
    
   
  
      
    t_dense = t_dense.to(device)
    base_lr = lr  # your existing lr, e.g., 1e-3
    
    if nf:
        flow_lr = base_lr * 1  # reduce flow LR by 10x, for example
        
        # Extract flow parameters from both encoders
        flow_params = list(encoder1.flow.parameters())
        
        # Extract all encoder parameters combined
        all_encoder_params = list(encoder1.parameters()) 
        
        # Non-flow encoder parameters = all encoder params - flow params
        non_flow_encoder_params = [p for p in all_encoder_params if id(p) not in {id(fp) for fp in flow_params}]
    
        
        main_params = [
            {"params": list(func.parameters()) + list(reducer.parameters()) + list(initial_encoder.parameters()) + non_flow_encoder_params, "lr": base_lr},
            {"params": flow_params, "lr": flow_lr},  # lower LR for flow params
            {"params": list(noise.parameters()), "lr": base_lr},
        ]
    else:
         main_params = [
             {"params": list(func.parameters()) + list(reducer.parameters()) + list(initial_encoder.parameters()) +list(encoder1.parameters()) , "lr": base_lr},
             {"params": list(noise.parameters()), "lr": base_lr},
         ]
         


    
    MAX_TIME = estimate_max_time(df)
    MAX_DOSE = estimate_max_dose(df)
    conc_mean, conc_std = dataset.conc_mean, dataset.conc_std
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    dose_times_lists = df['Dose times'].apply(ast.literal_eval).tolist()
    all_times = np.concatenate(dose_times_lists)
    
    unique_times_np = np.unique(all_times)
    unique_times = torch.from_numpy(unique_times_np).to(dtype=torch.float32, device=device)
    dose_times_tensor = unique_times / MAX_TIME
    
    t_dense = torch.unique(torch.cat([t_dense.to(device), dose_times_tensor]))



    if nf:
        print(f"{'---- NF training initialized ----' if nf else '---- AE training initialized ----'}")
    else:
        print(f"{'---- AE training initialized ----' if ae else '---- VAE training initialized ----'}")

    print(f"Total epochs: {n_epochs}")
    print(f"Warmup epochs (noise): {warmup_epochs_noise}")
    print(f"Warmup epochs (IIV): {warmup_epochs_iiv}")
    print(f"Smoothing (EMA) starts after epoch: {smoothing_start_epoch}")
    print("========================================")


    
    for epoch in range(n_epochs):
        first_batch = True

        # Log when key stages start
        if epoch == warmup_epochs_noise:
            print(f"[Epoch {epoch}] ➤ Noise training activated.")
        
        if epoch == warmup_epochs_iiv:
            print(f"[Epoch {epoch}] ➤ Full KL regularization activated.")
        
        if epoch == smoothing_start_epoch:
            print(f"[Epoch {epoch}] ➤ EMA smoothing activated.")
            
       
                        

        total_loss = 0.0
        total_kl = 0.0
        total_recon = 0.0
        total_mse = 0.0
        z_individual_list = []
        logvar_q_list = []
        mu_q_list = []

        trajectory_records = []

        # === Enable or disable noise learning ===
        for param in noise.parameters():
            param.requires_grad = epoch >= warmup_epochs_noise

        for id_list, t_padded, x_padded, mask, dose_tensor, dose_times_list in dataloader:
            t_padded, x_padded, mask = t_padded.to(device), x_padded.to(device), mask.to(device)
            dose_tensor = dose_tensor.to(device)
            dose_times_list = [dt.to(device) for dt in dose_times_list]
            batch_size = t_padded.size(0)

            dose_times_padded, dose_times_mask = pad_dose_times(dose_times_list)

            batch_size, max_len = dose_times_padded.shape
            dose_tensor.shape == [batch_size, 1]  # e.g., [20, 1]
            # dose_tensor: [10]
            dose_tensor_expanded = dose_tensor.unsqueeze(1).repeat(1, dose_times_padded.size(1))  # [10, 4]
            dose_tensor_expanded = dose_tensor_expanded.unsqueeze(-1)  # [10, 4, 1]
            
            # dose_times_padded: [10, 4]
            dose_times_expanded = dose_times_padded.unsqueeze(-1)  # [10, 4, 1]
            
            # Now concatenate on last dim:
            dose_features = torch.cat([dose_times_expanded, dose_tensor_expanded], dim=-1)  # [10, 4, 2]
           

            # Result: [batch_size, max_len, dose_dim]

         #  print(dose_tensor_expanded)

            # === Prepare data for each group ===
            t_low, x_low = t_padded, x_padded
      
            
            if max_points_visible > 0 :
                
                t_low_list, x_low_list = truncate_time_series(t_padded, x_padded, max_points_visible) 
                t_low = pad_sequence(t_low_list, batch_first=True).to(device)
                x_low = pad_sequence(x_low_list, batch_first=True).to(device)
  

            
          
            if nf:
                    z_refined, mu_q_low, logvar_q_low, log_det_low = encoder1(t_low, x_low)
            else:
                    mu_q_low, logvar_q_low = encoder1(t_low, x_low)
                    std_q = torch.exp(0.5 * logvar_q)
                    z_refined= mu_q_low + std_q[mask_low] * torch.randn_like(mu_q_low)
          

            if ae:
                z_refined=mu_q_low
         
            # === ODE Prediction ===
            x0_1 = initial_encoder(x_padded[:, 0].unsqueeze(1))
            mask_drop=None
            if onlymedian:
               z_refined= torch.zeros_like(z_refined)
               x0 = torch.cat([x0_1, z_refined], dim=1)
            else:
                if remove_encoder:
                    z_refined, mask_drop = dropout_rows(z_refined,p,device)
                    x0 = torch.cat([x0_1,z_refined +  torch.randn_like(z_refined) * 0.01], dim=1)

                else:
                    x0 = torch.cat([x0_1, z_refined+ torch.randn_like(z_refined) * 0.01], dim=1)

          

            #print(dose_tensor.unsqueeze(-1))
        
            dose_mask_expanded = dose_times_expanded.unsqueeze(-1)  # shape [batch, max_len, 1]

            ode_func = ODEWrapper(func, dose_times_expanded, dose_tensor_expanded, dose_times_mask)

            if epoch == 0 and first_batch:
                print(f" ODE Solving starting")
            pred = odeint(ode_func, x0, t_dense, method='dopri5')
            if epoch == 0 and first_batch:
                print(f" ODE Solving finished")
            pred_batch = pred.permute(1, 0, 2)  # [batch, time, features]

            # === Interpolate and Calculate Loss ===
            t_dense_exp = t_dense.unsqueeze(0).repeat(batch_size, 1)
            test = reducer(pred_batch[:, :, :latent_dim])
            pred_interp = batch_linear_interpolate_1d(test, t_dense_exp, t_padded)
            recon_loss_noise, mse = noise.nll(destandardize_concentration(x_padded,conc_mean,conc_std), pred_interp, mask)
         
           
      

            mu_std_low = torch.zeros_like(mu_q_low)
            logvar_std_low = torch.zeros_like(logvar_q_low)
            
 
        
            if ae:
                kl_weight = 0
                free_bits_on = 0  # optional: safe default if KL isn't used
               
            else:
                free_bits_on = 0 if epoch+1 >= warmup_epochs_iiv else free_bits* (1 - min(1.0, epoch / warmup_epochs_iiv))
                kl_weight = 1 if epoch+1 >= warmup_epochs_iiv else min(1.0,   epoch / warmup_epochs_iiv)

            
            if ae:
                loss = recon_loss_noise
                KL_loss=0
            else:
                if nf:
                
                    KL_loss_low = kl_divergence_NF(epoch,warmup_epochs_iiv,
                        mu_q_low, logvar_q_low,
                        mu_std_low, logvar_std_low,
                        log_det=log_det_low,
                        free_bits=free_bits_on, keep_mask=mask_drop)
                        
                      
                    KL_loss = KL_loss_low #+ KL_loss_high
                    
                else:
                    KL_loss = kl_divergence_gaussians(mu_q_low, logvar_q_low, mu_std_low, logvar_std_low, free_bits_on,mask)
                              
                loss = recon_loss_noise + kl_weight * KL_loss

            

            # === Backprop ===
            total_loss +=  loss
            total_mse += mse
            if not ae:
                total_kl += KL_loss

            
            
            
            total_recon +=  recon_loss_noise
            optimizer.zero_grad()
            if epoch == 0 and first_batch:
                print(f"Backprop")
            loss.backward()
            # === Gradient check ===
            if epoch == 0 and first_batch:
                for name, model in models.items():
                    for param_name, param in model.named_parameters():
                        if param.requires_grad:
                            if param.grad is None:
                                print(f"[WARNING] No gradient for {name}.{param_name}")
            
        

            if epoch == 0 and first_batch:
              print(f"Optimization")
            optimizer.step()
            first_batch = False


            residual = x_padded - pred_interp
    
            # === Store for analysis ===
            z_individual_list.append(z_refined.detach())
            mu_q_list.append(mu_q_low.detach())
            logvar_q_list.append(logvar_q_low.detach())

            for i in range(batch_size):
                trajectory_records.append((
                    id_list[i],
                    t_padded[i, mask[i]],
                    x_padded[i, mask[i]],
                    dose_tensor[i],
                    dose_times_list[i],
                    z_refined[i].detach()
                ))

        # === Global latent and EMA updates ===
        z_all = torch.cat(z_individual_list, dim=0)
        mu_all = torch.cat(mu_q_list, dim=0)
        logvar_all = torch.cat(logvar_q_list, dim=0)

        if epoch % 10 == 0:
            with torch.no_grad():
                mu_all = torch.cat(z_individual_list, dim=0)               # [N, D]
                logvar_all = torch.cat(logvar_q_list, dim=0)               # [N, D]
                std_all = torch.exp(0.5 * logvar_all)                      # [N, D]
            
                mu_mean = mu_all.mean(dim=0).mean().item()
                mu_std = mu_all.std(dim=0).mean().item()
                std_mean = std_all.mean(dim=0).mean().item()
                std_std = std_all.std(dim=0).mean().item()
            
                print(f"[Epoch {epoch}] μ mean: {mu_mean:.4f}, μ std: {mu_std:.4f}, "
                      f"σ mean: {std_mean:.4f}, σ std: {std_std:.4f}")


        with torch.no_grad():
            if epoch > smoothing_start_epoch:
                func.update_ema(alpha=0.1)
                reducer.update_ema(alpha=0.1)
                initial_encoder.update_ema(alpha=0.1)
                encoder1.update_ema(alpha=0.1)
                encoder1.update_ema(alpha=0.1)
             

        scheduler.step(total_loss)

        # === Logging ===
        if epoch % print_epoch == 0:
            with torch.no_grad():
                # Assuming main_params[0] corresponds to the main parameters with base_lr
                main_lr = optimizer.param_groups[0]['lr']
          
                if ae:
                    print(
                        f"Epoch {epoch}, "
                        f"MSE {total_mse.item():.4f} "
                        f"-LL: {total_recon.item():.4f}, "
                        f"Add. error: {(torch.exp(noise.log_sigma_add)).item():.4f}, "
                        f"Prop. error: {torch.exp(noise.log_sigma_prop).item():.4f}, "
                        f"lr: {main_lr:.6f}")
                    
                    
                else:    
                    print(
                        f"Epoch {epoch}, "
                        f"MSE {total_mse.item():.4f} "
                        f"loss: {total_loss.item():.4f}, "
                        f"-LL: {total_recon.item():.4f}, "
                        f"KL loss: {total_kl.item():.4f}, "
                        f"Add. error: {(torch.exp(noise.log_sigma_add)).item():.4f}, "
                        f"Prop. error: {torch.exp(noise.log_sigma_prop).item():.4f}, "
                        f"lr: {main_lr:.6f}")
        if plot_from_training_records_enable:
            if epoch % plot_epoch == 0:
                plot_from_training_records(batch_size,device, df=df, dataset=dataset, latent_dim=latent_dim,
                    records=trajectory_records,
                    func=func,
                    reducer=reducer,
                   initial_encoder=initial_encoder,
                    ODEWrapper=ODEWrapper,
                    t_dense=t_dense,
                    max_plots=max_plots,
                    nr_row=nr_row,
                    nr_col=nr_col,)




class TrajectoryDataset(Dataset):
    def __init__(self, path, compartment='C2'):
        self.df = pd.read_csv(path)
        self.compartment = compartment

        # Calculate max dose and time here
        self.max_dose = self.estimate_max_dose()
        self.max_time = self.estimate_max_time()

        # Normalization functions as instance methods
        def normalize_dose(dose): 
            return dose / self.max_dose
        def normalize_time(time): 
            return time / self.max_time

        # Normalize columns
        self.df['Dose_norm'] = normalize_dose(self.df['Dose'])
        self.df['Time_norm'] = normalize_time(self.df['Time'])
        self.conc_mean = self.df[self.compartment].mean()
        self.conc_std = self.df[self.compartment].std()
        self.df['C2_norm'] = (self.df[self.compartment] - self.conc_mean) / self.conc_std

        # Parse dose times column (string of list)
        self.df['DoseTimesParsed'] = self.df['Dose times'].apply(ast.literal_eval)
        self.df['DoseTimesNorm'] = self.df['DoseTimesParsed'].apply(lambda lst: [t / self.max_time for t in lst])

        # Add DoseLabel (0,1,2,...)
        unique_doses = sorted(self.df['Dose'].unique())
        dose_to_label = {dose: i for i, dose in enumerate(unique_doses)}
        self.df['DoseLabel'] = self.df['Dose'].map(dose_to_label)
        self.dose_to_label = dose_to_label

        # Extract trajectories
        self.trajectories = self._extract_trajectories()

    def estimate_max_dose(self):
        return self.df['Dose'].max()

    def estimate_max_time(self):
        return self.df['Time'].max()

    def _extract_trajectories(self):
        start_idxs = self.df[self.df['Time'] == 0].index.tolist() + [len(self.df)]
        trajectories = []
        for i in range(len(start_idxs) - 1):
            group = self.df.iloc[start_idxs[i]:start_idxs[i+1]]
            t = torch.tensor(group['Time_norm'].values, dtype=torch.float32)
            x = torch.tensor(group['C2_norm'].values, dtype=torch.float32)
            dose = torch.tensor(group['Dose_norm'].values[0], dtype=torch.float32)
            dose_times = torch.tensor(group['DoseTimesNorm'].values[0], dtype=torch.float32)
            subject_id = group['ID'].iloc[0]
            dose_label = torch.tensor(group['DoseLabel'].values[0], dtype=torch.long)
            trajectories.append((t, x, dose, dose_times, subject_id))
        return trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]



def collate_fn(batch):
    t_list, x_list, dose_list, dose_times_list, id_list = zip(*batch)

    t_padded = pad_sequence(t_list, batch_first=True)
    x_padded = pad_sequence(x_list, batch_first=True)

    max_len = t_padded.size(1)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, t in enumerate(t_list):
        mask[i, :len(t)] = 1

    dose_tensor = torch.stack(dose_list)
 

    return id_list, t_padded, x_padded, mask, dose_tensor, dose_times_list





def kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p, free_bits=0.0, mask=None):
    """
    KL[q||p] between two diagonal Gaussians with optional mask and free bits.
    
    Args:
        mu_q, logvar_q: [B, D] approximate posterior
        mu_p, logvar_p: [B, D] prior
        free_bits: float, minimum KL per dim to prevent collapse
        mask: optional tensor of shape [B], [B, 1], or [B, D] (1=keep, 0=drop)
    
    Returns:
        Scalar: mean KL divergence with masking and free bits
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    
    kl_per_dim = 0.5 * ((var_q + (mu_q - mu_p) ** 2) / var_p - 1 + logvar_p - logvar_q)  # [B, D]

    # Apply free bits per dim
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    # Apply mask (optional)
    if mask is not None:
        mask = mask.to(kl_per_dim.device)
        if mask.dim() == 1:
            mask = mask.unsqueeze(1)  # [B] → [B, 1]
        kl_per_dim = kl_per_dim * mask  # broadcasted masking
        denom = mask.sum().clamp(min=1.0)
    else:
        denom = torch.numel(kl_per_dim)

    return kl_per_dim.sum() / denom


def kl_divergence_NF(
    epoch,
    warmup_epochs_iiv,
    mu_q,
    logvar_q,
    mu_p,
    logvar_p,
    log_det=None,
    free_bits=0.0,
    keep_mask=None,  # <-- New
):
    """
    KL[q||p] between two diagonal Gaussians with optional flow correction,
    free bits, and per-sample dropout masking.

    Args:
        mu_q, logvar_q: [B, D] posterior (before flow)
        mu_p, logvar_p: [B, D] prior
        log_det: [B] log-det from normalizing flow (optional)
        free_bits: float, min KL per sample
        keep_mask: [B] or [B, 1], binary mask (1=keep, 0=drop); if None, all included

    Returns:
        scalar: mean KL with flow correction, masking, and free bits
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    # KL per dim: [B, D]
    kl_per_dim = 0.5 * ((var_q + (mu_q - mu_p) ** 2) / var_p - 1 + logvar_p - logvar_q)
    kl = kl_per_dim.sum(dim=1)  # [B]

    # Flow correction
    if log_det is not None:
        flow_weight = 1.0 if epoch + 1 >= warmup_epochs_iiv else min(1.0, epoch / warmup_epochs_iiv)
        kl = kl - flow_weight * log_det

    # Free bits (optional)
    if free_bits > 0.0:
        kl = torch.clamp(kl, min=free_bits)

    # Clamp at 0 to avoid negatives
    kl = torch.clamp(kl, min=0.0)

    # === Mask out dropped samples ===
    if keep_mask is not None:
        keep_mask = keep_mask.view(-1)  # ensure shape [B]
        kl = kl * keep_mask
        denom = keep_mask.sum().clamp(min=1.0)  # avoid div by 0
    else:
        denom = kl.size(0)

    return kl.sum() / denom

