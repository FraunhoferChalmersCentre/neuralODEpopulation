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
import gc
import os
import random
import time
import copy
from collections import defaultdict
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint as odeint
# from torchdiffeq import odeint_adjoint as odeint

#from lib.utils.model_validation import plot_individual_fits
   
 

import os


@torch.no_grad()
def plot_individual_fits_random(global_max_dose,
                         global_mean, 
                         global_std,
                         enable_ae_training,
                         enable_nf_training,
                         test_dataset,
                         latent_dim,
                         noise,
                         encoder,
                         func,
                         reducer,
                         initial_encoder,
                         ODEWrapper,
                         t_dense,
                         max_individuals=6,
                         n_samples=50,
                         device="cpu",
                         truncation=0.5,
                         add_noise=False):

    MAX_TIME = max(entry[0].max().item() for entry in test_dataset)
    conc_mean, conc_std = global_mean, global_std
    t_dense = t_dense.to(device)
    mse_tot = 0.0
    # Group dataset indices by dose
   # Group dataset indices by dose
 # Determine number of groups
    dose_to_indices = {}
    for idx, entry in enumerate(test_dataset):
        dose_val = entry[2].item()
        dose_to_indices.setdefault(dose_val, []).append(idx)
    
    num_groups = len(dose_to_indices)
    max_per_group = max_individuals // num_groups  # integer division
    
    selected_indices = []
    for dose_val, indices in dose_to_indices.items():
        n_sample = min(max_per_group, len(indices))
        sampled = np.random.choice(indices, n_sample, replace=False)
        selected_indices.extend(sampled)
    
    # If max_individuals is odd, add one more from the largest group
    while len(selected_indices) < max_individuals:
        largest_group = max(dose_to_indices.items(), key=lambda x: len(x[1]))[1]
        extra = np.random.choice(largest_group, 1, replace=False)
        selected_indices.extend(extra)

    
    # Now `selected_indices` has roughly half of each dose group


    ncols = 3
    nrows = math.ceil(len(selected_indices) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for i, idx in enumerate(selected_indices):
        entry = test_dataset[idx]
        t, x, dose_tensor, dose_times_tensor, subject_id, x_low_normalized = entry

        t, x = t.to(device), x.to(device)
        dose_tensor = dose_tensor.to(device).unsqueeze(0)
        dose_val = dose_tensor.item()
        dose_times_tensor = dose_times_tensor.to(device)
        x_low_normalized = x_low_normalized.to(device)
        t_trunc = t.unsqueeze(0)
        x_trunc = x.unsqueeze(0)
     #   x_dose_norm_list = x_dose_norm.unsqueeze(0)
        
        if truncation > 0 :
    
            t_low_list, x_low_list = truncate_time_series(t, x_low_normalized, truncation) 
            t_low_encoder = pad_sequence(t_low_list, batch_first=True).to(device)
            x_low_normalized = pad_sequence(x_low_list, batch_first=True).to(device)

      
        # Sample latent z
        with torch.no_grad():
            if enable_ae_training:
                z, mu, logvar, _ = encoder(t_low_encoder, x_low_normalized)
                z = mu.repeat(n_samples, 1)
            elif enable_nf_training:
                _, _, mu, logvar, _ = encoder(t_low_encoder, x_low_normalized)
                _, z, _ = encoder.sample(mu, logvar, num_samples=n_samples)
            else:
              
                _, _, mu, logvar, _ = encoder(t_low_encoder.T, x_low_normalized.T)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn(n_samples, *std.shape).to(device)
                z = mu + eps * std
                z = z.squeeze(1)

        x0_latent = initial_encoder(x_trunc[:, 0].unsqueeze(1)).repeat(n_samples, 1)
        x0 = torch.cat([x0_latent, z.squeeze(0)], dim=1)

        dose_times_padded = torch.nn.utils.rnn.pad_sequence([dose_times_tensor], batch_first=True).to(device)
        dose_mask = (dose_times_padded != 0).to(device)
        dose_times_padded = dose_times_padded.repeat(n_samples, 1)
        dose_mask = dose_mask.repeat(n_samples, 1)
        dose_rep = dose_tensor.repeat(n_samples, 1)

        ode_func = ODEWrapper(func, dose_times_padded.unsqueeze(-1), dose_rep.unsqueeze(-1), dose_mask)
        pred = odeint(ode_func, x0, t_dense, method='rk4')
        pred = reducer(pred[:, :, :latent_dim])
        pred = destandardize_concentration(pred, conc_mean, conc_std).squeeze(-1)
        mask = pred != 0
        
        if t_trunc.dim() == 1:
            t_trunc3 = t_trunc.unsqueeze(0).repeat(n_samples, 1)
        else:
            t_trunc3 = t_trunc.repeat(n_samples, 1)

       # t_trunc3 = t_trunc.unsqueeze(0).repeat(n_samples, 1)  # shape [n_samples, len(t_trunc)]
     
        t_dense3 = t_dense.unsqueeze(0).repeat(n_samples, 1)  # [n_samples, len(t_dense)]
       

        # Make t_trunc3 match n_samples
        t_trunc3 = t_trunc.repeat(n_samples, 1)  # [n_samples, 11]
       
        # Make t_dense3 match pred's second dimension (time)
        pred_interp = batch_linear_interpolate_1d(pred.T, t_dense3, t_trunc3).mean(dim=0)
   
        _, mse = noise.nll(destandardize_concentration(x_trunc.squeeze(0) , conc_mean, conc_std), pred_interp,)
       
        if add_noise:
            # Create a mask where pred != 0
            mask = pred != 0
            # Only add noise where pred is non-zero
            pred_noisy = pred.clone()
            pred_noisy[mask] = noise.sample(pred[mask], n_samples=1).squeeze(0)
            # Ensure no negative values
            pred = torch.clamp(pred_noisy, min=0)

                    
            
            
        perc10 = torch.quantile(pred, 0.025, dim=1).detach().cpu().numpy()
        median = torch.quantile(pred, 0.50, dim=1).detach().cpu().numpy()
        perc90 = torch.quantile(pred, 0.975, dim=1).detach().cpu().numpy()

        ax = axes[i // ncols][i % ncols]
        t_hours = t_dense.cpu().numpy() * MAX_TIME

        # Interpolate to common time grid
        common_time = np.linspace(0, t_hours.max(), 100)
        perc10_interp = np.interp(common_time, np.linspace(0, t_hours.max(), len(perc10)), perc10)
        perc90_interp = np.interp(common_time, np.linspace(0, t_hours.max(), len(perc90)), perc90)
        median_interp = np.interp(common_time, np.linspace(0, t_hours.max(), len(median)), median)

        ax.fill_between(common_time, perc10_interp, perc90_interp, alpha=0.3, color="blue", label="Predicted 10-90%")
        ax.plot(common_time, median_interp, color="blue", label="Predicted median")
        ax.scatter(t.cpu().numpy() * MAX_TIME, x.cpu().numpy() * conc_std + conc_mean,
                   color='red', label="Observed")

        ax.set_title(f"ID {subject_id} | Dose {global_max_dose * dose_val:.2f}")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Concentration")
        ax.grid(True)
        ax.legend()
        mse_tot+=mse
    # Hide unused subplots
    total_axes = nrows * ncols
    for j in range(len(selected_indices), total_axes):
        fig.delaxes(axes[j // ncols][j % ncols])

    plt.tight_layout()
    plt.show()
    
    return mse_tot

@torch.no_grad()
def plot_individual_fits(iteration_for,global_max_dose,global_max_time,
                         global_mean, 
                         global_std,
                         enable_ae_training,
                         enable_nf_training,
                         test_dataset,
                         latent_dim,
                         noise,
                         encoder,
                         func,
                         reducer,
                         initial_encoder,
                         ODEWrapper,
                         t_dense,
                         max_individuals=6,
                         n_samples=50,
                         device="cpu",
                         truncation=0.5,
                         add_noise=False):

    MAX_TIME = max(entry[0].max().item() for entry in test_dataset)
    conc_mean, conc_std = global_mean, global_std
    t_dense = t_dense.to(device)
    mse_tot = []
    r2_tot= []
    preds = []
    obs = []
    id_list = []
    encoder.eval()
    initial_encoder.eval()
    reducer.eval()
    func.eval()
    # Group dataset indices by dose
   # Group dataset indices by dose
 # Determine number of groups
    dose_to_indices = {}
    for idx, entry in enumerate(test_dataset):
        dose_val = entry[2].item()
        dose_to_indices.setdefault(dose_val, []).append(idx)
    
    # Select all individuals deterministically
    selected_indices = []
    for dose_val, indices in dose_to_indices.items():
        selected_indices.extend(sorted(indices))  # pick all individuals in sorted order

    
    # Now `selected_indices` has roughly half of each dose group


    ncols = 3
    nrows = math.ceil(len(selected_indices) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for i, idx in enumerate(selected_indices):
        entry = test_dataset[idx]
        t, x, dose_tensor, dose_times_tensor, subject_id, x_low_normalized = entry

        t, x = t.to(device), x.to(device)
        dose_tensor = dose_tensor.to(device).unsqueeze(0)
        dose_val = dose_tensor.item()
        dose_times_tensor = dose_times_tensor.to(device)
        x_low_normalized = x_low_normalized.to(device)
        t_trunc = t.unsqueeze(0)
        x_trunc = x.unsqueeze(0)
     #   x_dose_norm_list = x_dose_norm.unsqueeze(0)
        
        if truncation > 0 :
    
            t_low_list, x_low_list, truncation_mask = truncate_time_series(t, x_low_normalized, truncation) 
            t_low_encoder = pad_sequence(t_low_list, batch_first=True).to(device)
            x_low_normalized = pad_sequence(x_low_list, batch_first=True).to(device)

      
        # Sample latent z
        with torch.no_grad():
            if enable_ae_training:
                _,_, mu, _, _ = encoder(t_low_encoder.T, x_low_normalized.T)
                z = mu.repeat(n_samples, 1)
            elif enable_nf_training:
                _, _, mu, logvar, _ = encoder(t_low_encoder, x_low_normalized)
                _, z, _ = encoder.sample(mu, logvar, num_samples=n_samples)
            else :   
                _, _, mu, logvar, _ = encoder(t_low_encoder.T, x_low_normalized.T)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn(n_samples, *std.shape).to(device)
                z = mu + eps * std
                z = z.squeeze(1)

        x0_latent = initial_encoder(x_trunc[:, 0].unsqueeze(1)).repeat(n_samples, 1)
        x0 = torch.cat([x0_latent, z.squeeze(0)], dim=1)

        dose_times_padded = torch.nn.utils.rnn.pad_sequence([dose_times_tensor], batch_first=True).to(device)
        dose_mask = (dose_times_padded != 0).to(device)
        dose_times_padded = dose_times_padded.repeat(n_samples, 1)
        dose_mask = dose_mask.repeat(n_samples, 1)
        dose_rep = dose_tensor.repeat(n_samples, 1)
        
     
        
    #    print("dose_time_1x1:", dose_times_padded.unsqueeze(-1), flush=True)
     #  print("dose_rep_1x1:", dose_rep.unsqueeze(-1), flush=True)
     #   print("dose_mask_1x1:", dose_mask, flush=True)


        
        ode_func = ODEWrapper(func, dose_times_padded.unsqueeze(-1), dose_rep.unsqueeze(-1), dose_mask)

        pred = odeint(ode_func, x0, t_dense, method='rk4')
        pred = reducer(pred[:, :, :latent_dim])
        pred = destandardize_concentration(pred, conc_mean, conc_std).squeeze(-1)
        
     

       # x0_latent = initial_encoder(x_trunc[:, 0].unsqueeze(1)) #.repeat(n_samples, 1)
     
        #x0 = torch.cat([x0_latent, z], dim=1)
        

        
        
        mask = pred != 0
        
        if t_trunc.dim() == 1:
            t_trunc3 = t_trunc.unsqueeze(0).repeat(n_samples, 1)
        else:
            t_trunc3 = t_trunc.repeat(n_samples, 1)

        t_dense3 = t_dense.unsqueeze(0).repeat(n_samples, 1)  # [n_samples, len(t_dense)]
       

        t_trunc3 = t_trunc.repeat(n_samples, 1)  # [n_samples, 11]
        
        mse_mask = [~m for m in truncation_mask]
        masks_tensor = torch.stack(mse_mask)  # shape: [batch_size, seq_len]
        mask_flat = masks_tensor.view(-1)  # True for valid points
       
        # Make t_dense3 match pred's second dimension (time)
        pred_median, _ = pred.median(dim=1)  # take median across samples

        pred_interp = batch_linear_interpolate_1d(pred_median.T.unsqueeze(0), t_dense3[:1], t_trunc3[:1])
        pred_interp = pred_interp.squeeze(0)  # remove batch dim after interpolation
       
        y_true = destandardize_concentration(x_trunc.squeeze(0), conc_mean, conc_std)
       
        y_pred = pred_interp  # already destandardized
        
         # Flatten both to 1D tensors
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        

        y_true_masked = y_true_flat[mask_flat]
     
        y_pred_masked = y_pred_flat[mask_flat]

   # Compute sums only on masked points
       # ss_res_total += torch.sum((y_true_masked - y_pred_masked) ** 2).item()
        #ss_tot_total += torch.sum((y_true_masked - y_true_masked.mean()) ** 2).item()
       
        r2=r2_score(y_true_masked.cpu(), y_pred_masked.cpu())
        mse=mean_squared_error(y_true_masked.cpu(), y_pred_masked.cpu())
       
       #_, mse = noise.nll(y_true, pred_interp)
        if add_noise:
            # Create a mask where pred != 0
            mask = pred != 0
            # Only add noise where pred is non-zero
            pred_noisy = pred.clone()
            pred_noisy[mask] = noise.sample(pred[mask], n_samples=1).squeeze(0)
            # Ensure no negative values
            pred = torch.clamp(pred_noisy, min=0)

                    
            
            
        perc10 = torch.quantile(pred, 0.025, dim=1).detach().cpu().numpy()
        median = torch.quantile(pred, 0.50, dim=1).detach().cpu().numpy()
        perc90 = torch.quantile(pred, 0.975, dim=1).detach().cpu().numpy()

        ax = axes[i // ncols][i % ncols]
        t_hours = t_dense.cpu().numpy() * global_max_time

        # Interpolate to common time grid
        common_time = np.linspace(0, t_hours.max(), 100)
        perc10_interp = np.interp(common_time, np.linspace(0, t_hours.max(), len(perc10)), perc10)
        perc90_interp = np.interp(common_time, np.linspace(0, t_hours.max(), len(perc90)), perc90)
        median_interp = np.interp(common_time, np.linspace(0, t_hours.max(), len(median)), median)

        ax.fill_between(common_time, perc10_interp, perc90_interp, alpha=0.3, color="blue", label="Predicted 10-90%")
        ax.plot(common_time, median_interp, color="blue", label="Predicted median")
        ax.scatter(t.cpu().numpy() * global_max_time, x.cpu().numpy() * conc_std + conc_mean,
                   color='red', label="Observed")

        ax.set_title(f"ID {subject_id} | Dose {global_max_dose * dose_val:.2f}")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Concentration")
        ax.grid(True)
        ax.legend()
        mse_tot.append(mse)
        r2_tot.append(r2)
        preds.append(y_pred_masked.cpu())
        obs.append(y_true_masked.cpu())
        id_list.append(np.array([subject_id] * len(y_true_masked.cpu())))

    # Hide unused subplots
    total_axes = nrows * ncols
    for j in range(len(selected_indices), total_axes):
        fig.delaxes(axes[j // ncols][j % ncols])

    plt.tight_layout()
    plt.show()
 
    mean_mse=np.mean(mse_tot)
    mean_r2=np.mean(r2_tot)
    median_mse = np.median(mse_tot)
    median_r2 = np.median(r2_tot)
    
    preds_flat = np.concatenate([p.numpy() for p in preds])
    obs_flat = np.concatenate([o.numpy() for o in obs])
    ids_flat = np.concatenate(id_list)  # works with strings now

    
    residuals = preds_flat - obs_flat

    # Create DataFrame
    res = pd.DataFrame({
        "Prediction": preds_flat,
        "Observation": obs_flat,
        "Residual": residuals,
        "Iteration": iteration_for,  # Add the iteration index here
        "ID": ids_flat
    })

    
    
    
    return mean_mse, mean_r2, median_mse, median_r2, res

def prepare_datasets_and_loaders(data_path, all_ids, i, global_max_dose, global_max_time,
                                 global_mean, global_std, device, batch_fraction=0.05):

    """
    Splits the dataset into train/val/test, creates DataLoaders, and generates the combined time+dose tensor.

    Returns:
        train_dataset, val_dataset, test_dataset, train_loader, val_loader, combined
    """


    n_ids = len(all_ids)
    random.shuffle(all_ids)

    train_ids = all_ids[: int(0.7 * n_ids)]
    val_ids   = all_ids[int(0.7 * n_ids): int(0.8 * n_ids)]
    test_ids  = all_ids[int(0.8 * n_ids):]

    train_dataset = TrajectoryDataset_paracetamol(
    data_path,  # <--- pass path, not df
    augment_with_prefixes=True,
    augment_dose_times=False,
    max_dose=global_max_dose,
    max_time=global_max_time,
    conc_mean=global_mean,
    conc_std=global_std,
    subset_ids=train_ids
    )
    val_dataset = TrajectoryDataset_paracetamol(
        data_path,
        augment_with_prefixes=False,
        augment_dose_times=False,
        max_dose=global_max_dose,
        max_time=global_max_time,
        conc_mean=global_mean,
        conc_std=global_std,
        subset_ids=val_ids
    )
    test_dataset = TrajectoryDataset_paracetamol(
        data_path,
        augment_with_prefixes=False,
        augment_dose_times=False,
        max_dose=global_max_dose,
        max_time=global_max_time,
        conc_mean=global_mean,
        conc_std=global_std,
        subset_ids=test_ids
    )


    export_training_data(test_dataset, train_dataset, global_mean, global_std, global_max_time, i)

    # Create DataLoaders
    batch_size = max(1, int(len(train_dataset) * batch_fraction))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=2*len(val_ids), shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=2*len(test_ids), shuffle=False, collate_fn=collate_fn)

    # Combined dose + time tensor
    doses = torch.tensor([3], dtype=torch.float32) / 24
    time_points = torch.linspace(0, 1, steps=120)
    combined = torch.cat((time_points, doses)).to(device)
    combined, _ = torch.sort(combined)  # ensure ascending order

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, combined


def run_model_variant(variant_name, train_dataset, val_dataset, test_dataset,
                      models, main_params, optimizer, scheduler, combined,
                      global_max_dose, global_max_time, global_mean, global_std,
                      latent_dim, noise, encoder, func, reducer, initial_encoder,
                      ODEWrapper, device, metrics, residuals, iteration,
                      n_epochs, train_loader, val_loader, base_dir,
                      warmup_noise=0, warmup_iiv=0, enable_ae=False, enable_nf=False,
                      free_bits=0):
    """
    Train, evaluate, append metrics and residuals for a given model variant.

    Args:
        base_dir (str): directory to save metrics/residuals
        ...
        other args as before
    """
    # Run training loop
    _, mse_validation = train_loop_model_paracetamol(
        train_dataset, val_dataset, global_max_time, global_max_dose,
        global_mean, global_std, main_params, val_loader, train_loader,
        models, optimizer, scheduler, func, reducer, initial_encoder, encoder,
        noise, t_dense=combined, n_epochs=n_epochs,
        warmup_epochs_noise=warmup_noise, warmup_epochs_iiv=warmup_iiv,
        smoothing_start_epoch=1000, traing_against_validation=True,
        enable_ae_training=enable_ae, enable_nf_training=enable_nf,
        enable_onlymedian_training=False, plot_from_training_records_enable=False,
        free_bits=free_bits, max_points_visible=0.6, print_epoch=10,
        plot_epoch=10, max_plots=30, nr_col=10, nr_row=3
    )

    # Evaluate predictions
    mse_mean, r2_mean, mse_median, r2_median, res_df = plot_individual_fits(
        iteration_for=iteration, global_max_dose=global_max_dose,
        global_max_time=global_max_time, global_mean=global_mean,
        global_std=global_std, enable_ae_training=enable_ae,
        enable_nf_training=enable_nf, test_dataset=test_dataset,
        latent_dim=latent_dim, noise=noise, encoder=encoder, func=func,
        reducer=reducer, initial_encoder=initial_encoder, ODEWrapper=ODEWrapper,
        t_dense=combined, max_individuals=6, n_samples=1000, device=device,
        truncation=0.6, add_noise=True
    )

    # Append metrics and residuals
    append_metrics(metrics, residuals, variant_name, mse_mean, r2_mean,
                   mse_median, r2_median, mse_validation, res_df)

    # Save metrics/residuals
    export_all_metrics_and_residuals(metrics, residuals, base_dir)


    
    
    
def compute_global_stats(df):
    """
    Compute global statistics for the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame with columns 'Dose', 'Time', 'C2'
    
    Returns:
        global_max_dose, global_max_time, global_mean, global_std, global_max_value
    """
    global_max_dose = df['Dose'].max()
    global_max_time = df['Time'].max()
    global_mean = df['C2'].mean()
    global_std = df['C2'].std()
    global_max_value = df['C2'].max()
    
    return global_max_dose, global_max_time, global_mean, global_std, global_max_value



def export_all_metrics_and_residuals(metrics, residuals, base_dir, variants=["", "_ae", "_ae_noise"]):
    os.makedirs(base_dir, exist_ok=True)
    
    for var in variants:
        # Export metrics
        metrics_df = pd.DataFrame(metrics[var])
        metrics_file = os.path.join(base_dir, f"metrics_raw{var}.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Saved metrics: {metrics_file}")

        # Export residuals
        residuals_file = os.path.join(base_dir, f"residuals_NODE{var}.csv")
        if residuals[var]:  # only concat if list is non-empty
            res_all = pd.concat(residuals[var], ignore_index=True)
            res_all.to_csv(residuals_file, index=False)
        else:
            # create empty CSV with standard columns if no residuals yet
            res_all = pd.DataFrame(columns=["Prediction", "Observation", "Residual", "Iteration", "ID"])
            res_all.to_csv(residuals_file, index=False)
        print(f"Saved residuals: {residuals_file}")
        
        
        
def append_metrics(metrics,residuals, variant, mse_mean, r2_mean, mse_median, r2_median, mse_validation, res_df):
    """
    Append metrics and residuals safely for a given variant.
    """
    metrics[variant]["mse_mean"].append(mse_mean)
    metrics[variant]["r2_mean"].append(r2_mean)
    metrics[variant]["mse_median"].append(mse_median)
    metrics[variant]["r2_median"].append(r2_median)
    metrics[variant]["mse_validation"].append(mse_validation)
    residuals[variant].append(res_df)


def load_all_metrics_and_residuals_as_lists(base_dir, variants=["", "_ae", "_ae_noise"]):
    """
    Loads metrics and residuals CSVs for all specified variants,
    and converts all metrics to Python lists so they can be appended.

    Returns:
        metrics: dict of dicts (all lists)
        residuals: dict of lists
        already_done: int, max length of mse_mean lists across variants
    """
    from lib.utils.my_utils import load_existing_metrics, load_existing_residuals

    metrics = {}
    residuals = {}

    ensure_result_files(base_dir)  # make sure all files exist

    for var in variants:
        metrics_file = os.path.join(base_dir, f"metrics_raw{var}.csv")
        loaded = load_existing_metrics(metrics_file, 
                                       ["mse_mean","r2_mean","mse_median","r2_median","mse_validation"])
        # Convert each series/list to pure Python list
        metrics[var] = {k: list(v) for k, v in loaded.items()}

        residuals_file = os.path.join(base_dir, f"residuals_NODE{var}.csv")
        res = load_existing_residuals(residuals_file)
        residuals[var] = list(res)  # ensure list

    # Compute already_done
    already_done = max(len(metrics[var]["mse_mean"]) for var in variants)

    return metrics, residuals, already_done







def ensure_result_files(base_dir: str):
    os.makedirs(base_dir, exist_ok=True)
    print(f"[INFO] Base dir: {os.path.abspath(base_dir)}")

    files_and_headers = {
        "metrics_raw.csv": ["mse_mean","r2_mean","mse_median","r2_median","mse_validation"],
        "residuals_NODE.csv": ["id","time","residual"],
        "metrics_raw_ae.csv": ["mse_mean","r2_mean","mse_median","r2_median","mse_validation"],
        "residuals_NODE_ae.csv": ["id","time","residual"],
        "metrics_raw_ae_noise.csv": ["mse_mean","r2_mean","mse_median","r2_median","mse_validation"],
        "residuals_NODE_ae_noise.csv": ["id","time","residual"],
    }

    for fname, headers in files_and_headers.items():
        path = os.path.join(base_dir, fname)
        if not os.path.exists(path):
            print(f"[INFO] Creating {path}")
            pd.DataFrame(columns=headers).to_csv(path, index=False)
        else:
            print(f"[INFO] Already exists: {path}")

def load_existing_residuals(filepath):
    """Load residuals from a CSV if it exists, else return an empty list."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return [df]   # we keep as list to match append/concat later
    else:
        return []
    
def load_existing_metrics(filepath, expected_columns):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Ensure expected columns exist
        for col in expected_columns:
            if col not in df.columns:
                raise ValueError(f"Missing column {col} in {filepath}")
        return {col: df[col].tolist() for col in expected_columns}
    else:
        return {col: [] for col in expected_columns}

class ODEWrapper(nn.Module):
    def __init__(self, func, dose_times, dose_amounts, dose_mask, keep_mask=None):
        super().__init__()
        self.func = func
        self.dose_times = dose_times          # [batch, max_len]
        self.dose_amounts = dose_amounts      # [batch, max_len]
        self.dose_mask = dose_mask            # [batch, max_len]
        self.keep_mask = keep_mask            # [batch, 1] — binary mask (optional)

    def forward(self, t, x):
        # Pass keep_mask if provided
        if self.keep_mask is not None:
            return self.func(t, x, self.dose_times, self.dose_amounts, self.dose_mask, self.keep_mask)
        else:
            return self.func(t, x, self.dose_times, self.dose_amounts, self.dose_mask)

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




def truncate_time_series(t_batch, x_batch, truncation_time):
    """
    Truncate a batch of time series to a specified cutoff time.

    This function iterates over a batch of time–value pairs and keeps only the
    entries where time ≤ `truncation_time`. It is useful when limiting observed
    data fed to an encoder (e.g., for semi-supervised training).

    Args:
        t_batch (list[torch.Tensor]): List of time tensors, each shaped [T_i].
        x_batch (list[torch.Tensor]): List of value tensors corresponding to t_batch, each shaped [T_i, ...].
        truncation_time (float): Time cutoff. All entries with time > cutoff are removed.

    Returns:
        tuple:
            truncated_t (list[torch.Tensor]): Truncated time tensors.
            truncated_x (list[torch.Tensor]): Truncated value tensors.
            masks (list[torch.BoolTensor]): Boolean masks indicating kept indices for each sequence.
    """
    truncated_t, truncated_x, masks = [], [], []

    for t_i, x_i in zip(t_batch, x_batch):
        # Boolean mask: True where time ≤ cutoff
        mask = t_i <= truncation_time

        # Apply mask to time and values
        truncated_t.append(t_i[mask])
        truncated_x.append(x_i[mask])
        masks.append(mask)

    return truncated_t, truncated_x, masks



def standardize_concentration(conc, mean, std): return (conc - mean) / std
def destandardize_concentration(norm_conc, mean, std): return norm_conc * std + mean



    
def plot_from_training_records(
    batch_size,
    device,
    global_max_time,
    global_max_dose,
    global_mean,
    global_std,
    latent_dim,
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
    """
    Plot model predictions against actual observed concentration data from training records.

    This function:
      1. Prepares batch data from a set of training records (subjects, doses, latent states, etc.).
      2. Pads dose times and amounts for batching.
      3. Encodes initial conditions with `initial_encoder` and concatenates latent states.
      4. Uses an ODE solver (`torchdiffeq.odeint`) with a wrapped function (`ODEWrapper`) to
         generate predictions across dense time points.
      5. Applies a reducer on latent dimensions and plots predicted vs. actual concentration curves.

    Args:
        batch_size (int): Number of samples per batch.
        device (torch.device): Device for computation (CPU/GPU).
        global_max_time (float): Scaling factor for time normalization → hours.
        global_max_dose (float): Scaling factor for dose normalization → mg.
        global_mean (float): Global mean used for denormalizing outputs.
        global_std (float): Global std used for denormalizing outputs.
        latent_dim (int): Dimensionality of latent representations.
        records (list): Training records, each containing
                        (subject_id, times, concentrations, doses, dose_times, z_refined).
        func (nn.Module): Neural ODE dynamics function.
        reducer (nn.Module): Reduces latent ODE outputs to prediction space.
        initial_encoder (nn.Module): Encodes initial concentration values.
        ODEWrapper (callable): Wraps ODE function with dose information.
        t_dense (torch.Tensor): Dense time grid for simulation.
        max_plots (int): Maximum number of subjects to plot.
        nr_row (int): Number of subplot rows.
        nr_col (int): Number of subplot columns.
        compartment (str): Compartment label for y-axis (default "central").

    Returns:
        None. Displays a matplotlib figure with predicted vs. actual concentration curves.
    """

    # Make sure we're using the same device as the model
    device = next(func.parameters()).device
    n = min(len(records), max_plots)  # number of subjects to plot

    # Containers for batch preparation
    subject_ids, ts, xs = [], [], []
    doses, dose_times_list, z_refined_list = [], [], []

    # --- Extract and preprocess each subject's data ---
    for i in range(n):
        subject_id, t_real, x_real, dose, dose_times, z_refined = records[i]
        subject_ids.append(subject_id)
        ts.append(t_real)
        xs.append(x_real)

        # Ensure all dose-related data are tensors
        dose_tensor = torch.as_tensor(dose).unsqueeze(-1) if not isinstance(dose, torch.Tensor) else dose
        doses.append(dose_tensor if dose_tensor.dim() > 0 else dose_tensor.unsqueeze(0))

        dose_times_tensor = torch.as_tensor(dose_times) if not isinstance(dose_times, torch.Tensor) else dose_times
        dose_times_list.append(dose_times_tensor)

        z_refined_tensor = torch.as_tensor(z_refined) if not isinstance(z_refined, torch.Tensor) else z_refined
        z_refined_list.append(z_refined_tensor)

    # --- Pad sequences for batching ---
    dose_times_padded = torch.nn.utils.rnn.pad_sequence(dose_times_list, batch_first=True, padding_value=0)
    doses_padded = torch.nn.utils.rnn.pad_sequence(doses, batch_first=True, padding_value=0)

    # Ensure doses are 2D [batch_size, max_len]
    if doses_padded.dim() == 3:
        doses_padded = doses_padded.squeeze(-1)

    # Mask: valid dose entries
    dose_mask = (dose_times_padded != 0)
    z_refined = torch.stack(z_refined_list)  # [batch_size, latent_dim]

    # --- Encode initial states ---
    x0_list = []
    for x in xs:
        out = initial_encoder(x[0].unsqueeze(0))  # encode first observation
        if out.dim() == 1:
            out = out.unsqueeze(0)
        x0_list.append(out)
    x0_tensor = torch.cat(x0_list, dim=0)

    # Concatenate encoded states with refined latent z
    x0 = torch.cat([x0_tensor, z_refined], dim=1)

    # --- Move everything to device ---
    doses_padded = doses_padded.to(device)
    dose_times_padded = dose_times_padded.to(device)
    dose_mask = dose_mask.to(device)
    z_refined = z_refined.to(device)
    x0 = x0.to(device)
    t_dense = t_dense.to(device)

    # Expand dose information for ODE solver
    dose_amounts_expanded = doses_padded.repeat(1, dose_times_padded.size(1)).unsqueeze(-1)
    dose_times_expanded = dose_times_padded.unsqueeze(-1)

    # --- Solve ODE ---
    ode_func = ODEWrapper(func, dose_times_expanded, dose_amounts_expanded, dose_mask)
    pred = odeint(ode_func, x0, t_dense, method='rk4')

    # Reduce latent states to prediction space
    x_pred = reducer(pred[:, :, :latent_dim])

    # --- Plot results ---
    fig, axs = plt.subplots(nr_row, nr_col, figsize=(nr_col * 6, nr_row * 5), sharex=True)
    axs = axs.flatten()

    for i in range(n):
        ax = axs[i]

        # Interpolated ground truth for alignment
        x_interp = torch.tensor(
            np.interp(t_dense.cpu().numpy(), ts[i].cpu().numpy(), xs[i].cpu().numpy()),
            dtype=torch.float32,
            device=ts[i].device,
        )

        # Actual observations
        ax.plot(
            ts[i].cpu().numpy() * global_max_time,
            (xs[i].cpu().numpy() * global_std) + global_mean,
            'o-', label='Actual'
        )

        # Model predictions
        ax.plot(
            t_dense.cpu().numpy() * global_max_time,
            (x_pred[:, i].detach().cpu().numpy() * global_std) + global_mean,
            '-', label='Predicted'
        )

        ax.set_title(f"Individual {subject_ids[i]} - Dose: {doses_padded[i].sum().item() * global_max_dose:.0f} mg")
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







@torch.no_grad()

def evaluate_on_val(
    global_mean,
    global_std,
    global_max_time,
    global_max_dose,
    enable_nf_training,
    enable_ae_training,
    max_points_visible,
    func,
    noise,
    reducer,
    dataset_val,
    dataloader_val,
    batch_size,
    device,
    encoder,
    initial_encoder,
    t_dense,
):
    """
    Evaluate model on a validation dataset by computing total MSE and NLL.
    Reuses preprocess, encode_latent, and prepare_ode_input functions.
    """

    device = next(func.parameters()).device
    latent_dim = func.dim_latent

    total_mse = 0.0
    total_LL = 0.0

    for batch in dataloader_val:
        id_list, t_padded, x_padded, mask, dose_tensor, dose_times_list, x_normalized = batch

        # --- Preprocess batch ---
        t_padded, x_padded, mask, t_encoder, x_encoder, dose_tensor_expanded, dose_times_expanded, dose_times_mask = preprocess_batch(
            (id_list, t_padded, x_padded, mask, dose_tensor, dose_times_list, x_normalized),
            device,
            max_points_visible=max_points_visible
        )

        # --- Encode latent ---
        z_refined, mu_q, logvar_q, log_det = encode_latent(
            encoder,
            t_encoder,
            x_encoder,
            enable_nf=enable_nf_training,
            enable_ae=enable_ae_training
        )

        # --- Prepare initial state and ODE ---
        x0, ode_func = prepare_ode_input(
            initial_encoder,
            x_padded,
            z_refined,
            func,
            dose_times_expanded,
            dose_tensor_expanded,
            dose_times_mask
        )

        # --- Compute predictions ---
        pred = odeint(ode_func, x0, t_dense, method="rk4")
        pred_batch = pred.permute(1, 0, 2)

        # --- Interpolate predictions to observed times ---
        t_dense_exp = t_dense.unsqueeze(0).repeat(x_padded.size(0), 1)
        pred_interp = batch_linear_interpolate_1d(reducer(pred_batch[:, :, :latent_dim]), t_dense_exp, t_padded)

        # --- Compute reconstruction loss ---
        recon_loss_noise, mse = noise.nll(
            destandardize_concentration(x_padded, global_mean, global_std),
            destandardize_concentration(pred_interp, global_mean, global_std),
            mask
        )

        # --- Accumulate metrics ---
        total_mse += mse.item()
        total_LL += recon_loss_noise.item()

    return total_mse, total_LL









        

    
def initialize_training(models, func, dataloader, t_dense, global_max_time):
    """
    Initialize variables and settings before starting training loop.

    Args:
        func: The ODE function model
        dataloader: training dataloader
        t_dense: dense time tensor
        global_max_time: maximum time for scaling

    Returns:
        device: torch device
        latent_dim: latent dimension of ODE
        dim_parameter_encoder: parameter encoder dimension
        best_val_mse: initial best validation MSE
        best_val_LL: initial best validation log-likelihood
        epochs_no_improve: counter for early stopping
        patience: patience for early stopping
        batch_size: batch size from dataloader
        t_dense: dense time tensor on correct device
        mse_history: list to store MSE per epoch
    """
    device = next(func.parameters()).device
    dim_parameter_encoder = func.dim_parameter_encoder
    latent_dim = func.dim_latent
    best_val_mse = float("inf")
    best_val_LL = float("inf")
    best_model_state = {k: v.state_dict() for k, v in models.items()}  # initial model state

    epochs_no_improve = 0
    patience = 30  # stop if no improvement for 30 epochs

    batch_size = dataloader.batch_size
    t_dense = t_dense.to(device)

    mse_history = []

    return (device, latent_dim, dim_parameter_encoder, best_val_mse, best_val_LL, 
            epochs_no_improve, patience, batch_size, t_dense, mse_history)

import time

def initialize_epoch_metrics():
    """
    Initialize variables at the start of each epoch for tracking losses, MSE, KL, etc.

    Returns:
        first_batch: boolean flag for first batch
        start_time: epoch start timestamp
        total_transform: accumulated log-det sum
        total_loss: accumulated total loss
        total_kl: accumulated KL divergence
        total_recon: accumulated reconstruction loss
        total_mse: accumulated MSE
        z_individual_list: list to store latent vectors per batch
        logvar_q_list: list to store log variance per batch
        mu_q_list: list to store mean vectors per batch
        trajectory_records: list to store trajectories for plotting
    """
    first_batch = True
   

    total_transform = 0.0
    total_loss = 0.0
    total_kl = 0.0
    total_recon = 0.0
    total_mse = 0.0
    z_individual_list = []
    logvar_q_list = []
    mu_q_list = []
    trajectory_records = []

    return (first_batch, total_transform, total_loss, total_kl, 
            total_recon, total_mse, z_individual_list, logvar_q_list, mu_q_list, trajectory_records)
    

    

def preprocess_batch(batch, device, max_points_visible=0):
    id_list, t_padded, x_padded, mask, dose_tensor, dose_times_list, x_normalized = batch
    
    t_padded, x_padded, mask, x_normalized = (
        t_padded.to(device), x_padded.to(device), mask.to(device), x_normalized.to(device)
    )
    dose_tensor = dose_tensor.to(device)
    dose_times_list = [dt.to(device) for dt in dose_times_list]
    
    dose_times_padded, dose_times_mask = pad_dose_times(dose_times_list)
    dose_tensor_expanded = dose_tensor.unsqueeze(1).repeat(1, dose_times_padded.size(1)).unsqueeze(-1)
    dose_times_expanded = dose_times_padded.unsqueeze(-1)
    
    if max_points_visible > 0:
        t_list, x_list, _ = truncate_time_series(t_padded, x_normalized, max_points_visible)
        t_encoder = pad_sequence(t_list, batch_first=True).to(device)
        x_encoder = pad_sequence(x_list, batch_first=True).to(device)
    else:
        t_encoder, x_encoder = t_padded, x_normalized
    
    return t_padded, x_padded, mask, t_encoder, x_encoder, dose_tensor_expanded, dose_times_expanded, dose_times_mask




def log_training_epoch(end_time, epoch, total_mse, total_loss, total_recon, total_kl, total_transform,
                       optimizer, noise, dataset, batch_size, val_mse=None, val_LL=None,
                       enable_ae_training=False, enable_nf_training=False, print_epoch=1):
    """
    Log training metrics for the current epoch.

    Args:
        epoch: current epoch index
        total_mse: accumulated MSE for epoch
        total_loss: accumulated total loss
        total_recon: accumulated reconstruction loss
        total_kl: accumulated KL divergence
        total_transform: accumulated log-det sum
        optimizer: optimizer instance
        noise: noise model containing sigma_add
        dataset: training dataset (for scaling)
        batch_size: batch size
        val_mse: validation MSE (optional)
        val_LL: validation log-likelihood (optional)
        enable_ae_training: bool flag
        enable_nf_training: bool flag
        print_epoch: frequency of printing
    """
    if epoch % print_epoch != 0:
        return

    with torch.no_grad():
        main_lr = optimizer.param_groups[0]['lr']
       # end_time = time.time()
        add_error = float(noise.sigma_add)

        if enable_ae_training:
            print(f"Epoch {epoch}, "
                  f"MSE {total_mse/20:.1f} "
                  f"-LL: {total_recon/20:.1f}, "
                  f"Add. error: {add_error:.2f}, "
                  f"val_mse: {val_mse:.3f}, "
                  f"val_LL: {val_LL:.3f}, "
                  f"lr: {main_lr:.3f}, "
                  f"{end_time:.2f} seconds")
        elif enable_nf_training:
            print(f"Epoch {epoch}, "
                  f"MSE {total_mse/20:.1f} "
                  f"loss: {total_loss/20:.1f}, "
                  f"-LL: {batch_size/len(dataset)*total_recon:.1f}, "
                  f"KL: {batch_size/len(dataset)*total_kl:.8f}, "
                  f"log_det: {batch_size/len(dataset)*total_transform:.8f}, "
                  f"Add. error: {add_error:.2f}, "
                  f"lr: {main_lr:.3f}, "
                  f"{end_time:.1f} seconds")
        else:
            print(f"Epoch {epoch}, "
                  f"MSE {total_mse/20:.1f} "
                  f"loss: {total_loss/20:.1f}, "
                  f"-LL: {total_recon/20:.2f}, "
                  f"KL: {total_kl/20:.4f}, "
                  f"Add. error: {add_error:.2f}, "
                  f"1000 lr: {1000*main_lr:.1f}, "
                  f"{end_time:.1f} seconds")




def encode_latent(encoder, t_encoder, x_normalized, 
                  enable_nf=False, enable_ae=False, enable_onlymedian=False):
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
    if enable_nf:
        _, z_refined, mu_q, logvar_q, log_det = encoder(t_encoder, x_normalized)
    elif enable_ae:
        _, _, mu_q, logvar_q, _ = encoder(t_encoder, x_normalized)
        z_refined = mu_q
        log_det = 0
    elif enable_onlymedian:
        _, _, mu_q, logvar_q, _ = encoder(t_encoder, x_normalized)
        z_refined = torch.zeros_like(mu_q)
        log_det = 0
    else:
        _, _, mu_q, logvar_q, _ = encoder(t_encoder, x_normalized)
        std_q = torch.exp(0.5 * logvar_q)
        z_refined = mu_q + std_q * torch.randn_like(mu_q)
        log_det = 0

    return z_refined, mu_q, logvar_q, log_det


def prepare_ode_input(initial_encoder, x_padded, z_refined, func, dose_times_expanded, dose_tensor_expanded, dose_times_mask):
    x0_1 = initial_encoder(x_padded[:, 0].unsqueeze(1))
    x0 = torch.cat([
        x0_1 + torch.randn_like(x0_1) * 0.01,
        z_refined + torch.randn_like(z_refined) * 0.01
    ], dim=1)
    ode_func = ODEWrapper(func, dose_times_expanded, dose_tensor_expanded, dose_times_mask)
    return x0, ode_func



def compute_loss_and_metrics(t_padded,latent_dim,
    t_dense, x_padded, mask, x0, ode_func, reducer, noise, mu_q, logvar_q, log_det,
    epoch, warmup_epochs_iiv, free_bits, enable_ae_training, enable_nf_training,
    global_mean, global_std, max_points_visible
):
    """
    Run ODE forward, interpolate, compute reconstruction and KL/NF losses.
    Returns:
        loss, recon_loss, mse, kl_loss, log_det_sum, log_det_penalty, pred_interp
    """
    batch_size = x_padded.size(0)
   # latent_dim = mu_q.size(1)

    t_dense_exp = t_dense.unsqueeze(0).repeat(batch_size, 1)
    pred = odeint(ode_func, x0, t_dense, method='rk4')
    pred_batch = pred.permute(1, 0, 2)
    
    test = reducer(pred_batch[:, :, :latent_dim])
    pred_interp = batch_linear_interpolate_1d(test, t_dense_exp, t_padded)
    
    recon_loss_noise, mse = noise.nll(
        destandardize_concentration(x_padded, global_mean, global_std),
        destandardize_concentration(pred_interp, global_mean, global_std),
        mask
    )

    mu_std = torch.zeros_like(mu_q)
    logvar_std = torch.zeros_like(logvar_q)

    if enable_ae_training:
        kl_weight = 0
        KL_loss = 0
        log_det_penalty = torch.tensor(0)
        log_det_sum = torch.tensor(0)
        kl_gauss= 0 
    else:
        free_bits_on = 0 if epoch+1 >= warmup_epochs_iiv else free_bits * (1 - min(1.0, epoch / warmup_epochs_iiv))
        kl_weight = 1 if epoch+1 >= warmup_epochs_iiv else min(1.0, epoch / warmup_epochs_iiv)

        if enable_nf_training:
            KL_loss, log_det_sum, kl_gauss, log_det_penalty = kl_divergence_NF(
                epoch, warmup_epochs_iiv, mu_q, logvar_q, mu_std, logvar_std, log_det,
                free_bits=free_bits_on, log_det_penalty_lambda=1
            )
        else:
            KL_loss = kl_divergence_gaussians(mu_q, logvar_q, mu_std, logvar_std, free_bits_on)
            kl_gauss = KL_loss
            log_det_sum = torch.tensor(0)
            log_det_penalty = 0

    loss = recon_loss_noise + kl_weight * KL_loss + log_det_penalty

    return loss, recon_loss_noise, mse, KL_loss, log_det_sum, log_det_penalty, pred_interp, kl_gauss

def accumulate_epoch_metrics(
    total_transform, total_loss, total_mse, total_kl, total_recon,
    log_det_sum, loss, mse, kl_gauss, recon_loss_noise, enable_ae_training
):
    total_transform += log_det_sum.item()
    total_loss += loss.item()
    total_mse += mse.item()
    total_kl += kl_gauss.item() if not enable_ae_training else 0.0
    total_recon += recon_loss_noise.item()
    return total_transform, total_loss, total_mse, total_kl, total_recon


def run_backprop_step(loss, main_params, optimizer):
    optimizer.zero_grad()
    loss.backward()
    for param_group in main_params:
        torch.nn.utils.clip_grad_norm_(param_group["params"], max_norm=1)
    optimizer.step()
def validate_and_update_early_stop(
    epoch, warmup_epochs_noise, traing_against_validation,
    val_mse, val_LL, best_val_mse, best_val_LL, epochs_no_improve,
    patience, models
):
    """
    Handles validation metrics, early stopping, and best model state update.

    Returns:
        best_val_mse, best_val_LL, epochs_no_improve, stop_training (bool), best_model_state
    """
    stop_training = False
    best_model_state = {k: v.state_dict() for k, v in models.items()}  # <<< initialize here

    if epoch < warmup_epochs_noise:
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            epochs_no_improve = 0
            best_model_state = {k: v.state_dict() for k, v in models.items()}
        else:
            epochs_no_improve += 1
            print(f"[Early Stop Monitor] No improvement for {epochs_no_improve} epochs, current val MSE {val_mse:.1f}, best val MSE {best_val_mse:.1f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs due to MSE not increasing")
            for k, v in models.items():
                v.load_state_dict(best_model_state[k])
            print(f"Best validation MSE achieved: {best_val_mse:.4f}") 
            stop_training = True
    else:
        if val_LL < best_val_LL:
            best_val_LL = val_LL
            best_val_mse = val_mse
            epochs_no_improve = 0
            best_model_state = {k: copy.deepcopy(v.state_dict()) for k, v in models.items()}
        else:
            epochs_no_improve += 1
            print(f"[Early Stop Monitor] No improvement for {epochs_no_improve} epochs, current val LL {val_LL:.1f}, best val LL {best_val_LL:.1f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs due to LL not improving")
            for k, v in models.items():
                v.load_state_dict(best_model_state[k])
            print(f"Best validation MSE and LL achieved: {best_val_mse:.4f}, {best_val_LL:.4f}") 
            stop_training = True

    return best_val_mse, best_val_LL, epochs_no_improve, stop_training, best_model_state


def train_loop_model_paracetamol(
    dataset, dataset_val, global_max_time, global_max_dose,
    global_mean, global_std, main_params, dataloader_val, dataloader, models,
    optimizer, scheduler, func, reducer, initial_encoder, encoder, noise, t_dense,
    n_epochs, warmup_epochs_noise, warmup_epochs_iiv, smoothing_start_epoch,
    traing_against_validation, enable_ae_training, enable_nf_training,
    enable_onlymedian_training, plot_from_training_records_enable, free_bits,
    max_points_visible, print_epoch=1, plot_epoch=1, max_plots=4,
    nr_col=1, nr_row=5
):
    # === Initialize training ===
    device, latent_dim, dim_parameter_encoder, best_val_mse, best_val_LL, \
    epochs_no_improve, patience, batch_size, t_dense, mse_history = initialize_training(
        models, func, dataloader, t_dense, global_max_time
    )

    for epoch in range(n_epochs):
        # --- Initialize epoch metrics ---
        first_batch, total_transform, total_loss, total_kl, \
        total_recon, total_mse, z_individual_list, logvar_q_list, mu_q_list, \
        trajectory_records = initialize_epoch_metrics()
        start_time = time.time()

        # --- Enable/disable noise learning ---
        for param in noise.parameters():
            param.requires_grad = epoch >= warmup_epochs_noise

        for batch in dataloader:
            # Unpack batch
            id_list, t_padded, x_padded, mask, dose_tensor, dose_times_list, x_normalized = batch

            # --- Preprocess batch ---
            t_padded, x_padded, mask, t_encoder, x_encoder, \
            dose_tensor_expanded, dose_times_expanded, dose_times_mask = preprocess_batch(
                (id_list, t_padded, x_padded, mask, dose_tensor, dose_times_list, x_normalized),
                device, max_points_visible=max_points_visible
            )

            # --- Encode latent ---
            z_refined, mu_q, logvar_q, log_det = encode_latent(
                encoder, t_encoder, x_encoder,
                enable_nf=enable_nf_training,
                enable_ae=enable_ae_training,
                enable_onlymedian=enable_onlymedian_training
            )

            # --- Prepare ODE input ---
            x0, ode_func = prepare_ode_input(
                initial_encoder, x_padded, z_refined, func,
                dose_times_expanded, dose_tensor_expanded, dose_times_mask
            )

            # --- Compute loss and metrics ---
            loss, recon_loss_noise, mse, KL_loss, log_det_sum, log_det_penalty, \
            pred_interp, kl_gauss = compute_loss_and_metrics(
                t_padded=t_padded, latent_dim=latent_dim,
                t_dense=t_dense, x_padded=x_padded, mask=mask,
                x0=x0, ode_func=ode_func, reducer=reducer, noise=noise,
                mu_q=mu_q, logvar_q=logvar_q, log_det=log_det,
                epoch=epoch, warmup_epochs_iiv=warmup_epochs_iiv,
                free_bits=free_bits, enable_ae_training=enable_ae_training,
                enable_nf_training=enable_nf_training, global_mean=global_mean,
                global_std=global_std, max_points_visible=max_points_visible
            )

            # --- Accumulate metrics ---
            total_transform, total_loss, total_mse, total_kl, total_recon = accumulate_epoch_metrics(
                total_transform, total_loss, total_mse, total_kl, total_recon,
                log_det_sum, loss, mse, kl_gauss, recon_loss_noise, enable_ae_training
            )

            # --- Backpropagation ---
            run_backprop_step(loss, main_params, optimizer)
            first_batch = False

        # --- Post-epoch operations ---
        mse_history.append(total_mse)
        scheduler.step(total_mse)
        end_time = time.time()
        epoch_duration = end_time - start_time

        # --- Validation and early stopping ---
        if traing_against_validation:
            val_mse, val_LL = evaluate_on_val(
                global_mean, global_std, global_max_time, global_max_dose,
                enable_nf_training, enable_ae_training, max_points_visible,
                func, noise, reducer, dataset_val, dataloader_val, 10,
                device, encoder, initial_encoder, t_dense
            )

            best_val_mse, best_val_LL, epochs_no_improve, stop_training, best_model_state = \
                validate_and_update_early_stop(
                    epoch, warmup_epochs_noise, traing_against_validation,
                    val_mse, val_LL, best_val_mse, best_val_LL, epochs_no_improve,
                    patience, models
                )
            if stop_training:
                break

        # --- Logging ---
        
        log_training_epoch(epoch_duration,
            epoch, total_mse, total_loss, total_recon, total_kl, total_transform,
            optimizer, noise, dataset, batch_size,
            val_mse=val_mse, val_LL=val_LL,
            enable_ae_training=enable_ae_training,
            enable_nf_training=enable_nf_training,
            print_epoch=print_epoch
        )

        # --- Plot training records ---
        if plot_from_training_records_enable and epoch % plot_epoch == 0:
            plot_from_training_records(
                batch_size, device, global_max_time, global_max_dose, global_mean, global_std,
                latent_dim, records=trajectory_records, func=func, reducer=reducer,
                initial_encoder=initial_encoder, ODEWrapper=ODEWrapper, t_dense=t_dense,
                max_plots=max_plots, nr_row=nr_row, nr_col=nr_col
            )

        torch.cuda.empty_cache()
        gc.collect()

    return min(mse_history), best_val_mse

        
def export_training_data(test_dataset, train_dataset_raw, global_mean,global_std,global_max_time,i):
    
    all_t_training, all_x_training, id_training = [], [], []
    for sample in train_dataset_raw:
        t_i = sample[0]   # Time
        x_i = sample[1]   # C2
        id_i=sample[4]
        all_t_training.append(t_i)
        all_x_training.append(x_i)
        id_training.append(id_i)
        
    flat_ids, flat_t, flat_x, flat_time0, ignore = [], [], [], [], []

    for t_i, x_i, id_i in zip(all_t_training, all_x_training, id_training):
        # Flatten each sample
        t_list = t_i.tolist()
        x_list = x_i.tolist()
        
        flat_t.extend(t_list)
        flat_x.extend(x_list)
        flat_ids.extend([id_i] * len(t_list))
        
        # Create binary column: 1 if Time == 0 else 0
        flat_time0.extend([1 if np.round(t*global_max_time,4) == 3 else 0 for t in t_list])
        ignore.extend([0  for t in t_list])

  
                


    all_t, all_x, id_test = [], [], []
    for sample in test_dataset:
        t_i = sample[0]   # Time
        x_i = sample[1]   # C2
        id_i=sample[4]
        all_t.append(t_i)
        all_x.append(x_i)
        id_test.append(id_i)
    
    # Apply truncation
    truncation_time = 0.6
    truncated_t, truncated_x, masks = truncate_time_series(all_t, all_x, truncation_time)
    
    
    flat1_ids, flat1_t, flat1_x, flat1_time0, ignore_test = [], [], [], [], []

    for t_i, x_i, id_i in zip(truncated_t, truncated_x, id_test):
        t_list = t_i.tolist()
        x_list = x_i.tolist()
        
        flat_t.extend(t_list)
        flat_x.extend(x_list)
        flat_ids.extend([id_i] * len(t_list))
        ignore_test.extend([0  for t in t_list])
        
        # Binary column: 1 if Time == 0 else 0
        flat_time0.extend([1 if np.round(t*global_max_time,4) == 3 else 0 for t in t_list])
   # Destandardize and round
    DV_flat = np.round(destandardize_concentration(np.array(flat_x), global_mean, global_std), 4)
    DV_flat1 = np.round(destandardize_concentration(np.array(flat1_x), global_mean, global_std), 4)
    time_flat = np.round(np.array(flat_t) * global_max_time, 4)
    time_flat1 = np.round(np.array(flat1_t) * global_max_time, 4)
    
    flat_disc_ids, flat_disc_t, flat_disc_x, flat_disc_time0, ignore2 = [], [], [], [], []

    for t_i, x_i, mask_i, id_i in zip(all_t, all_x, masks, id_test):
        # Inverse the mask → gives us the points that were truncated away
        inv_mask = ~mask_i.bool()  # True for discarded points
    
        t_list = t_i[inv_mask].tolist()
        x_list = x_i[inv_mask].tolist()
    
        flat_disc_t.extend(t_list)
        flat_disc_x.extend(x_list)
        flat_disc_ids.extend([id_i] * len(t_list))
    
        # Dose column: 1 at time 0, else 0
        flat_disc_time0.extend([1 if np.round(t*global_max_time,4) == 3  else 0 for t in t_list])
        ignore2.extend([1 for t in t_list])

    # Destandardize concentrations
    DV_disc = np.round(destandardize_concentration(np.array(flat_disc_x), global_mean, global_std), 4)
    
    # Convert normalized times back to original scale
    time_disc = np.round(np.array(flat_disc_t) * global_max_time, 4)
 
    
    
    # Create a DataFrame
    df_export = pd.DataFrame({
        'ID': flat_ids + flat1_ids + flat_disc_ids,
        'Time': np.concatenate([time_flat, time_flat1, time_disc]),
        'DV': np.concatenate([DV_flat, DV_flat1, DV_disc]),
        'Amount': flat_time0 + flat1_time0+flat_disc_time0,
        'Censoring': ignore + ignore_test + ignore2
    })
            
    # Export to CSV
    filename = f"monolix_data_{i}.csv"
    df_export.to_csv(filename, index=False)
    
    
    flat_disc_ids, flat_disc_t, flat_disc_x, flat_disc_time0 = [], [], [], []

    for t_i, x_i, mask_i, id_i in zip(all_t, all_x, masks, id_test):
        # Inverse the mask → gives us the points that were truncated away
        inv_mask = ~mask_i.bool()  # True for discarded points
    
        t_list = t_i[inv_mask].tolist()
        x_list = x_i[inv_mask].tolist()
    
        flat_disc_t.extend(t_list)
        flat_disc_x.extend(x_list)
        flat_disc_ids.extend([id_i] * len(t_list))
    
        # Dose column: 1 at time 0, else 0
        flat_disc_time0.extend([1 if t == 0 else 0 for t in t_list])
    
    # Destandardize concentrations
    DV_disc = np.round(destandardize_concentration(np.array(flat_disc_x), global_mean, global_std), 4)
    
    # Convert normalized times back to original scale
    time_disc = np.round(np.array(flat_disc_t) * global_max_time, 4)
    print(time_disc)
    # Create DataFrame
    df_discarded = pd.DataFrame({
        'ID': flat_disc_ids,
        'Time': time_disc,
        'DV': DV_disc,
        'Amount': flat_disc_time0
    })
    
    # Export to CSV
  #  filename = f"test_data_{i}.csv"
  #  df_export.to_csv(filename, index=False)
    





class TrajectoryDataset(Dataset):
    def __init__(
        self,
        path,
        compartment='C2',
        augment_with_prefixes=False,
        augment_dose_times=False,
        dose_jitter_std=0.01,
        # Optional global normalization constants
        max_dose=None,
        max_time=None,
        conc_mean=None,
        conc_std=None,
        dose_max_abs_dict=None
    ):
        # === Step 1: Load data ===
        self.df = pd.read_csv(path)
        self.compartment = compartment
        self.augment = augment_with_prefixes
        self.augment_dose_times = augment_dose_times
        self.dose_jitter_std = dose_jitter_std

        # === Step 2: Final normalization parameters (once only) ===
        self.max_dose = max_dose if max_dose is not None else self.df['Dose'].max()
        self.max_time = max_time if max_time is not None else self.df['Time'].max()
        self.conc_mean = conc_mean if conc_mean is not None else self.df[self.compartment].mean()
        self.conc_std = conc_std if conc_std is not None else self.df[self.compartment].std()

        # === Step 3: Apply normalization once ===
        self.df['Dose_norm'] = self.df['Dose'] / self.max_dose
        self.df['Time_norm'] = self.df['Time'] / self.max_time
        self.df['C2_norm'] = (self.df[self.compartment] - self.conc_mean) / self.conc_std

       # === Step 4: Per-dose median normalization and per-trajectory cmax normalization ===

        # --- Step 4a: Normalize against dose-group median ---
        if dose_max_abs_dict is None:
            dose_group_median = {}
            for dose, df_dose in self.df.groupby('Dose'):
                median_val = float(df_dose[self.compartment].median())
                dose_group_median[dose] = median_val if median_val != 0 else 1.0
            self.dose_max_abs = dose_group_median
        else:
            self.dose_max_abs = dose_max_abs_dict
        
        # Column: normalized against dose median
        self.df['C2_dose_norm'] = self.df.apply(
            lambda row: row[self.compartment] / (self.dose_max_abs[row['Dose']] + 1e-8),
            axis=1
        )
        
    
         
                
        
        # === Step 5: Normalize dose times ===
        self.df['DoseTimesParsed'] = self.df['Dose times'].apply(ast.literal_eval)
        self.df['DoseTimesNorm'] = self.df['DoseTimesParsed'].apply(
            lambda lst: [t / self.max_time for t in lst]
        )

        # === Step 6: Dose label mapping ===
        unique_doses = sorted(self.df['Dose'].unique())
        self.dose_to_label = {dose: i for i, dose in enumerate(unique_doses)}
        self.df['DoseLabel'] = self.df['Dose'].map(self.dose_to_label)

        # === Step 7: Extract trajectories ===
        self.trajectories = self._extract_trajectories()

        # === Step 8: Generate augmented samples ===
        self.samples = self._generate_samples(self.trajectories)

    def _jitter_dose_times(self, dose_times):
        noise = torch.randn_like(dose_times) * self.dose_jitter_std
        jittered = dose_times + noise
        return torch.clamp(jittered, 0.0, 1.0)

    def _random_schedule_preserving_intervals(self, dose_times):
        dose_times = dose_times.flatten()
        shift = torch.rand(1) * 0.5
        new_dose_times = dose_times + shift
        return new_dose_times, shift

    def _extract_trajectories(self):
        start_idxs = self.df[self.df['Time'] == 0].index.tolist() + [len(self.df)]
        trajectories = []
        for i in range(len(start_idxs) - 1):
            group = self.df.iloc[start_idxs[i]:start_idxs[i + 1]]
            t = torch.tensor(group['Time_norm'].values, dtype=torch.float32)
            x_global = torch.tensor(group['C2_norm'].values, dtype=torch.float32)
            x_dose = torch.tensor(group['C2_dose_norm'].values, dtype=torch.float32)
            dose = torch.tensor(group['Dose_norm'].values[0], dtype=torch.float32)
            dose_times = torch.tensor(group['DoseTimesNorm'].values[0], dtype=torch.float32)
            subject_id = group['ID'].iloc[0]
            trajectories.append((t, x_global, dose, dose_times, subject_id, x_dose))
        return trajectories

    def _generate_samples(self, trajectories):
        augmented = []
        for t, x_global, dose, dose_times, subject_id, x_dose_norm in trajectories:
            subject_id_str = str(subject_id)

            # Optionally augment by shifting dose times
            if self.augment_dose_times:
                shifted_dose_times, t_shift = self._random_schedule_preserving_intervals(dose_times)
                shifted_t = t + t_shift
                mask = (shifted_t >= 0.0) & (shifted_t <= 1.0)
                if mask.sum() > 0:
                    augmented.append((
                        shifted_t[mask],
                        x_global[mask],
                        dose,
                        shifted_dose_times[(shifted_dose_times >= 0.0) & (shifted_dose_times <= 1.0)],
                        subject_id_str + "_shifted",
                        x_dose_norm[mask]
                    ))

            # Always include the original
            augmented.append((t, x_global, dose, dose_times, subject_id_str, x_dose_norm))

            # Prefix/suffix augmentation
            if self.augment:
                T = len(t)
                for end in range(1, T):
                    augmented.append((
                        t[:end],
                        x_global[:end],
                        dose,
                        dose_times,
                        subject_id_str + f"_prefix{end}",
                        x_dose_norm[:end]
                    ))
                for start in range(1, T - 1):
                    augmented.append((
                        t[start:],
                        x_global[start:],
                        dose,
                        dose_times,
                        subject_id_str + f"_suffix{start}",
                        x_dose_norm[start:]
                    ))
        return augmented

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class TrajectoryDataset_paracetamol(Dataset):
    def __init__(
        self,
        path,
        compartment='C2',
        augment_with_prefixes=False,
        augment_dose_times=False,
        dose_jitter_std=0.01,
        max_dose=None,
        max_time=None,
        conc_mean=None,
        conc_std=None,
        subset_ids=None  # only use IDs now
    ):
        # --- Step 1: Load data ---
        self.df = pd.read_csv(path, sep=";")
        self.compartment = compartment
        self.augment = augment_with_prefixes
        self.augment_dose_times = augment_dose_times
        self.dose_jitter_std = dose_jitter_std

        # --- Step 2: Normalization ---
        self.max_dose = max_dose if max_dose is not None else self.df['Dose'].max()
        self.max_time = max_time if max_time is not None else self.df['Time'].max()
        self.conc_mean = conc_mean if conc_mean is not None else self.df[self.compartment].mean()
        self.conc_std = conc_std if conc_std is not None else self.df[self.compartment].std()

        self.df['Dose_norm'] = self.df['Dose'] / self.max_dose
        self.df['Time_norm'] = self.df['Time'] / self.max_time
        self.df['C2_norm'] = (self.df[self.compartment] - self.conc_mean) / self.conc_std
        self.df['C2_dose_norm'] = self.df['C2_norm']
        self.df['DoseTimesParsed'] = self.df['Dose times'].apply(ast.literal_eval)
        self.df['DoseTimesNorm'] = self.df['DoseTimesParsed'].apply(lambda lst: [t/self.max_time for t in lst])

        # --- Step 3: Dose label mapping ---
        unique_doses = sorted(self.df['Dose'].unique())
        self.dose_to_label = {dose:i for i,dose in enumerate(unique_doses)}
        self.df['DoseLabel'] = self.df['Dose'].map(self.dose_to_label)

        # --- Step 4: Extract trajectories per occasion ---
        all_trajectories = self._extract_trajectories()

        # Store all subject IDs for splitting
        self.all_subject_ids = [traj[4] for traj in all_trajectories]  # traj[4] = subject ID

        # --- Step 5: Filter by subset_ids ---
        if subset_ids is not None:
            trajectories = [traj for traj in all_trajectories if traj[4] in subset_ids]
        else:
            trajectories = all_trajectories

        # --- Step 6: Generate samples ---
        self.samples = self._generate_samples(trajectories)

    def _jitter_dose_times(self, dose_times):
        noise = torch.randn_like(dose_times) * self.dose_jitter_std
        jittered = dose_times + noise
        return torch.clamp(jittered, 0.0, 1.0)

    def _random_schedule_preserving_intervals(self, dose_times):
        dose_times = dose_times.flatten()
        shift = torch.rand(1) * 0.5
        new_dose_times = dose_times + shift
        return new_dose_times, shift

    def _extract_trajectories(self):
      trajectories = []
      grouped = self.df.groupby('Occasion')  # one trajectory per occasion
  
      for occ_name, group in grouped:
          t = torch.tensor(group['Time_norm'].values, dtype=torch.float32)
          x_global = torch.tensor(group['C2_norm'].values, dtype=torch.float32)
          x_dose = torch.tensor(group['C2_dose_norm'].values, dtype=torch.float32)
          dose = torch.tensor(group['Dose_norm'].values[0], dtype=torch.float32)
          dose_times = torch.tensor(group['DoseTimesNorm'].values[0], dtype=torch.float32)
          subject_id = group['ID'].iloc[0]  # actual subject ID, for splitting
  
          trajectories.append((t, x_global, dose, dose_times, subject_id, x_dose))
  
      return trajectories

    def _generate_samples(self, trajectories):
        augmented = []
        for t, x_global, dose, dose_times, subject_id, x_dose_norm in trajectories:
            subject_id_str = str(subject_id)

            # Optionally augment by shifting dose times
            if self.augment_dose_times:
                shifted_dose_times, t_shift = self._random_schedule_preserving_intervals(dose_times)
                shifted_t = t + t_shift
                mask = (shifted_t >= 0.0) & (shifted_t <= 1.0)
                if mask.sum() > 0:
                    augmented.append((
                        shifted_t[mask],
                        x_global[mask],
                        dose,
                        shifted_dose_times[(shifted_dose_times >= 0.0) & (shifted_dose_times <= 1.0)],
                        subject_id_str + "_shifted",
                        x_dose_norm[mask]
                    ))

            # Always include the original
            augmented.append((t, x_global, dose, dose_times, subject_id_str, x_dose_norm))

            # Prefix/suffix augmentation
            if self.augment:
                T = len(t)
                for end in range(1, T):
                    augmented.append((
                        t[:end],
                        x_global[:end],
                        dose,
                        dose_times,
                        subject_id_str + f"_prefix{end}",
                        x_dose_norm[:end]
                    ))
                for start in range(1, T - 1):
                    augmented.append((
                        t[start:],
                        x_global[start:],
                        dose,
                        dose_times,
                        subject_id_str + f"_suffix{start}",
                        x_dose_norm[start:]
                    ))
        return augmented

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]





def collate_fn(batch):
    # Unpack: t, x_global, x_dose, dose, dose_times, id
    t_list, x_global_list, dose_list, dose_times_list, id_list, x_dose_list = zip(*batch)

    # Pad time & both x variants
    t_padded = pad_sequence(t_list, batch_first=True)
    x_global_padded = pad_sequence(x_global_list, batch_first=True)
    x_dose_padded = pad_sequence(x_dose_list, batch_first=True)

    # Build mask
    max_len = t_padded.size(1)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, t in enumerate(t_list):
        mask[i, :len(t)] = 1

    # Dose tensor
    dose_tensor = torch.stack(dose_list)

    return (
        id_list,
        t_padded,
        x_global_padded,  # z-score normalized
        mask,
        dose_tensor,
        dose_times_list,
        x_dose_padded,    # per-dose normalized
    )






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
    kl=kl_per_dim.sum(dim=1)
    # Apply free bits per dim
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    if mask is not None:
        # Flatten to [B], keep only valid samples
        kl=kl_per_dim.sum(dim=1)

        mask = mask.view(mask.shape[0], -1).any(dim=1).float()
        kl = kl * mask
        denom = mask.sum().clamp(min=1.0)
    else:
        denom = kl.shape[0]  # number of samples


    return kl.sum() / denom




def kl_divergence_NF(
    epoch,
    warmup_epochs_iiv,
    mu_q,
    logvar_q,
    mu_p,
    logvar_p,
    log_det=None,
    free_bits=0.0,
    keep_mask=None,
    log_det_penalty_lambda=0.0,  # new
):
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    kl_per_dim = 0.5 * ((var_q + (mu_q - mu_p) ** 2) / var_p - 1 + logvar_p - logvar_q)
    kl_gauss = kl_per_dim.sum(dim=1)  # [B]

    if log_det is not None:
        kl = kl_gauss - log_det
    else:
        kl = kl_gauss

    if free_bits > 0.0:
        kl = torch.clamp(kl, min=free_bits)

    kl = torch.clamp(kl, min=0.0)

    if keep_mask is not None:
        # Ensure shape is [B], not [B*D]
        if keep_mask.dim() > 1:
            keep_mask = keep_mask.any(dim=1)  # Collapse per-feature mask to per-sample mask
    
        keep_mask = keep_mask.view(-1)  # Now shape should be [B]
        kl = kl * keep_mask
        denom = keep_mask.sum().clamp(min=1.0)
        if log_det is not None:
            log_det = log_det * keep_mask
    else:
        denom = kl.size(0)

    # Compute log_det penalty
    if log_det is not None and log_det_penalty_lambda > 0:
        log_det_penalty = log_det_penalty_lambda * torch.mean(log_det ** 2)
    else:
        log_det_penalty = torch.tensor(0.0, device=kl.device)

    return kl.sum() / denom, log_det.sum() / denom if log_det is not None else None, kl_gauss.sum() / denom, log_det_penalty



def train_model_paracetamol(dataset, dataset_val,global_max_time,
  global_max_dose,global_mean, global_std, main_params, dataloader_val, dataloader,models, optimizer,scheduler, func, reducer, initial_encoder,
    encoder, noise, t_dense,
     n_epochs, warmup_epochs_noise,warmup_epochs_iiv,
    smoothing_start_epoch,traing_against_validation,
     enable_ae_training,enable_nf_training,enable_onlymedian_training,plot_from_training_records_enable, free_bits, max_points_visible, print_epoch=1,
    plot_epoch=1,
    max_plots=4,
    nr_col=1,
    nr_row=5):
    
    device = next(func.parameters()).device
    dim_parameter_encoder = func.dim_parameter_encoder
    latent_dim = func.dim_latent

    best_val_mse = float("inf")
    epochs_no_improve = 0
    patience = 30  # stop if no improvement for 10 epochs

    batch_size=dataloader.batch_size
      
    t_dense = t_dense.to(device)
 

    
    MAX_TIME = global_max_time
    MAX_DOSE = global_max_dose
 



    if enable_nf_training:
        print(f"{'---- NF training initialized ----' if enable_nf_training else '---- AE training initialized ----'}")
    else:
        print(f"{'---- AE training initialized ----' if enable_ae_training else '---- VAE training initialized ----'}")

    print(f"Total epochs: {n_epochs}")
    print(f"Warmup epochs (noise): {warmup_epochs_noise}")
    print(f"Warmup epochs (IIV): {warmup_epochs_iiv}")
    print(f"Smoothing (EMA) starts after epoch: {smoothing_start_epoch}")
    print("========================================")



    for epoch in range(n_epochs):
        first_batch = True
        start_time = time.time()


        # Log when key stages start
        if epoch == warmup_epochs_noise:
            print(f"[Epoch {epoch}] ➤ Noise training activated.")
        
        if epoch == warmup_epochs_iiv:
            print(f"[Epoch {epoch}] ➤ Full KL regularization activated.")
        
        if epoch == smoothing_start_epoch:
            print(f"[Epoch {epoch}] ➤ EMA smoothing activated.")
            
       
                        
        total_transform = 0.0
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
        dose_max_abs = 1 #compute_max_per_dose(dataloader, device)
        
        for id_list, t_padded, x_padded, mask, dose_tensor, dose_times_list, x_normalized in dataloader:
            t_padded, x_padded, mask,x_normalized = t_padded.to(device), x_padded.to(device), mask.to(device), x_normalized.to(device)
            dose_tensor = dose_tensor.to(device)
            dose_times_list = [dt.to(device) for dt in dose_times_list]
            batch_size = t_padded.size(0)

            dose_times_padded, dose_times_mask = pad_dose_times(dose_times_list)

            batch_size, max_len = dose_times_padded.shape
            dose_tensor.shape == [batch_size, 1]  # e.g., [20, 1]
         
            dose_tensor_expanded = dose_tensor.unsqueeze(1).repeat(1, dose_times_padded.size(1))  # [10, 4]
            dose_tensor_expanded = dose_tensor_expanded.unsqueeze(-1)  # [10, 4, 1]
            
         
            dose_times_expanded = dose_times_padded.unsqueeze(-1)  # [10, 4, 1]
            
      
            dose_features = torch.cat([dose_times_expanded, dose_tensor_expanded], dim=-1)  # [10, 4, 2]
           

         
            t, x, x_normalized = t_padded, x_padded, x_normalized
      
            
            
            if max_points_visible > 0 :
                
                t_list, x_list, _ = truncate_time_series(t, x_normalized, max_points_visible) 
                t_encoder = pad_sequence(t_list, batch_first=True).to(device)
                x_normalized = pad_sequence(x_list, batch_first=True).to(device)
  
          
       
          
            if enable_nf_training:
                   # x_drop, encoder_mask = block_mask_some_samples(x_low_normalized, block_fraction=0.99, sample_fraction=p_dropout)
                    _, z_refined, mu_q, logvar_q, log_det = encoder(t_encoder, x_normalized) #, encoder_mask)
           
            else:
                    _, _, mu_q, logvar_q,_ = encoder(t_encoder,x_normalized)
                    std_q = torch.exp(0.5 * logvar_q)
                    z_refined= mu_q + std_q * torch.randn_like(mu_q)
                    

            if enable_ae_training:
                z_refined=mu_q
    
            if enable_onlymedian_training:
               z_refined= torch.zeros_like(z_refined)
           
      
            x0_1 = initial_encoder(x_padded[:, 0].unsqueeze(1))
         
            x0 = torch.cat([x0_1+ torch.randn_like(x0_1) * 0.01, z_refined+ torch.randn_like(z_refined) * 0.01], dim=1)
         

            dose_mask_expanded = dose_times_expanded.unsqueeze(-1)  # shape [batch, max_len, 1]
            keep_mask1 = torch.ones(batch_size, 1, device=device)
 
            ode_func = ODEWrapper(func, dose_times_expanded, dose_tensor_expanded, dose_times_mask)
        

            # === Interpolate and Calculate Loss ===
            t_dense_exp = t_dense.unsqueeze(0).repeat(batch_size, 1)
 
            
            if epoch == 0 and first_batch:
                print(f" ODE Solving starting")
            pred = odeint(ode_func, x0, t_dense, method='rk4')
     

            if epoch == 0 and first_batch:
            
                print(f" ODE Solving finished")
            pred_batch = pred.permute(1, 0, 2)  # [batch, time, features]
    
            t_dense_exp = t_dense.unsqueeze(0).repeat(batch_size, 1)
    
            test = reducer(pred_batch[:, :, :latent_dim])
            pred_interp = batch_linear_interpolate_1d(test, t_dense_exp, t_padded)
            recon_loss_noise, mse = noise.nll(destandardize_concentration(x_padded, global_mean, global_std), destandardize_concentration(pred_interp, global_mean, global_std), mask)
         
  

            mu_std = torch.zeros_like(mu_q)
            logvar_std = torch.zeros_like(logvar_q)
            
            
       
            if enable_ae_training:
                kl_weight = 0
                KL_loss=0
                log_det_penalty=torch.tensor(0)  
                log_det_sum=torch.tensor(0)  
               
            else:
                free_bits_on = 0 if epoch+1 >= warmup_epochs_iiv else free_bits* (1 - min(1.0, epoch / warmup_epochs_iiv))
                kl_weight = 1 if epoch+1 >= warmup_epochs_iiv else min(1.0,   epoch / warmup_epochs_iiv)

                if enable_nf_training:
                 
                    KL_loss, log_det_sum, kl_gauss,log_det_penalty = kl_divergence_NF(epoch,warmup_epochs_iiv,
                                                                                      mu_q, logvar_q,mu_std, logvar_std,log_det,free_bits=free_bits_on,  log_det_penalty_lambda=1)
                else:
                    KL_loss = kl_divergence_gaussians(mu_q, logvar_q, mu_std, logvar_std, free_bits_on)
                    kl_gauss=KL_loss
                    log_det_sum=torch.tensor(0)  
                    log_det_penalty=0
 

            loss = recon_loss_noise + kl_weight * KL_loss + log_det_penalty

          

            # === Backprop ===
            total_transform +=log_det_sum.item()
            total_loss +=  loss.item()
            total_mse += mse.item()
            total_kl += kl_gauss.item() if not enable_ae_training else 0.0

      
            
            
            total_recon +=  recon_loss_noise.item()
            
            optimizer.zero_grad()
            if epoch == 0 and first_batch:
                print(f"Backprop")
            loss.backward()

        
            for param_group in main_params:
                torch.nn.utils.clip_grad_norm_(param_group["params"], max_norm=1)
                
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
            mu_q_list.append(mu_q.detach())
            logvar_q_list.append(logvar_q.detach())

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

               
        with torch.no_grad():
            if epoch > smoothing_start_epoch:
                func.update_ema(alpha=0.1)
                reducer.update_ema(alpha=0.1)
                initial_encoder.update_ema(alpha=0.1)
                encoder.update_ema(alpha=0.1)
             

        scheduler.step(total_mse)
        end_time = time.time()
       
        # === Logging ===
        if epoch % print_epoch == 0:
            with torch.no_grad():
                # Assuming main_params[0] corresponds to the main parameters with base_lr
                main_lr = optimizer.param_groups[0]['lr']
          
                if enable_ae_training:
                    end_time = time.time()
                   
                    print(
                        f"Epoch {epoch}, "
                        f"MSE {1/20 * total_mse:.1f} "
                        f"-LL: {1/20 * total_recon:.4f}, "
                        f"Add. error: {float(noise.sigma_add):.2f}, "
                        f"Prop. error: {float(noise.sigma_prop):.2f}, "

                        f"lr: {main_lr:.3f}",
                        f"{end_time - start_time:.2f} seconds")
                   
             
                    
                if enable_nf_training:   
                    end_time = time.time()

                    print(
                        f"Epoch {epoch}, "
                        f"MSE {1/20 * total_mse:.1f} "
                        f"loss: {1/20 * total_loss:.4f}, "
                        f"-LL: {batch_size/len(dataset) * total_recon:.1f}, "
                        f"KL: {batch_size/len(dataset) * total_kl:.8f}, "
                        f"log_det: {batch_size/len(dataset) * total_transform:.8f}, "
                        
                        f"Add. error: {float(noise.sigma_add):.2f}, "
                        f"Prop. error: {float(noise.sigma_prop):.2f}, "
                        f"lr: {main_lr:.3f}",
                        f"{end_time - start_time:.1f} seconds")
                if not enable_nf_training and not enable_ae_training:
 
                    end_time = time.time()

                    print(
                        f"Epoch {epoch}, "
                        f"MSE {1/20 * total_mse:.1f} "
                        f"loss: {1/20* total_loss:.2f}, "
                        f"-LL: {1/20 * total_recon:.2f}, "
                        f"KL: {1/20 * total_kl:.4f}, "
                        
                        f"Add. error: {float(noise.sigma_add):.2f}, "
                        f"Prop. error: {float(noise.sigma_prop):.2f}, "
                        f"1000 lr: {1000*main_lr:.1f}",
                        f"{end_time - start_time:.1f} seconds")
                  
                    
                
                    
                    
                    
                    
        if plot_from_training_records_enable:
            if epoch % plot_epoch == 0:
                plot_from_training_records(batch_size, device, global_max_time, global_max_dose, global_mean, global_std, latent_dim, records=trajectory_records, func=func, reducer=reducer, initial_encoder=initial_encoder, ODEWrapper=ODEWrapper, t_dense=t_dense, max_plots=max_plots, nr_row=nr_row, nr_col=nr_col)
            
        if traing_against_validation:
            val_mse = evaluate_on_val(global_mean, global_std, global_max_time, global_max_dose, enable_nf_training, enable_ae_training, max_points_visible, func, noise, reducer, dataset_val, dataloader_val, 10, device, encoder, initial_encoder, t_dense)

        if epoch % 10 == 0:
          print(f"MSE on validation set: {val_mse}")
      
    #   Initialize `best_val_mse` with a low number like 0.0 or use `float('-inf')` earlier in your code
        if val_mse < best_val_mse:
          best_val_mse = val_mse
          epochs_no_improve = 0
          best_model_state = {k: v.state_dict() for k, v in models.items()}
        else:
          epochs_no_improve += 1
          print(f"[Early Stop Monitor] No improvement for {epochs_no_improve} epochs")


      
        if epochs_no_improve >= patience:
         if epoch >= 500:
              print(f"Early stopping triggered after {epoch + 1} epochs due to MSE not increasing")
         break

        torch.cuda.empty_cache()
        gc.collect()
        
        

def train_model(
    global_mean,
    global_std,
    global_max_time,
    global_max_dose,
    p_dropout,
    main_params,
    dataloader_val,
    dataloader,
    models,
    optimizer,
    scheduler,
    func,
    reducer,
    initial_encoder,
    encoder,
    noise,
    t_dense,
    n_epochs,
    warmup_epochs_noise,
    warmup_epochs_iiv,
    smoothing_start_epoch,
    traing_against_validation,
    enable_ae_training,
    enable_nf_training,
    enable_onlymedian_training,
    plot_from_training_records_enable,
    free_bits,
    df,
    df_val,
    dataset,
    dataset_val,
    max_points_visible,
    print_epoch=1,
    plot_epoch=1,
    max_plots=4,
    nr_col=1,
    nr_row=5,
):
    """
    Train a neural ODE model for concentration–time data with flexible modes (AE, NF, VAE).

    Training procedure:
      1. Iterates through training batches, prepares dose/time features and encodes latent states.
      2. Solves ODE trajectories with `torchdiffeq.odeint`.
      3. Computes losses: reconstruction (NLL), KL divergence (VAE/NF), log-det penalty (NF).
      4. Backpropagation with gradient clipping.
      5. Optionally validates on a separate dataset and applies early stopping.
      6. Logs progress, learning rates, and can plot trajectories during training.

    Args:
        global_mean (float): Mean concentration (for denormalization).
        global_std (float): Std concentration (for denormalization).
        global_max_time (float): Time scaling factor.
        global_max_dose (float): Dose scaling factor.
        p_dropout (float): Dropout probability for encoder masking.
        main_params (list): Parameter groups for optimizer.
        dataloader_val (DataLoader): Validation dataloader.
        dataloader (DataLoader): Training dataloader.
        models (dict): Dictionary of model components (func, reducer, encoders, etc.).
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        func (nn.Module): Neural ODE function.
        reducer (nn.Module): Reduces ODE latent trajectories to outputs.
        initial_encoder (nn.Module): Encodes initial concentration state.
        encoder (nn.Module): Variational encoder (VAE/NF/AE).
        noise (nn.Module): Noise model for NLL computation.
        t_dense (torch.Tensor): Dense time grid for ODE solver.
        n_epochs (int): Number of training epochs.
        warmup_epochs_noise (int): Epochs before enabling noise learning.
        warmup_epochs_iiv (int): Epochs before enabling full KL regularization.
        smoothing_start_epoch (int): Epoch to start EMA smoothing.
        traing_against_validation (bool): If True, evaluates on validation set.
        enable_ae_training (bool): Train as AE (no KL).
        enable_nf_training (bool): Train with normalizing flows.
        enable_onlymedian_training (bool): Force latent z to zero (median).
        plot_from_training_records_enable (bool): Whether to plot trajectories during training.
        free_bits (float): Free-bits threshold for KL divergence.
        df (DataFrame): Training dataframe (for plotting).
        df_val (DataFrame): Validation dataframe (for plotting).
        dataset (Dataset): Training dataset.
        dataset_val (Dataset): Validation dataset.
        max_points_visible (int): Max observed points visible to encoder.
        print_epoch (int): Logging frequency.
        plot_epoch (int): Plotting frequency.
        max_plots (int): Max number of subjects to plot.
        nr_col (int): Number of subplot columns (plots).
        nr_row (int): Number of subplot rows (plots).

    Returns:
        None
    """

    device = next(func.parameters()).device
    latent_dim = func.dim_latent

    best_val_mse = float("inf")
    epochs_no_improve = 0
    patience = 30  # early stopping patience

    batch_size = dataloader.batch_size
    t_dense = t_dense.to(device)

    # --- Initialization Logs ---
    if enable_nf_training:
        print("---- NF training initialized ----")
    elif enable_ae_training:
        print("---- AE training initialized ----")
    else:
        print("---- VAE training initialized ----")

    print(f"Total epochs: {n_epochs}")
    print(f"Warmup epochs (noise): {warmup_epochs_noise}")
    print(f"Warmup epochs (IIV): {warmup_epochs_iiv}")
    print(f"Smoothing (EMA) starts after epoch: {smoothing_start_epoch}")
    print("========================================")

    # --- Training Loop ---
    for epoch in range(n_epochs):
        first_batch = True
        start_time = time.time()

        # Stage logs
        if epoch == warmup_epochs_noise:
            print(f"[Epoch {epoch}] ➤ Noise training activated.")
        if epoch == warmup_epochs_iiv:
            print(f"[Epoch {epoch}] ➤ Full KL regularization activated.")
        if epoch == smoothing_start_epoch:
            print(f"[Epoch {epoch}] ➤ EMA smoothing activated.")

        # Reset epoch accumulators
        total_transform = total_loss = total_kl = total_recon = total_mse = 0.0
        z_individual_list, logvar_q_list, mu_q_list, trajectory_records = [], [], [], []

        # Enable/disable noise training
        for param in noise.parameters():
            param.requires_grad = epoch >= warmup_epochs_noise

        # --- Mini-batch Training ---
        for id_list, t_padded, x_padded, mask, dose_tensor, dose_times_list, x_normalized in dataloader:
            # Move to device
            t_padded, x_padded, mask, x_normalized = (
                t_padded.to(device),
                x_padded.to(device),
                mask.to(device),
                x_normalized.to(device),
            )
            dose_tensor = dose_tensor.to(device)
            dose_times_list = [dt.to(device) for dt in dose_times_list]
            batch_size = t_padded.size(0)

            # --- Pad and expand dose times ---
            dose_times_padded, dose_times_mask = pad_dose_times(dose_times_list)
            dose_tensor_expanded = dose_tensor.unsqueeze(1).repeat(1, dose_times_padded.size(1)).unsqueeze(-1)
            dose_times_expanded = dose_times_padded.unsqueeze(-1)

            # --- Encode latent z ---
            if enable_nf_training:
                _, z_refined, mu_q, logvar_q, log_det = encoder(t_padded, x_normalized)
            else:
                _, _, mu_q, logvar_q, _ = encoder(t_padded, x_normalized)
                std_q = torch.exp(0.5 * logvar_q)
                z_refined = mu_q + std_q * torch.randn_like(mu_q)

            if enable_ae_training:
                z_refined = mu_q
            if enable_onlymedian_training:
                z_refined = torch.zeros_like(z_refined)

            # --- Encode initial state ---
            x0_1 = initial_encoder(x_padded[:, 0].unsqueeze(1))
            x0 = torch.cat([x0_1 + 0.01 * torch.randn_like(x0_1),
                            z_refined + 0.01 * torch.randn_like(z_refined)], dim=1)

            # --- Solve ODE ---
            ode_func = ODEWrapper(func, dose_times_expanded, dose_tensor_expanded, dose_times_mask)
            if epoch == 0 and first_batch:
                print("ODE Solving starting")
            pred = odeint(ode_func, x0, t_dense, method="rk4")
            if epoch == 0 and first_batch:
                print("ODE Solving finished")

            pred_batch = pred.permute(1, 0, 2)  # [batch, time, features]

            # --- Interpolate predictions ---
            t_dense_exp = t_dense.unsqueeze(0).repeat(batch_size, 1)
            pred_interp = batch_linear_interpolate_1d(
                reducer(pred_batch[:, :, :latent_dim]), t_dense_exp, t_padded
            )

            # --- Loss computation ---
            recon_loss_noise, mse = noise.nll(
                destandardize_concentration(x_padded, global_mean, global_std),
                destandardize_concentration(pred_interp, global_mean, global_std),
                mask,
            )
            mu_std_low = torch.zeros_like(mu_q)
            logvar_std_low = torch.zeros_like(logvar_q)

            if enable_ae_training:
                kl_weight, KL_loss, log_det_penalty, log_det_sum = 0, 0, torch.tensor(0), torch.tensor(0)
            else:
                free_bits_on = 0 if epoch + 1 >= warmup_epochs_iiv else free_bits * (1 - min(1.0, epoch / warmup_epochs_iiv))
                kl_weight = 1 if epoch + 1 >= warmup_epochs_iiv else min(1.0, epoch / warmup_epochs_iiv)
                if enable_nf_training:
                    KL_loss, log_det_sum, kl_gauss, log_det_penalty = kl_divergence_NF(
                        epoch, warmup_epochs_iiv, mu_q, logvar_q,
                        mu_std_low, logvar_std_low, log_det,
                        free_bits=free_bits_on, log_det_penalty_lambda=1,
                    )
                else:
                    KL_loss = kl_divergence_gaussians(mu_q, logvar_q, mu_std_low, logvar_std_low, free_bits_on)
                    kl_gauss, log_det_sum, log_det_penalty = KL_loss, torch.tensor(0), 0

            loss = recon_loss_noise + kl_weight * KL_loss + log_det_penalty

            # --- Backpropagation ---
            total_transform += log_det_sum.item()
            total_loss += loss.item()
            total_mse += mse.item()
            total_kl += kl_gauss.item() if not enable_ae_training else 0.0
            total_recon += recon_loss_noise.item()

            optimizer.zero_grad()
            if epoch == 0 and first_batch:
                print("Backprop")
            loss.backward()

            for param_group in main_params:
                torch.nn.utils.clip_grad_norm_(param_group["params"], max_norm=1)

            if epoch == 0 and first_batch:
                for name, model in models.items():
                    for param_name, param in model.named_parameters():
                        if param.requires_grad and param.grad is None:
                            print(f"[WARNING] No gradient for {name}.{param_name}")

            optimizer.step()
            first_batch = False

            # Store latent trajectories for plotting
            z_individual_list.append(z_refined.detach())
            mu_q_list.append(mu_q.detach())
            logvar_q_list.append(logvar_q.detach())
            for i in range(batch_size):
                trajectory_records.append((
                    id_list[i], t_padded[i, mask[i]], x_padded[i, mask[i]],
                    dose_tensor[i], dose_times_list[i], z_refined[i].detach()
                ))

        # --- EMA update ---
        with torch.no_grad():
            if epoch > smoothing_start_epoch:
                func.update_ema(alpha=0.1)
                reducer.update_ema(alpha=0.1)
                initial_encoder.update_ema(alpha=0.1)
                encoder.update_ema(alpha=0.1)

        scheduler.step(total_mse)
        end_time = time.time()

        # --- Logging ---
        if epoch % print_epoch == 0:
            main_lr = optimizer.param_groups[0]["lr"]
            if enable_ae_training:
                print(f"Epoch {epoch}, MSE {batch_size/len(dataset) * total_mse:.1f} "
                      f"-LL: {batch_size/len(dataset) * total_recon:.4f}, "
                      f"Add. error: {torch.exp(noise.log_sigma_add).item():.8f}, "
                      f"Prop. error: {torch.exp(noise.log_sigma_prop).item():.1f}, "
                      f"lr: {main_lr:.3f}, {end_time - start_time:.2f}s")
            elif enable_nf_training:
                print(f"Epoch {epoch}, MSE {batch_size/len(dataset) * total_mse:.1f} "
                      f"loss: {batch_size/len(dataset) * total_loss:.4f}, "
                      f"-LL: {batch_size/len(dataset) * total_recon:.1f}, "
                      f"KL: {batch_size/len(dataset) * total_kl:.8f}, "
                      f"log_det: {batch_size/len(dataset) * total_transform:.8f}, "
                      f"Add. error: {(torch.exp(noise.log_sigma_add)).item():.2f}, "
                      f"Prop. error: {torch.exp(noise.log_sigma_prop).item():.4f}, "
                      f"lr: {main_lr:.3f}, {end_time - start_time:.1f}s")
            else:
                print(f"Epoch {epoch}, MSE {batch_size/len(dataset) * total_mse:.1f} "
                      f"loss: {batch_size/len(dataset) * total_loss:.2f}, "
                      f"-LL: {batch_size/len(dataset) * total_recon:.2f}, "
                      f"KL: {batch_size/len(dataset) * total_kl:.4f}, "
                      f"Add. error: {(torch.exp(noise.log_sigma_add)).item():.2f}, "
                      f"Prop. error: {torch.exp(noise.log_sigma_prop).item():.2f}, "
                      f"1000 lr: {1000 * main_lr:.1f}, {end_time - start_time:.1f}s")

        # --- Plotting ---
        if plot_from_training_records_enable and epoch % plot_epoch == 0:
            plot_from_training_records(
                batch_size, device, df=df, dataset=dataset, latent_dim=latent_dim,
                records=trajectory_records, func=func, reducer=reducer,
                initial_encoder=initial_encoder, ODEWrapper=ODEWrapper,
                t_dense=t_dense, max_plots=max_plots, nr_row=nr_row, nr_col=nr_col,
            )

        # --- Validation ---
        if traing_against_validation:
            val_mse = evaluate_on_val(
                global_mean, global_std, global_max_time, global_max_dose,
                enable_nf_training, enable_ae_training, max_points_visible,
                func, noise, reducer, dataset_val, dataloader_val,
                10, device, encoder, initial_encoder, t_dense,
            )
            if epoch % 10 == 0:
                print(f"MSE on validation set: {val_mse:.4f}")

            # Early stopping
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                epochs_no_improve = 0
                best_model_state = {k: v.state_dict() for k, v in models.items()}
            else:
                epochs_no_improve += 1
                print(f"[Early Stop Monitor] No improvement for {epochs_no_improve} epochs")

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        torch.cuda.empty_cache()
        gc.collect()
@torch.no_grad()
def evaluate_on_val2(
    global_mean,
    global_std,
    global_max_time,
    global_max_dose,
    enable_nf_training,
    enable_ae_training,
    max_points_visible,
    func,
    noise,
    reducer,
    dataset_val,
    dataloader_val,
    batch_size,
    device,
    encoder,
    initial_encoder,
    t_dense,
):
    """
    Evaluate model performance on a validation dataset by computing mean squared error (MSE).

    This function:
      1. Iterates through the validation dataloader.
      2. Pads dose times and aligns dose features for each subject.
      3. Encodes latent states using a provided encoder (supports NF and AE training modes).
      4. Initializes the ODE system and simulates trajectories using `torchdiffeq.odeint`.
      5. Interpolates model predictions to observed time points.
      6. Computes negative log-likelihood (via `noise`) and MSE.
      7. Returns the aggregated MSE across the validation dataset.

    Args:
        global_mean (float): Mean concentration used for denormalization.
        global_std (float): Std concentration used for denormalization.
        global_max_time (float): Time scaling factor.
        global_max_dose (float): Dose scaling factor.
        enable_nf_training (bool): Whether to use normalizing-flow training mode.
        enable_ae_training (bool): Whether to use autoencoder-only training mode.
        max_points_visible (int): Max points per series for truncated input (for encoders).
        func (nn.Module): Neural ODE function (dynamics model).
        noise (nn.Module): Noise model for likelihood evaluation.
        reducer (nn.Module): Reduces latent ODE output to concentration space.
        dataset_val (Dataset): Validation dataset.
        dataloader_val (DataLoader): Validation dataloader.
        batch_size (int): Batch size.
        device (torch.device): Device for computation (CPU/GPU).
        encoder (nn.Module): Encoder for latent state inference.
        initial_encoder (nn.Module): Encoder for initial state.
        t_dense (torch.Tensor): Dense time grid for ODE solver.

    Returns:
        float: Total mean squared error across the validation dataset.
    """

    # Use model's device
    device = next(func.parameters()).device
    latent_dim = func.dim_latent

    conc_mean, conc_std = global_mean, global_std
    total_mse = 0.0
    total_LL = 0.0

    # --- Iterate over validation batches ---
    for id_list, t_padded, x_padded, mask, dose_tensor, dose_times_list, x_normalized in dataloader_val:
        # Move tensors to device
        t_padded, x_padded, mask, x_normalized = (
            t_padded.to(device),
            x_padded.to(device),
            mask.to(device),
            x_normalized.to(device),
        )
        dose_tensor = dose_tensor.to(device)
        dose_times_list = [dt.to(device) for dt in dose_times_list]
        batch_size = t_padded.size(0)

        # --- Pad dose times ---
        dose_times_padded, dose_times_mask = pad_dose_times(dose_times_list)

        # Expand doses to match dose_times
        dose_tensor_expanded = dose_tensor.unsqueeze(1).repeat(1, dose_times_padded.size(1)).unsqueeze(-1)
        dose_times_expanded = dose_times_padded.unsqueeze(-1)

        # Dose features: [batch_size, max_len, 2] (time + dose amount)
        dose_features = torch.cat([dose_times_expanded, dose_tensor_expanded], dim=-1)

        # --- Prepare inputs for encoder ---
        t_low, x_low, x_low_normalized = t_padded, x_padded, x_normalized

        if max_points_visible > 0:
            # Truncate series for encoder
            t_low_list, x_low_list, _ = truncate_time_series(t_low, x_low_normalized, max_points_visible)
            t_low_encoder = pad_sequence(t_low_list, batch_first=True).to(device)
            x_low_normalized = pad_sequence(x_low_list, batch_first=True).to(device)
        else:
            t_low_encoder = t_low

        # --- Encode latent representation ---
        if enable_nf_training:
            # NF encoder returns refined latent + distributions
            _, z_refined, mu_q, logvar_q, log_det = encoder(t_low, x_low_normalized)
        else:
            # VAE-style encoder
            _, _, mu_q, logvar_q, _ = encoder(t_low_encoder, x_low_normalized)
            std_q = torch.exp(0.5 * logvar_q)
            z_refined = mu_q + std_q * torch.randn_like(mu_q) * 0  # (set to 0 → deterministic)

        if enable_ae_training:
            z_refined = mu_q

        # --- Encode initial state ---
        x0_1 = initial_encoder(x_padded[:, 0].unsqueeze(1))
        x0 = torch.cat([x0_1, z_refined], dim=1)

        # --- Simulate ODE ---
        ode_func = ODEWrapper(func, dose_times_expanded, dose_tensor_expanded, dose_times_mask)
        pred = odeint(ode_func, x0, t_dense, method="rk4")
        pred_batch = pred.permute(1, 0, 2)  # [batch_size, time_steps, latent_dim]

        # --- Interpolate predictions to observed times ---
        t_dense_exp = t_dense.unsqueeze(0).repeat(batch_size, 1)
        pred_interp = batch_linear_interpolate_1d(
            reducer(pred_batch[:, :, :latent_dim]), t_dense_exp, t_padded
        )

        # --- Compute reconstruction loss ---
        recon_loss_noise, mse = noise.nll(
            destandardize_concentration(x_padded, conc_mean, conc_std),
            destandardize_concentration(pred_interp, conc_mean, conc_std),
            mask,
        )

        # (unused placeholder for KL terms)
        mu_std_low = torch.zeros_like(mu_q)
        logvar_std_low = torch.zeros_like(logvar_q)

        # Accumulate MSE
        total_mse += mse.item()
        total_LL += recon_loss_noise.item()

    return total_mse, total_LL