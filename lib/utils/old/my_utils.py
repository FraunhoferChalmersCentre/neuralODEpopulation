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
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint as odeint
# from torchdiffeq import odeint_adjoint as odeint

#from lib.utils.model_validation import plot_individual_fits
   
def prepare_optimizer(models,device, lr=0.001):
    main_params = [
        {"params": list(models["func"].parameters()) +
                   list(models["reducer"].parameters()) +
                   list(models["initial_encoder"].parameters()) +
                   list(models["encoder"].parameters()), "lr": lr},
        {"params": list(models["noise"].parameters()), "lr": lr}
    ]
    optimizer = torch.optim.Adam(main_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=7, min_lr=1e-4
    )
    
    for name, model in models.items():
          model.to(device)
          print(f"{name} is on {next(model.parameters()).device}")
    
    return optimizer, scheduler, main_params


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
        t, x, dose_tensor, dose_times_tensor, subject_id, x_low_normalized, occasion = entry


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
        
     



        
        ode_func = ODEWrapper(func, dose_times_padded.unsqueeze(-1), dose_rep.unsqueeze(-1), dose_mask)

        pred = odeint(ode_func, x0, t_dense, method='rk4')
        pred = reducer(pred[:, :, :latent_dim])
        pred = destandardize_concentration(pred, conc_mean, conc_std).squeeze(-1)
        
     


        
        
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

        ax.set_title(f"ID {occasion} | Dose {global_max_dose * dose_val:.2f}")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Concentration")
        ax.grid(True)
        ax.legend()
        mse_tot.append(mse)
        r2_tot.append(r2)
        preds.append(y_pred_masked.cpu())
        obs.append(y_true_masked.cpu())
        id_list.append(np.array([occasion] * len(y_true_masked.cpu())))

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

def prepare_datasets_and_loaders(data_path,base_dir, all_ids, i,already_done, global_max_dose, global_max_time,
                                 global_mean, global_std, device, batch_fraction=0.05,trunctation=1):

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
    
    
    train_export_dataset = TrajectoryDataset_paracetamol(
    data_path,  # <--- pass path, not df
    augment_with_prefixes=False,
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


    export_training_data(test_dataset, train_export_dataset, global_mean, global_std, global_max_time, i+already_done, trunctation, base_dir)

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
def prepare_datasets_and_loaders_simulated(data_path_train, data_path_val, data_path_test,
                                        
                                           global_max_dose, global_max_time,
                                           global_mean, global_std, device,
                                           batch_fraction=0.05, trunctation=1):
    """
    Loads train/val/test datasets from CSV paths, creates DataLoaders, and generates
    the combined time+dose tensor.

    Returns:
        train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, combined
    """

    # --- Load datasets ---
    train_dataset = TrajectoryDataset(
        data_path_train,
        augment_with_prefixes=False,
        augment_dose_times=False,
        max_dose=global_max_dose,
        max_time=global_max_time,
        conc_mean=global_mean,
        conc_std=global_std
    )
    train_base_dataset = TrajectoryDataset(
        data_path_train,
        augment_with_prefixes=False,
        augment_dose_times=False,
        max_dose=global_max_dose,
        max_time=global_max_time,
        conc_mean=global_mean,
        conc_std=global_std
    )

    val_dataset = TrajectoryDataset(
        data_path_val,
        augment_with_prefixes=False,
        augment_dose_times=False,
        max_dose=global_max_dose,
        max_time=global_max_time,
        conc_mean=global_mean,
        conc_std=global_std
    )

    test_dataset = TrajectoryDataset(
        data_path_test,
        augment_with_prefixes=False,
        augment_dose_times=False,
        max_dose=global_max_dose,
        max_time=global_max_time,
        conc_mean=global_mean,
        conc_std=global_std
    )



    # --- Create DataLoaders ---
    batch_size_train = max(1, int(len(train_dataset) * batch_fraction))
    batch_size_val = max(1, int(len(val_dataset)))
    batch_size_test = max(1, int(len(test_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, collate_fn=collate_fn)

    # --- Combined dose + time tensor ---
    df = pd.read_csv(data_path_train)
    dose_times_lists = df['Dose times'].apply(ast.literal_eval).tolist()
    all_times = np.concatenate(dose_times_lists)
    unique_times_np = np.unique(all_times)
   

   # doses = torch.tensor([3], dtype=torch.float32) / 24
    time_points = torch.linspace(0, 1, steps=120)
    dose_times_tensor = torch.from_numpy(unique_times_np).float().to(time_points.device) / global_max_time
    combined = torch.cat((time_points, dose_times_tensor)).to(device)
    combined, _ = torch.sort(combined)  # ensure ascending order

    return train_dataset, val_dataset, test_dataset,train_base_dataset,  train_loader, val_loader, test_loader, combined, batch_size_train, batch_size_val, batch_size_test


def run_model_variant(variant_name, train_dataset, val_dataset, test_dataset,
                      models, main_params, optimizer, scheduler, combined,
                      global_max_dose, global_max_time, global_mean, global_std,
                      latent_dim, noise, encoder, func, reducer, initial_encoder,
                      ODEWrapper, device, metrics, residuals, iteration,
                      n_epochs, train_loader, val_loader, base_dir,
                      warmup_noise=0, warmup_iiv=0, enable_ae=False, enable_nf=False,plot_from_training_records_enable=False,
                      free_bits=0):
    """
    Train, evaluate, append metrics and residuals for a given model variant.

    Args:
        base_dir (str): directory to save metrics/residuals
        ...
        other args as before
    """
    # Run training loop
    mse_train, mse_validation = train_loop_model_paracetamol(
        train_dataset, val_dataset, global_max_time, global_max_dose,
        global_mean, global_std, main_params, val_loader, train_loader,
        models, optimizer, scheduler, func, reducer, initial_encoder, encoder,
        noise, t_dense=combined, n_epochs=n_epochs,
        warmup_epochs_noise=warmup_noise, warmup_epochs_iiv=warmup_iiv,
        smoothing_start_epoch=1000, traing_against_validation=True,
        enable_ae_training=enable_ae, enable_nf_training=enable_nf,
        enable_onlymedian_training=False, plot_from_training_records_enable=plot_from_training_records_enable,
        free_bits=free_bits, max_points_visible=0.6, print_epoch=10,
        plot_epoch=1, max_plots=30, nr_col=10, nr_row=3
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
    return mse_train

    
    
    
def compute_global_stats(df):
    """
    Compute global statistics for the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame with columns 'Dose', 'Time', 'C2'
    
    Returns:
        global_max_dose, global_max_time, global_mean, global_std, global_max_value
    """
    print(df)
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
        occ_list, t_padded, x_padded, mask, dose_tensor, dose_times_list, x_normalized = batch

        # --- Preprocess batch ---
        occ_list, t_padded, x_padded, mask, t_encoder, x_encoder, \
        dose_tensor_expanded, dose_times_expanded, dose_times_mask = preprocess_batch(
            batch, device, max_points_visible=max_points_visible
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
    occ_list, t_padded, x_padded, mask, dose_tensor, dose_times_list, x_normalized = batch
    
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

    return (
        occ_list,       # 1
        t_padded,       # 2
        x_padded,       # 3
        mask,           # 4
        t_encoder,      # 5
        x_encoder,      # 6
        dose_tensor_expanded,  # 7
        dose_times_expanded,   # 8
        dose_times_mask         # 9
    )





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
            #      f"val_mse: {val_mse:.3f}, "
            #      f"val_LL: {val_LL:.3f}, "
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
    # Remove first 4 measurements
    t_encoder_trimmed = t_encoder[:, 4:]
    x_normalized_trimmed = x_normalized[:, 4:]

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
        num_batches = len(dataloader)

        # --- Enable/disable noise learning ---
        for param in noise.parameters():
            param.requires_grad = epoch >= warmup_epochs_noise
            
        for batch in dataloader:
            # Unpack batch
            occ_list, t_padded, x_padded, mask, dose_tensor, dose_times_list, x_normalized = batch



            # --- Preprocess batch ---
            occ_list, t_padded, x_padded, mask, t_encoder, x_encoder, \
    dose_tensor_expanded, dose_times_expanded, dose_times_mask = preprocess_batch(
        batch, device, max_points_visible=max_points_visible
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
        for i in range(len(occ_list)):
                # Use the original padded sequences
                t_i = t_padded[i, :mask[i].sum()].detach()  # mask ensures only valid points
                x_i = x_padded[i, :mask[i].sum()].detach()
            
                # Sort by time just in case
                t_i_sorted, sort_idx = torch.sort(t_i)
                x_i_sorted = x_i[sort_idx]
            
                trajectory_records.append((
                    occ_list[i],
                    t_i_sorted,
                    x_i_sorted,
                    dose_tensor[i].detach(),
                    dose_times_list[i].detach() if isinstance(dose_times_list[i], torch.Tensor) else dose_times_list[i],
                    z_refined[i].detach()
                ))


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

    return min(mse_history)/num_batches, best_val_mse

        
def export_training_data(test_dataset, train_dataset_raw, global_mean, global_std, global_max_time, i,truncation, base_dir):
    def flatten_samples(times, values, ids, censoring_flag=0, truncate_mask=None):
        """Flatten time series data into lists for DataFrame export."""
        flat_t, flat_x, flat_ids, flat_time0, censoring = [], [], [], [], []
        
        for t_i, x_i, id_i in zip(times, values, ids):
            if truncate_mask is not None:
                # Apply mask to keep only selected points
                mask = truncate_mask.pop(0)
                t_i, x_i = t_i[mask], x_i[mask]

            t_list, x_list = t_i.tolist(), x_i.tolist()
            
            flat_t.extend(t_list)
            flat_x.extend(x_list)
            flat_ids.extend([id_i] * len(t_list))
            # Mark dosing event (example: time==3 after scaling)
            flat_time0.extend([1 if np.round(t * global_max_time, 4) == 3 else 0 for t in t_list])
            censoring.extend([censoring_flag] * len(t_list))
        
        return flat_ids, flat_t, flat_x, flat_time0, censoring

    # --- TRAINING DATA ---
    train_t, train_x, train_ids = zip(*[(s[0], s[1], s[6]) for s in train_dataset_raw])
    ids_train, t_train, x_train, time0_train, censor_train = flatten_samples(train_t, train_x, train_ids)

    # --- TEST DATA (after truncation) ---
    test_t, test_x, test_ids = zip(*[(s[0], s[1], s[6]) for s in test_dataset])
    truncation_time = 0.6
    truncated_t, truncated_x, masks = truncate_time_series(test_t, test_x, truncation_time)
    ids_test, t_test, x_test, time0_test, censor_test = flatten_samples(truncated_t, truncated_x, test_ids)

    # --- DISCARDED (censored) TEST POINTS ---
    disc_t, disc_x, disc_ids, time0_disc, censor_disc = [], [], [], [], []
    for t_i, x_i, mask, id_i in zip(test_t, test_x, masks, test_ids):
        inv_mask = ~mask.bool()
        ids_d, t_d, x_d, time0_d, censor_d = flatten_samples(
            [t_i[inv_mask]], [x_i[inv_mask]], [id_i],
            censoring_flag=1
        )
        disc_ids.extend(ids_d)
        disc_t.extend(t_d)
        disc_x.extend(x_d)
        time0_disc.extend(time0_d)
        censor_disc.extend(censor_d)

    # --- DESTANDARDIZE + SCALE TIME ---
    def process_values(times, values):
        times = np.round(np.array(times) * global_max_time, 4)
        values = np.round(destandardize_concentration(np.array(values), global_mean, global_std), 4)
        return times, values

    time_train, dv_train = process_values(t_train, x_train)
    time_test, dv_test = process_values(t_test, x_test)
    time_disc, dv_disc = process_values(disc_t, disc_x)

    # --- EXPORT ---
    df_export = pd.DataFrame({
        'ID': ids_train + ids_test + disc_ids,
        'Time': np.concatenate([time_train, time_test, time_disc]),
        'DV': np.concatenate([dv_train, dv_test, dv_disc]),
        'Amount': time0_train + time0_test + time0_disc,
        'Censoring': censor_train + censor_test + censor_disc
    })

    os.makedirs(base_dir, exist_ok=True)
    filename = os.path.join(base_dir, f"monolix_data_{i}.csv")
    df_export.to_csv(filename, index=False)




   
    





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
            subject_id = group['ID'].iloc[0]
            occasion = occ_name   # ✅ keep occasion
    
            # Add both subject_id and occasion
            trajectories.append((t, x_global, dose, dose_times, subject_id, x_dose, occasion))

        return trajectories

  
    def _generate_samples(self, trajectories):
        augmented = []
        for t, x_global, dose, dose_times, subject_id, x_dose_norm, occasion in trajectories:
            subject_id_str = str(subject_id)
            occasion_str = str(occasion)
    
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
                        x_dose_norm[mask],
                        occasion_str
                    ))
    
            # Always include the original
            augmented.append((t, x_global, dose, dose_times, subject_id_str, x_dose_norm, occasion_str))
    
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
                        x_dose_norm[:end],
                        occasion_str
                    ))
                for start in range(1, T - 1):
                    augmented.append((
                        t[start:],
                        x_global[start:],
                        dose,
                        dose_times,
                        subject_id_str + f"_suffix{start}",
                        x_dose_norm[start:],
                        occasion_str
                    ))
        return augmented


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]





def collate_fn(batch):
    # Unpack dataset tuples
    # (t, x_global, dose, dose_times, id, x_dose, occasion)
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
        id_list,         # ✅ only Occasion kept
        t_padded,
        x_global_padded,  # z-score normalized
        mask,
        dose_tensor,
        dose_times_list,
        x_dose_padded     # per-dose normalized
    )



# def collate_fn(batch):
#     # Unpack dataset tuples
#     # (t, x_global, dose, dose_times, id, x_dose, occasion)
#     t_list, x_global_list, dose_list, dose_times_list, _, x_dose_list, subject_id = zip(*batch)

#     # Pad time & both x variants
#     t_padded = pad_sequence(t_list, batch_first=True)
#     x_global_padded = pad_sequence(x_global_list, batch_first=True)
#     x_dose_padded = pad_sequence(x_dose_list, batch_first=True)

#     # Build mask
#     max_len = t_padded.size(1)
#     mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
#     for i, t in enumerate(t_list):
#         mask[i, :len(t)] = 1

#     # Dose tensor
#     dose_tensor = torch.stack(dose_list)

#     return (
#         subject_id,         # ✅ only Occasion kept
#         t_padded,
#         x_global_padded,  # z-score normalized
#         mask,
#         dose_tensor,
#         dose_times_list,
#         x_dose_padded     # per-dose normalized
#     )




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









def train_model(
    global_mean,
    global_std,
    global_max_time,
    global_max_dose,
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
    Refactored training loop using helper functions.
    """

    # --- Initialization ---
    (
        device, latent_dim, dim_parameter_encoder,
        best_val_mse, best_val_LL,
        epochs_no_improve, patience, batch_size,
        t_dense, mse_history
    ) = initialize_training(models, func, dataloader, t_dense, global_max_time)

    print(f"Training initialized | total epochs={n_epochs}, "
          f"warmup_noise={warmup_epochs_noise}, warmup_iiv={warmup_epochs_iiv}, "
          f"smoothing starts={smoothing_start_epoch}")
    print("========================================")

    # --- Main Loop ---
    for epoch in range(n_epochs):
        # Reset epoch accumulators
        (
            first_batch, total_transform, total_loss, total_kl,
            total_recon, total_mse, z_individual_list,
            logvar_q_list, mu_q_list, trajectory_records
        ) = initialize_epoch_metrics()

        # Enable/disable noise params
        for param in noise.parameters():
            param.requires_grad = epoch >= warmup_epochs_noise

        start_time = time.time()

        # --- Iterate over mini-batches ---
        for batch in dataloader:
            (
                occ_list, t_padded, x_padded, mask,
                t_encoder, x_encoder, dose_tensor_expanded,
                dose_times_expanded, dose_times_mask
            ) = preprocess_batch(batch, device, max_points_visible=max_points_visible)

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

            # --- Forward + Loss ---
            loss, recon_loss_noise, mse, KL_loss, log_det_sum, log_det_penalty, pred_interp, kl_gauss = compute_loss_and_metrics(
                t_padded, latent_dim, t_dense, x_padded, mask, x0, ode_func,
                reducer, noise, mu_q, logvar_q, log_det,
                epoch, warmup_epochs_iiv, free_bits,
                enable_ae_training, enable_nf_training,
                global_mean, global_std, max_points_visible
            )

            # --- Backprop ---
            total_transform, total_loss, total_mse, total_kl, total_recon = accumulate_epoch_metrics(
                total_transform, total_loss, total_mse, total_kl, total_recon,
                log_det_sum, loss, mse, kl_gauss, recon_loss_noise,
                enable_ae_training
            )

            run_backprop_step(loss, main_params, optimizer)

            # Store trajectories for plotting
            for i in range(x_padded.size(0)):
                trajectory_records.append((
                    occ_list[i], t_padded[i, mask[i]], x_padded[i, mask[i]],
                    dose_tensor_expanded[i], dose_times_expanded[i], z_refined[i].detach()
                ))

        # --- EMA update ---
        if epoch > smoothing_start_epoch:
            func.update_ema(alpha=0.1)
            reducer.update_ema(alpha=0.1)
            initial_encoder.update_ema(alpha=0.1)
            encoder.update_ema(alpha=0.1)

        # --- LR schedule ---
        scheduler.step(total_mse)
        end_time = time.time()

        # --- Logging ---
        log_training_epoch(
            end_time - start_time, epoch, total_mse, total_loss, total_recon,
            total_kl, total_transform, optimizer, noise, dataset, batch_size,
            print_epoch=print_epoch,
            enable_ae_training=enable_ae_training,
            enable_nf_training=enable_nf_training
        )

        # --- Optional Plotting ---
        if plot_from_training_records_enable and epoch % plot_epoch == 0:
            plot_from_training_records(
                batch_size, device, df=df, dataset=dataset, latent_dim=latent_dim,
                records=trajectory_records, func=func, reducer=reducer,
                initial_encoder=initial_encoder, ODEWrapper=ODEWrapper,
                t_dense=t_dense, max_plots=max_plots, nr_row=nr_row, nr_col=nr_col
            )

        # --- Validation & Early Stop ---
        if traing_against_validation:
            val_mse = evaluate_on_val(
                global_mean, global_std, global_max_time, global_max_dose,
                enable_nf_training, enable_ae_training, max_points_visible,
                func, noise, reducer, dataset_val, dataloader_val,
                10, device, encoder, initial_encoder, t_dense,
            )
            if epoch % 10 == 0:
                print(f"[Validation] Epoch {epoch}: MSE={val_mse:.4f}")

            best_val_mse, best_val_LL, epochs_no_improve, stop_training, best_model_state = validate_and_update_early_stop(
                epoch, warmup_epochs_noise, traing_against_validation,
                val_mse, None, best_val_mse, best_val_LL,
                epochs_no_improve, patience, models
            )

            if stop_training:
                break

        # --- Cleanup ---
        torch.cuda.empty_cache()
        gc.collect()
