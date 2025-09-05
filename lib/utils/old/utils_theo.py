# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 22:01:22 2025

@author: Baaz
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 16:27:00 2025

@author: Baaz
"""

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
#from torchdiffeq import odeint_adjoint as odeint

#from lib.utils.model_validation import plot_individual_fits

def plot_trajectories_by_sex_DV(dataset):
    """
    Plots all trajectories of raw DV vs TIME in a dataset, split by SEX.
    Uses 'rows' to get the full time series.
    """
    # Separate trajectories by sex
    sex_groups = {}
    for traj in dataset:
        sex = traj['rows'][0]['SEX']  # Assuming all rows have the same SEX
        if sex not in sex_groups:
            sex_groups[sex] = []
        sex_groups[sex].append(traj)

    # Plot each sex in separate figures
    for sex, trajectories in sex_groups.items():
        plt.figure(figsize=(8, 5))
        for traj in trajectories:
            # Extract TIME and DV
            t = [row['TIME'] for row in traj['rows']]
            dv = [pd.to_numeric(row['DV'], errors='coerce') for row in traj['rows']]

            # Remove NaNs
            t = np.array(t)
            dv = np.array(dv)
            mask = ~np.isnan(dv)
            t = t[mask]
            dv = dv[mask]

            # Sort by time
            sorted_idx = np.argsort(t)
            t_sorted = t[sorted_idx]
            dv_sorted = dv[sorted_idx]

            plt.plot(t_sorted, dv_sorted, alpha=0.5)

        plt.xlabel("Time")
        plt.ylabel("DV")
        plt.yscale('log')
        plt.title(f"Trajectories for SEX={sex}")
        plt.show()




















    







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




















        

 
import matplotlib.pyplot as plt
import torch






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
            occ_list, t_padded, x_padded, mask, t_encoder, x_encoder, dose_tensor = preprocess_batch(
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
        # if plot_from_training_records_enable and epoch % plot_epoch == 0:
        #     plot_from_training_records(
        #         batch_size, device, df=df, dataset=dataset, latent_dim=latent_dim,
        #         records=trajectory_records, func=func, reducer=reducer,
        #         initial_encoder=initial_encoder, ODEWrapper=ODEWrapper,
        #         t_dense=t_dense, max_plots=max_plots, nr_row=nr_row, nr_col=nr_col
        #     )

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
