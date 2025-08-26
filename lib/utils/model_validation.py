# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 16:24:41 2025

@author: Baaz
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 10:21:33 2025

@author: Baaz
"""

import os
import ast
import math
import random
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

import torch
from torch.utils.data import DataLoader
from torchdiffeq import odeint as odeint

from lib.utils.my_utils import  collate_fn, ODEWrapper, torch_linear_interpolate2, batch_linear_interpolate_1d, destandardize_concentration, pad_dose_times, truncate_time_series, pad_sequence




import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
@torch.no_grad()
def compute_residuals(global_mean, global_std,func, encoder, reducer, initial_encoder, noise, dataloader, t_dense, device):
    """
    Compute residuals and generate plots for a trained model.

    Args:
        func, encoder, reducer, initial_encoder, noise: model components
        dataloader: torch DataLoader for dataset
        conc_mean, conc_std: concentration normalization parameters
        t_dense: tensor of dense time points
        device: torch device
        normal_std: standard deviation for overlaying normal PDF on histogram

    Returns:
        residuals_list: list of residual tensors for each batch
        predictions_list: list of model predictions for each batch
        targets_list: list of ground truth targets for each batch
        Also generates three plots:
            - Histogram of residuals
            - Residuals vs time
            - Residuals vs measured values
    """
    encoder.eval()
    initial_encoder.eval()
    reducer.eval()
    func.eval()
    residuals_list = []
    predictions_list = []
    targets_list = []
    times_list = []

    for id_list, t_padded, x_padded, mask, dose_tensor, dose_times_list, x_normalized in dataloader:
        t_padded, x_padded, mask, x_normalized = (
            t_padded.to(device), x_padded.to(device), mask.to(device), x_normalized.to(device)
        )
        
        # Skip first 3 measurements for each individual
       # t_padded = t_padded[:, 3:]
       # x_padded = x_padded[:, 3:]
      #  x_normalized = x_normalized[:, 3:]
        #mask = mask[:, 3:]
        
        batch_size = t_padded.size(0)
    
        # Encode
        _, _, mu_q, logvar_q, _ = encoder(t_padded, x_normalized, mask=mask)
        std_q = torch.exp(0.5 * logvar_q)
        z_refined = mu_q + std_q * 0 * torch.randn_like(std_q)
    
        # Initial encoding
        x0_1 = initial_encoder(x_padded[:, 0].unsqueeze(1))
        x0 = torch.cat([x0_1, z_refined], dim=1)
    
        # Dose preprocessing
        dose_times_padded, dose_times_mask = pad_dose_times([dt.to(device) for dt in dose_times_list])
        dose_tensor_expanded = torch.zeros_like(dose_times_padded, device=device, dtype=dose_tensor.dtype)
        for i, dt in enumerate(dose_tensor):
            dose_tensor_expanded[i, :dt.numel()] = dt.to(device)
        dose_tensor_expanded = dose_tensor_expanded.unsqueeze(-1)
        dose_times_expanded = dose_times_padded.unsqueeze(-1)
        dose_features = torch.cat([dose_times_expanded, dose_tensor_expanded], dim=-1)
    
        # Solve ODE
        ode_func = ODEWrapper(func, dose_times_expanded, dose_tensor_expanded, dose_times_mask)
        pred = odeint(ode_func, x0, t_dense, method='rk4')
        pred_batch = pred.permute(1, 0, 2)
    
        # Reduce latent and interpolate
        reduced = destandardize_concentration(reducer(pred_batch[:, :, :func.dim_latent]), global_mean, global_std)
        t_dense_exp = t_dense.unsqueeze(0).repeat(batch_size, 1)
        pred_interp = batch_linear_interpolate_1d(reduced, t_dense_exp, t_padded)
    
        # Residuals
        t_padded = t_padded[:, 4:]
        x_padded = x_padded[:, 4:]
        pred_interp= pred_interp[:,4:]
        residual = destandardize_concentration(x_padded, global_mean, global_std) - pred_interp
    
        residuals_list.append(residual.cpu())
        predictions_list.append(pred_interp.cpu())
        targets_list.append(destandardize_concentration(x_padded, global_mean, global_std).cpu())
        times_list.append(t_padded.cpu())


    # Concatenate all batches
   # Concatenate all batches safely
        residuals_all = torch.cat([r.flatten() for r in residuals_list]).cpu()
        targets_all = torch.cat([t.flatten() for t in targets_list]).cpu()
        times_all = torch.cat([tt.flatten() for tt in times_list]).cpu()
        preds_all = torch.cat([p.flatten() for p in predictions_list]).cpu()

        residuals_all=residuals_all.to(device)
        
        
   
    # Histogram with normal PDF overlay
    # Concatenate all batches
    residuals_all = torch.cat([r.flatten() for r in residuals_list]).cpu()
    targets_all = torch.cat([t.flatten() for t in targets_list]).cpu()
    times_all = torch.cat([tt.flatten() for tt in times_list]).cpu()
    preds_all = torch.cat([p.flatten() for p in predictions_list]).cpu()
    
    # Histogram of raw residuals
   # plt.figure(figsize=(6, 4))
  #  plt.hist(residuals_all.numpy(), bins=25, density=True, alpha=0.6, color='g', label='Residuals')
  #  plt.show()
    
    # Standardized residuals using trained noise
    sigma_add = noise.sigma_add.cpu()
    sigma_prop = noise.sigma_prop.cpu()  # will be 0 if proportional noise is off
    sigma_total = torch.sqrt(sigma_add**2 + (sigma_prop * preds_all)**2)
    std_resid = residuals_all / sigma_total
    

    
    
    # Plot histogram
    plt.figure(figsize=(6, 4))
    plt.hist(std_resid.numpy(), bins=25, density=True, alpha=0.6, label="Std Residuals (combined error)")
    x = np.linspace(-4, 4, 200)
    plt.plot(x, norm.pdf(x, 0, 1), 'r--', lw=2, label="N(0,1)")
    plt.xlabel("Standardized Residual")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Standardized Residuals (combined additive + proportional error)")
    plt.show()
    # Residuals vs time
    plt.figure(figsize=(6, 4))
    plt.scatter(times_all.numpy(), residuals_all.numpy(), alpha=0.5, s=10)
    plt.axhline(0, color='r', linestyle='--')
    plt.title("Residuals vs Time")
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.show()

    # Residuals vs measurements
    plt.figure(figsize=(6, 4))
    plt.scatter(targets_all.numpy(), residuals_all.numpy(), alpha=0.5, s=10)
    plt.axhline(0, color='r', linestyle='--')
    plt.title("Residuals vs Measurements")
    plt.xlabel("Measured Value")
    plt.ylabel("Residual")
    plt.show()
    
    residual_std = residuals_all.std(unbiased=True)   # sample std (N-1 in denom)
    residual_std_population = residuals_all.std(unbiased=False)  # population std

    return residual_std
    




 
        
def vpc(global_max_dose,  global_mean, 
  global_std,onlymedian,
        df_training, df, dataset, latent_dim,
      dim_parameters, initial_encoder, func, reducer, noise, ODEWrapper,
    t_dense,compartment, num_simulated_total=500,
    add_noise_to_prediction: bool = False  # 🔧 NEW ARGUMENT
):
   with torch.no_grad():

    initial_encoder.eval()
    reducer.eval()
    func.eval()
    
    plot_data = []

    device = next(func.parameters()).device
    t_dense=t_dense.to(device)
    MAX_TIME = estimate_max_time(df_training)
    MAX_DOSE = estimate_max_dose(df_training)
    MAX_TIME_test = estimate_max_time(df)
    MAX_DOSE_test = estimate_max_dose(df)
   # conc_mean, conc_std = dataset.conc_mean, dataset.conc_std
    conc_mean, conc_std = global_mean, global_std

    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=collate_fn, num_workers=0)
    dose_times_lists = df['Dose times'].apply(ast.literal_eval).tolist()
    all_times = np.concatenate(dose_times_lists)
    
    unique_times_np = np.unique(all_times)
    dose_times_tensor = torch.from_numpy(unique_times_np).float().to(t_dense.device) / MAX_TIME
    #t_dense = torch.unique(torch.cat([t_dense, dose_times_tensor]))

    
    #t_dense = torch.unique(torch.cat([t_dense.to(device), dose_times_tensor]))
    
    unique_doses = sorted(set(entry[2].item() for entry in dataset))
    num_doses = len(unique_doses)
    num_simulated_per_dose = max(1, num_simulated_total // num_doses)

    for dose_value in unique_doses:
        dose_filtered_dataset = [entry for entry in dataset if entry[2].item() == dose_value]
        if len(dose_filtered_dataset) == 0:
            print(f"⚠️ No data found for dose {dose_value}. Skipping plot.")
            continue
        indices = np.random.choice(len(dose_filtered_dataset), num_simulated_per_dose, replace=True)
        batch_entries = [dose_filtered_dataset[i] for i in indices]
        t_dense2 = t_dense
        

        t_reals, x_trues, doses, dose_times_list, z_samples = [], [], [], [], []
   
        for entry in batch_entries:
            t_real, x_true, dose, dose_times = entry[:4]

            t_reals.append(t_real)
            x_trues.append(x_true)
            doses.append(dose)
            dose_times_list.append(dose_times)
            sample_shape = torch.Size([dim_parameters])
            new_sample = torch.randn(sample_shape)
            z_samples.append(new_sample)
        
        doses  = [dose * MAX_DOSE_test / MAX_DOSE for dose in doses]
        dose_times=dose_times.to(device)
       # t_dense2 = torch.unique(torch.cat([t_dense2, dose_times]))
        doses_tensor = torch.stack(doses).to(device)
        dose_times_padded = torch.nn.utils.rnn.pad_sequence(dose_times_list, batch_first=True).to(device)
        dose_mask = (dose_times_padded != 0).to(device)
        z_tensor = torch.stack(z_samples).to(device)
        if onlymedian:
            z_tensor=torch.zeros_like(z_tensor)
            
        x0_list = []
        x_trues=x_trues
        for x in x_trues:
            x=x.to(device)
            out = initial_encoder(x[0].unsqueeze(0))  # expect shape [1, 4]
            if out.dim() == 1:
                out = out.unsqueeze(0)  # convert [4] -> [1, 4]
            x0_list.append(out)
        x0_tensor = torch.cat(x0_list, dim=0)  # now shape [6, 4]
        

                # Expand doses_tensor to match the shape of dose_times_padded
        # dose_times_padded: [B, max_doses], so we want doses_tensor: [B, max_doses]
        if doses_tensor.dim() == 1:
            doses_tensor = doses_tensor.unsqueeze(1)  # [B] -> [B, 1]
        doses_tensor = doses_tensor.expand(-1, dose_times_padded.size(1))  # [B, max_doses]

     
        ode_func = ODEWrapper(func, dose_times_padded.unsqueeze(-1),doses_tensor.unsqueeze(-1), dose_mask)
        x0 = torch.cat([x0_tensor, z_tensor], dim=1)  # shape [batch_size, 6]
        x0=x0.to(device)

        pred = odeint(ode_func, x0, t_dense.to(device), method='rk4')  # [time, batch, latent_dim+1]
        
        x_pred = destandardize_concentration(reducer(pred[:, :, :latent_dim]), conc_mean,conc_std)  # [time, batch, state_dim]

        # ✅ Optionally add noise
        if add_noise_to_prediction:
            # Create mask: only where predicted values are > 0
            mask = x_pred > 0
        
            # Sample noise normally
            noisy_x_pred = noise.sample(x_pred, n_samples=1).squeeze(0)
        
            # Apply noise only where x_pred > 0
            x_pred = torch.where(mask, noisy_x_pred, x_pred)
        
            # Clamp any negative values to 0
            x_pred = torch.clamp(x_pred, min=0)


        # Compute quantiles
        perc10_sim = torch.quantile(x_pred.squeeze(-1), 0.10, dim=1)
        median_sim = torch.quantile(x_pred.squeeze(-1), 0.50, dim=1)
        perc90_sim = torch.quantile(x_pred.squeeze(-1), 0.90, dim=1)

        interp_all = [
            torch_linear_interpolate2(t_real.to(t_dense2.device), x_true.to(t_dense2.device), t_dense2)
            for t_real, x_true in zip(t_reals, x_trues)
        ]

        data_matrix = torch.stack(interp_all)

        perc10_data = torch.quantile(data_matrix, 0.10, dim=0)
        median_data = torch.quantile(data_matrix, 0.50, dim=0)
        perc90_data = torch.quantile(data_matrix, 0.90, dim=0)

        # De-standardize
        perc10_sim_real = perc10_sim
        median_sim_real = median_sim
        perc90_sim_real = perc90_sim

        perc10_data_real = destandardize_concentration(perc10_data, conc_mean, conc_std)
        median_data_real = destandardize_concentration(median_data, conc_mean, conc_std)
        perc90_data_real = destandardize_concentration(perc90_data, conc_mean, conc_std)
        time_hours = t_dense.cpu().numpy() * MAX_TIME_test

        plot_data.append({
        "dose_value": dose_value * global_max_dose,
        "time_hours": time_hours,
        "perc10_sim": perc10_sim_real.detach().cpu().numpy(),
        "median_sim": median_sim_real.detach().cpu().numpy(),
        "perc90_sim": perc90_sim_real.detach().cpu().numpy(),
        "perc10_data": perc10_data_real.detach().cpu().numpy(),
        "median_data": median_data_real.detach().cpu().numpy(),
        "perc90_data": perc90_data_real.detach().cpu().numpy()
    })

        # ---- Plot all in a grid ---- #
        n_plots = len(plot_data)
        n_cols = 3
        n_rows = math.ceil(n_plots / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 8 * n_rows), sharex=True, sharey=True)
        axes = axes.flatten()  # Flatten to 1D for easy iteration
        for i, data in enumerate(plot_data):
            ax = axes[i]
            ax.plot(data["time_hours"], data["perc10_sim"], label="Simulated 10th", color="blue", linestyle="--")
            ax.plot(data["time_hours"], data["median_sim"], label="Simulated median", color="blue", marker="o", markersize=3)
            ax.plot(data["time_hours"], data["perc90_sim"], label="Simulated 90th", color="blue", linestyle="--")
        
            ax.plot(data["time_hours"], data["perc10_data"], label="Raw 10th", color="orange", linestyle="--")
            ax.plot(data["time_hours"], data["median_data"], label="Raw median", color="orange")
            ax.plot(data["time_hours"], data["perc90_data"], label="Raw 90th", color="orange", linestyle="--")
        
            ax.set_xlim(0, MAX_TIME)
            ax.set_ylim(0, 180)
            ax.set_title(f"Dose {data['dose_value']:.0f}")
            ax.set_xlabel("Time (hours)")
            ax.set_ylabel(f"Concentration ({compartment})")
            ax.grid(True)
            if i == 0:
                ax.legend()
        
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
        
def vpc_extended(
    global_max_dose, global_mean, global_std, onlymedian,
    df_training, df, dataset, latent_dim,
    dim_parameters, initial_encoder, func, reducer, noise, ODEWrapper,
    t_dense, compartment, num_simulated_total=500,
    num_vpc_datasets=1000, add_noise_to_prediction: bool = False
):
    device = next(func.parameters()).device
    t_dense = t_dense.to(device)
    MAX_TIME = estimate_max_time(df_training)
    MAX_DOSE = estimate_max_dose(df_training)
    MAX_TIME_test = estimate_max_time(df)
    MAX_DOSE_test = estimate_max_dose(df)
    initial_encoder.eval()
    reducer.eval()
    func.eval()
    conc_mean, conc_std = global_mean, global_std
    unique_doses = sorted(set(entry[2].item() for entry in dataset))
    num_doses = len(unique_doses)
    num_simulated_per_dose = max(1, num_simulated_total // num_doses)

    all_vpc_results = []

    for dose_value in unique_doses:
        dose_filtered_dataset = [entry for entry in dataset if entry[2].item() == dose_value]
        if len(dose_filtered_dataset) == 0:
            print(f"⚠️ No data found for dose {dose_value}. Skipping.")
            continue

        # Collect percentiles across multiple VPC datasets
        perc10_all, median_all, perc90_all = [], [], []

        for vpc_iter in range(num_vpc_datasets):
            indices = np.random.choice(len(dose_filtered_dataset), num_simulated_per_dose, replace=True)
            batch_entries = [dose_filtered_dataset[i] for i in indices]

            t_reals, x_trues, doses, dose_times_list, z_samples = [], [], [], [], []

            for entry in batch_entries:
                t_real, x_true, dose, dose_times = entry[:4]
                t_reals.append(t_real)
                x_trues.append(x_true)
                doses.append(dose)
                dose_times_list.append(dose_times)
                z_samples.append(torch.randn(dim_parameters))

            doses = [dose * MAX_DOSE_test / MAX_DOSE for dose in doses]
            dose_times_padded = torch.nn.utils.rnn.pad_sequence(dose_times_list, batch_first=True).to(device)
            dose_mask = (dose_times_padded != 0).to(device)
            doses_tensor = torch.tensor(doses, device=device).unsqueeze(1).expand(-1, dose_times_padded.size(1))
            z_tensor = torch.stack(z_samples).to(device)
            if onlymedian:
                z_tensor = torch.zeros_like(z_tensor)

            x0_list = []
            for x in x_trues:
                x = x.to(device)
                out = initial_encoder(x[0].unsqueeze(0))
                if out.dim() == 1:
                    out = out.unsqueeze(0)
                x0_list.append(out)
            x0_tensor = torch.cat(x0_list, dim=0)

            ode_func = ODEWrapper(func, dose_times_padded.unsqueeze(-1), doses_tensor.unsqueeze(-1), dose_mask)
            x0 = torch.cat([x0_tensor, z_tensor], dim=1).to(device)

            pred = odeint(ode_func, x0, t_dense, method='rk4')
            x_pred = destandardize_concentration(reducer(pred[:, :, :latent_dim]), conc_mean, conc_std)

            if add_noise_to_prediction:
                mask = x_pred > 0
                noisy_x_pred = noise.sample(x_pred, n_samples=1).squeeze(0)
                x_pred = torch.where(mask, noisy_x_pred, x_pred)
                x_pred = torch.clamp(x_pred, min=0)

            # Percentiles for this VPC dataset
            perc10_sim = torch.quantile(x_pred.squeeze(-1), 0.10, dim=1).cpu().numpy()
            median_sim = torch.quantile(x_pred.squeeze(-1), 0.50, dim=1).cpu().numpy()
            perc90_sim = torch.quantile(x_pred.squeeze(-1), 0.90, dim=1).cpu().numpy()

            perc10_all.append(perc10_sim)
            median_all.append(median_sim)
            perc90_all.append(perc90_sim)

        # Convert to arrays
        perc10_all = np.stack(perc10_all)
        median_all = np.stack(median_all)
        perc90_all = np.stack(perc90_all)

        # Compute percentiles of percentiles
        perc10_vpc = np.percentile(perc10_all, [10, 50, 90], axis=0)
        median_vpc = np.percentile(median_all, [10, 50, 90], axis=0)
        perc90_vpc = np.percentile(perc90_all, [10, 50, 90], axis=0)

        time_hours = t_dense.cpu().numpy() * MAX_TIME_test

        all_vpc_results.append({
            "dose_value": dose_value * global_max_dose,
            "time_hours": time_hours,
            "perc10_vpc": perc10_vpc,
            "median_vpc": median_vpc,
            "perc90_vpc": perc90_vpc
        })

    # ---- Plotting ---- #
    n_plots = len(all_vpc_results)
    n_cols = 3
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 8 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, data in enumerate(all_vpc_results):
        ax = axes[i]

        # Shaded areas
        ax.fill_between(data["time_hours"], data["perc10_vpc"][0], data["perc10_vpc"][2], color='blue', alpha=0.2)
        ax.fill_between(data["time_hours"], data["median_vpc"][0], data["median_vpc"][2], color='green', alpha=0.2)
        ax.fill_between(data["time_hours"], data["perc90_vpc"][0], data["perc90_vpc"][2], color='red', alpha=0.2)

        # Median lines
        ax.plot(data["time_hours"], data["perc10_vpc"][1], color='blue', linestyle='--', label='10th percentile')
        ax.plot(data["time_hours"], data["median_vpc"][1], color='green', marker='o', markersize=3, label='Median')
        ax.plot(data["time_hours"], data["perc90_vpc"][1], color='red', linestyle='--', label='90th percentile')

        ax.set_xlim(0, MAX_TIME)
        ax.set_ylim(0, 180)
        ax.set_title(f"Dose {data['dose_value']:.0f}")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(f"Concentration ({compartment})")
        ax.grid(True)
        if i == 0:
            ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
 

def vpc_with_encoder(global_max_value,global_max_dose, global_mean, global_std,
                     enable_ae_training, enable_nf_training,
                     df_training, df, dataset, latent_dim,
                     dim_parameters, initial_encoder, encoder,
                     func, reducer, noise, ODEWrapper,
                     t_dense, compartment, num_simulated_total=500,
                     add_noise_to_prediction: bool = False):

    with torch.no_grad():
        encoder.eval()
        initial_encoder.eval()
        reducer.eval()
        func.eval()
        device = next(func.parameters()).device

        MAX_TIME = estimate_max_time(df_training)
        MAX_DOSE = estimate_max_dose(df_training)

        conc_mean, conc_std = global_mean, global_std

        dose_times_lists = df['Dose times'].apply(ast.literal_eval).tolist()
        all_times = np.concatenate(dose_times_lists)
        unique_times_np = np.unique(all_times)
        dose_times_tensor = torch.from_numpy(unique_times_np).float().to(t_dense.device) / MAX_TIME
        t_dense = torch.unique(torch.cat([t_dense, dose_times_tensor]))

        unique_doses = sorted(set(entry[2].item() for entry in dataset))
        num_doses = len(unique_doses)
        num_simulated_per_dose = max(1, num_simulated_total // num_doses)

        plot_data = []

        for dose_value in unique_doses:
            dose_filtered_dataset = [entry for entry in dataset if entry[2].item() == dose_value]
            if len(dose_filtered_dataset) == 0:
                print(f"⚠️ No data found for dose {dose_value}. Skipping plot.")
                continue

            indices = np.random.choice(len(dose_filtered_dataset), num_simulated_per_dose, replace=True)
            batch_entries = [dose_filtered_dataset[i] for i in indices]
            t_dense2 = t_dense.to(device)  #torch.linspace(0, 1, steps=120).to(device)

            t_reals, x_trues, doses, dose_times_list, x_dose_norm_list = [], [], [], [], []
            for entry in batch_entries:
                t_real, x_true, dose, dose_times, subject_id, x_dose_norm = entry
                t_reals.append(t_real)
                x_trues.append(x_true)
                doses.append(dose)
                dose_times_list.append(dose_times)
                x_dose_norm_list.append(x_dose_norm)

            dose_times = dose_times.to(device)
            t_dense2 = torch.unique(torch.cat([t_dense2, dose_times]))

            doses_tensor = torch.stack(doses).to(device)
            dose_times_padded = torch.nn.utils.rnn.pad_sequence(dose_times_list, batch_first=True).to(device)
            dose_mask = (dose_times_padded != 0).to(device)

            t_real_padded = torch.nn.utils.rnn.pad_sequence(t_reals, batch_first=True).to(device)
            x_true_padded = torch.nn.utils.rnn.pad_sequence(x_trues, batch_first=True).to(device)
            x_dose_norm_list_padded = torch.nn.utils.rnn.pad_sequence(x_dose_norm_list, batch_first=True).to(device)

            # Latent sampling
            if enable_nf_training:
                _, z_tensor, mu_q, logvar_q, _ = encoder(t_real_padded, x_dose_norm_list_padded)
            else:
                _, _, mu_q, logvar_q, _ = encoder(t_real_padded, x_dose_norm_list_padded)
                std_q = torch.exp(0.5 * logvar_q)
                eps = torch.randn_like(std_q)
                z_tensor = mu_q + eps * std_q

            # Encode initial conditions
            x0_list = []
            for x in x_trues:
                out = initial_encoder(x[0].unsqueeze(0).to(device))
                if out.dim() == 1:
                    out = out.unsqueeze(0)
                x0_list.append(out)
            x0_tensor = torch.cat(x0_list, dim=0)

            if doses_tensor.dim() == 1:
                doses_tensor = doses_tensor.unsqueeze(1)
            doses_tensor = doses_tensor.expand(-1, dose_times_padded.size(1))

            if enable_ae_training:
                z_tensor = mu_q

            ode_func = ODEWrapper(func, dose_times_padded.unsqueeze(-1), doses_tensor.unsqueeze(-1), dose_mask)
            x0 = torch.cat([x0_tensor, z_tensor], dim=1).to(device)

            pred = odeint(ode_func, x0, t_dense.to(device), method='rk4')

            x_pred = destandardize_concentration(reducer(pred[:, :, :latent_dim]), conc_mean, conc_std)

            if add_noise_to_prediction:
                x_pred = noise.sample(x_pred, n_samples=1).squeeze(0)

            perc10_sim = torch.quantile(x_pred.squeeze(-1), 0.10, dim=1)
            median_sim = torch.quantile(x_pred.squeeze(-1), 0.50, dim=1)
            perc90_sim = torch.quantile(x_pred.squeeze(-1), 0.90, dim=1)

            interp_all = [
                torch_linear_interpolate2(t_real.to(t_dense2.device), x_true.to(t_dense2.device), t_dense2)
                for t_real, x_true in zip(t_reals, x_trues)
            ]
            data_matrix = torch.stack(interp_all)

            perc10_data = torch.quantile(data_matrix, 0.10, dim=0)
            median_data = torch.quantile(data_matrix, 0.50, dim=0)
            perc90_data = torch.quantile(data_matrix, 0.90, dim=0)

            perc10_sim_real = perc10_sim
            median_sim_real = median_sim
            perc90_sim_real = perc90_sim
            perc10_data_real = destandardize_concentration(perc10_data, conc_mean, conc_std)
            median_data_real = destandardize_concentration(median_data, conc_mean, conc_std)
            perc90_data_real = destandardize_concentration(perc90_data, conc_mean, conc_std)

            time_hours = t_dense.cpu().numpy() * MAX_TIME

            plot_data.append({
                "dose_value": dose_value * global_max_dose,
                "time_hours": time_hours,
                "perc10_sim": perc10_sim_real.detach().cpu().numpy(),
                "median_sim": median_sim_real.detach().cpu().numpy(),
                "perc90_sim": perc90_sim_real.detach().cpu().numpy(),
                "perc10_data": perc10_data_real.detach().cpu().numpy(),
                "median_data": median_data_real.detach().cpu().numpy(),
                "perc90_data": perc90_data_real.detach().cpu().numpy()
            })

        # ======= Plot all doses in one figure =======
        n_plots = len(plot_data)
        n_cols = 3
        n_rows = math.ceil(n_plots / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 8 * n_rows), sharex=True, sharey=True)
        axes = axes.flatten()
      
        
        for i, data in enumerate(plot_data):
            ax = axes[i]
            ax.plot(data["time_hours"], data["perc10_sim"], label="Simulated 10th", color="blue", linestyle="--")
            ax.plot(data["time_hours"], data["median_sim"], label="Simulated median", color="blue", marker="o", markersize=3)
            ax.plot(data["time_hours"], data["perc90_sim"], label="Simulated 90th", color="blue", linestyle="--")

            ax.plot(data["time_hours"], data["perc10_data"], label="Raw 10th", color="orange", linestyle="--")
            ax.plot(data["time_hours"], data["median_data"], label="Raw median", color="orange")
            ax.plot(data["time_hours"], data["perc90_data"], label="Raw 90th", color="orange", linestyle="--")

            ax.set_xlim(0, MAX_TIME)
            ax.set_ylim(0, global_max_value)
            ax.set_title(f"Dose {data['dose_value']:.0f}")
            ax.set_xlabel("Time (hours)")
            ax.set_ylabel(f"Concentration ({compartment})")
            ax.grid(True)
            if i == 0:
                ax.legend()

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()






def plot_encoder_mu_vs_params(df, dataset, encoder, latent_dim, dim_parameter_encoder, device=None):
    """
    Plots encoder latent means (mu_q) vs ka/cl for the first `dim_parameter_encoder` dimensions,
    with points colored by dose group and R² values displayed.

    Parameters
    ----------
    df : pandas.DataFrame
        Original dataset containing 'ID', 'ka', 'cl', and 'Dose' columns.
    dataset : list
        List of tuples: (t_real, x_true, dose, dose_times, subject_id, x_dose_norm)
    encoder : torch.nn.Module
        Trained encoder model.
    latent_dim : int
        Total latent space dimensionality.
    dim_parameter_encoder : int
        Number of latent dimensions to plot against ka and cl.
    device : torch.device or None
        Device to run encoder on. If None, inferred from encoder.
    """
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import r2_score

    if device is None:
        device = next(encoder.parameters()).device

    mu_list = []
    ka_list = []
    cl_list = []
    dose_list = []

    with torch.no_grad():
        for t_real, x_true, dose, dose_times, subject_id, x_dose_norm in dataset:
            # Prepare data
            t_real_t = t_real.unsqueeze(0).to(device)
            x_dose_norm_t = x_dose_norm.unsqueeze(0).to(device)

            # Run encoder
            _, _, mu_q, _, _ = encoder(t_real_t, x_dose_norm_t)
            mu_list.append(mu_q.squeeze(0).cpu().numpy())

            # Handle subject_id type
            if torch.is_tensor(subject_id):
                subj_id_val = subject_id.item()
            else:
                try:
                    subj_id_val = int(subject_id)
                except ValueError:
                    subj_id_val = subject_id  # keep string if needed

            # Retrieve ka, cl, dose from df
            row = df.loc[df["ID"] == subj_id_val]
            if row.empty:
                raise ValueError(f"Subject ID {subj_id_val} not found in dataframe.")
            ka_list.append(row["ka"].values[0])
            cl_list.append(row["cl"].values[0])
            dose_list.append(row["Dose"].values[0])

    mu_array = np.array(mu_list)
    ka_array = np.array(ka_list)
    cl_array = np.array(cl_list)
    dose_array = np.array(dose_list)

    # Create a colormap for doses
    unique_doses = np.unique(dose_array)
    cmap = plt.get_cmap("tab10")
    dose_to_color = {dose: cmap(i % 10) for i, dose in enumerate(unique_doses)}
    colors = [dose_to_color[d] for d in dose_array]

    # ===== Plot grid =====
    fig_rows = dim_parameter_encoder
    fig, axes = plt.subplots(fig_rows, 2, figsize=(10, 4 * fig_rows))

    if fig_rows == 1:
        axes = np.array([axes])

    for i in range(dim_parameter_encoder):
        # ka vs mu
        r, _ = pearsonr(mu_array[:, i], ka_array)
        r2_ka = r**2
        axes[i, 0].scatter(ka_array, mu_array[:, i], c=colors, alpha=0.7, edgecolors='k')
        axes[i, 0].set_xlabel("ka")
        axes[i, 0].set_ylabel(f"mu_q[{i}]")
        axes[i, 0].set_title(f"ka vs mu_q[{i}] (R²={r2_ka:.2f})")
    
        # cl vs mu
        r, _ = pearsonr(mu_array[:, i], cl_array)
        r2_cl = r**2
        axes[i, 1].scatter(cl_array, mu_array[:, i], c=colors, alpha=0.7, edgecolors='k')
        axes[i, 1].set_xlabel("cl")
        axes[i, 1].set_ylabel(f"mu_q[{i}]")
        axes[i, 1].set_title(f"cl vs mu_q[{i}] (R²={r2_cl:.2f})")

    # Legend for doses
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(d),
                          markerfacecolor=cmap(i % 10), markersize=8)
               for i, d in enumerate(unique_doses)]
    fig.legend(handles, [str(d) for d in unique_doses], title="Dose Group", loc="upper right")

    plt.tight_layout()
    plt.show()

    return mu_array, ka_array, cl_array, dose_array


import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def rf_predict_params_from_encoder_validation(
        df_train, dataset_train, df_val, dataset_val,
        encoder, latent_dim, dim_parameter_encoder,
        device=None, n_estimators=200, random_state=42):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    if device is None:
        device = next(encoder.parameters()).device

    def extract_mu_q(df, dataset):
        mu_list, ka_list, cl_list = [], [], []

        encoder.eval()
        with torch.no_grad():
            for t_real, x_true, dose, dose_times, subject_id, x_dose_norm in dataset:
                t_real_t = t_real.unsqueeze(0).to(device)
                x_dose_norm_t = x_dose_norm.unsqueeze(0).to(device)

                _, _, mu_q, _, _ = encoder(t_real_t, x_dose_norm_t)
                mu_array = mu_q.squeeze(0).cpu().numpy()
                mu_list.append(mu_array[:dim_parameter_encoder])  # use only specified dims

                # Get ka/cl for this subject
                try:
                    subj_id_val = int(subject_id)
                except:
                    subj_id_val = subject_id
                row = df[df["ID"] == subj_id_val]
                ka_list.append(row["ka"].values[0])
                cl_list.append(row["cl"].values[0])

        X = np.stack(mu_list)
        y_ka = np.array(ka_list)
        y_cl = np.array(cl_list)
        return X, y_ka, y_cl

    # Extract features and labels
    X_train, y_ka_train, y_cl_train = extract_mu_q(df_train, dataset_train)
    X_val, y_ka_val, y_cl_val = extract_mu_q(df_val, dataset_val)

    # Train Random Forests
    rf_ka = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_cl = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    rf_ka.fit(X_train, y_ka_train)
    rf_cl.fit(X_train, y_cl_train)

    # Predictions on validation
    y_ka_pred = rf_ka.predict(X_val)
    y_cl_pred = rf_cl.predict(X_val)

    r2_ka = r2_score(y_ka_val, y_ka_pred)
    r2_cl = r2_score(y_cl_val, y_cl_pred)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_ka_val, y_ka_pred, c='blue', alpha=0.7, edgecolors='k')
    axes[0].plot([y_ka_val.min(), y_ka_val.max()], [y_ka_val.min(), y_ka_val.max()], 'r--')
    axes[0].set_xlabel("True ka (validation)")
    axes[0].set_ylabel("Predicted ka")
    axes[0].set_title(f"ka prediction (R²={r2_ka:.2f})")

    axes[1].scatter(y_cl_val, y_cl_pred, c='green', alpha=0.7, edgecolors='k')
    axes[1].plot([y_cl_val.min(), y_cl_val.max()], [y_cl_val.min(), y_cl_val.max()], 'r--')
    axes[1].set_xlabel("True cl (validation)")
    axes[1].set_ylabel("Predicted cl")
    axes[1].set_title(f"cl prediction (R²={r2_cl:.2f})")

    plt.tight_layout()
    plt.show()

    return rf_ka, rf_cl, y_ka_pred, y_cl_pred

def rf_predict_params_from_encoderanddose_validation(
        df_train, dataset_train, df_val, dataset_val,
        encoder, latent_dim, dim_parameter_encoder,
        device=None, n_estimators=200, random_state=42):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    if device is None:
        device = next(encoder.parameters()).device

    def extract_mu_q_with_dose(df, dataset):
        mu_list, dose_list, ka_list, cl_list = [], [], [], []

        encoder.eval()
        with torch.no_grad():
            for t_real, x_true, dose, dose_times, subject_id, x_dose_norm in dataset:
                t_real_t = t_real.unsqueeze(0).to(device)
                x_dose_norm_t = x_dose_norm.unsqueeze(0).to(device)

                _, _, mu_q, _, _ = encoder(t_real_t, x_dose_norm_t)
                mu_array = mu_q.squeeze(0).cpu().numpy()
                mu_list.append(mu_array[:dim_parameter_encoder])  # use only specified dims

                # Include dose as feature (flatten if needed)
                dose_list.append(dose.numpy().flatten())

                # Get ka/cl for this subject
                try:
                    subj_id_val = int(subject_id)
                except:
                    subj_id_val = subject_id
                row = df[df["ID"] == subj_id_val]
                ka_list.append(row["ka"].values[0])
                cl_list.append(row["cl"].values[0])

        # Concatenate latent features and dose
        X = np.hstack([np.stack(mu_list), np.stack(dose_list)])
        y_ka = np.array(ka_list)
        y_cl = np.array(cl_list)
        return X, y_ka, y_cl

    # Extract features and labels
    X_train, y_ka_train, y_cl_train = extract_mu_q_with_dose(df_train, dataset_train)
    X_val, y_ka_val, y_cl_val = extract_mu_q_with_dose(df_val, dataset_val)

    # Train Random Forests
    rf_ka = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_cl = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    rf_ka.fit(X_train, y_ka_train)
    rf_cl.fit(X_train, y_cl_train)

    # Predictions on validation
    y_ka_pred = rf_ka.predict(X_val)
    y_cl_pred = rf_cl.predict(X_val)

    r2_ka = r2_score(y_ka_val, y_ka_pred)
    r2_cl = r2_score(y_cl_val, y_cl_pred)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_ka_val, y_ka_pred, c='blue', alpha=0.7, edgecolors='k')
    axes[0].plot([y_ka_val.min(), y_ka_val.max()], [y_ka_val.min(), y_ka_val.max()], 'r--')
    axes[0].set_xlabel("True ka (validation)")
    axes[0].set_ylabel("Predicted ka")
    axes[0].set_title(f"ka prediction (R²={r2_ka:.2f})")

    axes[1].scatter(y_cl_val, y_cl_pred, c='green', alpha=0.7, edgecolors='k')
    axes[1].plot([y_cl_val.min(), y_cl_val.max()], [y_cl_val.min(), y_cl_val.max()], 'r--')
    axes[1].set_xlabel("True cl (validation)")
    axes[1].set_ylabel("Predicted cl")
    axes[1].set_title(f"cl prediction (R²={r2_cl:.2f})")

    plt.tight_layout()
    plt.show()

    return rf_ka, rf_cl, y_ka_pred, y_cl_pred


