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
import matplotlib.gridspec as gridspec
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr, norm

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import torch
from torch.utils.data import DataLoader
from torchdiffeq import odeint


from lib.utils.my_utils import (
    collate_fn, collate_fn,  ODEWrapper, torch_linear_interpolate2, batch_linear_interpolate_1d,
    destandardize_concentration, pad_dose_times, truncate_time_series, pad_sequence
)


def plot_node_latent_vs_reduced(
    dataset, idx, latent_dim, dim_parameters, 
    initial_encoder, encoder, func, reducer, noise, ODEWrapper,
    t_dense, global_max_time, global_max_dose, 
    global_mean, global_std, onlymedian=False,
    add_noise_to_prediction: bool = False,
    enable_ae_training=False, enable_nf_training=False
):
    """
    Plot NODE latent states vs reduced states for one individual using encoder.
    """


    device = next(func.parameters()).device
    t_dense = t_dense.to(device)

    # ---- Pick individual ---- #
    entry = dataset[idx]
    t_real, x_true, dose, dose_times, subject_id, x_norm = entry
    t_real, x_true, x_norm = t_real.to(device), x_true.to(device), x_norm.to(device)
    dose = dose * global_max_dose / global_max_dose
    dose_times = dose_times.to(device)

    with torch.no_grad():
        # ---- Set models to eval ---- #
        encoder.eval()
        initial_encoder.eval()
        reducer.eval()
        func.eval()

        conc_mean, conc_std = global_mean, global_std

        # ---- Encode initial condition ---- #
        x0_encoded = initial_encoder(x_true[0].unsqueeze(0))
        if x0_encoded.dim() == 1:
            x0_encoded = x0_encoded.unsqueeze(0)

        # ---- Latent sampling using encoder ---- #
        t_real_padded = t_real.unsqueeze(0)  # batch dim
        x_norm_padded = x_norm.unsqueeze(0)

        if enable_nf_training:
            _, z_tensor, _, _, _ = encoder(t_real_padded, x_norm_padded)
        else:
            _, _, mu_q, logvar_q, _ = encoder(t_real_padded, x_norm_padded)
            std_q = torch.exp(0.5 * logvar_q)
            eps = torch.randn_like(std_q)
            z_tensor = mu_q + eps * std_q

        if enable_ae_training:
            z_tensor = mu_q

        # ---- Combine initial state and latent ---- #
        x0 = torch.cat([x0_encoded, z_tensor], dim=1)

        # ---- Dosing ---- #
        doses_tensor = dose.repeat(dose_times.size(0)).unsqueeze(0).to(device)
        dose_mask = (dose_times != 0).unsqueeze(0).to(device)
        ode_func = ODEWrapper(func, dose_times.unsqueeze(0).unsqueeze(-1),
                              doses_tensor.unsqueeze(-1), dose_mask)

        # ---- Solve NODE ---- #
        pred = odeint(ode_func, x0, t_dense, method="rk4")  # [T, 1, latent_dim+z]
        latent_states = pred[:, 0, :latent_dim]
        reduced_states = reducer(latent_states)
        reduced_states = destandardize_concentration(reduced_states, conc_mean, conc_std)

        # ---- Convert to numpy ---- #
        latent_np = latent_states.cpu().numpy()
        reduced_np = reduced_states.cpu().numpy().squeeze()
        time_hours = t_dense.cpu().numpy() * global_max_time

        # ---- Put everything in a DataFrame ---- #
        df = pd.DataFrame({
            "latent1": latent_np[:, 0],
            "latent2": latent_np[:, 1] if latent_np.shape[1] >= 2 else np.zeros_like(latent_np[:, 0]),
            "reduced": reduced_np,
            "time": time_hours
        })

        # ---- Sort by reduced concentration ---- #
        df_sorted = df.sort_values(by="reduced").reset_index(drop=True)
        print(df_sorted)

        # ---- Plotting ---- #
    
        fig3d = plt.figure(figsize=(10, 8))
        ax1 = fig3d.add_subplot(111, projection="3d")
        
        if latent_np.shape[1] >= 2:
            ax1.plot(latent_np[:, 0], latent_np[:, 1], reduced_np, color="blue", label="NODE trajectory")
            sc = ax1.scatter(latent_np[:, 0], latent_np[:, 1], reduced_np, c=time_hours, cmap="viridis", s=20)
            ax1.set_xlabel("Latent dim 1")
            ax1.set_ylabel("Latent dim 2")
            ax1.set_zlabel("Concentration state")
            ax1.set_title("3D: Latent1 vs Latent2 vs Concentration")
            ax1.view_init(elev=30, azim=60)
        else:
            ax1.text(0.5, 0.5, 0.5, "Need >=2 latent dims", transform=ax1.transAxes)
        
        plt.show()
        
        
        # ---- 3x1 grid for contour + 2D latent plots ---- #
        fig, axes = plt.subplots(1,3, figsize=(20, 6))
        plt.subplots_adjust(hspace=0.35)
        
        # ---- Top: contour plot ---- #
        ax2 = axes[0]
        if latent_np.shape[1] >= 2:
            triang = tri.Triangulation(latent_np[:, 0], latent_np[:, 1])
            contour = ax2.tricontourf(triang, reduced_np, cmap="viridis")
            cbar = fig.colorbar(contour, ax=ax2, label="Concentration")
            ax2.set_xlabel("Latent dim 1")
            ax2.set_ylabel("Latent dim 2")
            ax2.set_title("Contour: Latent1 vs Latent2 vs Concentration")
            ax2.set_xlim(latent_np[:, 0].min(), latent_np[:, 0].max())
            ax2.set_ylim(latent_np[:, 1].min(), latent_np[:, 1].max())
        else:
            ax2.text(0.5, 0.5, "Need >=2 latent dims", transform=ax2.transAxes)
        
        # ---- Middle: latent1 vs reduced ---- #
        ax3 = axes[1]
        sort_idx = np.argsort(reduced_np)
        reduced_sorted = reduced_np[sort_idx]
        latent1_sorted = latent_np[sort_idx, 0]
        
        ax3.plot(reduced_sorted, latent1_sorted, marker='o', color="red")
        ax3.set_xlabel("Concentration")
        ax3.set_ylabel("Latent dim 1")
        ax3.set_title("Concentration vs Latent 1")
        ax3.set_xlim(reduced_sorted.min(), reduced_sorted.max())
        ax3.set_ylim(latent1_sorted.min(), latent1_sorted.max())
        ax3.set_xticks(np.linspace(reduced_sorted.min(), reduced_sorted.max(), 5))
        ax3.set_yticks(np.linspace(latent1_sorted.min(), latent1_sorted.max(), 5))
        
        # ---- Bottom: latent2 vs reduced ---- #
        ax4 = axes[2]
        if latent_np.shape[1] >= 2:
            latent2_sorted = latent_np[sort_idx, 1]
            ax4.plot(reduced_sorted, latent2_sorted, marker='o', color="green")
            ax4.set_xlabel("Concentration")
            ax4.set_ylabel("Latent dim 2")
            ax4.set_title("Concentration vs Latent 2")
            ax4.set_xlim(reduced_sorted.min(), reduced_sorted.max())
            ax4.set_ylim(latent2_sorted.min(), latent2_sorted.max())
            ax4.set_xticks(np.linspace(reduced_sorted.min(), reduced_sorted.max(), 5))
            ax4.set_yticks(np.linspace(latent2_sorted.min(), latent2_sorted.max(), 5))
        else:
            ax4.text(0.5, 0.5, "Only 1 latent dim", transform=ax4.transAxes)
        
        plt.show()
        
                









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
    
def generate_plot_data(df, dataset, t_dense, global_max_time, global_max_dose, global_mean, global_std,
                   latent_dim, dim_parameters, initial_encoder, encoder, func, reducer, noise,
                   ODEWrapper, num_simulated_total=500, add_noise_to_prediction=False,
                   enable_ae_training=False, enable_nf_training=False):

              with torch.no_grad():
                encoder.eval()
                initial_encoder.eval()
                reducer.eval()
                func.eval()
                device = next(func.parameters()).device
        
                conc_mean, conc_std = global_mean, global_std
        
                # Collect unique dose times
                
                dose_times_lists = df['Dose times'].apply(ast.literal_eval).tolist()
                all_times = np.concatenate(dose_times_lists)
                unique_times_np = np.unique(all_times)
                dose_times_tensor = torch.from_numpy(unique_times_np).float().to(t_dense.device) / global_max_time
                t_dense_local = torch.unique(torch.cat([t_dense, dose_times_tensor])).to(device)
        
                unique_doses = sorted(set(entry[2].item() for entry in dataset))
                num_doses = len(unique_doses)
                num_simulated_per_dose = max(1, num_simulated_total // num_doses)
        
                plot_data_local = []
        
                for dose_value in unique_doses:
                    dose_filtered_dataset = [entry for entry in dataset if entry[2].item() == dose_value]
                    if len(dose_filtered_dataset) == 0:
                        continue
                    indices = np.random.choice(len(dose_filtered_dataset), num_simulated_per_dose, replace=True)
                    batch_entries = [dose_filtered_dataset[i] for i in indices]
        
                    t_reals, x_trues, doses, dose_times_list, x_dose_norm_list = [], [], [], [], []
                    for entry in batch_entries:
                        t_real, x_true, dose, dose_times, _, x_dose_norm = entry
                        t_reals.append(t_real)
                        x_trues.append(x_true)
                        doses.append(dose)
                        dose_times_list.append(dose_times)
                        x_dose_norm_list.append(x_dose_norm)
        
                    dose_times_padded = torch.nn.utils.rnn.pad_sequence(dose_times_list, batch_first=True).to(device)
                    dose_mask = (dose_times_padded != 0).to(device)
                    doses_tensor = torch.stack(doses).to(device).unsqueeze(1).expand(-1, dose_times_padded.size(1))
        
                    t_real_padded = torch.nn.utils.rnn.pad_sequence(t_reals, batch_first=True).to(device)
                    x_dose_norm_list_padded = torch.nn.utils.rnn.pad_sequence(x_dose_norm_list, batch_first=True).to(device)
        
                    # Latent sampling from encoder
                    if enable_nf_training:
                        _, z_tensor, mu_q, logvar_q, _ = encoder(t_real_padded, x_dose_norm_list_padded)
                    else:
                        _, _, mu_q, logvar_q, _ = encoder(t_real_padded, x_dose_norm_list_padded)
                        std_q = torch.exp(0.5 * logvar_q)
                        eps = torch.randn_like(std_q)
                        z_tensor =  eps
        
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
                 
                    x0 = torch.cat([x0_tensor, z_tensor], dim=1).to(device)
                    ode_func = ODEWrapper(func, dose_times_padded.unsqueeze(-1), doses_tensor.unsqueeze(-1), dose_mask)
                    pred = odeint(ode_func, x0, t_dense_local, method='rk4')
        
                    x_pred = destandardize_concentration(reducer(pred[:, :, :latent_dim]), conc_mean, conc_std)
        
                    if add_noise_to_prediction:
                        mask = x_pred > 0
                        x_pred = torch.where(mask, noise.sample(x_pred, n_samples=1).squeeze(0), x_pred)
                        x_pred = torch.clamp(x_pred, min=0)
        
                    perc10_sim = torch.quantile(x_pred.squeeze(-1), 0.10, dim=1)
                    median_sim = torch.quantile(x_pred.squeeze(-1), 0.50, dim=1)
                    perc90_sim = torch.quantile(x_pred.squeeze(-1), 0.90, dim=1)
        
                    interp_all = [
                        torch_linear_interpolate2(t_real.to(t_dense_local.device),
                                                  x_true.to(t_dense_local.device), t_dense_local)
                        for t_real, x_true in zip(t_reals, x_trues)
                    ]
                    data_matrix = torch.stack(interp_all)
                    perc10_data = torch.quantile(data_matrix, 0.10, dim=0)
                    median_data = torch.quantile(data_matrix, 0.50, dim=0)
                    perc90_data = torch.quantile(data_matrix, 0.90, dim=0)
        
                    plot_data_local.append({
                        "dose_value": dose_value * global_max_dose,
                        "time_hours": t_dense_local.cpu().numpy() * global_max_time,
                        "perc10_sim": perc10_sim.detach().cpu().numpy(),
                        "median_sim": median_sim.detach().cpu().numpy(),
                        "perc90_sim": perc90_sim.detach().cpu().numpy(),
                        "perc10_data": destandardize_concentration(perc10_data, conc_mean, conc_std).detach().cpu().numpy(),
                        "median_data": destandardize_concentration(median_data, conc_mean, conc_std).detach().cpu().numpy(),
                        "perc90_data": destandardize_concentration(perc90_data, conc_mean, conc_std).detach().cpu().numpy()
                    })
        
              return plot_data_local


def vpc_all(global_max_dose, global_max_time, global_mean, global_std,
                 onlymedian, dataset_1, dataset_2, df1,df2,
                 latent_dim, dim_parameters,encoder_ae, initial_encoder_ae, func_ae, reducer_ae, noise_ae,
                 encoder_vae, initial_encoder_vae, func_vae, reducer_vae, noise_vae,
                 ODEWrapper, t_dense, compartment, num_simulated_total=500,
                 add_noise_to_prediction=False):


    # --- Generate data ---
    plot_data_test_1_AE = generate_plot_data(
                    df=df1,
                    dataset=dataset_1,
                    t_dense=t_dense,
                    global_max_time=global_max_time,
                    global_max_dose=global_max_dose,
                    global_mean=global_mean,
                    global_std=global_std,
                    latent_dim=latent_dim,
                    dim_parameters=dim_parameters,
                    initial_encoder=initial_encoder_ae,
                    encoder=encoder_ae,
                    func=func_ae,
                    reducer=reducer_ae,
                    noise=noise_ae,
                    ODEWrapper=ODEWrapper,
                    num_simulated_total=num_simulated_total,
                    add_noise_to_prediction=False,
                    enable_ae_training=True,
                    enable_nf_training=False
                )
                
    plot_data_test_2_AE = generate_plot_data(
                    df=df2,
                    dataset=dataset_2,
                    t_dense=t_dense,
                    global_max_time=global_max_time,
                    global_max_dose=global_max_dose,
                    global_mean=global_mean,
                    global_std=global_std,
                    latent_dim=latent_dim,
                    dim_parameters=dim_parameters,
                    initial_encoder=initial_encoder_ae,
                    encoder=encoder_ae,
                    func=func_ae,
                    reducer=reducer_ae,
                    noise=noise_ae,
                    ODEWrapper=ODEWrapper,
                    num_simulated_total=num_simulated_total,
                    add_noise_to_prediction=False,
                    enable_ae_training=True,
                    enable_nf_training=False
                    )
        
        
    plot_data_test_1_VAE = generate_plot_data(
                    df=df1,
                    dataset=dataset_1,
                    t_dense=t_dense,
                    global_max_time=global_max_time,
                    global_max_dose=global_max_dose,
                    global_mean=global_mean,
                    global_std=global_std,
                    latent_dim=latent_dim,
                    dim_parameters=dim_parameters,
                    initial_encoder=initial_encoder_vae,
                    encoder=encoder_vae,
                    func=func_vae,
                    reducer=reducer_vae,
                    noise=noise_vae,
                    ODEWrapper=ODEWrapper,
                    num_simulated_total=num_simulated_total,
                    add_noise_to_prediction=False,
                    enable_ae_training=False,
                    enable_nf_training=False
                    )
        
    plot_data_test_2_VAE = generate_plot_data(
                    df=df2,
                    dataset=dataset_2,
                    t_dense=t_dense,
                    global_max_time=global_max_time,
                    global_max_dose=global_max_dose,
                    global_mean=global_mean,
                    global_std=global_std,
                    latent_dim=latent_dim,
                    dim_parameters=dim_parameters,
                    initial_encoder=initial_encoder_vae,
                    encoder=encoder_vae,
                    func=func_vae,
                    reducer=reducer_vae,
                    noise=noise_vae,
                    ODEWrapper=ODEWrapper,
                    num_simulated_total=num_simulated_total,
                    add_noise_to_prediction=False,
                    enable_ae_training=False,
                    enable_nf_training=False
                    )



    # --- Plot 2x2 grid ---
    fig = plt.figure(figsize=(18, 12))
    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2, width_ratios=[1, 1.5])

    panel_labels = ['a', 'b', 'c', 'd']
    plots_per_panel = [2, 3, 2, 3]  # a,c=val; b,d=test
    panel_to_data = [plot_data_test_1_AE, plot_data_test_2_AE, plot_data_test_1_VAE, plot_data_test_2_VAE]

    for i in range(4):
        n_plots = plots_per_panel[i]
        data_set = panel_to_data[i]
    
        # --- Only keep doses 50, 150, 250 for c and d ---
        if i in [1, 3]:  # panels b and d
            filtered_doses = [50, 150, 250]
            data_set = [d for d in data_set if d["dose_value"] in filtered_doses]

        n_plots = len(data_set)
    
        if n_plots == 0:
            print(f"⚠️ Panel {panel_labels[i]} has no data to plot. Skipping.")
            continue
    
        if n_plots <= 2:
            inner = gridspec.GridSpecFromSubplotSpec(1, n_plots, subplot_spec=outer[i], wspace=0.3, hspace=0.3)
        else:
            inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[i], wspace=0.3, hspace=0.3)
      
        ax_panel = plt.Subplot(fig, outer[i])
        ax_panel.axis('off')
        ax_panel.text(-0.1, 1.05, panel_labels[i], transform=ax_panel.transAxes,
                      fontsize=25, fontweight='bold', va='top', ha='left')
        fig.add_subplot(ax_panel)
      
        for j in range(n_plots):
            data = data_set[j]
            ax = plt.Subplot(fig, inner[j])
            
            # --- Plot simulated and observed data ---
            ax.plot(data["time_hours"], data["median_sim"], color="blue", marker="o", markersize=3, label="Sim median")
            ax.plot(data["time_hours"], data["perc10_sim"], color="blue", linestyle="--", label="Sim 10th")
            ax.plot(data["time_hours"], data["perc90_sim"], color="blue", linestyle="--", label="Sim 90th")
      
            ax.plot(data["time_hours"], data["median_data"], color="orange", label="Obs median")
            ax.plot(data["time_hours"], data["perc10_data"], color="orange", linestyle="--", label="Obs 10th")
            ax.plot(data["time_hours"], data["perc90_data"], color="orange", linestyle="--", label="Obs 90th")
            
            ax.set_title(f"Dose {data['dose_value']:.0f}", fontsize=20)
            ax.set_xlim(0, 24)  # 🔹 Set x-axis limit
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.grid(True)
            fig.add_subplot(ax)



    # Shared labels
    fig.text(0.5, 0.05, 'Time (hours)', ha='center', fontsize=20)
    fig.text(0.05, 0.5, f'Concentration', va='center', rotation='vertical', fontsize=20)

    plt.tight_layout(pad=1.0)
    plt.savefig("vpc_plot.png", dpi=300)  # dpi=300 gives high resolution

    # Optional: save as PDF
    plt.savefig("vpc_plot.pdf")
    
    # Show the plot
    plt.show()






        
def vpc(global_max_dose, global_max_time,  global_mean, 
  global_std,onlymedian,
        df_training, df, dataset, latent_dim,
      dim_parameters, initial_encoder,encoder, func, reducer, noise, ODEWrapper,
    t_dense,compartment,enable_nf_training, enable_ae_training,  num_simulated_total=500,
    add_noise_to_prediction: bool = False  # 🔧 NEW ARGUMENT
):
        plot_data = generate_plot_data(
          df=df,
          dataset=dataset,
          t_dense=t_dense,
          global_max_time=global_max_time,
          global_max_dose=global_max_dose,
          global_mean=global_mean,
          global_std=global_std,
          latent_dim=latent_dim,
          dim_parameters=dim_parameters,
          initial_encoder=initial_encoder,
          encoder=encoder,
          func=func,
          reducer=reducer,
          noise=noise,
          ODEWrapper=ODEWrapper,
          num_simulated_total=num_simulated_total,
          add_noise_to_prediction=add_noise_to_prediction,
          enable_ae_training=enable_ae_training,
          enable_nf_training=enable_nf_training
      )
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
        
            ax.set_xlim(0, global_max_time)
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



def rf_predict_params_from_encoder_validation(
        df_train, dataset_train, df_val, dataset_val,
        encoder, latent_dim, dim_parameter_encoder,
        device=None, n_estimators=200, random_state=42):
  

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
    # Plot predictions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
   
    axes[0].scatter(y_ka_val, y_ka_pred, c='blue', alpha=0.7, edgecolors='k')
    axes[0].plot([y_ka_val.min(), y_ka_val.max()],
                 [y_ka_val.min(), y_ka_val.max()], 'r--')
    axes[0].set_xlabel("True ka (validation)")
    axes[0].set_ylabel("Predicted ka")
    axes[0].set_title(f"ka prediction (R²={r2_ka:.2f})")
   
    axes[1].scatter(y_cl_val, y_cl_pred, c='green', alpha=0.7, edgecolors='k')
    axes[1].plot([y_cl_val.min(), y_cl_val.max()],
                 [y_cl_val.min(), y_cl_val.max()], 'r--')
    axes[1].set_xlabel("True cl (validation)")
    axes[1].set_ylabel("Predicted cl")
    axes[1].set_title(f"cl prediction (R²={r2_cl:.2f})")
    
    plt.tight_layout()
    plt.show()
   
    # ----------------------------------------------------------
    # Plot individual latent parameters against ka and cl
    # ----------------------------------------------------------
    n_params = X_val.shape[1]

    
    # --- Create outer 2x2 grid ---
    fig = plt.figure(figsize=(18, 12))
    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    
    panel_labels = ['a', 'b', 'c', 'd']
    
    # (a) mu[0] vs ka/cl
    inner_a = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
    ax_panel_a = plt.Subplot(fig, outer[0]); ax_panel_a.axis('off')
    ax_panel_a.text(-0.1, 1.05, panel_labels[0], transform=ax_panel_a.transAxes,
                    fontsize=20, fontweight='bold', va='top', ha='left')
    fig.add_subplot(ax_panel_a)
    
    ax_a = plt.Subplot(fig, inner_a[0])
    ax_a.scatter(X_val[:, 0], y_ka_val, c='blue', alpha=0.7, edgecolors='k', label="ka")
    ax_a.scatter(X_val[:, 0], y_cl_val, c='green', alpha=0.7, edgecolors='k', label="cl")
    ax_a.set_xlabel("mu[0]", fontsize=20)
    ax_a.set_ylabel("True value", fontsize=20)
    ax_a.legend(fontsize=16)
    fig.add_subplot(ax_a)
    
    # (b) mu[1] vs ka/cl
    inner_b = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1])
    ax_panel_b = plt.Subplot(fig, outer[1]); ax_panel_b.axis('off')
    ax_panel_b.text(-0.1, 1.05, panel_labels[1], transform=ax_panel_b.transAxes,
                    fontsize=20, fontweight='bold', va='top', ha='left')
    fig.add_subplot(ax_panel_b)
    
    ax_b = plt.Subplot(fig, inner_b[0])
    ax_b.scatter(X_val[:, 1], y_ka_val, c='blue', alpha=0.7, edgecolors='k', label="ka")
    ax_b.scatter(X_val[:, 1], y_cl_val, c='green', alpha=0.7, edgecolors='k', label="cl")
    ax_b.set_xlabel("mu[1]", fontsize=20)
    ax_b.set_ylabel("True value", fontsize=20)
    ax_b.legend(fontsize=16)
    fig.add_subplot(ax_b)
    
    # (c) RF ka prediction
    inner_c = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[2])
    ax_panel_c = plt.Subplot(fig, outer[2]); ax_panel_c.axis('off')
    ax_panel_c.text(-0.1, 1.05, panel_labels[2], transform=ax_panel_c.transAxes,
                    fontsize=20, fontweight='bold', va='top', ha='left')
    fig.add_subplot(ax_panel_c)
    
    ax_c = plt.Subplot(fig, inner_c[0])
    ax_c.scatter(y_ka_val, y_ka_pred, c='blue', alpha=0.7, edgecolors='k')
    ax_c.plot([y_ka_val.min(), y_ka_val.max()],
              [y_ka_val.min(), y_ka_val.max()], 'r--')
    ax_c.set_xlabel("True ka", fontsize=20)
    ax_c.set_ylabel("Predicted ka", fontsize=20)
    # Add R² inside plot
    ax_c.text(0.05, 0.9, f"R² = {r2_ka:.2f}", transform=ax_c.transAxes,
              fontsize=20, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    fig.add_subplot(ax_c)
    
    # (d) RF cl prediction
    inner_d = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[3])
    ax_panel_d = plt.Subplot(fig, outer[3]); ax_panel_d.axis('off')
    ax_panel_d.text(-0.1, 1.05, panel_labels[3], transform=ax_panel_d.transAxes,
                    fontsize=20, fontweight='bold', va='top', ha='left')
    fig.add_subplot(ax_panel_d)
    
    ax_d = plt.Subplot(fig, inner_d[0])
    ax_d.scatter(y_cl_val, y_cl_pred, c='green', alpha=0.7, edgecolors='k')
    ax_d.plot([y_cl_val.min(), y_cl_val.max()],
              [y_cl_val.min(), y_cl_val.max()], 'r--')
    ax_d.set_xlabel("True cl", fontsize=20)
    ax_d.set_ylabel("Predicted cl", fontsize=20)
    # Add R² inside plot
    ax_d.text(0.05, 0.9, f"R² = {r2_cl:.2f}", transform=ax_d.transAxes,
              fontsize=20, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax_d.tick_params(axis='both', which='major', labelsize=16)
    ax_c.tick_params(axis='both', which='major', labelsize=16)
    ax_a.tick_params(axis='both', which='major', labelsize=16)
    ax_b.tick_params(axis='both', which='major', labelsize=16)


    fig.add_subplot(ax_d)
    
    
    
    plt.tight_layout()
    plt.show()


