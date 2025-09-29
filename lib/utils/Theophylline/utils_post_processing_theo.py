# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 17:13:41 2025

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

from lib.utils.Theophylline.utils_preprocess_theo import standardize_concentration, destandardize_concentration, collate_fn
from lib.utils.Theophylline.utils_shared_theo import encode_latent, preprocess_batch, prepare_ode_input, make_predictions, prepare_ode_input_eval
from torch.nn.utils.rnn import pad_sequence

from contextlib import contextmanager
import pandas as pd


def plot_individual_fits(dataset,
    latent_dim, t_dense, models, dataloader, device,
    global_mean, global_std, global_max_time,
    enable_nf, enable_ae,enable_vae, enable_onlymedian,
    truncation, max_plots=25, nr_row=5, nr_col=5,
    num_simulated_total=200,  # instead of n_samples
    ci_lower=0.05, ci_upper=0.95,
    add_noise_to_prediction=False
):
    """
    Plots individual predictions with median and confidence intervals for a trained model.
    Uses simulation-based sampling (like generate_plot_data) for uncertainty estimates.
    """
    encoder = models['encoder']
    initial_encoder = models['initial_encoder']
    func = models['func']
    reducer = models['reducer']
    noise = models.get('noise', None)

    fig, axes = plt.subplots(nr_row, nr_col, figsize=(5*nr_col, 4*nr_row))
    axes = axes.flatten() if nr_row*nr_col > 1 else [axes]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for i, data in enumerate(dataloader):
        if i >= max_plots:
            break

        # Preprocess batch (single subject)
        id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list, masks_list = preprocess_batch(
            data, device, truncation=truncation
        )

        # Repeat the same subject num_simulated_total times
        t_padded_exp = t_padded.expand(num_simulated_total, *t_padded.shape[1:])
        x_padded_exp = x_padded.expand(num_simulated_total, *x_padded.shape[1:])
        dose_tensor_exp = dose_tensor.expand(num_simulated_total, *dose_tensor.shape[1:])

        # Encode latent (resample)
        z_refined, mu_q, logvar_q, log_det = encode_latent(
            encoder,
            t_encoder,
            x_encoder,
            enable_nf=enable_nf,
            enable_ae=enable_ae,
            enable_onlymedian=enable_onlymedian
        )

        # If VAE mode, resample latent codes
        if enable_vae:
            eps = torch.randn(num_simulated_total, z_refined.size(-1), device=z_refined.device)
            z_refined = mu_q + eps * torch.exp(0.5 * logvar_q)
        else:
            eps = torch.randn(num_simulated_total, z_refined.size(-1), device=z_refined.device)
            z_refined = mu_q + 0*eps * torch.exp(0.5 * logvar_q)

        # Prepare ODE input
        x0, ode_func,mu_IC,logvar_IC = prepare_ode_input(
            initial_encoder,
            x_padded_exp,
            z_refined,
            func,
            dose_tensor_exp,
            dose_times_list, # * num_simulated_total,  # repeat dose_times list
            enable_vae
        )
        
        
        if enable_vae:
            eps = torch.randn_like(mu_IC, device=z_refined.device)  # shape [batch_size, latent_dim]
            x0_new = mu_IC + eps * torch.exp(0.5 * logvar_IC)          # shape [batch_size, latent_dim]
            x0[:, :x0_new.size(1)] = x0_new

            
       
        
      #  print(x0)

        # Make predictions
        pred_interp, pred_batch = make_predictions(
            t_padded_exp, t_dense, x0, ode_func,
            reducer, latent_dim, global_mean, global_std
        )
        
        # for k in range(100):
        #     plt.plot(
        #         t_dense.detach().cpu().numpy(),
        #         pred_batch[k].detach().cpu().numpy(),
        #         alpha=0.5
        #     )

        
     

        if add_noise_to_prediction and noise is not None:
            mask = pred_batch > 0
            pred_batch = torch.where(mask, noise.sample(pred_batch, n_samples=1).squeeze(0), pred_batch)
            pred_batch = torch.clamp(pred_batch, min=0)

        # Compute quantiles across simulations
        pred_lower = torch.quantile(pred_batch, ci_lower, dim=0).detach().cpu().numpy()
        pred_median = torch.quantile(pred_batch, 0.5, dim=0).detach().cpu().numpy()
        pred_upper = torch.quantile(pred_batch, ci_upper, dim=0).detach().cpu().numpy()
        t_dense_np = t_dense.detach().cpu().numpy() * global_max_time

        # Observed values
        t_encoder_np = t_encoder.squeeze(0).detach().cpu().numpy() * global_max_time
        x_encoder_np = (x_encoder.squeeze(0).detach().cpu().numpy() * global_std + global_mean)

        t_cut_np = t_cut.squeeze(0).detach().cpu().numpy() * global_max_time
       # print(t_cut_np)
        x_cut_np = (x_cut.squeeze(0).detach().cpu().numpy() * global_std + global_mean)

        # Plot
        ax = axes[i]
        ax.plot(t_encoder_np, x_encoder_np, 'o', color='blue', label='Observed (used)')
        if len(t_cut_np) > 0:
            ax.plot(t_cut_np, x_cut_np, 'o', color='red', label='Removed')
        ax.plot(t_dense_np, pred_median, '-', color='green', label='Predicted median')
        ax.fill_between(t_dense_np, pred_lower, pred_upper, color='green', alpha=0.3, label=f'{int((ci_upper-ci_lower)*100)}% CI')

        ax.set_title(f'Individual {id_list}')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Concentration')
        ax.legend()

    plt.tight_layout()
    plt.show()




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




def generate_plot_data(models, dataloader, dataset, t_dense, global_max_time, global_max_dose, global_mean, global_std,
                   latent_dim, dim_parameters, initial_encoder, encoder, func, reducer, noise,
                    num_simulated_total, add_noise_to_prediction,enable_onlymedian,
                   enable_ae, enable_nf, enable_vae, truncation=1):

              with torch.no_grad():
                   # with ema_eval(models):
                     
                    encoder.eval()
                    initial_encoder.eval()
                    reducer.eval()
                    func.eval()
                   
                    device = next(func.parameters()).device
            
            
                    # Collect unique dose times
                    
           
                    unique_doses = sorted(set(entry['amt'].item() for entry in dataset))
                    
                   # print(unique_doses)
                    num_doses = len(unique_doses)
                    num_simulated_per_dose = max(1, num_simulated_total // num_doses)
            
                    plot_data_local = []
            
                    for dose_value in unique_doses:
                            dose_filtered_dataset = [entry for entry in dataset if entry['amt'].item() == dose_value]
                            
                            
                            
                            if len(dose_filtered_dataset) == 0:
                                continue
                            indices = np.random.choice(len(dose_filtered_dataset), num_simulated_per_dose, replace=True)
                            batch_entries = [dose_filtered_dataset[i] for i in indices]
                            
                            
                            max_len = max([len(entry['t']) for entry in batch_entries])
                            masks = torch.zeros((len(batch_entries), max_len), dtype=torch.bool)
                            for i, entry in enumerate(batch_entries):
                                masks[i, :len(entry['t'])] = 1
                           

                            batch = (
                                [entry['subject_id'] for entry in batch_entries],                                  # id_list
                                pad_sequence([entry['t'] for entry in batch_entries], batch_first=True),           # t_padded
                                pad_sequence([entry['x_global'] for entry in batch_entries], batch_first=True),    # x_global_padded
                                masks,                                                                             # mask
                                torch.stack([entry['amt'] for entry in batch_entries]),                             # dose_tensor
                                [entry['dose_times'] for entry in batch_entries],                                   # dose_times_list
                                                                                            # x_dose_padded
                            )
                        
                    
                            # Preprocess
                            id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list, masks_list= preprocess_batch(
                                batch, device, truncation=truncation
                            )
                            
                        
                            
                          #  print(x_encoder)
                            
                            
                            # Encode latent
                            z_refined, mu_q, logvar_q, log_det = encode_latent(
                                encoder, t_encoder, x_encoder, enable_nf=enable_nf, enable_ae=enable_ae, enable_onlymedian=enable_onlymedian
                            )
                            
                            if enable_vae:
                           
                                z_refined =  torch.randn_like(z_refined)

                    
                      
                          
                            # Prepare ODE input
                            x0, ode_func,_,_ = prepare_ode_input(
                                initial_encoder,
                                x_padded,
                                z_refined,
                                func,
                                dose_tensor,
                                dose_times_list,
                                enable_vae
                            )
                            
                            if enable_vae:
                                
                                x0 =  eps = torch.randn_like(x0)

                    
                            
                         
                            # Make predictions
                            pred_interp, pred_batch = make_predictions(
                                t_padded, t_dense, x0, ode_func, reducer, latent_dim, global_mean, global_std
                            )
                            
                           
                          #  print(pred_batch)
                        
                            if add_noise_to_prediction:
                                mask = pred_batch > 0
                                pred_batch = torch.where(mask, noise.sample(pred_batch, n_samples=1).squeeze(0), pred_batch)
                                pred_batch = torch.clamp(pred_batch, min=0)
                
                            perc10_sim = torch.quantile(pred_batch, 0.10, dim=0)  # shape [121]
                            median_sim = torch.quantile(pred_batch, 0.50, dim=0)
                            perc90_sim = torch.quantile(pred_batch, 0.90, dim=0)
                                        
                                
                    
                         #   print(median_sim)
                    
                            interp_all = [
                            linear_interpolate_with_extrapolation(entry['t'].to(t_dense.device),
                                                      entry['x_global'].to(t_dense.device),
                                                      t_dense)
                            for entry in dose_filtered_dataset
                                            ]
                            data_matrix = torch.stack(interp_all)  # shape: [num_subjects, num_time_points]
    
                            perc10_data = torch.quantile(data_matrix, 0.10, dim=0)
                            median_data = torch.quantile(data_matrix, 0.50, dim=0)
                            perc90_data = torch.quantile(data_matrix, 0.90, dim=0)
    
    
                            plot_data_local.append({
                                "dose_value": dose_value * global_max_dose,
                                "time_hours": t_dense.cpu().numpy() * global_max_time,
                                "perc10_sim": perc10_sim.detach().cpu().numpy(),
                                "median_sim": median_sim.detach().cpu().numpy(),
                                "perc90_sim": perc90_sim.detach().cpu().numpy(),
                                "perc10_data": destandardize_concentration(perc10_data, global_mean, global_std).detach().cpu().numpy(),
                                "median_data": destandardize_concentration(median_data, global_mean, global_std).detach().cpu().numpy(),
                                "perc90_data": destandardize_concentration(perc90_data, global_mean, global_std).detach().cpu().numpy()
                            })
            
                 
                    return plot_data_local
          
            
          
def vpc(models, dataloader, global_max_dose, global_max_time,  global_mean, 
  global_std, dataset, latent_dim,
      dim_parameters, initial_encoder,encoder, func, reducer, noise, 
    t_dense,compartment,onlymedian, enable_nf, enable_ae, enable_vae_training, add_noise_to_prediction,  
    num_simulated_total,truncation
):
        plot_data = generate_plot_data(models=models, dataloader=dataloader,
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
          num_simulated_total=num_simulated_total,
          add_noise_to_prediction=add_noise_to_prediction,
          enable_ae=enable_ae,
          enable_nf=enable_nf,
          enable_vae=enable_vae_training,
          enable_onlymedian=onlymedian,
          truncation=truncation
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
            ax.set_ylim(0, 20)
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
          

def compute_residuals(dataset, latent_dim, global_mean, global_std,global_max_time, func, encoder, reducer, initial_encoder, noise, dataloader, t_dense, device, truncation):
    """
    Compute residuals and generate plots for a trained model.

    Returns:
        residuals_all: concatenated residuals over all batches
        times_all: corresponding time points
        predictions_all: model predictions
        targets_all: ground truth
    """
    encoder.eval()
    initial_encoder.eval()
    reducer.eval()
    func.eval()
    noise.eval()
    
    residuals_list = []
    predictions_list = []
    targets_list = []
    times_list = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch
            occ_list, t_padded, x_global_padded, mask, dose_tensor, dose_times_list = batch
    
            # Preprocess
            id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list, list123 = preprocess_batch(
                batch, device, truncation=truncation
            )
    
            # Encode latent
            z_refined, mu_q, logvar_q, log_det = encode_latent(
                encoder, t_encoder, x_encoder, enable_nf=False, enable_ae=True, enable_onlymedian=False
            )
    
            # Prepare ODE input
            x0, ode_func, _, _ = prepare_ode_input_eval(
                initial_encoder,
                x_padded,
                z_refined,
                func,
                dose_tensor,
                dose_times_list
            )
            
    
            # Make predictions
            pred_interp, pred_batch = make_predictions(
                t_padded, t_dense, x0, ode_func, reducer, latent_dim, global_mean, global_std
            )
            # noise is an instance of TrainableNoise
            sigma_add = noise.sigma_add        # additive component, shape [size]
            sigma_prop = noise.sigma_prop      # proportional component, shape [size] or 0 if disabled
            sigma_total = torch.sqrt(sigma_add**2 + (sigma_prop * pred_interp)**2)

            # Compute residuals
            targets = destandardize_concentration(x_padded, global_mean, global_std)
            residuals = (targets - pred_interp)/sigma_total
         #   print('residuals', residuals)
         #   print('id', id_list)
       #     print('mask',mask )
          #  print()
    
    
            # Flatten and collect all values (no mask)
            residuals_list.append(residuals.view(-1))
            times_list.append(t_padded.view(-1))
            predictions_list.append(pred_interp.view(-1))
            targets_list.append(targets.view(-1))

    # Concatenate all batches
    residuals_all = torch.cat(residuals_list)
   
    times_all = torch.cat(times_list)
    predictions_all = torch.cat(predictions_list)
    targets_all = torch.cat(targets_list)
    
    # --- Plotting ---
    residuals_np = residuals_all.cpu().numpy()
    times_np = times_all.cpu().numpy()
    targets_np = targets_all.cpu().numpy()
    
    # Gaussian additive noise from noise model
    sigma_add = noise.sigma_add.detach().cpu().numpy().item()  # assuming size=1
    x_vals = np.linspace(residuals_np.min(), residuals_np.max(), 500)
    pdf_vals = norm.pdf(x_vals, loc=0.0, scale=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Histogram + PDF
    axes[0].hist(residuals_np, bins=25, density=True, alpha=0.7, label="Residuals")
    axes[0].plot(x_vals, pdf_vals, 'r--', label=f'Gaussian Noise σ={1}')
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Histogram of residuals")
    axes[0].legend()
    
    # Residuals vs time
    axes[1].scatter(global_max_time*times_np, residuals_np, alpha=0.5, s=5)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs Time")
    
    # Residuals vs observed values
    axes[2].scatter(targets_np, residuals_np, alpha=0.5, s=5)
    axes[2].set_xlabel("Observed value")
    axes[2].set_ylabel("Residual")
    axes[2].set_title("Residuals vs Observed value")
    
    # Optional empty subplot
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()

    return residuals_all, times_all, predictions_all, targets_all


def vpc_true(
    models, dataset, t_dense, global_max_time, global_max_dose,
    global_mean, global_std, latent_dim, dim_parameters,
    initial_encoder, encoder, func, reducer, noise, add_noise_to_prediction=False,
    enable_onlymedian=True, enable_ae=False, enable_nf=False,
    enable_vae=False, truncation=1, num_repeats=100
):
    import torch, gc, matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
  
    gc.collect()
  


    
    encoder.eval()
    initial_encoder.eval()
    reducer.eval()
    func.eval()
    device = next(func.parameters()).device

    dataloader = DataLoader(dataset, batch_size=12, shuffle=False, collate_fn=collate_fn)

    for batch in dataloader:
        # --- Preprocess batch ---
        id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list, masks_list = preprocess_batch(
            batch, device, truncation=truncation
        )

        # --- Encode latent ---
        z_refined, mu_q, logvar_q, log_det = encode_latent(
            encoder, t_encoder, x_encoder,
            enable_nf=enable_nf, enable_ae=enable_ae, enable_onlymedian=enable_onlymedian
        )

        if enable_vae:
            z_refined = torch.randn_like(z_refined)

        # --- Prepare ODE input ---
        x0, ode_func, _, _ = prepare_ode_input(
            initial_encoder, x_padded, z_refined, func, dose_tensor, dose_times_list, enable_vae
        )
        if enable_vae:
            x0  = torch.randn_like(x0)

        # --- Simulate multiple studies for CI ---
        all_perc10 = []
        all_perc50 = []
        all_perc90 = []

        for _ in range(num_repeats):
            # Simulate trajectories
            x0  = torch.randn_like(x0)
            _, pred_batch = make_predictions(
                t_padded, t_dense, x0, ode_func, reducer, latent_dim, global_mean, global_std
            )

            if add_noise_to_prediction:
                mask_pred = pred_batch > 0
                pred_batch = torch.where(mask_pred, noise.sample(pred_batch, n_samples=1).squeeze(0), pred_batch)
                pred_batch = torch.clamp(pred_batch, min=0)

            # Compute percentiles for this repeat
            all_perc10.append(torch.quantile(pred_batch, 0.05, dim=0).detach().cpu())
            all_perc50.append(torch.quantile(pred_batch, 0.50, dim=0).detach().cpu())
            all_perc90.append(torch.quantile(pred_batch, 0.95, dim=0).detach().cpu())

        # Stack repeats
        # --- Stack repeats along new dimension ---
        all_perc10 = torch.stack(all_perc10, dim=0)  # shape [num_repeats, time_points]
        all_perc50 = torch.stack(all_perc50, dim=0)
        all_perc90 = torch.stack(all_perc90, dim=0)
        
        # --- Compute median percentiles across repeats ---
        perc10 = torch.median(all_perc10, dim=0).values      # shape [time_points]
        perc50 = torch.median(all_perc50, dim=0).values
        perc90 = torch.median(all_perc90, dim=0).values
        
        # --- Compute 95% CI of percentiles across repeats ---
        perc10_ci_lower = torch.quantile(all_perc10, 0.025, dim=0)
        perc10_ci_upper = torch.quantile(all_perc10, 0.975, dim=0)
        perc50_ci_lower = torch.quantile(all_perc50, 0.025, dim=0)
        perc50_ci_upper = torch.quantile(all_perc50, 0.975, dim=0)
        perc90_ci_lower = torch.quantile(all_perc90, 0.025, dim=0)
        perc90_ci_upper = torch.quantile(all_perc90, 0.975, dim=0)



        # --- Plot ---
        plt.figure(figsize=(10,6))

        t_dense_np = t_dense.detach().cpu().numpy()
        plt.plot(t_dense_np, perc50.detach().cpu().numpy(), color='black', label='Median prediction')
        plt.plot(t_dense_np, perc10.detach().cpu().numpy(), color='black', linestyle='--', label='5th percentile')
        plt.plot(t_dense_np, perc90.detach().cpu().numpy(), color='black', linestyle='--', label='95th percentile')

        plt.fill_between(t_dense_np, perc50_ci_lower.detach().cpu().numpy(), perc50_ci_upper.detach().cpu().numpy(), color='red', alpha=0.3, label='95% CI median')
        plt.fill_between(t_dense_np, perc10_ci_lower.detach().cpu().numpy(), perc90_ci_upper.detach().cpu().numpy(), color='blue', alpha=0.2, label='95% CI outer percentiles')

        # Plot observed data (destandardized)
        for i in range(x_padded.shape[0]):
            t_obs_np = t_padded[i, :mask[i].sum()].detach().cpu().numpy()
            x_obs_np = destandardize_concentration(x_padded[i, :mask[i].sum()], global_mean, global_std).detach().cpu().numpy()
            plt.scatter(t_obs_np, x_obs_np, color='gray', s=10, alpha=0.5)

        plt.xlabel('Time')
        plt.ylabel('Concentration / output')
        plt.title('VPC by batch')
        plt.legend()
        plt.show()

        # Free memory
        del pred_batch, x0, z_refined
        torch.cuda.empty_cache()
        gc.collect()

        


def analyze_model_with_vpc(
    models,
    dataset,
    t_dense,
    global_max_time,
    global_max_dose,
    global_mean,
    global_std,
    latent_dim,
    dim_parameters,
    initial_encoder,
    encoder,
    func,
    reducer,
    noise,
    device,
    add_noise_to_prediction=False,
    enable_onlymedian=True,
    enable_ae=False,
    enable_nf=False,
    enable_vae=False,
    truncation=1,
    num_repeats=100,
):
    """
    Runs residual analysis and VPC, and combines the plots in a 2x2 grid.
    VPC is placed in the top-left corner, and observed data is shown in the VPC.
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from torch.utils.data import DataLoader
    import gc

    # -------------------------
    # Residuals
    # -------------------------
    encoder.eval()
    initial_encoder.eval()
    reducer.eval()
    func.eval()
    noise.eval()

    residuals_list, predictions_list, targets_list, times_list = [], [], [], []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in dataloader:
            id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list, _ = preprocess_batch(
                batch, device, truncation=truncation
            )

            z_refined, mu_q, logvar_q, log_det = encode_latent(
                encoder, t_encoder, x_encoder,
                enable_nf=False, enable_ae=True, enable_onlymedian=False
            )

            x0, ode_func, _, _ = prepare_ode_input_eval(
                initial_encoder, x_padded, z_refined, func, dose_tensor, dose_times_list
            )

            pred_interp, pred_batch = make_predictions(
                t_padded, t_dense, x0, ode_func, reducer,
                latent_dim, global_mean, global_std
            )

            sigma_add = noise.sigma_add
            sigma_prop = noise.sigma_prop
            sigma_total = torch.sqrt(sigma_add**2 + (sigma_prop * pred_interp)**2)

            targets = destandardize_concentration(x_padded, global_mean, global_std)
            residuals = (targets - pred_interp) / sigma_total

            residuals_list.append(residuals.view(-1))
            times_list.append(t_padded.view(-1))
            predictions_list.append(pred_interp.view(-1))
            targets_list.append(targets.view(-1))

    residuals_all = torch.cat(residuals_list)
    times_all = torch.cat(times_list)
    predictions_all = torch.cat(predictions_list)
    targets_all = torch.cat(targets_list)

    residuals_np = residuals_all.cpu().numpy()
    times_np = times_all.cpu().numpy()
    targets_np = targets_all.cpu().numpy()

    x_vals = np.linspace(residuals_np.min(), residuals_np.max(), 500)
    pdf_vals = norm.pdf(x_vals, loc=0.0, scale=1.0)

    # -------------------------
    # VPC
    # -------------------------
    gc.collect()
    dataloader_vpc = DataLoader(dataset, batch_size=12, shuffle=False, collate_fn=collate_fn)

    for batch in dataloader_vpc:
        id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list, masks_list = preprocess_batch(
            batch, device, truncation=truncation
        )

        z_refined, mu_q, logvar_q, log_det = encode_latent(
            encoder, t_encoder, x_encoder,
            enable_nf=enable_nf, enable_ae=enable_ae, enable_onlymedian=enable_onlymedian
        )

        if enable_vae:
            z_refined = torch.randn_like(z_refined)

        x0, ode_func, _, _ = prepare_ode_input(
            initial_encoder, x_padded, z_refined, func,
            dose_tensor, dose_times_list, enable_vae
        )

        if enable_vae:
            x0 = torch.randn_like(x0)

        all_perc10, all_perc50, all_perc90 = [], [], []

        for _ in range(num_repeats):
            x0 = torch.randn_like(x0)
            _, pred_batch = make_predictions(
                t_padded, t_dense, x0, ode_func, reducer,
                latent_dim, global_mean, global_std
            )

            if add_noise_to_prediction:
                mask_pred = pred_batch > 0
                pred_batch = torch.where(
                    mask_pred, noise.sample(pred_batch, n_samples=1).squeeze(0), pred_batch
                )
                pred_batch = torch.clamp(pred_batch, min=0)

            all_perc10.append(torch.quantile(pred_batch, 0.05, dim=0).detach().cpu())
            all_perc50.append(torch.quantile(pred_batch, 0.50, dim=0).detach().cpu())
            all_perc90.append(torch.quantile(pred_batch, 0.95, dim=0).detach().cpu())

        all_perc10 = torch.stack(all_perc10, dim=0)
        all_perc50 = torch.stack(all_perc50, dim=0)
        all_perc90 = torch.stack(all_perc90, dim=0)

        perc10 = torch.median(all_perc10, dim=0).values
        perc50 = torch.median(all_perc50, dim=0).values
        perc90 = torch.median(all_perc90, dim=0).values

        perc10_ci_lower = torch.quantile(all_perc10, 0.025, dim=0)
        perc10_ci_upper = torch.quantile(all_perc10, 0.975, dim=0)
        perc50_ci_lower = torch.quantile(all_perc50, 0.025, dim=0)
        perc50_ci_upper = torch.quantile(all_perc50, 0.975, dim=0)
        perc90_ci_lower = torch.quantile(all_perc90, 0.025, dim=0)
        perc90_ci_upper = torch.quantile(all_perc90, 0.975, dim=0)

        break

    # -------------------------
    # Combined Plotting (2x2 grid, VPC in top-left)
    # -------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # VPC plot (top left, axes[0])
    t_dense_np = t_dense.detach().cpu().numpy()
    axes[0].plot(t_dense_np, perc50.cpu().numpy(), color='black', label='Median prediction')
    axes[0].plot(t_dense_np, perc10.cpu().numpy(), color='black', linestyle='--', label='5th percentile')
    axes[0].plot(t_dense_np, perc90.cpu().numpy(), color='black', linestyle='--', label='95th percentile')
    axes[0].fill_between(t_dense_np, perc50_ci_lower.cpu().numpy(), perc50_ci_upper.cpu().numpy(),
                         color='red', alpha=0.3, label='95% CI median')
    axes[0].fill_between(t_dense_np, perc10_ci_lower.cpu().numpy(), perc90_ci_upper.cpu().numpy(),
                         color='blue', alpha=0.2, label='95% CI outer percentiles')

    # Overlay observed data in VPC
    for i in range(x_padded.shape[0]):
        t_obs_np = t_padded[i, :mask[i].sum()].cpu().numpy()
        x_obs_np = destandardize_concentration(
            x_padded[i, :mask[i].sum()], global_mean, global_std
        ).cpu().numpy()
        axes[0].scatter(t_obs_np, x_obs_np, color='gray', s=10, alpha=0.5)

    axes[0].set_title("Visual Predictive Check")
    axes[0].legend()

    # Histogram of residuals
    axes[1].hist(residuals_np, bins=25, density=True, alpha=0.7, label="Residuals")
    axes[1].plot(x_vals, pdf_vals, 'r--', label='N(0,1)')
    axes[1].set_title("Histogram of residuals")
    axes[1].legend()

    # Residuals vs time
    axes[2].scatter(global_max_time * times_np, residuals_np, alpha=0.5, s=5)
    axes[2].set_title("Residuals vs Time")
    axes[2].set_xlabel("Time")

    # Residuals vs observed
    axes[3].scatter(targets_np, residuals_np, alpha=0.5, s=5)
    axes[3].set_title("Residuals vs Observed")
    axes[3].set_xlabel("Observed value")

    plt.tight_layout()
    plt.show()

    return {
        "residuals": residuals_all,
        "times": times_all,
        "predictions": predictions_all,
        "targets": targets_all,
        "vpc": {
            "perc10": perc10,
            "perc50": perc50,
            "perc90": perc90,
            "ci": {
                "perc10": (perc10_ci_lower, perc10_ci_upper),
                "perc50": (perc50_ci_lower, perc50_ci_upper),
                "perc90": (perc90_ci_lower, perc90_ci_upper),
            }
        }
    }
