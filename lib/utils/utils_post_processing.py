import os
import math
import gc
import ast
import re
from contextlib import contextmanager
from itertools import combinations

# ===== Third-Party Libraries =====
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

# ===== Local Project Imports =====
from lib.utils.utils_shared import (
    destandardize_concentration,
    use_ema,
    normalize_encoder_input,
    preprocess_batch,
    encode_latent,
    prepare_ode_input,
    make_predictions,
    ODEWrapper
)
from lib.utils.utils_preprocess import make_collate_fn, compute_global_stats

def compute_test_metrics(
     dataset, models,models_med,
    df,
    enable_vae, enable_ae, enable_onlymedian, t_dense,
    truncation=1,normalization=True
):
    """
    Computes mean/median R² and MSE across subjects,
    plus residuals on the cut (removed) values only.

    Returns:
        r2_mean, r2_median, mse_mean, mse_median, residuals_df
    """
    encoder = models['encoder'].eval()
    func = models['func'].eval()
    reducer = models['reducer'].eval()
    
    encoder_med = models_med['encoder'].eval()
    func_med = models_med['func'].eval()
    reducer_med = models_med['reducer'].eval()
    
    global_max_dose, global_max_time, global_mean, global_std, global_max_value, global_min_value=compute_global_stats(df)
    device = next(encoder.parameters()).device


    all_targets, all_predictions, all_ids, all_iterations = [], [], [], []
    per_subject_r2, per_subject_mse = [], []
    collate_fn = make_collate_fn(dataset.global_max_len)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for batch in dataloader:
        # Preprocess batch (single subject)
        id_list,treatment_list, t_padded, x_padded,  t_encoder, x_encoder,t_cut, x_cut, mask,mask_encoder, cov, dose_tensor, dose_times_list, evid = preprocess_batch(
            batch, device, truncation=truncation
        )
       


          
        if normalization:
              x_encoder = normalize_encoder_input( x_encoder, t_encoder, x_padded,cov, dose_tensor, dose_times_list, evid,
               encoder_med, func_med, reducer_med,
               t_dense, global_mean, global_std)
              
         
        k_param,z0,  mu_q, L_q, mu_p, L_p,mask_dropout, _= encode_latent(
              encoder,
              t_padded,
              x_padded,
              cov,
              dose_tensor,
              mask_encoder,
              enable_vae=False,
              enable_ae=True,
              enable_onlymedian=False
              
          )
          
         # print(mu_p)

        ode_func= prepare_ode_input(
              x_padded,
              k_param,
              func,
              cov,
              dose_tensor,
              dose_times_list,
              mask_dropout,
              evid,
              enable_ae,
              enable_onlymedian,
              
          )
        pred_interp, pred_batch = make_predictions(
            t_cut, t_dense, k_param, ode_func, reducer,global_mean,global_std
        )
   
  
        # Denormalize true and predicted cut values
        x_cut_true = (x_cut.squeeze(0).detach().cpu().numpy() * global_std + global_mean)
        x_cut_pred = pred_interp.squeeze(0).detach().cpu().numpy()
        n = len(x_cut_true)  # how many cut points this subject has
        id_val = int(id_list.item())  # actual subject ID
        

        # Collect global arrays
        all_targets.extend(x_cut_true.tolist())
        all_predictions.extend(x_cut_pred.tolist())
        all_ids.extend([id_val] * n)   # repeat ID for each measurement


        r2=r2_score(all_targets,all_predictions)
        mse=mean_squared_error(all_targets,all_predictions)
        
  

    return r2, mse, all_targets, all_predictions, all_ids






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
    
  
    
def export_all_metrics_and_residuals(metrics, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    
  
        # Export metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_file = os.path.join(base_dir, f"metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Saved metrics: {metrics_file}")

    



def estimate_coverage(
    models,
    encoder_med, initial_encoder_med, func_med, reducer_med,
    dataset,
    device,
    t_dense,
    global_mean,
    global_std,
    truncation=0,
    max_individuals=None,
    n_samples=100,
    ci_lower=0.05,
    ci_upper=0.95,
    enable_vae=False,
    enable_ae=False,
    enable_onlymedian=False,
    add_noise=True,
    use_ema_models=False,
    normalization=False
):
    """
    Estimate coverage of predictive intervals for individuals in dataset.
    
    Args:
        models: dict with {'encoder','initial_encoder','func','reducer','noise'}
        dataset: dataset of individuals
        device: torch.device
        t_dense: time grid for predictions
        global_mean/global_std: normalization constants
        truncation: int, cut length
        max_individuals: int, limit how many individuals to process
        n_samples: int, number of latent samples per individual
        ci_lower, ci_upper: float, lower and upper quantiles
        enable_vae/ae/onlymedian: encoder options
        add_noise: bool, whether to sample from noise model
        use_ema_models: bool, whether to use EMA weights

    Returns:
        dict with:
            - coverage_per_individual: list of coverages (NaN if invalid)
            - overall_point_coverage: float
            - mean_individual_coverage: float
            - std_individual_coverage: float
    """
    encoder = models['encoder']
    initial_encoder = models['initial_encoder']
    func = models['func']
    reducer = models['reducer']
    noise = models['noise']
    collate_fn = make_collate_fn(dataset.global_max_len)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    coverage_list = []      # per-individual coverage
    point_inside_list = []  # numerator for overall coverage
    point_total_list = []   # denominator for overall coverage

    encoder.eval()
    initial_encoder.eval()
    func.eval()

    for i, data in enumerate(dataloader):
        if max_individuals is not None and i >= max_individuals:
            break

        # --- Preprocess batch ---
        id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list = preprocess_batch(
            data, device, truncation=truncation
        )

        latent_dim = encoder.latent_dim if hasattr(encoder, 'latent_dim') else 2

        # Encode latent n_samples times
        k_param_samples_list = []
        with use_ema(encoder) if use_ema_models else contextmanager(lambda: (yield))():
            if normalization:
                x_encoder = normalize_encoder_input( x_encoder, t_encoder, x_padded, dose_tensor, dose_times_list,
                 encoder_med, initial_encoder_med, func_med, reducer_med,
                 t_dense, latent_dim, global_mean, global_std)
            for _ in range(n_samples):
                k_param_samples, _, _, _ = encode_latent(
                    encoder, t_encoder, x_encoder,
                    enable_vae=enable_vae,
                    enable_ae=enable_ae,
                    enable_onlymedian=enable_onlymedian
                )
                k_param_samples_list.append(k_param_samples)

        k_param_samples = torch.stack(k_param_samples_list, dim=1)  # [1, n_samples, latent_dim]
        k_param_flat = k_param_samples.view(-1, latent_dim)

        # Repeat inputs for each sample
        x_padded_exp = x_padded.repeat(n_samples, 1)
        t_padded_exp = t_padded.repeat(n_samples, 1)
        dose_tensor_exp = dose_tensor.repeat(n_samples)
        dose_times_list_exp = []
        for dt in dose_times_list:
            dose_times_list_exp.extend([dt] * n_samples)
        
        # Prepare ODE input
        x0_samples_list = []
        with use_ema(initial_encoder) if use_ema_models else contextmanager(lambda: (yield))():
           
            x0, ode_func, _, _ = prepare_ode_input(
                initial_encoder,
                t_padded_exp,          # <-- unexpanded version, same as for encoder loop
                k_param_flat,      # latent parameters already sampled
                func,
                dose_tensor_exp,       # unexpanded dose
                dose_times_list_exp,   # unexpanded times
                enable_ae
            )
            
 

        # Make predictions
        with use_ema(func) if use_ema_models else contextmanager(lambda: (yield))():
            pred_interp, pred_batch = make_predictions(
                    t_padded_exp,
                    t_dense,
                    x0,  # flatten
                    ode_func,
                    reducer,
                    latent_dim=latent_dim,
                    global_mean=global_mean,
                    global_std=global_std
                )

        if add_noise:
            mask_valid = pred_batch > 0
            pred_batch = torch.where(
                mask_valid,
                noise.sample(pred_batch, n_samples=1).squeeze(0),
                pred_batch
            )
            pred_batch = torch.clamp(pred_batch, min=0)

        # Compute predictive intervals
        pred_lower = torch.quantile(pred_batch, ci_lower, dim=0).detach().cpu().numpy()
        pred_upper = torch.quantile(pred_batch, ci_upper, dim=0).detach().cpu().numpy()

        # Use full raw x for coverage
        t_dense_np = t_dense.detach().cpu().numpy()
        t_raw_np = t_padded.squeeze(0).detach().cpu().numpy()  # full trajectory times
        x_raw_np = destandardize_concentration(
            x_padded.squeeze(0).detach().cpu().numpy(), global_mean, global_std
        )
        
        # Compute coverage for this individual
        if len(t_raw_np) > 0:
            lower_interp = np.interp(t_raw_np, t_dense_np, pred_lower)
            upper_interp = np.interp(t_raw_np, t_dense_np, pred_upper)
        
            inside_mask = (x_raw_np >= lower_interp) & (x_raw_np <= upper_interp)
            n_inside = inside_mask.sum()
            n_total = len(x_raw_np)
        
            coverage = n_inside / n_total if n_total > 0 else float("nan")
        
            point_inside_list.append(n_inside)
            point_total_list.append(n_total)
        else:
            coverage = float("nan")


        coverage_list.append(coverage)

    # --- Summary statistics ---
    coverage_array = np.array([c for c in coverage_list if not np.isnan(c)])
    if len(coverage_array) > 0:
        mean_cov = coverage_array.mean()
        std_cov = coverage_array.std()
    else:
        mean_cov, std_cov = float("nan"), float("nan")

    if point_total_list:
        total_inside = np.sum(point_inside_list)
        total_points = np.sum(point_total_list)
        overall_cov = total_inside / total_points
    else:
        overall_cov = float("nan")

    return {
        "mean_individual_coverage": mean_cov,
        "std_individual_coverage": std_cov,
        "overall_point_coverage": overall_cov,
    }









def plot_individual_fits(
    models_eval,
    models_median,
    dataset,
    t_dense,
    df,
    truncation=1,
    max_plots=25,
    n_samples=100,
    ci_lower=0.05,
    ci_upper=0.95,
    nr_row=5,
    nr_col=5,
    enable_vae=False,
    enable_ae=False,
    enable_onlymedian=False,
    add_noise=True,
    use_ema_models=False,
    normalization=True,
    fontsize=14,  # Added a parameter to control font size
    export=False,
    pdf_filename="individual_fits.csv"
):

    # --- Set global font sizes ---
    mpl.rcParams.update({
        'axes.titlesize': fontsize + 2,
        'axes.labelsize': fontsize,
        'xtick.labelsize': fontsize - 2,
        'ytick.labelsize': fontsize - 2,
        'legend.fontsize': fontsize,
    })

    encoder = models_eval['encoder'].eval()

    func = models_eval['func'].eval()
    reducer = models_eval['reducer'].eval()
    noise = models_eval['noise'].eval()
    
    
    encoder_med = models_median['encoder'].eval()
    func_med = models_median['func'].eval()
    reducer_med = models_median['reducer'].eval()
    
    global_max_dose, global_max_time, global_mean, global_std, global_max_value, global_min_value=compute_global_stats(df)
    device = next(encoder.parameters()).device

    
    
    
    
    collate_fn = make_collate_fn(dataset.global_max_len)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    fig, axes = plt.subplots(nr_row, nr_col, figsize=(5*nr_col, 4*nr_row))
    axes = axes.flatten() if nr_row*nr_col > 1 else [axes]

   
    
    
    dim_total = encoder.dim_parameter_dynamic+encoder.dim_latent
    population_preds = []
    coverage_list = []
    point_inside_list = []
    point_total_list = []

    for i, data in enumerate(dataloader):
        if i >= max_plots:
            break

        # --- Preprocess batch ---
        id_list,_, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask,cov, dose_tensor, dose_times_list,evid = preprocess_batch(
            data, device, truncation=truncation
        )
        

        k_param_samples_list = []
     
        if normalization:
            x_encoder_norm = normalize_encoder_input( x_padded, t_padded, x_padded,cov, dose_tensor, mask, dose_times_list,evid,
             encoder_med, func_med, reducer_med,
             t_dense, global_mean, global_std)
        else:
            x_encoder_norm=x_encoder
        
        with use_ema(encoder):# if use_ema_models else contextmanager(lambda: (yield))():  
            for _ in range(n_samples):
                k_param_samples,_, _, _, _,_,_,_ = encode_latent(
                    encoder, t_padded, x_padded,cov,dose_tensor,mask,
                    enable_vae=enable_vae,
                    enable_ae=enable_ae,
                    enable_onlymedian=enable_onlymedian
                )
                k_param_samples_list.append(k_param_samples)
                

        k_param_samples = torch.stack(k_param_samples_list, dim=1)
        k_param_flat = k_param_samples.view(-1, dim_total)
    
        x_padded_exp = x_padded.repeat(n_samples, 1)
        t_padded_exp = t_padded.repeat(n_samples, 1)
        dose_tensor_exp = dose_tensor.unsqueeze(0).repeat(n_samples, *([1]*dose_tensor.dim()))
        
        dose_tensor_exp = dose_tensor.repeat(n_samples, 1)      # [n_samples, num_features]
        evid_exp = evid.repeat(n_samples, 1)                    # [n_samples, num_doses]
       
        
      
        dose_times_list_exp = []
        for dt in dose_times_list:
            dose_times_list_exp.extend([dt]*n_samples)
            
        dose_times_tensor = torch.stack(dose_times_list_exp, dim=0)  # [n_samples, num_doses]
    


        with use_ema(func): # if use_ema_models else contextmanager(lambda: (yield))():

                ode_func= prepare_ode_input(

                    x_padded_exp,
                    k_param_flat,
                    func,
                    cov,
                    dose_tensor_exp,
                    dose_times_list_exp,
                    evid_exp,
                    enable_ae,
                    enable_onlymedian
                
                )
                
        with use_ema(reducer): # if use_ema_models else contextmanager(lambda: (yield))():
            pred_interp, pred_batch = make_predictions(
                t_padded_exp, t_dense, k_param_flat, ode_func, reducer,

                global_mean=global_mean,
                global_std=global_std
            )

        if add_noise:
            mask_valid = pred_batch > 0
            pred_batch = torch.where(mask_valid, noise.sample(pred_batch, n_samples=1).squeeze(0), pred_batch)
            pred_batch = torch.clamp(pred_batch, min=0)

        pred_median = pred_batch.median(dim=0).values.detach().cpu().numpy()
        pred_lower = torch.quantile(pred_batch, ci_lower, dim=0).detach().cpu().numpy()
        pred_upper = torch.quantile(pred_batch, ci_upper, dim=0).detach().cpu().numpy()
        t_dense_np = t_dense.detach().cpu().numpy()

        ax = axes[i]
        t_encoder_np = t_encoder.squeeze(0).detach().cpu().numpy()
        t_cut_np = t_cut.squeeze(0).detach().cpu().numpy()
        x_encoder_np = destandardize_concentration(x_encoder.squeeze(0).detach().cpu().numpy(), global_mean, global_std)
        x_cut_np = destandardize_concentration(x_cut.squeeze(0).detach().cpu().numpy(), global_mean, global_std)

        if len(t_encoder_np) > 0:
            lower_interp = np.interp(t_encoder_np, t_dense_np, pred_lower)
            upper_interp = np.interp(t_encoder_np, t_dense_np, pred_upper)
            inside_mask = (x_encoder_np >= lower_interp) & (x_encoder_np <= upper_interp)
            n_inside = inside_mask.sum()
            n_total = len(x_encoder_np)
            coverage = n_inside / n_total if n_total > 0 else float('nan')
            point_inside_list.append(n_inside)
            point_total_list.append(n_total)
        else:
            coverage = float('nan')

        coverage_list.append(coverage)

        # --- Plotting ---
        ax.plot(t_encoder_np, x_encoder_np, 'o', color='blue', label='Observed')
        if len(t_cut_np) > 0:
            ax.plot(t_cut_np, x_cut_np, 'o', color='red', label='Removed')
        ax.plot(t_dense_np, pred_median, '-', color='green', label='Predicted median')
        ax.fill_between(t_dense_np, pred_lower, pred_upper, color='green', alpha=0.3, label='95% CI')
        ax.set_title(f'Test Individual {i+1}')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Concentration (mg/L)')

        population_preds.append(pred_batch.detach().cpu())

    # if population_preds:
    #     population_preds = torch.cat(population_preds, dim=0)
    #     population_median = population_preds.median(dim=0).values.numpy()
    #     for ax in axes[:i+1]:
    #         ax.plot(t_dense_np, population_median, '--', color='black', label='Population median')

    # --- Common legend ---
    handles, labels = axes[-1].get_legend_handles_labels()
    if len(handles) > 0:  # only create legend if there are plotted lines
        fig.legend(
            handles, 
            labels, 
            frameon=False,
            loc='lower center', 
            ncol=max(1, len(labels)),  # ensure at least 1 column
            bbox_to_anchor=(0.5, 0.05)
        )

    plt.tight_layout(rect=[0, 0.15, 1, 1])  # more space at bottom for legend
    plt.show()
    if export:
        pp = PdfPages(pdf_filename)
        pp.savefig(fig)
        pp.close()
        plt.close(fig)

    coverage_array = np.array([c for c in coverage_list if not np.isnan(c)])
    if len(coverage_array) > 0:
        mean_cov = coverage_array.mean()
        std_cov = coverage_array.std()
        print(f"Individual coverage: {mean_cov:.1%} ± {std_cov:.1%}")
    else:
        print("No valid individual coverage values computed.")

    if point_total_list:
        total_inside = np.sum(point_inside_list)
        total_points = np.sum(point_total_list)
        overall_cov = total_inside / total_points
        print(f"Overall point coverage: {overall_cov:.1%}")
    else:
        print("No valid overall coverage computed.")














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


def plot_VPC_and_residuals(models_eval, models_median, dataloader, t_dense,
                                                   df,
                                                   truncation=1, num_repeats=50, show_confidence_intervals=True, export=False, pdf_filename="vpcres.csv"):
    """
    Compute residuals and generate VPC plot showing:
    - Predicted percentiles with confidence intervals
    - Observed scatter points
    - Observed percentiles
    Returns: residuals_all, predictions_all, targets_all, times_all
    """
    
    encoder = models_eval['encoder'].eval()
    func = models_eval['func'].eval()
    reducer = models_eval['reducer'].eval()
    noise=models_eval['noise'].eval()


    device = next(encoder.parameters()).device
    global_max_dose, global_max_time, global_mean, global_std, global_max_value, global_min_value=compute_global_stats(df)

    # ---- Step 1: Collect residuals and predictions ----
    residuals_list,std_residuals_list, predictions_list, targets_list, times_list = [], [], [], [],[]

    all_times = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            id_list,treatment_list, t_padded, x_padded,  t_encoder, x_encoder,t_cut, x_cut, mask_encoder, cov, dose_tensor, dose_times_list, evid = preprocess_batch(
                batch, device, truncation=truncation
            )
            # Encode latent
            k_param,z0,  mu_q, L_q, mu_p, L_p,mask_dropout, _= encode_latent(
                encoder, t_encoder, x_encoder,   cov,
                   dose_tensor,mask_encoder,
                enable_vae=False, enable_ae=True, enable_onlymedian=False
            )

            # Prepare ODE input and predict
            ode_func= prepare_ode_input(

                x_padded,
                k_param,
                func,
                cov,
                dose_tensor,
                dose_times_list,
                  evid,
                enable_ae=True
                
            )
            pred_interp, pred_batch = make_predictions(t_padded, t_dense, k_param, ode_func, reducer, global_mean, global_std)
            targets = destandardize_concentration(x_padded, global_mean, global_std)
            residuals = targets - pred_interp
      
          
        
            
            
            with torch.no_grad():
                sigma_total = noise._compute_sigma_total(pred_interp)   # SAME SHAPE as pred_interp
            
            # ---- Standardized residuals: (obs - pred) / sigma_total ----
            std_residuals = residuals / sigma_total
            
            
            
       
            
          #  mask_obs = mask_obs.unsqueeze(1).expand_as(residuals)  # shape [batch_size, seq_len]
            
            # Flatten both before appending
            std_residuals_list.append(std_residuals.view(-1))
            residuals_list.append(residuals.view(-1))
            predictions_list.append(pred_interp.view(-1))
            targets_list.append(targets.view(-1))
            times_list.append(t_padded.view(-1))
            
            # For VPC scatter
            all_times.append(t_padded.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())


    residuals_all = torch.cat(residuals_list)
    predictions_all = torch.cat(predictions_list)
    targets_all = torch.cat(targets_list)
    times_all = torch.cat(times_list)

    all_times = np.concatenate(all_times)
    all_targets = np.concatenate(all_targets)
    
    std_residuals_all = torch.cat(std_residuals_list)
 
        

    # ---- Step 2: VPC simulation with repeats (predicted percentiles) ----
    all_repeats_data = []
    for r in range(num_repeats):
        plot_data_r = generate_plot_data(
            models_median=models_median,
            models_eval=models_eval,
            dataloader=dataloader,
            dataset=dataloader.dataset,
            t_dense=t_dense,
            df=df,
            add_noise_to_prediction=True,
            enable_onlymedian=False,
            enable_ae=False,
            enable_vae=True,
            normalization=True,
            truncation=truncation
        )
        all_repeats_data.append(plot_data_r)

    treatments = sorted(set(d["treatment"] for d in all_repeats_data[0]))
    plot_by_treat = {t: [] for t in treatments}
    for t in treatments:
        plot_by_treat[t] = [
            next(d for d in repeat if d["treatment"] == t) for repeat in all_repeats_data
        ]

    # ---- Step 3: Plotting ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    axes = axes.flatten()
    color_pred, color_obs, color_obs_percentiles = "blue", "grey", "green"

    # --- [0,0] VPC with observed scatter + observed percentiles ---
    ax = axes[0]


    # for i in range(num_bins):
    #     mask = bin_indices == i
    #     if np.sum(mask) > 0:
    #         bin_centers.append(bins[i] + (bins[i+1]-bins[i])/2 if i < num_bins-1 else bins[i])
    #         data_in_bin = all_targets[mask]
    #         for perc in obs_percentiles:
    #             obs_percentiles[perc].append(np.percentile(data_in_bin, perc))

    # Scatter all observed data points
    ax.scatter(all_times*global_max_time, all_targets, color=color_obs, s=20, alpha=0.4, label="Observed points")

    # Plot observed percentiles
    # for perc, color in zip([5,25,50,75,95], ["#66c2a5","#fc8d62","#8da0cb","#fc8d62","#66c2a5"]):
    #     ax.plot(bin_centers, obs_percentiles[perc], linestyle='--', color=color, linewidth=1.5, label=f"Observed {perc}th percentile")

  #  Plot predicted percentiles
    # Plot predicted percentiles
    # Plot predicted percentiles
    
    handled_labels = set()  # To avoid duplicate legend entries
    
    for i, (treat, repeat_dicts) in enumerate(plot_by_treat.items()):
        times = repeat_dicts[0]["time_hours"]
        percentile_keys = ["perc10_sim", "median_sim", "perc90_sim"]
        sim_stats = {}
    
        for k in percentile_keys:
            arr = np.stack([d.get(k, np.nan*np.ones_like(times)) for d in repeat_dicts], axis=0)
            sim_stats[k] = {"median": np.nanmedian(arr, axis=0)}
            if show_confidence_intervals:
                sim_stats[k]["lower"] = np.nanpercentile(arr, 2.5, axis=0)
                sim_stats[k]["upper"] = np.nanpercentile(arr, 97.5, axis=0)
    
        for k in percentile_keys:
            med = sim_stats[k]["median"]
            lower = sim_stats[k].get("lower")
            upper = sim_stats[k].get("upper")
    
            # Confidence interval fill
            if lower is not None and upper is not None:
                ci_color = "red" if k == "median_sim" else "blue"
                ci_label = "95 % CI" if k == "median_sim" else "95 % CI"
                if ci_label not in handled_labels:
                    ax.fill_between(times, lower, upper, color=ci_color, alpha=0.2, label=ci_label)
                    handled_labels.add(ci_label)
                else:
                    ax.fill_between(times, lower, upper, color=ci_color, alpha=0.2)
    
            # Line color and style
            if k == "median_sim":
                line_color = "black"
                line_style = "-"
                line_width = 2.5
                line_label = "Median simulation"
            else:
                line_color = "grey"
                line_style = "--"
                line_width = 2
                line_label = "10th/90th simulation"
    
            if line_label not in handled_labels:
                ax.plot(times, med, color=line_color, linestyle=line_style, linewidth=line_width, label=line_label)
                handled_labels.add(line_label)
            else:
                ax.plot(times, med, color=line_color, linestyle=line_style, linewidth=line_width)
    
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Concentration (mg/L)")
    ax.set_title("Visual Predictive Check")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper center', frameon=False, bbox_to_anchor=(0.5, -0.1), ncol=2)

    
    # --- [0,1] Residual histogram ---
    ax_hist = axes[1]
    residuals_np = std_residuals_all.cpu().numpy()
    x_vals = np.linspace(residuals_np.min(), residuals_np.max(), 500)
    pdf_vals = norm.pdf(x_vals, loc=0.0, scale=1)
    ax_hist.hist(residuals_np, bins=25, density=True, alpha=0.7, color='lightblue', label='IWRES')
    ax_hist.plot(x_vals, pdf_vals, 'r--', linewidth=2, label='Standard Gaussian')
    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Residual histogram")
    ax_hist.legend(fontsize=12, loc='upper center', frameon=False, bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    # --- [1,0] Observation vs Prediction ---
    ax_res_time = axes[2]
    ax_res_time.scatter(targets_all.cpu().numpy(), predictions_all.cpu().numpy(),
                        alpha=0.5, s=5, color='black', label="Predictions/Observations")
    lims = [
        np.min([targets_all.cpu().numpy(), predictions_all.cpu().numpy()]),
        np.max([targets_all.cpu().numpy(), predictions_all.cpu().numpy()])
    ]
    ax_res_time.plot(lims, lims, 'r--', linewidth=1.5, label='y = x')
    ax_res_time.set_xlim(lims)
    ax_res_time.set_ylim(lims)
    ax_res_time.set_xlabel("Observation")
    ax_res_time.set_ylabel("Prediction")
    ax_res_time.set_title("Observation vs Prediction")
    ax_res_time.grid(True, alpha=0.3)
    ax_res_time.legend(fontsize=12, loc='upper center', frameon=False, bbox_to_anchor=(0.5, -0.1), ncol=2)

    
    # # --- [1,1] Residuals vs Observed ---
    # ax_res_obs = axes[3]
    # ax_res_obs.scatter(targets_all.cpu().numpy(), std_residuals_all.cpu().numpy(), alpha=0.5, s=5, color='black')
    # ax_res_obs.set_xlabel("Observed Concentration")
    # ax_res_obs.set_ylabel("Residual")
    # ax_res_obs.set_title("Residuals vs Observed")
    # ax_res_obs.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    if export:
        pp = PdfPages(pdf_filename)
        pp.savefig(fig)
        pp.close()
        plt.close(fig)

    # ---- Return the original 4 outputs ----
   # return residuals_all, predictions_all, targets_all, times_all







    # Split dataloader by treatment
def split_dataloader_by_treatment(dataloader, dataset):
    treatment_values = sorted(dataset.df['TREATMENT'].unique())
    loaders_by_treatment = {}
    for t in treatment_values:
        subset_indices = [i for i, traj in enumerate(dataset.samples) if traj['treatment'] == t]
        if not subset_indices:
            continue
        subset = Subset(dataset, subset_indices)
        loaders_by_treatment[t] = DataLoader(
            subset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            collate_fn=getattr(dataloader, 'collate_fn', None),
            num_workers=getattr(dataloader, 'num_workers', 0)
        )
    return loaders_by_treatment

    


def plot_encoder_histograms(
   models_norm,models_eval,
    t_dense, df,
    dataset, dataloader,
 truncation=None, normalization=True
):
    """
    Plots encoder latent outputs:
    - Histograms of μ and σ for each latent dimension, grouped by treatment.
    - Pairwise scatter plots of latent dimensions, colored by treatment.
    - Shows prior and empirical posterior correlations as text on scatter plots.
    """

    encoder = models_eval['encoder'].eval()

    encoder_med = models_norm['encoder'].eval()
    func_med = models_norm['func'].eval()
    reducer_med = models_norm['reducer'].eval()

    global_max_dose, global_max_time, global_mean, global_std, global_max_value, global_min_value=compute_global_stats(df)

    device = next(encoder.parameters()).device

    loaders_by_treatment = split_dataloader_by_treatment(dataloader, dataset)

    mu_list, sigma_list, treatment_list_all = [], [], []


    with torch.no_grad():
        for treatment_value, treatment_loader in loaders_by_treatment.items():
            for batch in treatment_loader:
                # --- Unpack batch ---
                id_tensor, treatment_tensor, t_padded, x_padded, mask, cov,dose_tensor, dose_times_padded, evid = batch

                # Move all tensors to device
                id_tensor = id_tensor.to(device)
                treatment_tensor = treatment_tensor.to(device)
                t_padded = t_padded.to(device)
                x_padded = x_padded.to(device)
                mask = mask.to(device)
                dose_tensor = dose_tensor.to(device)
                dose_times_padded = dose_times_padded.to(device)
                evid = evid.to(device)
                cov=cov.to(device)

                # Preprocess batch
                _, treatment_list, t_encoder, x_encoder, *_ = preprocess_batch(
                    batch, device, truncation=truncation
                )

                # Optional normalization
                if normalization:
                    x_encoder = normalize_encoder_input(
                        x_encoder, t_encoder, x_padded, cov, dose_tensor, mask,dose_times_padded, evid,
                        encoder_med, func_med, reducer_med,
                        t_dense, global_mean, global_std
                    )

                # Encode latent with EMA weights
                with use_ema(encoder):
                    k_param,_, mu_q, L_q, mu_p, L_p,_, repeat_factor = encode_latent(
                        encoder, t_encoder, x_encoder,cov, dose_tensor,mask,
                        enable_vae=True, enable_ae=False, enable_onlymedian=False
                    )
                
                    mu_q=mu_q -mu_p
          
          
                    if L_q.dim() == 2:
                        L_q = torch.diag_embed(L_q)  # convert [B, D] -> [B, D, D]
                    
                    Sigma_q = L_q @ L_q.transpose(-1, -2)
                    sigma_q = torch.sqrt(torch.diagonal(Sigma_q, dim1=-2, dim2=-1))  # [B, D]
                 #   print(mu_q)
                    mu_list.append(mu_q.cpu().numpy())
                    sigma_list.append(sigma_q.cpu().numpy())
                    treatment_list_all.append(treatment_list.cpu().numpy())

    # Aggregate
    mu_array = np.vstack(mu_list)        # [n_samples, D]
    
 #   print(mu_array)
    sigma_array = np.vstack(sigma_list)  # [n_samples, D]
    treatment_array = np.concatenate(treatment_list_all).ravel()
    n_dims = mu_array.shape[1]

    # Empirical posterior correlation
    emp_corr = np.corrcoef(mu_array, rowvar=False)

    # Prior correlation
    with torch.no_grad():
        _,_, _, _, mu_p, L_p, _,_= encode_latent(
            encoder, t_encoder, x_encoder, cov, dose_tensor, mask,
            enable_vae=True, enable_ae=False, enable_onlymedian=False
        )
        L_p = L_p.to(device)  # [B, D, D]
    
        if L_p.dim() == 3:
            Sigma_p = torch.bmm(L_p, L_p.transpose(1, 2))  # [B, D, D]
            if Sigma_p.shape[0] == 1:
                Sigma_p = Sigma_p[0]  # remove batch dim
            else:
                # if multiple batches, you may need to select one or average
                Sigma_p = Sigma_p[0]  # for example, take first batch
        else:
            Sigma_p = L_p @ L_p.T  # [D, D]
    
        Sigma_p = Sigma_p.detach().cpu().numpy()

        prior_corr = np.zeros((n_dims, n_dims))
        for i in range(n_dims):
            for j in range(n_dims):
                prior_corr[i, j] = Sigma_p[i, j] / np.sqrt(Sigma_p[i, i] * Sigma_p[j, j])

    # --- 1. Histograms ---
    fig, axes = plt.subplots(n_dims, 2, figsize=(16, 6*n_dims)) if n_dims > 1 else plt.subplots(1, 2, figsize=(16,6))
    if n_dims == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_dims):
        # μ histogram
        ax_mu = axes[i, 0]
        for t_val in np.unique(treatment_array):
            idx = treatment_array == t_val
            ax_mu.hist(mu_array[idx, i], bins=30, alpha=0.5, label=f"Treatment {t_val}")
        ax_mu.set_title(f"μ dim {i}")
        ax_mu.set_xlabel("μ value")
        ax_mu.set_ylabel("Frequency")
        ax_mu.grid(True, linestyle="--", alpha=0.6)
        ax_mu.legend()

        # σ histogram
        ax_sigma = axes[i, 1]
        for t_val in np.unique(treatment_array):
            idx = treatment_array == t_val
            ax_sigma.hist(sigma_array[idx, i], bins=30, alpha=0.5, label=f"Treatment {t_val}")
        ax_sigma.set_title(f"σ dim {i}")
        ax_sigma.set_xlabel("σ value")
        ax_sigma.set_ylabel("Frequency")
        ax_sigma.grid(True, linestyle="--", alpha=0.6)
        ax_sigma.legend()

    plt.tight_layout()
    plt.show()

    # --- 2. Pairwise scatter plots of μ with correlations ---
    pairs = list(combinations(range(n_dims), 2))
    n_pairs = len(pairs)
    if n_pairs > 0:
        ncols = 2
        nrows = (n_pairs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6*nrows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx, (d1, d2) in enumerate(pairs):
            ax = axes[idx]
            for t_val in np.unique(treatment_array):
                idx_t = treatment_array == t_val
                ax.scatter(mu_array[idx_t, d1], mu_array[idx_t, d2], alpha=0.6, label=f"Treatment {t_val}", s=50)
            ax.set_xlabel(f"μ{d1}")
            ax.set_ylabel(f"μ{d2}")
            ax.set_title(f"μ{d1} vs μ{d2}")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()

            # Add correlation text
            ax.text(0.05, 0.95, f"Emp. corr: {emp_corr[d1,d2]:.2f}\nPrior corr: {prior_corr[d1,d2]:.2f}",
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

        for j in range(idx+1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()










from matplotlib.backends.backend_pdf import PdfPages


def plot_single_model_encoders_and_regression(
    df_train, df_val, dataset_train, dataset_val,
   models_eval, models_median,t_dense,
   truncation=1, use_ema_models=False, normalization=False,   single_figure=False,export=False, pdf_filename="encoder_regression_plots.pdf"
):
    """
    Plot regression and encoder outputs using μ for regression,
    colored by treatment.
    Fully robust to latent_dim = 1 or higher.
    """

    gc.collect()
    torch.cuda.empty_cache()

    encoder = models_eval['encoder'].eval()

    func = models_eval['func'].eval()
    reducer = models_eval['reducer'].eval()
    noise = models_eval['noise'].eval()
    
    
    encoder_med = models_median['encoder'].eval()
    func_med = models_median['func'].eval()
    reducer_med = models_median['reducer'].eval()
    
    global_max_dose, global_max_time, global_mean, global_std, global_max_value, global_min_value=compute_global_stats(df_train)
    device = next(encoder.parameters()).device
    latent_dim=encoder.total_parameters
  
    # ---------------- Extract latents helper ----------------
    def extract_latents(df, dataset, encoder):
        mus, param_values_list, id_list_all, treatment_list_all = [], [], [], []
        collate_fn = make_collate_fn(dataset.global_max_len)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)
        param_names = ast.literal_eval(df["PARAM_NAMES"].iloc[0])
        n_params = len(param_names)

        ctx = use_ema(encoder) if use_ema_models else contextmanager(lambda: (yield))()
        with ctx:
            encoder.eval()
            with torch.no_grad():
                for batch in dataloader:
                    id_tensor, treatment_tensor, t_padded, x_padded, mask,cov, dose_tensor, dose_times_padded, evid_padded = batch
                    id_tensor = id_tensor.to(device)
                    treatment_tensor = treatment_tensor.to(device)
                    t_padded = t_padded.to(device)
                    x_padded = x_padded.to(device)
                    mask = mask.to(device)
                    dose_tensor = dose_tensor.to(device)
                    dose_times_padded = dose_times_padded.to(device)
                    evid_padded = evid_padded.to(device)
                    cov=cov.to(device)

                    # Preprocess
                    id_list, treatment_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, cov,dose_tensor, dose_times_list, evid = preprocess_batch(
                        batch, device, truncation=truncation
                    )

                    # Optional normalization
                    if normalization:
                        
              
                        x_encoder = normalize_encoder_input(
                            x_encoder, t_encoder, x_padded,cov, dose_tensor,mask, dose_times_padded, evid_padded,
                            encoder_med, func_med, reducer_med,
                            t_dense, global_mean, global_std
                        )

                    k_param,z0,  mu_q, logvar_q, mu_p, logvar_p,mask2,  repeat_factor= encode_latent(
                        encoder, t_encoder, x_encoder,cov, dose_tensor,mask,
                        enable_vae=False, enable_ae=True, enable_onlymedian=False
                    )
                    mu_q=mu_q -mu_p
                    mu_q = mu_q.cpu().numpy()
                    treatment_cpu = treatment_tensor.cpu().numpy()

                    for i, sid in enumerate(id_tensor):

                        mus.append(mu_q[i, :latent_dim])
                        treatment_list_all.append(treatment_cpu[i])
                        row = df[df["ID"] == int(sid)]
                        if row.empty:
                            continue
                        s = row["PARAM_VALUES"].values[0]

                        # Remove np.float64( ... )
                        cleaned = re.sub(r'np\.float64\(', '', s)
                        cleaned = cleaned.replace(')', '')
                        
                        param_values = ast.literal_eval(cleaned)
                                                
                        
                        
                       # param_values = ast.literal_eval(row["PARAM_VALUES"].values[0])
                        param_values = param_values[:n_params]
                        param_values_list.append(param_values)
                        id_list_all.append(sid.item())

        X = np.vstack(mus)
        if X.ndim == 1:
            X = X[:, None]  # ensure 2D shape (N, 1)
        y = np.array(param_values_list)
        if y.ndim == 1:
            y = y[:, None]
        treatments = np.array(treatment_list_all)
        return X, y, treatments, param_names

    # ---------------- Extract latents ----------------
    X_train, y_train, treatments_train, param_names = extract_latents(df_train, dataset_train, encoder)
    X_val, y_val, treatments_val, _ = extract_latents(df_val, dataset_val, encoder)

    n_params = y_train.shape[1]
    actual_latent_dim = X_val.shape[1]  # safe in case latent_dim is inconsistent

    # ---------------- Fit regressors ----------------
    rf_list = []
    y_pred_val = np.zeros_like(y_val, dtype=float)
    for i in range(n_params):
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

        rf.fit(X_train, y_train[:, i])
        rf_list.append(rf)
        y_pred_val[:, i] = rf.predict(X_val)

    # ---------------- Plot per parameter ----------------
    fs = 18
    marker_size = 90
    unique_treatments = np.unique(treatments_val)
    n_treatments = len(unique_treatments)

    # Generate complementary/distinct colors dynamically
    base_colors = ['blue', 'orange']

    # If more than 2 groups, generate extra colors dynamically
    if n_treatments > 2:
        extra_colors = plt.cm.tab20(np.linspace(0, 1, n_treatments - 2))
        colors = np.vstack([np.array([[0, 0, 1, 1], [1, 0.55, 0, 1]]), extra_colors])
    else:
        colors = np.array([[0, 0, 1, 1], [1, 0.55, 0, 1]])[:n_treatments]
    
    # Map treatments to colors
    treatment_color_map = {t: c for t, c in zip(unique_treatments, colors)}
    if single_figure:
        # Single large figure
        n_rows = n_params
        n_cols = actual_latent_dim + 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_params == 1:
            axes = np.expand_dims(axes, axis=0)  # ensure 2D
        if actual_latent_dim == 1:
            axes = np.expand_dims(axes, axis=1)
        fig.suptitle("Latent Individual Parameters", fontsize=fs+6, fontweight='bold')

        for i in range(n_params):
            # Latent scatter plots
            for d in range(actual_latent_dim):
                ax = axes[i, d]
                y_true = y_val[:, i]
                mu_vals = X_val[:, d]
                for t_val in unique_treatments:
                    mask = treatments_val == t_val
                    ax.scatter(mu_vals[mask], y_true[mask], alpha=0.8,
                               color=treatment_color_map[t_val], s=marker_size)
                ax.set_xlabel(f"μ{d+1}", fontsize=fs)
                ax.set_ylabel(f"True {param_names[i]}", fontsize=fs)
                ax.set_title(f"Latent dim {d+1}", fontsize=fs)
                ax.tick_params(axis='both', labelsize=fs-2)
                ax.grid(True, linestyle="--", alpha=0.6)

            # Regression plot
            ax_last = axes[i, -1]
            y_true = y_val[:, i]
            y_pred = y_pred_val[:, i]
            r2 = r2_score(y_true, y_pred)
            for t_val in unique_treatments:
                mask = treatments_val == t_val
                ax_last.scatter(y_pred[mask], y_true[mask], alpha=0.8, s=marker_size,
                                edgecolors='k', color=treatment_color_map[t_val])
            min_val, max_val = min(y_pred.min(), y_true.min()), max(y_pred.max(), y_true.max())
            ax_last.plot([min_val, max_val], [min_val, max_val], "r--")
            ax_last.set_xlabel("Predicted", fontsize=fs)
            ax_last.set_ylabel(f"True {param_names[i]}", fontsize=fs)
            ax_last.set_title("Regression", fontsize=fs)
            ax_last.tick_params(axis='both', labelsize=fs-2)
            ax_last.text(0.05, 0.9, f"R² = {r2:.2f}", transform=ax_last.transAxes,
                         fontsize=fs-2, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        # Legend below
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=treatment_color_map[t],
                              markersize=10, label=f"Treatment {int(global_max_dose*(t+1)/2)} mg/kg") for t in unique_treatments]
        fig.legend(handles=handles, loc='lower center',   frameon=False, ncol=len(unique_treatments), fontsize=fs)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Save as PDF
        if export:
            pp = PdfPages(pdf_filename)
            
        pp.savefig(fig)
        pp.close()
        plt.close(fig)
    else:    
        for i in range(n_params):
            fig, axes = plt.subplots(1, actual_latent_dim + 1, figsize=(5*(actual_latent_dim+1), 5))
            fig.suptitle(f"Parameter: {param_names[i]}", fontsize=fs+6, fontweight='bold')
    
            # Ensure axes is iterable
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
    
            # μ vs true parameter plots
            for d in range(actual_latent_dim):
                ax = axes[d]
                y_true = y_val[:, i]
                mu_vals = X_val[:, d]
                for t_val in unique_treatments:
                    mask = treatments_val == t_val
                    ax.scatter(mu_vals[mask], y_true[mask], alpha=0.8, label=f"Treatment {t_val}",
                               color=treatment_color_map[t_val], s=marker_size)
                ax.set_xlabel(f"μ{d+1}", fontsize=fs)
                ax.set_ylabel(f"True {param_names[i]}", fontsize=fs)
                ax.set_title(f"Latent individual dim {1+d}", fontsize=fs)
                ax.tick_params(axis='both', labelsize=fs-2)
                ax.grid(True, linestyle="--", alpha=0.6)
    
            # Regression plot
            ax_last = axes[-1]
            y_true = y_val[:, i]
            y_pred = y_pred_val[:, i]
            r2 = r2_score(y_true, y_pred)
            for t_val in unique_treatments:
                mask = treatments_val == t_val
                ax_last.scatter(y_pred[mask], y_true[mask], alpha=0.8, s=marker_size,
                                edgecolors='k', color=treatment_color_map[t_val], label=f"Treatment {t_val}")
            min_val, max_val = min(y_pred.min(), y_true.min()), max(y_pred.max(), y_true.max())
            ax_last.plot([min_val, max_val], [min_val, max_val], "r--")
            ax_last.set_xlabel("Predicted", fontsize=fs)
            ax_last.set_ylabel(f"True {param_names[i]}", fontsize=fs)
            ax_last.set_title("Regression", fontsize=fs)
            ax_last.tick_params(axis='both', labelsize=fs-2)
            ax_last.text(0.05, 0.9, f"R² = {r2:.2f}", transform=ax_last.transAxes,
                         fontsize=fs-2, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    
            # Legend below
            handles = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=treatment_color_map[t],
                                  markersize=10, label=f"Treatment {global_max_dose*(t+1)/2} mg/kg") for t in unique_treatments]
            fig.legend(handles=handles, loc='lower center', ncol=len(unique_treatments), fontsize=fs)
            plt.tight_layout(rect=[0, 0.15, 1, 0.95])
            plt.show()
    












def split_dataloader_by_treatment(dataloader, dataset):
    """Split an existing dataset/dataloader into one dataloader per treatment."""
    
    treatment_values = sorted(dataset.df['TREATMENT'].unique())
    loaders_by_treatment = {}

    for t in treatment_values:
        subset_indices = [i for i, traj in enumerate(dataset.samples) if traj['treatment'] == t]
        if not subset_indices:
            continue
        subset = Subset(dataset, subset_indices)
        loaders_by_treatment[t] = DataLoader(
            subset,
            batch_size=1000,
            shuffle=False,
            collate_fn=dataloader.collate_fn,
            num_workers=getattr(dataloader, 'num_workers', 0)
        )
    return loaders_by_treatment



def generate_plot_data(
      models_median,
      models_eval,
    dataloader, dataset, t_dense, df, add_noise_to_prediction,
    enable_onlymedian, enable_ae, enable_vae, normalization, truncation,
    use_ema_models=True
):
    
    plot_data_local = []
    global_max_dose, global_max_time, global_mean, global_std, global_max_value, global_min_value=compute_global_stats(df)
    print()
    encoder = models_eval['encoder'].eval()
    func = models_eval['func'].eval()
    reducer = models_eval['reducer'].eval()
    noise=models_eval['noise'].eval()

    
    
    encoder_med = models_median['encoder'].eval()
    func_med = models_median['func'].eval()
    reducer_med = models_median['reducer'].eval()
    
    device = next(func.parameters()).device

    

    # === Create dataloaders grouped by treatment ===
    loaders_by_treatment = split_dataloader_by_treatment(dataloader, dataset)
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()

        # Put models in eval mode
        encoder.eval(); 
        reducer.eval(); func.eval()
        encoder_med.eval();
        reducer_med.eval(); func_med.eval()

        for treatment_value, treatment_loader in loaders_by_treatment.items():

            all_preds, all_times, all_data = [], [], []
            treatment_name = dataset.df.loc[
              dataset.df["TREATMENT"] == treatment_value,
              "TREATMENT_NAME"
          ].iloc[0]
            for batch in treatment_loader:
                (
                    id_tensor,
                    t_padded,
                    x_padded,
                    mask,
                    cov,
                    dose_tensor,
                    dose_times_padded,
                    evid_padded,
                    treatment_tensor
                ) = batch
                
                

                id_list, treatment_list, t_padded, x_padded, t_encoder, x_encoder, \
                t_cut, x_cut, mask,cov_tensor, dose_tensor, dose_times_list, evid = preprocess_batch(
                    batch, device, truncation=truncation
                )
                
               # print(cov_tensor)
                # === Optional normalization ===
                if normalization:
                    
                    if isinstance(evid, list):
                        evid = torch.stack(evid, dim=0).to(device)  # shape: [batch, n_doses]
                    x_encoder = normalize_encoder_input(
                        x_encoder, t_encoder, x_padded, cov_tensor, dose_tensor,mask, dose_times_list,evid,
                        encoder_med,  func_med, reducer_med,
                        t_dense, global_mean, global_std
                    )

                # === Encode latent ===
                with use_ema(encoder):
                    k_param,_, mu_q, logvar_q, mu_p,L_p,mask_dropout,_ = encode_latent(
                        encoder, t_encoder, x_encoder,cov_tensor,dose_tensor,mask,
                        enable_vae=enable_vae,
                        enable_ae=enable_ae,
                        enable_onlymedian=enable_onlymedian,
                        warmup_epochs_iiv=0,
                        epoch=0,
                        min_batch_size=1,
                        augment=False,
                        sample_posterior=False
                    )
                    
               
                with use_ema(func):
                        ode_func= prepare_ode_input(
                            x_padded,
                            k_param,
                            func,
                            cov_tensor,
                            dose_tensor,
                            dose_times_list,
                            evid,
                            enable_ae,
                            enable_onlymedian
                        )
              
               

                      
              #  if enable_onlymedian:
                 #   k_param = torch.zeros_like(k_param)
                # === Predict ===
                with use_ema(reducer):
                    pred_interp, pred_batch = make_predictions(
                        t_padded, t_dense, k_param, ode_func,
                        reducer, global_mean, global_std
                    )

                if add_noise_to_prediction:
                    mask_pred = pred_batch > 0
                    pred_batch = torch.where(
                        mask_pred,
                        noise.sample(pred_batch, n_samples=1).squeeze(0),
                        pred_batch
                    )
                    pred_batch = torch.clamp(pred_batch, min=0)

                all_preds.append(pred_batch.detach().cpu())
                all_times.append(t_padded.detach().cpu())
                all_data.append(x_padded.detach().cpu())

            # === Aggregate results across batches ===
            pred_batch = torch.cat(all_preds, dim=0)
            t_padded = torch.cat(all_times, dim=0)
            data_matrix = torch.cat(all_data, dim=0)
            unique_times = torch.unique(t_padded[t_padded >= 0])
            perc10_list, median_list, perc90_list = [], [], []

            for t in unique_times:
                mask = t_padded == t  # shape [N, T], True where subject has this time
                values = destandardize_concentration(data_matrix[mask], global_mean, global_std)  # flatten all measurements at this time
                if len(values) > 0:
                    perc10_list.append(torch.quantile(values, 0.10))
                    median_list.append(torch.quantile(values, 0.50))
                    perc90_list.append(torch.quantile(values, 0.90))
                else:
                    perc10_list.append(torch.tensor(float('nan')))
                    median_list.append(torch.tensor(float('nan')))
                    perc90_list.append(torch.tensor(float('nan')))
            perc10_data = torch.stack(perc10_list)
            median_data = torch.stack(median_list)
            perc90_data = torch.stack(perc90_list)
                 

            perc10_sim = torch.quantile(pred_batch, 0.10, dim=0)
            median_sim = torch.quantile(pred_batch, 0.50, dim=0)
            perc90_sim = torch.quantile(pred_batch, 0.90, dim=0)

            # data_matrix_destd = destandardize_concentration(data_matrix, global_mean, global_std).T
            # perc10_data = torch.quantile(data_matrix_destd, 0.10, dim=1)
            # median_data = torch.quantile(data_matrix_destd, 0.50, dim=1)
            # perc90_data = torch.quantile(data_matrix_destd, 0.90, dim=1)

            plot_data_local.append({
                "treatment": treatment_name,
                "time_hours": t_dense.cpu().numpy() * global_max_time,
                "time_hours_data": unique_times.cpu().numpy() * global_max_time,
                "perc10_sim": perc10_sim.numpy(),
                "median_sim": median_sim.numpy(),
                "perc90_sim": perc90_sim.numpy(),
                "perc10_data": perc10_data.numpy(),
                "median_data": median_data.numpy(),
                "perc90_data": perc90_data.numpy(),
            })

    torch.cuda.empty_cache()
    gc.collect()
    return plot_data_local



    
def vpc(
    models_norm,models_eval, dataset, t_dense, df,
    add_noise_to_prediction=False,
    enable_onlymedian=True, enable_ae=False,
    enable_vae=False, normalization=True, truncation=1, num_repeats=100,
    use_ema_models=False,cols=2,export=False,pdf_filename="vpc.pdf",
    fontsize=14,
    show_confidence_intervals=True  # toggle all CI shading
):
    

    gc.collect()

    # --- Font sizes ---
    mpl.rcParams.update({
        'axes.titlesize': fontsize + 2,
        'axes.labelsize': fontsize,
        'xtick.labelsize': fontsize - 2,
        'ytick.labelsize': fontsize - 2,
        'legend.fontsize': fontsize,
    })


    collate_fn = make_collate_fn(dataset.global_max_len)
    global_max_dose, global_max_time, global_mean, global_std, global_max_value, global_min_value=compute_global_stats(df)

    dataloader = DataLoader(dataset, batch_size=12, shuffle=False, collate_fn=collate_fn)

    # ============================================================
    # --- Run generate_plot_data num_repeats times and collect ---
    # ============================================================
    all_repeats_data = []
    for r in range(num_repeats):
        plot_data_r = generate_plot_data(
            models_norm,
            models_eval,
            dataloader=dataloader, dataset=dataset, t_dense=t_dense,
            df=df,
            add_noise_to_prediction=add_noise_to_prediction,
            enable_onlymedian=enable_onlymedian, enable_ae=enable_ae,
            enable_vae=enable_vae, normalization=normalization, truncation=truncation,
            use_ema_models=use_ema_models
        )
        all_repeats_data.append(plot_data_r)

    # ============================================================
    # --- Group repeats by treatment ---
    # ============================================================
    treatments = sorted(set(d["treatment"] for d in all_repeats_data[0]))
    plot_by_treat = {t: [] for t in treatments}
    for t in treatments:
        plot_by_treat[t] = [
            next(d for d in repeat if d["treatment"] == t) for repeat in all_repeats_data
        ]

    # ============================================================
    # --- Create subplots ---
    # ============================================================
    n_plots = len(treatments)
    n_cols = cols
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(24, 8 * n_rows),
        sharex=True, sharey=True
    )
    axes = axes.flatten()

    color_pred = "blue"
    color_obs = "orange"

    # ============================================================
    # --- Loop over treatments ---
    # ============================================================
    for i, (treat, repeat_dicts) in enumerate(plot_by_treat.items()):
        ax = axes[i]
        times = repeat_dicts[0]["time_hours"]

        # Percentiles to track
        percentile_keys = [
            "perc5_sim", "perc10_sim", "perc25_sim", "median_sim",
            "perc75_sim", "perc90_sim", "perc95_sim"
        ]
        available_keys = [k for k in percentile_keys if k in repeat_dicts[0]]

        # --- Stack across repeats for each percentile ---
        sim_stats = {}
        for k in available_keys:
            arr = np.stack([d[k] for d in repeat_dicts], axis=0)  # (num_repeats, time)
            sim_stats[k] = {
                "median": np.median(arr, axis=0)
            }
            if show_confidence_intervals:
                sim_stats[k]["lower"] = np.percentile(arr, 2.5, axis=0)
                sim_stats[k]["upper"] = np.percentile(arr, 97.5, axis=0)

        # --- Observed data ---
        obs = repeat_dicts[0]
        ax.plot(obs["time_hours_data"], obs["median_data"],
                color=color_obs, linewidth=2.5, label="Observed median")
        ax.plot(obs["time_hours_data"], obs["perc10_data"],
                color=color_obs, linestyle="--", linewidth=2, label="Observed 10th")
        ax.plot(obs["time_hours_data"], obs["perc90_data"],
                color=color_obs, linestyle="--", linewidth=2, label="Observed 90th")

        # --- Helper to plot percentile lines + CI ---
        def plot_with_ci(x, median, lower, upper, label, alpha, style="-", width=2):
            if show_confidence_intervals and lower is not None and upper is not None:
                ax.fill_between(x, lower, upper, color=color_pred, alpha=alpha, linewidth=0)
            ax.plot(x, median, color=color_pred, linestyle=style, linewidth=width, label=label)

        # --- Plot each percentile with distinct styles and alphas ---
        style_map = {
            "perc5_sim": (":", 1.5, 0.10),
            "perc10_sim": ("--", 2, 0.12),
            "perc25_sim": ("-.", 2, 0.15),
            "median_sim": ("-", 2.5, 0.20),
            "perc75_sim": ("-.", 2, 0.15),
            "perc90_sim": ("--", 2, 0.12),
            "perc95_sim": (":", 1.5, 0.10),
        }

        for k, (style, width, alpha) in style_map.items():
            if k in sim_stats:
                plot_with_ci(times,
                             sim_stats[k]["median"],
                             sim_stats[k].get("lower"),
                             sim_stats[k].get("upper"),
                             f"Predicted {k.replace('_sim', '').replace('perc', '')}th",
                             alpha,
                             style=style,
                             width=width)

        # --- Formatting ---
        ax.set_xlim(0, global_max_time)
        ax.set_title(f"Treatment {treat}", fontsize=28, fontweight="bold")
        ax.set_xlabel("Time (hours)", fontsize=22)
        ax.set_ylabel("Concentration (mg/L)", fontsize=22)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=18, width=1.5)

    # --- Remove unused axes ---
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fontsize_legend = fontsize + 10   # adjust as desired

    legend_elements = [
        # Predictions
        Line2D([0], [0], color="blue", linestyle="--", linewidth=3, label="10th Prediction"),
        Line2D([0], [0], color="orange", linestyle="--", linewidth=3, label="10th Observation"),
        Line2D([0], [0], color="blue", linestyle="-",  linewidth=3, label="Median Prediction"),
        Line2D([0], [0], color="orange", linestyle="-",  linewidth=3, label="Median Observation"),
        Line2D([0], [0], color="blue", linestyle="--", linewidth=3, label="90th Prediction"),
    
        # Observed
      

        Line2D([0], [0], color="orange", linestyle="--", linewidth=3, label="90th Observation"),
    ]    
    # --- Legend ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=3,                       # <-- FORCE 3 per row
    fontsize=fontsize_legend,
    bbox_to_anchor=(0.5, -0.01),
    frameon=False,
    columnspacing=1.2,            # tidy spacing between columns
    handletextpad=0.5,            # spacing between line and label
    borderpad=0.7,
    labelspacing=1.0              # spacing between rows
)


    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.show()
    if export:
        pp = PdfPages(pdf_filename)
        pp.savefig(fig)
        pp.close()
        plt.close(fig)
               
def vpc_dual(
    models_norm, models_eval1, models_eval2, dataset, t_dense, df,
    add_noise_to_prediction=False,
    enable_onlymedian=True, enable_ae=False,
    enable_vae=False, normalization=True, truncation=1, num_repeats=100,
    use_ema_models=False, cols=2, export=False, pdf_filename="vpc_dual.pdf",
    fontsize=14, show_confidence_intervals=True
):
    import gc
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.lines import Line2D
    from matplotlib.backends.backend_pdf import PdfPages
    from torch.utils.data import DataLoader
    import numpy as np
    import math

    gc.collect()

    mpl.rcParams.update({
        'axes.titlesize': fontsize + 2,
        'axes.labelsize': fontsize,
        'xtick.labelsize': fontsize - 2,
        'ytick.labelsize': fontsize - 2,
        'legend.fontsize': fontsize,
    })

    collate_fn = make_collate_fn(dataset.global_max_len)
    global_max_dose, global_max_time, global_mean, global_std, global_max_value, global_min_value = compute_global_stats(df)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=False, collate_fn=collate_fn)

    # ---------------- collect data for both models ----------------
    def collect_data(models_eval):
        all_repeats_data = []
        for r in range(num_repeats):
            plot_data_r = generate_plot_data(
                models_norm, models_eval,
                dataloader=dataloader, dataset=dataset, t_dense=t_dense,
                df=df, add_noise_to_prediction=add_noise_to_prediction,
                enable_onlymedian=enable_onlymedian, enable_ae=enable_ae,
                enable_vae=enable_vae, normalization=normalization, truncation=truncation,
                use_ema_models=use_ema_models
            )
            all_repeats_data.append(plot_data_r)
        treatments = sorted(set(d["treatment"] for d in all_repeats_data[0]))
        plot_by_treat = {t: [next(d for d in repeat if d["treatment"] == t) for repeat in all_repeats_data] for t in treatments}
        return treatments, plot_by_treat

    treatments1, plot_by_treat1 = collect_data(models_eval1)
    treatments2, plot_by_treat2 = collect_data(models_eval2)

    # ---------------- create figure with two rows (one per model) ----------------
    n_cols = cols
    n_rows1 = math.ceil(len(treatments1) / n_cols)
    n_rows2 = math.ceil(len(treatments2) / n_cols)
    total_rows = n_rows1 + n_rows2
    fig, axes = plt.subplots(total_rows, n_cols, figsize=(24, 8 * total_rows), sharex=True, sharey=True, gridspec_kw={'hspace': 0.6})  # vertical space between rows)
    axes = axes.flatten()

    color_pred = "blue"
    color_obs = "orange"

    # ---------------- helper to plot a single model ----------------
    def plot_model(plot_by_treat, start_idx):
        for i, (treat, repeat_dicts) in enumerate(plot_by_treat.items()):
            ax = axes[start_idx + i]
            times = repeat_dicts[0]["time_hours"]

            percentile_keys = ["perc5_sim", "perc10_sim", "perc25_sim", "median_sim", "perc75_sim", "perc90_sim", "perc95_sim"]
            available_keys = [k for k in percentile_keys if k in repeat_dicts[0]]

            sim_stats = {}
            for k in available_keys:
                arr = np.stack([d[k] for d in repeat_dicts], axis=0)
                sim_stats[k] = {"median": np.median(arr, axis=0)}
                if show_confidence_intervals:
                    sim_stats[k]["lower"] = np.percentile(arr, 2.5, axis=0)
                    sim_stats[k]["upper"] = np.percentile(arr, 97.5, axis=0)

            obs = repeat_dicts[0]
            ax.plot(obs["time_hours_data"], obs["median_data"], color=color_obs, linewidth=2.5, label="Observed median")
            ax.plot(obs["time_hours_data"], obs["perc10_data"], color=color_obs, linestyle="--", linewidth=2, label="Observed 10th")
            ax.plot(obs["time_hours_data"], obs["perc90_data"], color=color_obs, linestyle="--", linewidth=2, label="Observed 90th")

            def plot_with_ci(x, median, lower, upper, label, alpha, style="-", width=2):
                if show_confidence_intervals and lower is not None and upper is not None:
                    ax.fill_between(x, lower, upper, color=color_pred, alpha=alpha, linewidth=0)
                ax.plot(x, median, color=color_pred, linestyle=style, linewidth=width, label=label)

            style_map = {
                "perc5_sim": (":", 1.5, 0.10), "perc10_sim": ("--", 2, 0.12), "perc25_sim": ("-.", 2, 0.15),
                "median_sim": ("-", 2.5, 0.20), "perc75_sim": ("-.", 2, 0.15), "perc90_sim": ("--", 2, 0.12),
                "perc95_sim": (":", 1.5, 0.10)
            }
            for k, (style, width, alpha) in style_map.items():
                if k in sim_stats:
                    plot_with_ci(times, sim_stats[k]["median"], sim_stats[k].get("lower"), sim_stats[k].get("upper"),
                                 f"Predicted {k.replace('_sim','').replace('perc','')}th", alpha, style=style, width=width)

            ax.set_xlim(0, global_max_time)
            ax.set_title(f"Treatment {treat}", fontsize=22, fontweight="bold")  # Only treatment title
            ax.set_xlabel("Time (hours)", fontsize=18)
            ax.set_ylabel("Concentration (mg/L)", fontsize=18)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)

        return start_idx + len(plot_by_treat)

    # ---------------- plot both models ----------------
    next_idx = 0
    row_start1 = next_idx
    next_idx = plot_model(plot_by_treat1, next_idx)
    row_start2 = next_idx
    next_idx = plot_model(plot_by_treat2, next_idx)

    for j in range(next_idx, len(axes)):
        fig.delaxes(axes[j])

    # ---------------- Add large panel titles ----------------
    # Model 1 title
    fig.text(0.5, 1 - 0.05, "Variational Autoencoder", ha='center', va='top', fontsize=32, fontweight='bold')
    # Model 2 title
    fig.text(0.5, 0.45, "Empirical Bayes Variational Autoencoder", ha='center', va='bottom', fontsize=32, fontweight='bold')

    # ---------------- legend ----------------
    fontsize_legend = fontsize + 8
    legend_elements = [
        Line2D([0], [0], color="blue", linestyle="--", linewidth=3, label="10th Prediction"),
        Line2D([0], [0], color="orange", linestyle="--", linewidth=3, label="10th Observation"),
        Line2D([0], [0], color="blue", linestyle="-", linewidth=3, label="Median Prediction"),
        Line2D([0], [0], color="orange", linestyle="-", linewidth=3, label="Median Observation"),
        Line2D([0], [0], color="blue", linestyle="--", linewidth=3, label="90th Prediction"),
        Line2D([0], [0], color="orange", linestyle="--", linewidth=3, label="90th Observation"),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=fontsize_legend, bbox_to_anchor=(0.5, -0.03),
               frameon=False, columnspacing=1.2, handletextpad=0.5,
               borderpad=0.7, labelspacing=1.0)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    # ---------------- after plotting both models ----------------




    
    plt.show()

    if export:
        pp = PdfPages(pdf_filename)
        pp.savefig(fig)
        pp.close()
        plt.close(fig)
