# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 07:10:20 2025
@author: Baaz
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import gc
import torch
import numpy as np
import math

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import torch
import numpy as np
import matplotlib.pyplot as plt

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ===== Standard Library =====
import math
import gc
import ast
from contextlib import contextmanager
from itertools import combinations

# ===== Third-Party Libraries =====
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import norm
import os

# ===== Local Project Imports =====
from lib.utils.utils_shared import (
    destandardize_concentration,
    use_ema,
    normalize_encoder_input,
    preprocess_batch,
    encode_latent,
    prepare_ode_input,
    make_predictions,
)
from lib.utils.utils_preprocess import make_collate_fn

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
    models,
    encoder_med, func_med, reducer_med,
    dataset,
    device,
    t_dense,
    global_mean,
    global_std,
    truncation=0,
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
    fontsize=14  # Added a parameter to control font size
):

    # --- Set global font sizes ---
    mpl.rcParams.update({
        'axes.titlesize': fontsize + 2,
        'axes.labelsize': fontsize,
        'xtick.labelsize': fontsize - 2,
        'ytick.labelsize': fontsize - 2,
        'legend.fontsize': fontsize,
    })

    encoder = models['encoder']

    func = models['func']
    reducer = models['reducer']
    noise = models['noise']
    collate_fn = make_collate_fn(dataset.global_max_len)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    fig, axes = plt.subplots(nr_row, nr_col, figsize=(5*nr_col, 4*nr_row))
    axes = axes.flatten() if nr_row*nr_col > 1 else [axes]

    encoder.eval()

    func.eval()
    dim_total = encoder.total_parameters + encoder.latent_dim
    population_preds = []
    coverage_list = []
    point_inside_list = []
    point_total_list = []

    for i, data in enumerate(dataloader):
        if i >= max_plots:
            break

        # --- Preprocess batch ---
        id_list,_, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list,evid = preprocess_batch(
            data, device, truncation=truncation
        )


        k_param_samples_list = []
     
        if normalization:
            x_encoder_norm = normalize_encoder_input( x_encoder, t_encoder, x_padded, dose_tensor, dose_times_list,evid,
             encoder_med, func_med, reducer_med,
             t_dense, global_mean, global_std)
        
        with use_ema(encoder):# if use_ema_models else contextmanager(lambda: (yield))():  
            for _ in range(n_samples):
                k_param_samples, _, _, _,_,_ = encode_latent(
                    encoder, t_encoder, x_encoder_norm,dose_tensor,
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
                    dose_tensor_exp,
                    dose_times_list_exp,
                    evid_exp,
                    enable_ae
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
        ax.set_title(f'Ind {i} | Coverage: {coverage:.1%}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')

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
            loc='lower center', 
            ncol=max(1, len(labels)),  # ensure at least 1 column
            bbox_to_anchor=(0.5, 0.1)
        )

    plt.tight_layout(rect=[0, 0.15, 1, 1])  # more space at bottom for legend
    plt.show()

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


def compute_residuals(func, encoder, reducer, noise, dataloader, t_dense,
                                                   global_mean, global_std, global_max_time,
                                                   truncation=1, num_repeats=50, show_confidence_intervals=True):
    """
    Compute residuals and generate VPC plot showing:
    - Predicted percentiles with confidence intervals
    - Observed scatter points
    - Observed percentiles
    Returns: residuals_all, predictions_all, targets_all, times_all
    """
    device = next(encoder.parameters()).device
    encoder.eval()
    reducer.eval()
    func.eval()

    # ---- Step 1: Collect residuals and predictions ----
    residuals_list, predictions_list, targets_list, times_list = [], [], [], []

    all_times = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            id_list, treatment_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list, evid = preprocess_batch(batch, device, truncation=truncation)

            # Encode latent
            k_param, mu_q, logvar_q, mu_p, logvar_p, _ = encode_latent(
                encoder, t_encoder, x_encoder,
                enable_vae=False, enable_ae=True, enable_onlymedian=False
            )

            # Prepare ODE input and predict
            ode_func = prepare_ode_input(x_padded, k_param, func, dose_tensor, dose_times_list, evid, enable_ae=True)
            pred_interp, pred_batch = make_predictions(t_padded, t_dense, k_param, ode_func, reducer, global_mean, global_std)
            targets = destandardize_concentration(x_padded, global_mean, global_std)
            residuals = targets - pred_interp

            residuals_list.append(residuals.view(-1))
            predictions_list.append(pred_interp.view(-1))
            targets_list.append(targets.view(-1))
            times_list.append(t_padded.view(-1))

            all_times.append(t_padded.view(-1).cpu().numpy())
            all_targets.append(targets.view(-1).cpu().numpy())

    residuals_all = torch.cat(residuals_list)
    predictions_all = torch.cat(predictions_list)
    targets_all = torch.cat(targets_list)
    times_all = torch.cat(times_list)

    all_times = np.concatenate(all_times)
    all_targets = np.concatenate(all_targets)

    # ---- Step 2: VPC simulation with repeats (predicted percentiles) ----
    all_repeats_data = []
    for r in range(num_repeats):
        plot_data_r = generate_plot_data(
            func_med=func,
            reducer_med=reducer,
            encoder_med=encoder,
            noise_med=noise,
            models={'encoder': encoder, 'func': func, 'reducer': reducer},
            dataloader=dataloader,
            dataset=dataloader.dataset,
            t_dense=t_dense,
            global_mean_time=1.0,
            global_mean_dose=1.0,
            global_mean=global_mean,
            global_std=global_std,
            encoder=encoder,
            func=func,
            reducer=reducer,
            noise=noise,
            ODEWrapper=None,
            num_simulated_total=1,
            add_noise_to_prediction=False,
            enable_onlymedian=True,
            enable_ae=False,
            enable_vae=False,
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
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    color_pred, color_obs, color_obs_percentiles = "blue", "orange", "green"

    # --- [0,0] VPC with observed scatter + observed percentiles ---
    ax = axes[0]

    # Compute observed percentiles over time bins
    num_bins = 20
    bins = np.linspace(0, global_max_time, num_bins)
    bin_indices = np.digitize(all_times*global_max_time, bins) - 1
    obs_percentiles = {perc: [] for perc in [5,25,50,75,95]}
    bin_centers = []

    for i in range(num_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_centers.append(bins[i] + (bins[i+1]-bins[i])/2 if i < num_bins-1 else bins[i])
            data_in_bin = all_targets[mask]
            for perc in obs_percentiles:
                obs_percentiles[perc].append(np.percentile(data_in_bin, perc))

    # Scatter all observed data points
    ax.scatter(all_times*global_max_time, all_targets, color=color_obs, s=20, alpha=0.4, label="Observed points")

    # Plot observed percentiles
    for perc, color in zip([5,25,50,75,95], ["#66c2a5","#fc8d62","#8da0cb","#fc8d62","#66c2a5"]):
        ax.plot(bin_centers, obs_percentiles[perc], linestyle='--', color=color, linewidth=1.5, label=f"Observed {perc}th percentile")

    # Plot predicted percentiles
    for i, (treat, repeat_dicts) in enumerate(plot_by_treat.items()):
        times = repeat_dicts[0]["time_hours"]
        percentile_keys = ["perc5_sim","perc10_sim","perc25_sim","median_sim","perc75_sim","perc90_sim","perc95_sim"]
        sim_stats = {}
        for k in percentile_keys:
            arr = np.stack([d.get(k, np.nan*np.ones_like(times)) for d in repeat_dicts], axis=0)
            sim_stats[k] = {"median": np.nanmedian(arr, axis=0)}
            if show_confidence_intervals:
                sim_stats[k]["lower"] = np.nanpercentile(arr, 2.5, axis=0)
                sim_stats[k]["upper"] = np.nanpercentile(arr, 97.5, axis=0)

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
            med = sim_stats[k]["median"]
            lower = sim_stats[k].get("lower")
            upper = sim_stats[k].get("upper")
            if lower is not None and upper is not None:
                ax.fill_between(times*global_max_time, lower, upper, color=color_pred, alpha=alpha)
            ax.plot(times*global_max_time, med, color=color_pred, linestyle=style, linewidth=width)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Concentration")
    ax.set_title("VPC: Observed scatter + percentiles + predicted percentiles")
    ax.grid(True)
    ax.legend(fontsize=8)

    # --- [0,1] Residual histogram ---
    ax_hist = axes[1]
    residuals_np = residuals_all.cpu().numpy()
    sigma_add = noise.sigma_add.detach().cpu().numpy().item()
    x_vals = np.linspace(residuals_np.min(), residuals_np.max(), 500)
    pdf_vals = norm.pdf(x_vals, loc=0.0, scale=sigma_add)
    ax_hist.hist(residuals_np, bins=25, density=True, alpha=0.7, color='gray', label='Residuals')
    ax_hist.plot(x_vals, pdf_vals, 'r--', linewidth=2, label=f'Gaussian Noise σ={sigma_add:.3f}')
    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Residual histogram")
    ax_hist.legend()

    # --- [1,0] Residuals vs Time ---
    ax_res_time = axes[2]
    ax_res_time.scatter(times_all.cpu().numpy()*global_max_time, residuals_all.cpu().numpy(), alpha=0.5, s=5, color='red')
    ax_res_time.set_xlabel("Time (hours)")
    ax_res_time.set_ylabel("Residual")
    ax_res_time.set_title("Residuals vs Time")
    ax_res_time.grid(True, alpha=0.3)

    # --- [1,1] Residuals vs Observed ---
    ax_res_obs = axes[3]
    ax_res_obs.scatter(targets_all.cpu().numpy(), residuals_all.cpu().numpy(), alpha=0.5, s=5, color='green')
    ax_res_obs.set_xlabel("Observed Concentration")
    ax_res_obs.set_ylabel("Residual")
    ax_res_obs.set_title("Residuals vs Observed")
    ax_res_obs.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ---- Return the original 4 outputs ----
    return residuals_all, predictions_all, targets_all, times_all








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
    encoder_med, func_med, reducer_med,
    t_dense, global_mean, global_std,
    dataset, dataloader, encoder,
    device=None, truncation=None, normalization=True
):
    """
    Plots encoder latent outputs:
    - Histograms of μ and σ for each latent dimension, grouped by treatment.
    - Pairwise scatter plots of latent dimensions, colored by treatment.
    - Shows prior and empirical posterior correlations as text on scatter plots.
    """

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import combinations

    if device is None:
        device = next(encoder.parameters()).device

    loaders_by_treatment = split_dataloader_by_treatment(dataloader, dataset)

    mu_list, sigma_list, treatment_list_all = [], [], []

    encoder.eval()
    with torch.no_grad():
        for treatment_value, treatment_loader in loaders_by_treatment.items():
            for batch in treatment_loader:
                # --- Unpack batch ---
                id_tensor, treatment_tensor, t_padded, x_padded, mask, dose_tensor, dose_times_padded, evid = batch

                # Move all tensors to device
                id_tensor = id_tensor.to(device)
                treatment_tensor = treatment_tensor.to(device)
                t_padded = t_padded.to(device)
                x_padded = x_padded.to(device)
                mask = mask.to(device)
                dose_tensor = dose_tensor.to(device)
                dose_times_padded = dose_times_padded.to(device)
                evid = evid.to(device)

                # Preprocess batch
                _, treatment_list, t_encoder, x_encoder, *_ = preprocess_batch(
                    batch, device, truncation=truncation
                )

                # Optional normalization
                if normalization:
                    x_encoder = normalize_encoder_input(
                        x_encoder, t_encoder, x_padded, dose_tensor, dose_times_padded, evid,
                        encoder_med, func_med, reducer_med,
                        t_dense, global_mean, global_std
                    )

                # Encode latent with EMA weights
                with use_ema(encoder):
                    k_param, mu_q, L_q, mu_p, L_p, _ = encode_latent(
                        encoder, t_encoder, x_encoder,dose_tensor,
                        enable_vae=True, enable_ae=False, enable_onlymedian=False
                    )

                    # Full covariance and σ
          
                    if L_q.dim() == 2:
                        # diagonal-only case
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
        mu_p, L_p = encoder.get_prior()
        L_p = L_p.to(device)  # [1, D, D] or [B, D, D]
    
        if L_p.dim() == 3 and L_p.shape[0] == 1:
            Sigma_p = torch.bmm(L_p, L_p.transpose(1, 2))  # [1, D, D]
            Sigma_p = Sigma_p[0]  # remove batch dim
        else:
            Sigma_p = L_p @ L_p.T  # fallback
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














def plot_encoder_vs_samples(
    encoder_med, initial_encoder_med, func_med,
    t_dense, reducer_med, global_mean, global_std,
    dataset, encoder, latent_dim, device=None, truncation=None, normalization=True
):
    """
    Plots μ and σ from the encoder as a function of the number of samples per individual.
    Loops over dataset one individual at a time (batch size = 1).
    Uses EMA weights if available.
    """

    if device is None:
        device = next(encoder.parameters()).device
    collate_fn = make_collate_fn(dataset.global_max_len)
    # Use batch_size=1 to process individual by individual
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    all_sample_counts = []
    all_mu = []
    all_sigma = []

    # --- Apply EMA weights temporarily ---
    if hasattr(encoder, "apply_ema_weights"):
        encoder.apply_ema_weights()
    if hasattr(initial_encoder_med, "apply_ema_weights"):
        initial_encoder_med.apply_ema_weights()
    if hasattr(func_med, "apply_ema_weights"):
        func_med.apply_ema_weights()

    with torch.no_grad():
        for data in dataloader:
            # Preprocess single individual
            id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list = preprocess_batch(
                data, device, truncation=truncation
            )

            # Number of actual samples for this individual
            sample_count = len(t_encoder[0])
            all_sample_counts.append(sample_count)

            if normalization:
                # Encode latent using median model
                k_param, mu_q, logvar_q = encode_latent(
                    encoder_med,
                    t_encoder,
                    x_encoder,
                    enable_vae=False,
                    enable_ae=False,
                    enable_onlymedian=True
                )

                # Prepare ODE input and predictions
                x0, ode_func = prepare_ode_input(
                    initial_encoder_med,
                    x_padded,
                    k_param,
                    func_med,
                    dose_tensor,
                    dose_times_list,
                    False
                )

                pred_interp, pred_batch = make_predictions(
                    t_encoder, t_dense, x0, ode_func, reducer_med, latent_dim, global_mean, global_std
                )

                x_encoder = destandardize_concentration(x_encoder, global_mean, global_std) / pred_interp

            # Encode latent using EMA encoder
            k_param, mu_q, logvar_q,_ = encode_latent(
                encoder, t_encoder, x_encoder,
                enable_vae=True, enable_ae=False, enable_onlymedian=False
            )

            mu_array = k_param.cpu().numpy()[0]      # shape [latent_dim]
            sigma_array = torch.exp(0.5 * logvar_q).cpu().numpy()[0]

            all_mu.append(mu_array)
            all_sigma.append(sigma_array)

    # Convert to arrays
    all_mu = np.array(all_mu)
    all_sigma = np.array(all_sigma)
    all_sample_counts = np.array(all_sample_counts)

    # Plot μ and σ vs number of samples
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    labels = ["μ dim 0", "μ dim 1", "σ dim 0", "σ dim 1"]
    data_arrays = [all_mu[:, 0], all_mu[:, 1], all_sigma[:, 0], all_sigma[:, 1]]

    for ax, data, label in zip(axes, data_arrays, labels):
        ax.scatter(all_sample_counts, data, alpha=0.7)
        ax.set_xlabel("Number of samples", fontsize=18)
        ax.set_ylabel(label, fontsize=18)
        ax.set_title(f"{label} vs Number of Samples", fontsize=20)
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()




def plot_two_models_encoders_and_regression(
    df_train, df_val, dataset_train, dataset_val,
    encoder1, initial_encoder1, func1, reducer1,
    encoder2, initial_encoder2, func2, reducer2,
    encoder_med, initial_encoder_med, func_med, reducer_med,
    latent_dim, global_mean, global_std, t_dense,
    device, enable_vae, enable_ae, enable_onlymedian, truncation, normalization,
    dim_parameter_encoder=2
):
    """
    Compare 2 models: encoder1, encoder2 (+ their initial/func/reducer).
    Produces a 2x4 grid of plots: each row = one model, each column = one plot type.
    Includes both μ and σ values from the encoder in regression and plots.
    """
      
    gc.collect()
    torch.cuda.empty_cache()
    if device is None:
        device = next(encoder1.parameters()).device

    # ---------------- Extract latents helper ----------------
    def extract_latents(encoder_med, initial_encoder_med, func_med, reducer_med,df, dataset, encoder, init_enc, func, reducer, normalization=False):

        mus, sigmas, ka_list, cl_list, id_list_all = [], [], [], [], []
       # encoder.eval()
        encoder_med.eval()
        initial_encoder_med.eval()
        func_med.eval()
        collate_fn = make_collate_fn(dataset.global_max_len)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)

        with torch.no_grad():
            for data in dataloader:
                id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list = preprocess_batch(
                    data, device, truncation=truncation
                )
                
                if normalization:
                    x_encoder = normalize_encoder_input( x_encoder, t_encoder, x_padded, dose_tensor, dose_times_list,
                     encoder_med, initial_encoder_med, func_med, reducer_med,
                     t_dense, latent_dim, global_mean, global_std)

                # Encode latent
                k_param, mu_q, logvar_q,_ = encode_latent(
                    encoder, t_encoder, x_encoder,
                    enable_vae=False, enable_ae=True, enable_onlymedian=False
                )
                    
                    
              
        
                    
             
                for i, sid in enumerate(id_list):
                    mu = mu_q[i].cpu().numpy()[:dim_parameter_encoder]
                    sigma = np.exp(0.5 * logvar_q[i].cpu().numpy()[:dim_parameter_encoder])
                    mus.append(mu)
                    sigmas.append(sigma)
                    row = df[df["ID"] == int(sid)]
                    if row.empty:
                        continue
                    ka_list.append(row["ka"].values[0])
                    cl_list.append(row["cl"].values[0])
                    id_list_all.append(sid)

        # Features are [μ, σ]
        return np.vstack(mus), np.array(ka_list), np.array(cl_list), id_list_all
      #  return np.hstack([np.vstack(mus), np.vstack(sigmas)]), np.array(ka_list), np.array(cl_list), id_list_all

    # ---------------- Shared color mapping ----------------
    cmap = plt.get_cmap("tab10")
    unique_doses = np.unique(df_val["AMT"].values)
    dose_to_color = {d: cmap(i % 10) for i, d in enumerate(unique_doses)}

    # ---------------- Prepare plots ----------------
    
    fs=80
    fig, axes = plt.subplots(2, 4, figsize=(fs, fs/1.62))  # 2 rows, 4 cols
    
    # ---------- Font sizes ----------
    fontsize_labels = 80
    fontsize_ticks = 70
    fontsize_legend = 70
    fontsize_r2 = 70
    
    models = [
        (encoder_med, initial_encoder_med, func_med, reducer_med,encoder1, initial_encoder1, func1, reducer1, "Naïve", False),  # row 1
        (encoder_med, initial_encoder_med, func_med, reducer_med,encoder2, initial_encoder2, func2, reducer2, "Baseline normalized", True),   # row 2
    ]
    
    for row_idx, (encoder_med, initial_encoder_med, func_med, reducer_med, encoder, init_enc, func, reducer, model_name, normalization) in enumerate(models):
        # --- Extract latents (μ and σ) ---
        X_val, ka_val, cl_val, id_list_val = extract_latents(
    encoder_med, initial_encoder_med, func_med, reducer_med,  # <--- added
    df_val, dataset_val,
    encoder, init_enc, func, reducer,
    normalization
)

        X_train, ka_train, cl_train, _ = extract_latents(
            encoder_med, initial_encoder_med, func_med, reducer_med,  # <--- added
            df_train, dataset_train,
            encoder, init_enc, func, reducer,
            normalization
        )

    
        # --- Fit regressors (using μ and σ) ---
        # lr_ka = LinearRegression().fit(X_train, np.log(ka_train))
        # lr_cl = LinearRegression().fit(X_train, np.log(cl_train))
        # ka_pred = lr_ka.predict(X_val)
        # cl_pred = lr_cl.predict(X_val)

        # --- Fit regressors (using μ and σ) ---
        rf_ka = RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ).fit(X_train, np.log(ka_train))
        
        rf_cl = RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ).fit(X_train, np.log(cl_train))
        
        ka_pred = rf_ka.predict(X_val)
        cl_pred = rf_cl.predict(X_val)
        
        r2_ka = r2_score(np.log(ka_val), ka_pred)
        r2_cl = r2_score(np.log(cl_val), cl_pred)
        # r2_ka = r2_score(np.log(ka_val), ka_pred)
        # r2_cl = r2_score(np.log(cl_val), cl_pred)
    
        # --- Colors per subject dose ---
        dose_array = np.array([df_val.loc[df_val["ID"] == int(sid), "AMT"].values[0] for sid in id_list_val])
        colors = [dose_to_color[d] for d in dose_array]
    
        log_ka_val = np.log(ka_val)
        log_cl_val = np.log(cl_val)
    
        # -------- Four plots for this model (row) --------
        marker_size = 200  # size of scatter plot markers
        marker_size_x = 300
        line_width = 8
        ax0, ax1, ax2, ax3 = axes[row_idx, :]  # select full row
        
        # μ and σ vs ka
        ax0.scatter(X_val[:,0], log_ka_val, c=colors, s=marker_size, alpha=0.7, edgecolors='k', label='μ0')
        ax0.scatter(X_val[:,1], log_ka_val, c=colors, s=marker_size_x, alpha=0.7, linewidths=6, edgecolors='k', marker='x', label='μ1')
        #ax0.scatter(X_val[:,2], log_ka_val, c=colors, s=marker_size, alpha=0.7, edgecolors='k', marker='s', label='σ0')
      #  ax0.scatter(X_val[:,3], log_ka_val, c=colors, s=marker_size_x, alpha=0.7, linewidths=6, edgecolors='k', marker='^', label='σ1')
        ax0.set_xlabel("Latent μ/σ", fontsize=fontsize_labels)
        ax0.set_ylabel("log(ka)", fontsize=fontsize_labels)
        ax0.tick_params(axis='both', labelsize=fontsize_ticks)
        ax0.set_title(f"{model_name}", fontsize=fontsize_labels)

        # μ and σ vs CL
        ax1.scatter(X_val[:,0], log_cl_val, c=colors, s=marker_size, alpha=0.7, edgecolors='k', label='μ0')
        ax1.scatter(X_val[:,1], log_cl_val, c=colors, s=marker_size_x, alpha=0.7, linewidths=6, edgecolors='k', marker='x', label='μ1')
       # ax1.scatter(X_val[:,2], log_cl_val, c=colors, s=marker_size, alpha=0.7, edgecolors='k', marker='s', label='σ0')
       # ax1.scatter(X_val[:,3], log_cl_val, c=colors, s=marker_size_x, alpha=0.7, linewidths=6, edgecolors='k', marker='^', label='σ1')
        ax1.set_xlabel("Latent μ/σ", fontsize=fontsize_labels)
        ax1.set_ylabel("log(ke)", fontsize=fontsize_labels)
        ax1.tick_params(axis='both', labelsize=fontsize_ticks)
        ax1.set_title(f"{model_name}", fontsize=fontsize_labels)
        
        # Regression ka
        ax2.scatter(ka_pred, log_ka_val, c=colors, s=marker_size, alpha=0.7, edgecolors='k')
        ax2.plot([ka_pred.min(), ka_pred.max()], [ka_pred.min(), ka_pred.max()], 'r--', lw=line_width)
        ax2.set_xlabel("Predicted log(ka)", fontsize=fontsize_labels)
        ax2.set_ylabel("True log(ka)", fontsize=fontsize_labels)
        ax2.tick_params(axis='both', labelsize=fontsize_ticks)
        ax2.set_title(f"{model_name}", fontsize=fontsize_labels)
        ax2.text(0.05, 0.9, f"R² = {r2_ka:.2f}", transform=ax2.transAxes,
                 fontsize=fontsize_r2, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        
        # Regression CL
        ax3.scatter(cl_pred, log_cl_val, c=colors, s=marker_size, alpha=0.7, edgecolors='k')
        ax3.plot([cl_pred.min(), cl_pred.max()], [cl_pred.min(), cl_pred.max()], 'r--', lw=line_width)
        ax3.set_xlabel("Predicted log(ke)", fontsize=fontsize_labels)
        ax3.set_ylabel("True log(ke)", fontsize=fontsize_labels)
        ax3.tick_params(axis='both', labelsize=fontsize_ticks)
        ax3.set_title(f"{model_name}", fontsize=fontsize_labels)
        ax3.text(0.05, 0.9, f"R² = {r2_cl:.2f}", transform=ax3.transAxes,
                 fontsize=fontsize_r2, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    
    
    
    # Marker legend for latent μ0 / μ1 / σ0 / σ1
    marker_handles = [
        Line2D([0], [0], marker='o', color='w', label='μ0', markerfacecolor='gray', markersize=40, markeredgecolor='k'),
        Line2D([0], [0], marker='x', color='w', label='μ1', markerfacecolor='gray', markersize=40, markeredgecolor='k', lw=6),
       # Line2D([0], [0], marker='s', color='w', label='σ0', markerfacecolor='gray', markersize=40, markeredgecolor='k'),
        #Line2D([0], [0], marker='^', color='w', label='σ1', markerfacecolor='gray', markersize=40, markeredgecolor='k')
    ]
    
    # Dose legend
    dose_handles = [
        Line2D([0], [0], marker='o', color='w', label=str(d),
               markerfacecolor=cmap(i % 10), markersize=40) for i, d in enumerate(unique_doses)
    ]
    
    # Add combined legend below the figure
    fig.legend(handles=marker_handles + dose_handles,
               loc='upper center',
               bbox_to_anchor=(0.5, -0.05),  # below the plot
               ncol=6,  # adjust columns to fit
               title=" ",
               fontsize=fontsize_legend,
               title_fontsize=fontsize_legend)
    plt.subplots_adjust(hspace=0.2, wspace=0.3, bottom=0.005)  # increase vertical space between rows
    plt.show()
    plt.close(fig)
    del fig, axes
    
    # Clear cached memory
    torch.cuda.empty_cache()    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                del obj
        except:
            pass
    del encoder1, initial_encoder1, reducer1, func1
    del encoder2, initial_encoder2, reducer2, func2
    del encoder_med, initial_encoder_med, reducer_med, func_med
    try:
        del X_val, ka_val, cl_val, X_train, ka_train, cl_train, id_list_val
    except NameError:
        pass
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
  
    gc.collect()
    torch.cuda.empty_cache()

def plot_one_model_encoders_and_regression2(
    df_train, df_val, dataset_train, dataset_val,
    encoder, initial_encoder, func, reducer,
    encoder_med, initial_encoder_med, func_med, reducer_med,
    latent_dim, global_mean, global_std, t_dense,
    device, enable_vae, enable_ae, enable_onlymedian, truncation, normalization,
    dim_parameter_encoder=2
):
    """
    Compare 1 model (encoder + median normalization).
    Produces a 2x3 grid of plots:
        - Row 1: μ0 vs ka, μ1 vs ka, regression ka.
        - Row 2: μ0 vs cl, μ1 vs cl, regression cl.
    """

    gc.collect()
    torch.cuda.empty_cache()
    if device is None:
        device = next(encoder.parameters()).device

    # ---------------- Extract latents helper ----------------
    def extract_latents(encoder_med, initial_encoder_med, func_med, reducer_med,
                        df, dataset, encoder, init_enc, func, reducer, normalization=False):

        mus, ka_list, cl_list, id_list_all = [], [], [], []
        encoder_med.eval()
        initial_encoder_med.eval()
        func_med.eval()
        collate_fn = make_collate_fn(dataset.global_max_len)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)
        with torch.no_grad():
            for data in dataloader:
                id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list = preprocess_batch(
                    data, device, truncation=truncation
                )

                if normalization:
                    x_encoder = normalize_encoder_input(
                        x_encoder, t_encoder, x_padded, dose_tensor, dose_times_list,
                        encoder_med, initial_encoder_med, func_med, reducer_med,
                        t_dense, latent_dim, global_mean, global_std)

              #  Encode latent
                k_param, mu_q, logvar_q, _ = encode_latent(
                    encoder, t_encoder, x_encoder,
                    enable_vae=False, enable_ae=True, enable_onlymedian=False
                )

                for i, sid in enumerate(id_list):
                    mu = mu_q[i].cpu().numpy()[:dim_parameter_encoder]
                    mus.append(mu)
                    row = df[df["ID"] == int(sid)]
                    if row.empty:
                        continue
                    ka_list.append(row["ka"].values[0])
                    cl_list.append(row["cl"].values[0])
                    id_list_all.append(sid)

        return np.vstack(mus), np.array(ka_list), np.array(cl_list), id_list_all

    # ---------------- Shared color mapping ----------------
    cmap = plt.get_cmap("tab10")
    unique_doses = np.unique(df_val["AMT"].values)
    dose_to_color = {d: cmap(i % 10) for i, d in enumerate(unique_doses)}

    # ---------------- Prepare plots ----------------
    fs = 60
    fig, axes = plt.subplots(2, 3, figsize=(fs, fs/1.5))

    # Bigger fonts
    fontsize_labels = 60
    fontsize_ticks = 50
    fontsize_r2 = 55

    # --- Extract latents ---
    X_val, ka_val, cl_val, id_list_val = extract_latents(
        encoder_med, initial_encoder_med, func_med, reducer_med,
        df_val, dataset_val, encoder, initial_encoder, func, reducer,
        normalization=True
    )
    X_train, ka_train, cl_train, _ = extract_latents(
        encoder_med, initial_encoder_med, func_med, reducer_med,
        df_train, dataset_train, encoder, initial_encoder, func, reducer,
        normalization=True
    )

    # --- Fit regressors (Random Forests) ---
    rf_ka = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, np.log(ka_train))
    rf_cl = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, np.log(cl_train))
    ka_pred = rf_ka.predict(X_val)
    cl_pred = rf_cl.predict(X_val)

    r2_ka = r2_score(np.log(ka_val), ka_pred)
    r2_cl = r2_score(np.log(cl_val), cl_pred)

    # --- Colors per subject dose ---
    dose_array = np.array([df_val.loc[df_val["ID"] == int(sid), "AMT"].values[0] for sid in id_list_val])
    colors = [dose_to_color[d] for d in dose_array]

    log_ka_val = np.log(ka_val)
    log_cl_val = np.log(cl_val)

    # ------------------- Scatter Plots -------------------
    # ------------------- Size controls -------------------
    marker_size   = 500   # size of scatter markers
    line_width    = 8     # regression line
    fontsize_title = 100   # subplot title font
    fontsize_labels = 100  # axis labels
    fontsize_ticks  = 80  # axis tick labels
    fontsize_r2     = 100  # R² box text
    fontsize_legend = 100  # legend text
    legend_marker_size = 60
    
    # ------------------- Scatter plots -------------------
    axes[0, 0].scatter(X_val[:, 0], log_ka_val, c=colors,
                       s=marker_size, alpha=0.7, edgecolors='k')
    axes[0, 0].set_xlabel("μ0", fontsize=fontsize_labels)
    axes[0, 0].set_ylabel("ka", fontsize=fontsize_labels)
    axes[0, 0].tick_params(axis='both', labelsize=fontsize_ticks)
    
    axes[0, 1].scatter(X_val[:, 1], log_ka_val, c=colors,
                       s=marker_size, alpha=0.7, edgecolors='k')
    axes[0, 1].set_xlabel("μ1", fontsize=fontsize_labels)
    axes[0, 1].set_ylabel("ka", fontsize=fontsize_labels)
    axes[0, 1].tick_params(axis='both', labelsize=fontsize_ticks)
    
    axes[0, 2].scatter(ka_pred, log_ka_val, c=colors,
                       s=marker_size, alpha=0.7, edgecolors='k')
    axes[0, 2].plot([ka_pred.min(), ka_pred.max()],
                    [ka_pred.min(), ka_pred.max()],
                    'r--', lw=line_width)
    axes[0, 2].text(0.05, 0.9, f"R² = {r2_ka:.2f}",
                    transform=axes[0, 2].transAxes,
                    fontsize=fontsize_r2,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    axes[0, 2].set_xlabel("Pred ka", fontsize=fontsize_labels)
    axes[0, 2].set_ylabel("ka", fontsize=fontsize_labels)
    axes[0, 2].tick_params(axis='both', labelsize=fontsize_ticks)
    
    # Row 2: cl
    axes[1, 0].scatter(X_val[:, 0], log_cl_val, c=colors,
                       s=marker_size, alpha=0.7, edgecolors='k')
    axes[1, 0].set_xlabel("μ0", fontsize=fontsize_labels)
    axes[1, 0].set_ylabel("ke", fontsize=fontsize_labels)
    axes[1, 0].tick_params(axis='both', labelsize=fontsize_ticks)
    
    axes[1, 1].scatter(X_val[:, 1], log_cl_val, c=colors,
                       s=marker_size, alpha=0.7, edgecolors='k')
    axes[1, 1].set_xlabel("μ1", fontsize=fontsize_labels)
    axes[1, 1].set_ylabel("ke", fontsize=fontsize_labels)
    axes[1, 1].tick_params(axis='both', labelsize=fontsize_ticks)
    
    axes[1, 2].scatter(cl_pred, log_cl_val, c=colors,
                       s=marker_size, alpha=0.7, edgecolors='k')
    axes[1, 2].plot([cl_pred.min(), cl_pred.max()],
                    [cl_pred.min(), cl_pred.max()],
                    'r--', lw=line_width)
    axes[1, 2].text(0.05, 0.9, f"R² = {r2_cl:.2f}",
                    transform=axes[1, 2].transAxes,
                    fontsize=fontsize_r2,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    axes[1, 2].set_xlabel("Pred ke", fontsize=fontsize_labels)
    axes[1, 2].set_ylabel("ke", fontsize=fontsize_labels)
    axes[1, 2].tick_params(axis='both', labelsize=fontsize_ticks)
    
    # ------------------- Shared legend -------------------
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               label=f'Dose {d}', markerfacecolor=c,
               markersize=legend_marker_size, markeredgecolor='k')
        for d, c in dose_to_color.items()
    ]
    
    fig.legend(handles=legend_elements,
               loc="lower center",
               bbox_to_anchor=(0.5, -0.02),
               ncol=len(legend_elements),
               fontsize=fontsize_legend,
               frameon=False)
    
    # ------------------- Global font adjustments -------------------
    for ax in axes.flat:
        ax.title.set_fontsize(fontsize_title)
        ax.xaxis.label.set_size(fontsize_labels)
        ax.yaxis.label.set_size(fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_ticks)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend
    plt.show()
    plt.close(fig)




def plot_single_model_encoders_and_regression(
    df_train, df_val, dataset_train, dataset_val,
    encoder1, func1, reducer1,
    encoder_med, func_med, reducer_med,
     global_mean, global_std, t_dense,
    device=None, truncation=1, use_ema_models=False, normalization=False, dim_parameter_encoder=2
):
    """
    Plot regression and encoder outputs using μ for regression,
    colored by treatment.
    Fully robust to latent_dim = 1 or higher.
    """

    gc.collect()
    torch.cuda.empty_cache()
    if device is None:
        device = next(encoder1.parameters()).device
    latent_dim=encoder1.total_parameters
  
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
                    id_tensor, treatment_tensor, t_padded, x_padded, mask, dose_tensor, dose_times_padded, evid_padded = batch
                    id_tensor = id_tensor.to(device)
                    treatment_tensor = treatment_tensor.to(device)
                    t_padded = t_padded.to(device)
                    x_padded = x_padded.to(device)
                    mask = mask.to(device)
                    dose_tensor = dose_tensor.to(device)
                    dose_times_padded = dose_times_padded.to(device)
                    evid_padded = evid_padded.to(device)

                    # Preprocess
                    id_list, treatment_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask_dose, dose_tensor, dose_times_list, evid = preprocess_batch(
                        batch, device, truncation=truncation
                    )

                    # Optional normalization
                    if normalization:
                        x_encoder_norm = normalize_encoder_input(
                            x_encoder, t_padded, x_padded, dose_tensor, dose_times_padded, evid_padded,
                            encoder_med, func_med, reducer_med,
                            t_dense, global_mean, global_std
                        )

                    _, mu_q, logvar_q, _,_,_ = encode_latent(
                        encoder, t_padded, x_encoder_norm,dose_tensor,
                        enable_vae=False, enable_ae=True, enable_onlymedian=False
                    )

                    mu_q = mu_q.cpu().numpy()
                    treatment_cpu = treatment_tensor.cpu().numpy()

                    for i, sid in enumerate(id_tensor):

                        mus.append(mu_q[i, :latent_dim])
                        treatment_list_all.append(treatment_cpu[i])
                        row = df[df["ID"] == int(sid)]
                        if row.empty:
                            continue
                        param_values = ast.literal_eval(row["PARAM_VALUES"].values[0])
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
    X_train, y_train, treatments_train, param_names = extract_latents(df_train, dataset_train, encoder1)
    X_val, y_val, treatments_val, _ = extract_latents(df_val, dataset_val, encoder1)

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
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_treatments)))
    treatment_color_map = {t: c for t, c in zip(unique_treatments, colors)}

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
            ax.set_xlabel(f"μ{d}", fontsize=fs)
            ax.set_ylabel(f"True {param_names[i]}", fontsize=fs)
            ax.set_title(f"Latent dim {d}", fontsize=fs)
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
                              markersize=10, label=f"Treatment {t}") for t in unique_treatments]
        fig.legend(handles=handles, loc='lower center', ncol=len(unique_treatments), fontsize=fs)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()














def plot_single_model_encoder_means(
    df_train, df_val, dataset_train, dataset_val,
    encoder1, initial_encoder1, func1, reducer1,
    encoder_med, initial_encoder_med, func_med, reducer_med,
    latent_dim, global_mean, global_std, t_dense,
    device=None, truncation=0, dim_parameter_encoder=2,
    use_ema_models=False, normalization=False
):
    


    gc.collect()
    torch.cuda.empty_cache()
    if device is None:
        device = next(encoder1.parameters()).device

    # ---------------- Extract latents helper ----------------
    def extract_latents_and_params(df, dataset, encoder, dim_parameter_encoder):
        mus_list, sigmas_list, ka_list, cl_list = [], [], [], []
        collate_fn = make_collate_fn(dataset.global_max_len)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)

        with use_ema(encoder) if use_ema_models else contextmanager(lambda: (yield))():
            encoder.eval()
            with torch.no_grad():
                for data in dataloader:
                    id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list = preprocess_batch(
                        data, device, truncation=truncation
                    )
                    if normalization:
                        x_encoder = normalize_encoder_input(
                            x_encoder, t_encoder, x_padded, dose_tensor, dose_times_list,
                            encoder_med, initial_encoder_med, func_med, reducer_med,
                            t_dense, latent_dim, global_mean, global_std
                        )

                    # Encode latent
                    _, mu_q, logvar_q, _ = encode_latent(
                        encoder, t_encoder, x_encoder,
                        enable_vae=False, enable_ae=True, enable_onlymedian=False
                    )

                    for i, sid in enumerate(id_list):
                        mu = mu_q[i].cpu().numpy()[:dim_parameter_encoder]
                        sigma = np.exp(0.5 * logvar_q[i].cpu().numpy()[:dim_parameter_encoder])
                        mus_list.append(mu)
                        sigmas_list.append(sigma)
                        row = df[df["ID"] == int(sid)]
                        if row.empty:
                            continue
                        ka_list.append(row["ka"].values[0])
                        cl_list.append(row["cl"].values[0])

        return np.vstack(mus_list), np.vstack(sigmas_list), np.array(ka_list), np.array(cl_list)

    # ---------------- Extract latents ----------------
    mus_val, sigmas_val, ka_val, cl_val = extract_latents_and_params(df_val, dataset_val, encoder1, dim_parameter_encoder)

    # ---------------- Compute σ² and average precision ----------------
    variances = sigmas_val ** 2
    precisions = 1.0 / variances
    avg_precisions = precisions.mean(axis=0)

    print("Average precision (1/σ^2) per latent dimension:")
    for i, prec in enumerate(avg_precisions):
        print(f"  Dimension {i+1}: {prec:.4f}")

    # ---------------- Plot μ vs true parameters ----------------
    fs = 5
    fig, axes = plt.subplots(2, dim_parameter_encoder, figsize=(fs*dim_parameter_encoder, fs*2))
    parameter_names = ["ka", "cl"]
    true_params = [ka_val, cl_val]

    for i, param in enumerate(true_params):
        for j in range(dim_parameter_encoder):
            ax = axes[i, j] if dim_parameter_encoder > 1 else axes[i]
            ax.scatter(mus_val[:, j], param, c="skyblue", s=80, alpha=0.7, edgecolors="k")
            ax.plot([mus_val[:, j].min(), mus_val[:, j].max()],
                    [param.min(), param.max()], "r--")
            ax.set_xlabel(f"Latent dim {j+1} mean")
            ax.set_ylabel(parameter_names[i])
            ax.set_title(f"{parameter_names[i]} vs latent {j+1}")

    plt.tight_layout()
    plt.show()



 
    
def plot_single_model_encoders_and_regression_combined(
    df_train, df_val, dataset_train, dataset_val,
    encoder1, initial_encoder1, func1, reducer1,
    encoder_med, initial_encoder_med, func_med, reducer_med,
    latent_dim, global_mean, global_std, t_dense,
    device, truncation, dim_parameter_encoder=2,
    use_ema_models=False
):
    """
    Plot regression using both μ and σ as features for log(ka) and log(CL),
    optionally using EMA weights for encoders.
    """
 
    gc.collect()
    torch.cuda.empty_cache()
    if device is None:
        device = next(encoder1.parameters()).device

    # ---------------- Extract latents helper ----------------
    def extract_latents(df, dataset, encoder, dim_parameter_encoder):
        mus, sigmas, ka_list, cl_list, id_list_all = [], [], [], [], []
        collate_fn = make_collate_fn(dataset.global_max_len)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)

        with use_ema(encoder) if use_ema_models else contextmanager(lambda: (yield))():
            encoder.eval()
            with torch.no_grad():
                for data in dataloader:
                    id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list = preprocess_batch(
                        data, device, truncation=truncation
                    )
                    
                    

                    _, mu_q, logvar_q, _ = encode_latent(
                        encoder, t_encoder, x_encoder,
                        enable_vae=False, enable_ae=True, enable_onlymedian=False
                    )

                    for i, sid in enumerate(id_list):
                        mu = mu_q[i].cpu().numpy()[:dim_parameter_encoder]
                        sigma = np.exp(0.5 * logvar_q[i].cpu().numpy()[:dim_parameter_encoder])
                        mus.append(mu)
                        sigmas.append(sigma)
                        row = df[df["ID"] == int(sid)]
                        if row.empty:
                            continue
                        ka_list.append(row["ka"].values[0])
                        cl_list.append(row["cl"].values[0])
                        id_list_all.append(sid)

        X = np.hstack([np.vstack(mus), np.vstack(sigmas)])  # combine μ and σ
        return X, np.array(ka_list), np.array(cl_list), id_list_all

    # ---------------- Extract latents ----------------
    X_train, ka_train, cl_train, _ = extract_latents(df_train, dataset_train, encoder1, dim_parameter_encoder)
    X_val, ka_val, cl_val, _ = extract_latents(df_val, dataset_val, encoder1, dim_parameter_encoder)

    # ---------------- Fit regressors ----------------
    rf_ka = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_ka.fit(X_train, np.log(ka_train))
    rf_cl = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_cl.fit(X_train, np.log(cl_train))

    # ---------------- Predict ----------------
    ka_pred = rf_ka.predict(X_val)
    cl_pred = rf_cl.predict(X_val)

    # R2 scores
    r2_ka = r2_score(np.log(ka_val), ka_pred)
    r2_cl = r2_score(np.log(cl_val), cl_pred)

    # ---------------- Plot ----------------
    fs = 16
    fig, axes = plt.subplots(1, 2, figsize=(fs*2, fs))
    ax0, ax1 = axes.flatten()
    marker_size = 80

    log_ka_val = np.log(ka_val)
    log_cl_val = np.log(cl_val)

    # Predicted vs True log(ka)
    ax0.scatter(ka_pred, log_ka_val, c="gray", s=marker_size, alpha=0.7, edgecolors="k")
    ax0.plot([ka_pred.min(), ka_pred.max()], [ka_pred.min(), ka_pred.max()], "r--")
    ax0.set_xlabel("Predicted log(ka)")
    ax0.set_ylabel("True log(ka)")
    ax0.set_title("log(ka) regression")
    ax0.text(0.05, 0.9, f"R² = {r2_ka:.2f}", transform=ax0.transAxes,
             fontsize=14, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    # Predicted vs True log(CL)
    ax1.scatter(cl_pred, log_cl_val, c="gray", s=marker_size, alpha=0.7, edgecolors="k")
    ax1.plot([cl_pred.min(), cl_pred.max()], [cl_pred.min(), cl_pred.max()], "r--")
    ax1.set_xlabel("Predicted log(CL)")
    ax1.set_ylabel("True log(CL)")
    ax1.set_title("log(CL) regression")
    ax1.text(0.05, 0.9, f"R² = {r2_cl:.2f}", transform=ax1.transAxes,
             fontsize=14, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    plt.tight_layout()
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
            batch_size=dataloader.batch_size,
            shuffle=False,
            collate_fn=dataloader.collate_fn,
            num_workers=getattr(dataloader, 'num_workers', 0)
        )
    return loaders_by_treatment

def sample_from_prior(encoder, batch_size, device):
    """
    Draw latent samples z ~ N(mu_p, L_p L_p^T) from the learned prior.
    """
    with torch.no_grad():
        mu_p, L_p = encoder.get_prior(batch_size=batch_size)
        eps = torch.randn(batch_size, mu_p.size(1), device=device)
        z = mu_p + torch.einsum('bij,bj->bi', L_p, eps)
    return z


def generate_plot_data(
    func_med, reducer_med,  encoder_med, noise_med,
    models, dataloader, dataset, t_dense, global_mean_time, global_mean_dose,
    global_mean, global_std,
     encoder, func, reducer, noise,
    ODEWrapper, num_simulated_total, add_noise_to_prediction,
    enable_onlymedian, enable_ae, enable_vae, normalization, truncation,
    use_ema_models=True
):
    
    plot_data_local = []
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

            for batch in treatment_loader:
                (
                    id_tensor,
                    t_padded,
                    x_padded,
                    mask,
                    dose_tensor,
                    dose_times_padded,
                    evid_padded,
                    treatment_tensor
                ) = batch

                id_list, treatment_list, t_padded, x_padded, t_encoder, x_encoder, \
                t_cut, x_cut, mask, dose_tensor, dose_times_list, evid = preprocess_batch(
                    batch, device, truncation=truncation
                )

                # === Optional normalization ===
                if normalization:
                    
                    if isinstance(evid, list):
                        evid = torch.stack(evid, dim=0).to(device)  # shape: [batch, n_doses]
                    x_encoder = normalize_encoder_input(
                        x_encoder, t_encoder, x_padded, dose_tensor, dose_times_list,evid,
                        encoder_med,  func_med, reducer_med,
                        t_dense, global_mean, global_std
                    )

                # === Encode latent ===
                with use_ema(encoder):
                    k_param, mu_q, logvar_q, mu_p,L_p,_ = encode_latent(
                        encoder, t_encoder, x_encoder,dose_tensor,
                        enable_vae=enable_vae,
                        enable_ae=enable_ae,
                        enable_onlymedian=enable_onlymedian
                    )
                if enable_vae:
                     with use_ema(encoder):
                             B, D = mu_p.shape
                             eps = torch.randn(B, D, device=mu_p.device)
                             k_param = mu_p + torch.einsum('bij,bj->bi', L_p, eps) 
                             cov_p = L_p @ L_p.transpose(-1, -2)
                           #  print(cov_p)
                             std = torch.sqrt(torch.diagonal(cov_p, dim1=-2, dim2=-1))  # [B, D]
                             corr_p = cov_p / std.unsqueeze(-1) / std.unsqueeze(-2)
                            
                            # Print first batch
                        #     print(corr_p[0])
                                                 
                
                with use_ema(func):
                        ode_func= prepare_ode_input(
                            x_padded,
                            k_param,
                            func,
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
                "treatment": int(treatment_value),
                "time_hours": t_dense.cpu().numpy() * global_mean_time,
                "time_hours_data": unique_times.cpu().numpy() * global_mean_time,
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






def vpc_2(
    # Median model
    encoder_med, initial_encoder_med, func_med, reducer_med, noise_med,
    # Model 1
    encoder_1, initial_encoder_1, func_1, reducer_1, noise_1, models_1,
    # Model 2
    encoder_2, initial_encoder_2, func_2, reducer_2, noise_2, models_2,
    dataloader, dataset, val_loader, dataset_val, t_dense, latent_dim, dim_parameters,
    global_mean_time, global_mean_dose, global_mean, global_std,
    compartment, ODEWrapper,
    add_noise_to_prediction, num_simulated_total,
    onlymedian, enable_ae, enable_vae, normalization,
    truncation
):
    fontsize_labels = 20
    fontsize_ticks = 15
    fontsize_legend = 15

    # ---- Generate plot data ----
    plot_data_1 = generate_plot_data(func_med=func_med, reducer_med=reducer_med,
                                     initial_encoder_med=initial_encoder_med, encoder_med=encoder_med,
                                     noise_med=noise_med, models=models_1, dataloader=dataloader,
                                     dataset=dataset, t_dense=t_dense, global_mean_time=global_mean_time,
                                     global_mean_dose=global_mean_dose, global_mean=global_mean, global_std=global_std,
                                     latent_dim=latent_dim, dim_parameters=dim_parameters,
                                     initial_encoder=initial_encoder_1, encoder=encoder_1, func=func_1,
                                     reducer=reducer_1, noise=noise_1, ODEWrapper=ODEWrapper,
                                     num_simulated_total=num_simulated_total,
                                     add_noise_to_prediction=add_noise_to_prediction,
                                     enable_ae=True, enable_vae=False,
                                     enable_onlymedian=onlymedian, normalization=normalization,
                                     truncation=truncation)

    plot_data_2 = generate_plot_data(func_med=func_med, reducer_med=reducer_med,
                                     initial_encoder_med=initial_encoder_med, encoder_med=encoder_med,
                                     noise_med=noise_med, models=models_2, dataloader=dataloader,
                                     dataset=dataset, t_dense=t_dense, global_mean_time=global_mean_time,
                                     global_mean_dose=global_mean_dose, global_mean=global_mean, global_std=global_std,
                                     latent_dim=latent_dim, dim_parameters=dim_parameters,
                                     initial_encoder=initial_encoder_2, encoder=encoder_2, func=func_2,
                                     reducer=reducer_2, noise=noise_2, ODEWrapper=ODEWrapper,
                                     num_simulated_total=num_simulated_total,
                                     add_noise_to_prediction=add_noise_to_prediction,
                                     enable_ae=enable_ae, enable_vae=enable_vae,
                                     enable_onlymedian=onlymedian, normalization=normalization,
                                     truncation=truncation)

    plot_data_3 = generate_plot_data(func_med=func_med, reducer_med=reducer_med,
                                     initial_encoder_med=initial_encoder_med, encoder_med=encoder_med,
                                     noise_med=noise_med, models=models_1, dataloader=val_loader,
                                     dataset=dataset_val, t_dense=t_dense, global_mean_time=global_mean_time,
                                     global_mean_dose=global_mean_dose, global_mean=global_mean, global_std=global_std,
                                     latent_dim=latent_dim, dim_parameters=dim_parameters,
                                     initial_encoder=initial_encoder_1, encoder=encoder_1, func=func_1,
                                     reducer=reducer_1, noise=noise_1, ODEWrapper=ODEWrapper,
                                     num_simulated_total=num_simulated_total,
                                     add_noise_to_prediction=add_noise_to_prediction,
                                     enable_ae=True,  enable_vae=False,
                                     enable_onlymedian=onlymedian, normalization=normalization,
                                     truncation=truncation)

    plot_data_4 = generate_plot_data(func_med=func_med, reducer_med=reducer_med,
                                     initial_encoder_med=initial_encoder_med, encoder_med=encoder_med,
                                     noise_med=noise_med, models=models_2, dataloader=val_loader,
                                     dataset=dataset_val, t_dense=t_dense, global_mean_time=global_mean_time,
                                     global_mean_dose=global_mean_dose, global_mean=global_mean, global_std=global_std,
                                     latent_dim=latent_dim, dim_parameters=dim_parameters,
                                     initial_encoder=initial_encoder_2, encoder=encoder_2, func=func_2,
                                     reducer=reducer_2, noise=noise_2, ODEWrapper=ODEWrapper,
                                     num_simulated_total=num_simulated_total,
                                     add_noise_to_prediction=add_noise_to_prediction,
                                     enable_ae=enable_ae,  enable_vae=enable_vae,
                                     enable_onlymedian=onlymedian, normalization=normalization,
                                     truncation=truncation)


    

    # Reorder datasets and titles

    # Reorder datasets and titles

        
        # Reorder datasets and titles
    # ---- Figure sizing ----

    
    # ---- Figure sizing ----
    row_width = 40                  # width of each row in inches
    golden_ratio = 1.62
    fig_height_per_row = row_width / golden_ratio  # height per row to follow golden ratio
    n_rows = 4
    
    vertical_space = 0.4
    horizontal_space = 0.1
    
    # ---- Lines and fonts ----
    lw_sim, lw_raw, markersize_sim = 12, 12, 20
    fontsize_title = 80
    fontsize_ticks = 70
    fontsize_labels = 100
    fontsize_legend = 80
    
    # ---- Datasets and titles ----
    datasets = [plot_data_3, plot_data_4, plot_data_1, plot_data_2]
    titles = ["AE", "VAE", "AE ", "VAE"]
    
    # ---- Create figure ----
    fig = plt.figure(figsize=(100, fig_height_per_row * n_rows), constrained_layout=True)
    outer_gs = GridSpec(n_rows, 1, figure=fig, hspace=vertical_space)
    
    for row_idx, data_set in enumerate(datasets):
        n_doses = len(data_set)
        
        # Nested grid for doses
        row_gs = GridSpecFromSubplotSpec(1, n_doses,
                                         subplot_spec=outer_gs[row_idx],
                                         wspace=horizontal_space)
        
        for col_idx, data in enumerate(data_set):
            ax = fig.add_subplot(row_gs[0, col_idx])
            
            # Determine colors
            if row_idx in [0, 2]:  # AE datasets
                sim_color = 'red'
                raw_color = 'orange'
            else:  # VAE datasets
                sim_color = 'blue'
                raw_color = 'orange'
            
            # Plot simulated / predicted
            ax.plot(data["time_hours"], data["perc10_sim"], '--', color=sim_color, linewidth=lw_sim)
            ax.plot(data["time_hours"], data["median_sim"], '-', color=sim_color, markersize=markersize_sim, linewidth=lw_sim)
            ax.plot(data["time_hours"], data["perc90_sim"], '--', color=sim_color, linewidth=lw_sim)
            
            # Plot raw data
            ax.plot(data["time_hours_data"], data["perc10_data"], '--', color=raw_color, linewidth=lw_raw)
            ax.plot(data["time_hours_data"], data["median_data"], '-', color=raw_color, linewidth=lw_raw)
            ax.plot(data["time_hours_data"], data["perc90_data"], '--', color=raw_color, linewidth=lw_raw)
            
            # Titles and limits
            ax.set_title(f"Dose {data['dose_value']:.0f} ({titles[row_idx]})", fontsize=fontsize_title)
            ax.set_xlim(0, 36 if row_idx in [0,1] else 20)
            ax.set_ylim(0, 180)
            ax.tick_params(axis='both', labelsize=fontsize_ticks)
            ax.grid(True)
    
    # Shared labels
    fig.text(0.5, -0.03, "Time (hours)", ha='center', fontsize=fontsize_labels)
    fig.text(-0.03, 0.5, f"Concentration ($\mu$g/mL)", va='center', rotation='vertical', fontsize=fontsize_labels)
    
    # Legend
    legend_elements = [
    # Prediction (blue)
    Line2D([0], [0], color='blue', linestyle='--', linewidth=lw_sim, label='Prediction 10th'),
    Line2D([0], [0], color='blue', linestyle='-', linewidth=lw_sim, label='Prediction median'),
    Line2D([0], [0], color='blue', linestyle='--', linewidth=lw_sim, label='Prediction 90th'),

    # Reconstruction (red)
    Line2D([0], [0], color='red', linestyle='--', linewidth=lw_raw, label='Reconstruction 10th'),
    Line2D([0], [0], color='red', linestyle='-', linewidth=lw_raw, label='Reconstruction median'),
    Line2D([0], [0], color='red', linestyle='--', linewidth=lw_raw, label='Reconstruction 90th'),

    # Data (orange)
    Line2D([0], [0], color='orange', linestyle='--', linewidth=lw_raw, label='Data 10th'),
    Line2D([0], [0], color='orange', linestyle='-', linewidth=lw_raw, label='Data median'),
    Line2D([0], [0], color='orange', linestyle='--', linewidth=lw_raw, label='Data 90th'),
]


    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=fontsize_legend, bbox_to_anchor=(0.5, -0.1), markerscale=3)
    
    # Save high-res full-width figure
    plt.savefig("full_width_figure.png", dpi=300)
    plt.show()

        
            
    
       


def vpc_3(
    # Median model
    encoder_med, initial_encoder_med, func_med, reducer_med, noise_med,
    # Model 1
    encoder_1, initial_encoder_1, func_1, reducer_1, noise_1, models_1,
    # Model 2
    encoder_2, initial_encoder_2, func_2, reducer_2, noise_2, models_2,
    dataloader, dataset, val_loader, dataset_val, t_dense, latent_dim, dim_parameters,
    global_mean_time, global_mean_dose, global_mean, global_std,
    compartment, ODEWrapper,
    add_noise_to_prediction, num_simulated_total,
    onlymedian, enable_ae, enable_vae, normalization,
    truncation
):
    fontsize_labels = 20
    fontsize_ticks = 15
    fontsize_legend = 15

    # ---- Generate plot data ----
  

    plot_data_3 = generate_plot_data(func_med=func_med, reducer_med=reducer_med,
                                     initial_encoder_med=initial_encoder_med, encoder_med=encoder_med,
                                     noise_med=noise_med, models=models_1, dataloader=val_loader,
                                     dataset=dataset_val, t_dense=t_dense, global_mean_time=global_mean_time,
                                     global_mean_dose=global_mean_dose, global_mean=global_mean, global_std=global_std,
                                     latent_dim=latent_dim, dim_parameters=dim_parameters,
                                     initial_encoder=initial_encoder_1, encoder=encoder_1, func=func_1,
                                     reducer=reducer_1, noise=noise_1, ODEWrapper=ODEWrapper,
                                     num_simulated_total=num_simulated_total,
                                     add_noise_to_prediction=add_noise_to_prediction,
                                     enable_ae=True,  enable_vae=False,
                                     enable_onlymedian=onlymedian, normalization=normalization,
                                     truncation=truncation)

    plot_data_4 = generate_plot_data(func_med=func_med, reducer_med=reducer_med,
                                     initial_encoder_med=initial_encoder_med, encoder_med=encoder_med,
                                     noise_med=noise_med, models=models_2, dataloader=val_loader,
                                     dataset=dataset_val, t_dense=t_dense, global_mean_time=global_mean_time,
                                     global_mean_dose=global_mean_dose, global_mean=global_mean, global_std=global_std,
                                     latent_dim=latent_dim, dim_parameters=dim_parameters,
                                     initial_encoder=initial_encoder_2, encoder=encoder_2, func=func_2,
                                     reducer=reducer_2, noise=noise_2, ODEWrapper=ODEWrapper,
                                     num_simulated_total=num_simulated_total,
                                     add_noise_to_prediction=add_noise_to_prediction,
                                     enable_ae=False,  enable_vae=True,
                                     enable_onlymedian=False, normalization=normalization,
                                     truncation=truncation)


    
    
 
    # ---- Figure sizing ----
    fig_width = 60
    golden_ratio = 1.62
    fig_height = fig_width / golden_ratio
    
    # ---- Lines and fonts ----
    lw_sim, lw_raw, markersize_sim = 12, 12, 20
    fontsize_title = 80
    fontsize_ticks = 70
    fontsize_labels = 70
    fontsize_legend = 70
    
    # ---- Datasets and titles ----
    datasets = [plot_data_3, plot_data_4]  # AE, VAE
    titles = ["AE", "VAE"]
    
    # ---- Create 2x2 grid ----
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, wspace=0.1, hspace=0.1)
    
    for row_idx, data_set in enumerate(datasets):  # row: model
        for col_idx, data in enumerate(data_set):  # column: dose
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Colors
            sim_color = 'red' if row_idx == 0 else 'blue'
            raw_color = 'orange'
            
            # Simulated / predicted
            ax.plot(data["time_hours"], data["perc10_sim"], '--', color=sim_color, linewidth=lw_sim)
            ax.plot(data["time_hours"], data["median_sim"], '-', color=sim_color, markersize=markersize_sim, linewidth=lw_sim)
            ax.plot(data["time_hours"], data["perc90_sim"], '--', color=sim_color, linewidth=lw_sim)
            
            # Raw data
            ax.plot(data["time_hours_data"], data["perc10_data"], '--', color=raw_color, linewidth=lw_raw)
            ax.plot(data["time_hours_data"], data["median_data"], '-', color=raw_color, linewidth=lw_raw)
            ax.plot(data["time_hours_data"], data["perc90_data"], '--', color=raw_color, linewidth=lw_raw)
            
            # Titles and limits
            ax.set_title(f"{titles[row_idx]}, Dose {data['dose_value']:.0f}", fontsize=fontsize_title)
            ax.set_xlim(0, 36)
            ax.set_ylim(0, 180)
            ax.tick_params(axis='both', labelsize=fontsize_ticks)
            ax.grid(True)
    
    # Shared labels
    fig.text(0.5, -0.05, "Time", ha='center', fontsize=fontsize_labels)
    fig.text(-0.05, 0.5, "Concentration", va='center', rotation='vertical', fontsize=fontsize_labels)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='blue', linestyle='--', linewidth=lw_sim, label='Prediction 10th'),
        Line2D([0], [0], color='blue', linestyle='-', linewidth=lw_sim, label='Prediction median'),
        Line2D([0], [0], color='blue', linestyle='--', linewidth=lw_sim, label='Prediction 90th'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=lw_raw, label='Reconstruction 10th'),
        Line2D([0], [0], color='red', linestyle='-', linewidth=lw_raw, label='Reconstruction median'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=lw_raw, label='Reconstruction 90th'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=lw_raw, label='Data 10th'),
        Line2D([0], [0], color='orange', linestyle='-', linewidth=lw_raw, label='Data median'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=lw_raw, label='Data 90th'),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=fontsize_legend,
               bbox_to_anchor=(0.5, -0.3), markerscale=3)
    
    plt.savefig("vpc_2x2_grid.png", dpi=300)
    plt.show()
    

        

def vpc_4(
    # Median model
    encoder_med, initial_encoder_med, func_med, reducer_med, noise_med,
    # Model 1
    encoder_1, initial_encoder_1, func_1, reducer_1, noise_1, models_1,
    # Model 2
    encoder_2, initial_encoder_2, func_2, reducer_2, noise_2, models_2,
    dataloader, dataset, val_loader, dataset_val, t_dense, latent_dim, dim_parameters,
    global_mean_time, global_mean_dose, global_mean, global_std,
    compartment, ODEWrapper,
    add_noise_to_prediction, num_simulated_total,
    onlymedian, enable_ae, enable_vae, normalization,
    truncation
):
    fontsize_labels = 20
    fontsize_ticks = 15
    fontsize_legend = 15

    # ---- Generate plot data ----
    plot_data_3 = generate_plot_data(
        func_med=func_med, reducer_med=reducer_med,
        initial_encoder_med=initial_encoder_med, encoder_med=encoder_med,
        noise_med=noise_med, models=models_1, dataloader=val_loader,
        dataset=dataset_val, t_dense=t_dense, global_mean_time=global_mean_time,
        global_mean_dose=global_mean_dose, global_mean=global_mean, global_std=global_std,
        latent_dim=latent_dim, dim_parameters=dim_parameters,
        initial_encoder=initial_encoder_1, encoder=encoder_1, func=func_1,
        reducer=reducer_1, noise=noise_1, ODEWrapper=ODEWrapper,
        num_simulated_total=num_simulated_total,
        add_noise_to_prediction=add_noise_to_prediction,
        enable_ae=True, enable_vae=False,
        enable_onlymedian=onlymedian, normalization=normalization,
        truncation=truncation
    )

    plot_data_4 = generate_plot_data(
        func_med=func_med, reducer_med=reducer_med,
        initial_encoder_med=initial_encoder_med, encoder_med=encoder_med,
        noise_med=noise_med, models=models_2, dataloader=val_loader,
        dataset=dataset_val, t_dense=t_dense, global_mean_time=global_mean_time,
        global_mean_dose=global_mean_dose, global_mean=global_mean, global_std=global_std,
        latent_dim=latent_dim, dim_parameters=dim_parameters,
        initial_encoder=initial_encoder_2, encoder=encoder_2, func=func_2,
        reducer=reducer_2, noise=noise_2, ODEWrapper=ODEWrapper,
        num_simulated_total=num_simulated_total,
        add_noise_to_prediction=add_noise_to_prediction,
        enable_ae=False, enable_vae=True,
        enable_onlymedian=False, normalization=normalization,
        truncation=truncation
    )

  
    # ---- Figure sizing ----
    fig_width = 100
    golden_ratio = 1.62
    fig_height = fig_width / golden_ratio

    # ---- Lines and fonts ----
    lw_sim, lw_raw, markersize_sim = 14, 14, 22
    fontsize_title = 120
    fontsize_ticks = 100
    fontsize_labels = 120
    fontsize_legend = 100

    # ---- Datasets and titles ----
    datasets = [plot_data_3, plot_data_4]  # AE, VAE
    titles = ["AE", "VAE"]

    # ---- Create 2x5 grid ----
    n_doses = len(plot_data_3)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = GridSpec(2, n_doses, figure=fig, wspace=0.1, hspace=0.1)

    for row_idx, data_set in enumerate(datasets):  # row: model
        for col_idx, data in enumerate(data_set):  # column: dose
            ax = fig.add_subplot(gs[row_idx, col_idx])

            # Colors
            sim_color = 'red' if row_idx == 0 else 'blue'
            raw_color = 'orange'

            # Simulated / predicted
            ax.plot(data["time_hours"], data["perc10_sim"], '--', color=sim_color, linewidth=lw_sim)
            ax.plot(data["time_hours"], data["median_sim"], '-', color=sim_color, markersize=markersize_sim, linewidth=lw_sim)
            ax.plot(data["time_hours"], data["perc90_sim"], '--', color=sim_color, linewidth=lw_sim)

            # Raw data
            ax.plot(data["time_hours_data"], data["perc10_data"], '--', color=raw_color, linewidth=lw_raw)
            ax.plot(data["time_hours_data"], data["median_data"], '-', color=raw_color, linewidth=lw_raw)
            ax.plot(data["time_hours_data"], data["perc90_data"], '--', color=raw_color, linewidth=lw_raw)

            # Titles and limits
            ax.set_title(f"{titles[row_idx]}, Dose {data['dose_value']:.0f}", fontsize=fontsize_title)
            ax.set_xlim(0, 24)
            ax.set_ylim(0, 220)
            ax.tick_params(axis='both', labelsize=fontsize_ticks)
            ax.grid(True)

    # Shared labels
    fig.text(0.5, -0.05, "Time", ha='center', fontsize=fontsize_labels)
    fig.text(-0.05, 0.5, "Concentration", va='center', rotation='vertical', fontsize=fontsize_labels)

    # Legend
    legend_elements = [
        Line2D([0], [0], color='blue', linestyle='--', linewidth=lw_sim, label='Prediction 10th'),
        Line2D([0], [0], color='blue', linestyle='-', linewidth=lw_sim, label='Prediction median'),
        Line2D([0], [0], color='blue', linestyle='--', linewidth=lw_sim, label='Prediction 90th'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=lw_raw, label='Reconstruction 10th'),
        Line2D([0], [0], color='red', linestyle='-', linewidth=lw_raw, label='Reconstruction median'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=lw_raw, label='Reconstruction 90th'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=lw_raw, label='Data 10th'),
        Line2D([0], [0], color='orange', linestyle='-', linewidth=lw_raw, label='Data median'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=lw_raw, label='Data 90th'),
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=fontsize_legend,
               bbox_to_anchor=(0.5, -0.2), markerscale=3)

    plt.savefig("vpc_5x2_grid.png", dpi=300)
    plt.show()

                   




          
           

def vpc(func_med,
        reducer_med,
        encoder_med,
        noise_med,
        models,
        dataloader,
        global_mean_dose,
        global_mean_time,
        global_mean, 
        global_std,
        dataset,
        encoder,
        func,
        reducer,
        noise,
        ODEWrapper,
        t_dense,
          onlymedian,
        enable_ae,
        enable_vae,
        add_noise_to_prediction,  
        num_simulated_total,
        normalization,
        truncation
       ):
    
    plot_data = generate_plot_data(
        func_med,
        reducer_med,
        encoder_med,
        noise_med,
        models=models,
        dataloader=dataloader,
        dataset=dataset,
        t_dense=t_dense,
        global_mean_time=global_mean_time,
        global_mean_dose=global_mean_dose,
        global_mean=global_mean,
        global_std=global_std,
        encoder=encoder,
        func=func,
        reducer=reducer,
        noise=noise,
        ODEWrapper=ODEWrapper,
        num_simulated_total=num_simulated_total,
        add_noise_to_prediction=add_noise_to_prediction,
        enable_ae=enable_ae,
        enable_vae=enable_vae,
        enable_onlymedian=onlymedian,
        normalization=normalization,
        truncation=truncation
    )
    
    # ---- Plot all in a grid ---- #
    n_plots = len(plot_data)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 8 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten to 1D for easy iteration
    
    for i, data in enumerate(plot_data):
        ax = axes[i]
        
        ax.plot(data["time_hours"], data["perc10_sim"], label="Simulated 10th", color="blue", linestyle="--", linewidth=2)
        ax.plot(data["time_hours"], data["median_sim"], label="Simulated median", color="blue", marker="o", markersize=4, linewidth=2)
        ax.plot(data["time_hours"], data["perc90_sim"], label="Simulated 90th", color="blue", linestyle="--", linewidth=2)
        
        ax.plot(data["time_hours_data"], data["perc10_data"], label="Raw 10th", color="orange", linestyle="--", linewidth=2)
        ax.plot(data["time_hours_data"], data["median_data"], label="Raw median", color="orange", linewidth=2)
        ax.plot(data["time_hours_data"], data["perc90_data"], label="Raw 90th", color="orange", linestyle="--", linewidth=2)
        
        ax.set_xlim(0, 48)
        # ax.set_ylim(0, 180)
        ax.set_title(f"Treatment {data['treatment']}", fontsize=32, fontweight='bold')
        ax.set_xlabel("Time (hours)", fontsize=24)
        ax.set_ylabel(f"Tumor Volume", fontsize=28)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=24, width=1.5)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Add a single legend below all plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        fontsize=26,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at the bottom for legend
    plt.show()
    
            




    
def vpc_true(
    ODEWrapper,
    models, dataset, t_dense, global_max_time, global_mean, global_std, global_mean_dose,
    encoder, func, reducer, noise,
    func_med, reducer_med, encoder_med, noise_med,
    add_noise_to_prediction=False,
    enable_onlymedian=True, enable_ae=False,
    enable_vae=False, truncation=1, num_repeats=100,
    use_ema_models=False,
    fontsize=14,
    show_confidence_intervals=True   # toggle all CI shading
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

    device = next(func.parameters()).device
    encoder.eval()
    func.eval()
    reducer.eval()
    collate_fn = make_collate_fn(dataset.global_max_len)

    dataloader = DataLoader(dataset, batch_size=12, shuffle=False, collate_fn=collate_fn)

    # ============================================================
    # --- Run generate_plot_data num_repeats times and collect ---
    # ============================================================
    all_repeats_data = []
    for r in range(num_repeats):
        plot_data_r = generate_plot_data(
            func_med=func_med, reducer_med=reducer_med, encoder_med=encoder_med, noise_med=noise_med,
            models=models, dataloader=dataloader, dataset=dataset, t_dense=t_dense,
            global_mean_time=global_max_time, global_mean_dose=global_mean_dose,
            global_mean=global_mean, global_std=global_std,encoder=encoder, func=func, reducer=reducer, noise=noise,
            ODEWrapper=ODEWrapper, num_simulated_total=1,
            add_noise_to_prediction=add_noise_to_prediction,
            enable_onlymedian=enable_onlymedian, enable_ae=enable_ae,
            enable_vae=enable_vae, normalization=True, truncation=truncation,
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
    n_cols = 2
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
        ax.set_ylabel("Concentration", fontsize=22)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=18, width=1.5)

    # --- Remove unused axes ---
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # --- Legend ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        fontsize=22,
        frameon=False,
        bbox_to_anchor=(0.5, -0.12)
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.show()



    
    
        

    

    
    
    


        
        
    

