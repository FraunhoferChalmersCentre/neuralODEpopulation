# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 07:10:20 2025

@author: Baaz
"""

import math
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import pandas as pd


from scipy.stats import norm

from sklearn.metrics import  r2_score

from lib.utils.utils_preprocess import (
    destandardize_concentration,
    collate_fn,
)
from lib.utils.utils_shared import (
    encode_latent,
    preprocess_batch,
    prepare_ode_input,
    make_predictions
)

def estimate_coverage(
    models,
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
        with use_ema(initial_encoder) if use_ema_models else contextmanager(lambda: (yield))():
            x0, ode_func, _, _ = prepare_ode_input(
                initial_encoder,
                x_padded_exp,
                k_param_flat,
                func,
                dose_tensor_exp,
                dose_times_list_exp,
                enable_ae
            )

        # Make predictions
        with use_ema(func) if use_ema_models else contextmanager(lambda: (yield))():
            pred_interp, pred_batch = make_predictions(
                t_padded_exp, t_dense, x0, ode_func, reducer,
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

        t_dense_np = t_dense.detach().cpu().numpy()
        t_encoder_np = t_encoder.squeeze(0).detach().cpu().numpy()
        x_encoder_np = destandardize_concentration(
            x_encoder.squeeze(0).detach().cpu().numpy(), global_mean, global_std
        )

        # Compute coverage for this individual
        if len(t_encoder_np) > 0:
            lower_interp = np.interp(t_encoder_np, t_dense_np, pred_lower)
            upper_interp = np.interp(t_encoder_np, t_dense_np, pred_upper)

            inside_mask = (x_encoder_np >= lower_interp) & (x_encoder_np <= upper_interp)
            n_inside = inside_mask.sum()
            n_total = len(x_encoder_np)

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



from contextlib import contextmanager


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





def plot_individual_fits(
    models,
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
    use_ema_models=False
):
    encoder = models['encoder']
    initial_encoder = models['initial_encoder']
    func = models['func']
    reducer = models['reducer']
    noise = models['noise']

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    fig, axes = plt.subplots(nr_row, nr_col, figsize=(5*nr_col, 4*nr_row))
    axes = axes.flatten() if nr_row*nr_col > 1 else [axes]

    encoder.eval()
    initial_encoder.eval()
    func.eval()

    # === Collect predictions for population median ===
    population_preds = []
    coverage_list = []      # coverage per individual
    point_inside_list = []  # per-point coverage for overall calculation
    point_total_list = []   # total points

    for i, data in enumerate(dataloader):
        if i >= max_plots:
            break

        # --- Preprocess batch ---
        id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list = preprocess_batch(
            data, device, truncation=truncation
        )

        latent_dim = encoder.latent_dim if hasattr(encoder, 'latent_dim') else 2

        # Encode latent n_samples times
        k_param_samples_list = []
        with use_ema(encoder) if use_ema_models else contextmanager(lambda: (yield))():
            for _ in range(n_samples):
                k_param_samples, _, _,_ = encode_latent(
                    encoder, t_encoder, x_encoder,
                    enable_vae=enable_vae,
                    enable_ae=enable_ae,
                    enable_onlymedian=enable_onlymedian
                )
                k_param_samples_list.append(k_param_samples)

        k_param_samples = torch.stack(k_param_samples_list, dim=1)  # [1, n_samples, latent_dim]

        # Flatten batch and sample dimensions
        k_param_flat = k_param_samples.view(-1, latent_dim)
        x_padded_exp = x_padded.repeat(n_samples, 1)
        t_padded_exp = t_padded.repeat(n_samples, 1)

        # Expand dose tensors
        dose_tensor_exp = dose_tensor.repeat(n_samples)
        dose_times_list_exp = []
        for dt in dose_times_list:
            dose_times_list_exp.extend([dt]*n_samples)

        # Prepare ODE input
        with use_ema(initial_encoder) if use_ema_models else contextmanager(lambda: (yield))():
            x0, ode_func,_,_ = prepare_ode_input(
                initial_encoder,
                x_padded_exp,
                k_param_flat,
                func,
                dose_tensor_exp,
                dose_times_list_exp,
                enable_ae
            )
            

        # Make predictions
        with use_ema(func) if use_ema_models else contextmanager(lambda: (yield))():
            pred_interp, pred_batch = make_predictions(
                t_padded_exp, t_dense, x0, ode_func, reducer,
                latent_dim=latent_dim,
                global_mean=global_mean,
                global_std=global_std
            )

        # Add noise and clamp
        
        # --- (Rest of plotting and coverage code unchanged) ---
        population_preds.append(pred_batch.detach().cpu())
        
        
        
       # pred_batch=destandardize_concentration(pred_batch, global_mean, global_std )
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
        
            # Exclude the first point from coverage calculations
            inside_mask = (x_encoder_np >= lower_interp) & (x_encoder_np <= upper_interp)
            n_inside = inside_mask.sum()
            n_total = len(x_encoder_np)
            coverage = n_inside / n_total if n_total > 0 else float('nan')
        
            # Keep track for overall coverage
            point_inside_list.append(n_inside)
            point_total_list.append(n_total)
        else:
            coverage = float('nan')


        coverage_list.append(coverage)

        ax.plot(t_encoder_np, x_encoder_np, 'o', color='blue', label='Observed')
        if len(t_cut_np) > 0:
            ax.plot(t_cut_np, x_cut_np, 'o', color='red', label='Removed')
        ax.plot(t_dense_np, pred_median, '-', color='green', label='Predicted median')
        ax.fill_between(t_dense_np, pred_lower, pred_upper, color='green', alpha=0.3, label='95% CI')
        ax.set_title(f'Ind {i} | Coverage: {coverage:.1%}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()

    # --- Population median and coverage summary (unchanged) ---
    if population_preds:
        population_preds = torch.cat(population_preds, dim=0)
        population_median = population_preds.median(dim=0).values.numpy()
        for ax in axes[:i+1]:
            ax.plot(t_dense_np, population_median, '--', color='black', label='Population median')
            ax.legend()

    plt.tight_layout()
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

def compute_residuals(latent_dim, global_mean, global_std, func, encoder, reducer, initial_encoder, noise, dataloader, t_dense, device, truncation):
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
    
    residuals_list = []
    predictions_list = []
    targets_list = []
    times_list = []

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch
            occ_list, t_padded, x_global_padded, mask, dose_tensor, dose_times_list = batch
    
            # Preprocess
            id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list = preprocess_batch(
                batch, device, truncation=truncation
            )
    
            # Encode latent
            k_param, mu_q, logvar_q = encode_latent(
                encoder, t_encoder, x_encoder, enable_vae=False, enable_ae=True, enable_onlymedian=False
            )
    
            # Prepare ODE input
            x0, ode_func = prepare_ode_input(
                initial_encoder,
                x_padded,
                k_param,
                func,
                dose_tensor,
                dose_times_list,
                True
            )
    
            # Make predictions
            pred_interp, pred_batch = make_predictions(
                t_padded, t_dense, x0, ode_func, reducer, latent_dim, global_mean, global_std
            )
    
            # Compute residuals
            targets = destandardize_concentration(x_padded, global_mean, global_std)
            residuals = targets - pred_interp
    
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
    pdf_vals = norm.pdf(x_vals, loc=0.0, scale=sigma_add)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Histogram + PDF
    axes[0].hist(residuals_np, bins=25, density=True, alpha=0.7, label="Residuals")
    axes[0].plot(x_vals, pdf_vals, 'r--', label=f'Gaussian Noise σ={sigma_add:.3f}')
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Histogram of residuals")
    axes[0].legend()
    
    # Residuals vs time
    axes[1].scatter(times_np, residuals_np, alpha=0.5, s=5)
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

def plot_encoder_augmentation(
    encoder_med, initial_encoder_med, func_med,
    t_dense, reducer_med, global_mean, global_std,
    dataset, encoder, latent_dim,
    n_individuals=10, n_augment=10, noise_std=1.0, device=None, truncation=None, normalization=True
):
    """
    Randomly samples individuals from dataset, generates augmented time series, 
    encodes with encoder, and plots μ vs σ scatter in 1x2 grid for each latent dimension, colored by individual ID.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    if device is None:
        device = next(encoder.parameters()).device

    encoder.eval()

    # Sample n_individuals
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    sampled_individuals = []
    for i, data in enumerate(dataloader):
        if i >= n_individuals:
            break
        sampled_individuals.append(data)

    all_mu = []
    all_sigma = []
    all_ids = []

    with torch.no_grad():
        for ind_id, data in enumerate(sampled_individuals):
            # Preprocess single individual
            id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list = preprocess_batch(
                data, device, truncation=truncation
            )

            for _ in range(n_augment):
                # Add additive Gaussian noise
                x_aug = x_encoder + torch.randn_like(x_encoder) * noise_std

                if normalization:
                    # Encode latent using median model
                    k_param, mu_q, logvar_q = encode_latent(
                        encoder_med,
                        t_encoder,
                        x_aug,
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

                    # Normalize encoder input
                    x_aug = destandardize_concentration(x_aug, global_mean, global_std) / pred_interp

                # Encode latent
                k_param, mu_q, logvar_q = encode_latent(
                    encoder, t_encoder, x_aug,
                    enable_vae=False, enable_ae=False, enable_onlymedian=False
                )

                mu_array = mu_q.cpu().numpy()[0]      # [latent_dim]
                sigma_array = torch.exp(0.5 * logvar_q).cpu().numpy()[0]

                all_mu.append(mu_array)
                all_sigma.append(sigma_array)
                all_ids.append(ind_id)

    all_mu = np.array(all_mu)
    all_sigma = np.array(all_sigma)
    all_ids = np.array(all_ids)

    # 1x2 grid: plot μ vs σ for dim0 and dim1
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes = axes.flatten()
    labels = ["Dimension 0", "Dimension 1"]
    cmap = plt.get_cmap("tab10")

    for i, ax in enumerate(axes):
        for ind_id in range(n_individuals):
            idx = all_ids == ind_id
            ax.scatter(all_mu[idx, i], all_sigma[idx, i], alpha=0.7, label=f"ID {ind_id}", color=cmap(ind_id % 10))
        ax.set_xlabel("μ", fontsize=16)
        ax.set_ylabel("σ", fontsize=16)
        ax.set_title(f"{labels[i]}: μ vs σ", fontsize=18)  # <-- corrected here
        ax.grid(True, linestyle="--", alpha=0.5)
    
    

def plot_encoder_histograms(
    encoder_med, initial_encoder_med, func_med,
    t_dense, reducer_med, global_mean, global_std,
    dataset, encoder, latent_dim, device=None, truncation=None, normalization=True
):
    """
    Plots:
    1. Histograms of encoder outputs (mu and sigma) in a 2x2 grid
    2. Pairwise correlation scatter plots of (mu0, mu1, sigma0, sigma1) in a 2x2 grid
    Uses EMA weights if available (restores original weights after plotting).
    """
    if device is None:
        device = next(encoder.parameters()).device

    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)

    encoder.eval()
    with torch.no_grad():
        for data in dataloader:
            id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list = preprocess_batch(
                data, device, truncation=truncation
            )

            # --- Normalization step ---
            if normalization:
                with use_ema(encoder_med):
                    k_param, mu_q, logvar_q = encode_latent(
                        encoder_med, t_encoder, x_encoder,
                        enable_vae=False, enable_ae=False, enable_onlymedian=True
                    )

                with use_ema(initial_encoder_med):
                    x0, ode_func = prepare_ode_input(
                        initial_encoder_med, x_padded, k_param,
                        func_med, dose_tensor, dose_times_list, False
                    )

                with use_ema(func_med):
                    pred_interp, pred_batch = make_predictions(
                        t_encoder, t_dense, x0, ode_func,
                        reducer_med, latent_dim, global_mean, global_std
                    )

                x_encoder = destandardize_concentration(x_encoder, global_mean, global_std) / pred_interp

            # --- Final encoding with EMA encoder ---
            with use_ema(encoder):
                _, mu_q, logvar_q, _ = encode_latent(
                    encoder, t_encoder, x_encoder,
                    enable_vae=True, enable_ae=False, enable_onlymedian=False
                )

            mu_array = mu_q.cpu().numpy()
            sigma_array = torch.exp(0.5 * logvar_q).cpu().numpy()
            dose_array = dose_tensor.cpu().numpy()

    # === 1. Histogram plots ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    labels = ["μ dim 0", "μ dim 1", "σ dim 0", "σ dim 1"]
    data_arrays = [mu_array[:, 0], mu_array[:, 1], sigma_array[:, 0], sigma_array[:, 1]]

    for ax, data, label in zip(axes, data_arrays, labels):
        for dose in np.unique(dose_array):
            idx = dose_array == dose
            ax.hist(data[idx], bins=30, alpha=0.5, label=f"Dose {dose}")
        ax.set_title(label, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[0].legend(title="Dose", fontsize=12, title_fontsize=14)
    plt.tight_layout()
    plt.show()

    # === 2. Correlation scatter plots (2x2) ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    pairs = [
        (mu_array[:, 0], mu_array[:, 1], "μ₀", "μ₁"),
        (mu_array[:, 0], sigma_array[:, 0], "μ₀", "σ₀"),
        (mu_array[:, 1], sigma_array[:, 1], "μ₁", "σ₁"),
        (sigma_array[:, 0], sigma_array[:, 1], "σ₀", "σ₁"),
    ]

    for ax, (x, y, xlab, ylab) in zip(axes.flatten(), pairs):
        for dose in np.unique(dose_array):
            idx = dose_array == dose
            ax.scatter(x[idx], y[idx], alpha=0.6, label=f"Dose {dose}")
        ax.set_xlabel(xlab, fontsize=16)
        ax.set_ylabel(ylab, fontsize=16)
        ax.set_title(f"{xlab} vs {ylab}", fontsize=18)
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[0, 0].legend(title="Dose", fontsize=12, title_fontsize=14)
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
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)

        with torch.no_grad():
            for data in dataloader:
                id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list = preprocess_batch(
                    data, device, truncation=truncation
                )
                
                if normalization:
                    
                    
                    k_param, mu_q, logvar_q = encode_latent(
                        encoder_med,
                        t_encoder,
                        x_encoder,
                        enable_vae=False,
                        enable_ae=False,
                        enable_onlymedian=True
                    )

                    # Prepare ODE input
                    x0, ode_func = prepare_ode_input(
                        initial_encoder_med,
                        x_padded,
                        k_param,
                        func_med,
                        dose_tensor,
                        dose_times_list,
                        False
                    )

                    # Make predictions
                    pred_interp, pred_batch = make_predictions(
                        t_encoder, t_dense, x0, ode_func, reducer_med, latent_dim, global_mean, global_std
                    )

                    # Destandardize encoder input
                    x_encoder = destandardize_concentration(x_encoder, global_mean, global_std) / pred_interp

                # Encode latent
                k_param, mu_q, logvar_q = encode_latent(
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
        from sklearn.ensemble import RandomForestRegressor

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

def plot_single_model_encoders_and_regression(
    df_train, df_val, dataset_train, dataset_val,
    encoder1, initial_encoder1, func1, reducer1,
    encoder_med, initial_encoder_med, func_med, reducer_med,
    latent_dim, global_mean, global_std, t_dense,
    device=None, truncation=0, dim_parameter_encoder=2,
    use_ema_models=False
):
    """
    Plot regression and encoder outputs using both μ and σ for regression,
    optionally using EMA weights for the encoders.
    """
    import gc
    import torch
    from torch.utils.data import DataLoader
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    import numpy as np
    from contextlib import contextmanager

    gc.collect()
    torch.cuda.empty_cache()
    if device is None:
        device = next(encoder1.parameters()).device

    # ---------------- Extract latents helper ----------------
    def extract_latents(df, dataset, encoder, dim_parameter_encoder):
        mus, sigmas, ka_list, cl_list, id_list_all = [], [], [], [], []

        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)

        # Use your existing EMA context manager
        with use_ema(encoder) if use_ema_models else contextmanager(lambda: (yield))():
            encoder.eval()
            with torch.no_grad():
                for data in dataloader:
                    id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list = preprocess_batch(
                        data, device, truncation=truncation
                    )

                    # Encode latent
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
    ax0.set_title("log(ka) regression (EMA)" if use_ema_models else "log(ka) regression")
    ax0.text(0.05, 0.9, f"R² = {r2_ka:.2f}", transform=ax0.transAxes,
             fontsize=14, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    # Predicted vs True log(CL)
    ax1.scatter(cl_pred, log_cl_val, c="gray", s=marker_size, alpha=0.7, edgecolors="k")
    ax1.plot([cl_pred.min(), cl_pred.max()], [cl_pred.min(), cl_pred.max()], "r--")
    ax1.set_xlabel("Predicted log(CL)")
    ax1.set_ylabel("True log(CL)")
    ax1.set_title("log(CL) regression (EMA)" if use_ema_models else "log(CL) regression")
    ax1.text(0.05, 0.9, f"R² = {r2_cl:.2f}", transform=ax1.transAxes,
             fontsize=14, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

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
    import gc
    import torch
    from torch.utils.data import DataLoader
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    import numpy as np
    from contextlib import contextmanager

    gc.collect()
    torch.cuda.empty_cache()
    if device is None:
        device = next(encoder1.parameters()).device

    # ---------------- Extract latents helper ----------------
    def extract_latents(df, dataset, encoder, dim_parameter_encoder):
        mus, sigmas, ka_list, cl_list, id_list_all = [], [], [], [], []

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


 
def plot_single_model_encoders_and_regression(
    df_train, df_val, dataset_train, dataset_val,
    encoder1, initial_encoder1, func1, reducer1,
    encoder_med, initial_encoder_med, func_med, reducer_med,
    latent_dim, global_mean, global_std, t_dense,
    device, truncation, dim_parameter_encoder=2
):
    """
    Plot regression and encoder outputs for a single model.
    Includes regressions from both encoder means (μ) and stds (σ).
    Temporarily applies EMA weights if available.
    """
    gc.collect()
    torch.cuda.empty_cache()
    if device is None:
        device = next(encoder1.parameters()).device

    # ---------------- Apply EMA if available ----------------
    ema_modules = [
        encoder1, initial_encoder1, func1, reducer1,
        encoder_med, initial_encoder_med, func_med, reducer_med
    ]
    applied_ema = []
    for module in ema_modules:
        if hasattr(module, "apply_ema_weights"):
            module.apply_ema_weights()
            applied_ema.append(module)

    # ---------------- Extract latents helper ----------------
    def extract_latents(
        encoder_med, initial_encoder_med, func_med, reducer_med,
        df, dataset, encoder, init_enc, func, reducer
    ):
        mus, sigmas, ka_list, cl_list, id_list_all = [], [], [], [], []
        encoder_med.eval()
        initial_encoder_med.eval()
        func_med.eval()
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)

        with torch.no_grad():
            for data in dataloader:
                id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list = preprocess_batch(
                    data, device, truncation=truncation
                )

                # Encode latent
                k_param, mu_q, logvar_q, _ = encode_latent(
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

        return np.vstack(mus), np.vstack(sigmas), np.array(ka_list), np.array(cl_list), id_list_all

    # ---------------- Prepare plots ----------------
    fs = 100
    fig, axes = plt.subplots(2, 4, figsize=(fs, fs/1.5))  # two rows: μ and σ

    fontsize_labels = 60
    fontsize_ticks = 50
    fontsize_legend = 50
    fontsize_r2 = 50

    # Extract latents for train/val
    X_val_mu, X_val_sigma, ka_val, cl_val, id_list_val = extract_latents(
        encoder_med, initial_encoder_med, func_med, reducer_med,
        df_val, dataset_val,
        encoder1, initial_encoder1, func1, reducer1
    )
    X_train_mu, X_train_sigma, ka_train, cl_train, _ = extract_latents(
        encoder_med, initial_encoder_med, func_med, reducer_med,
        df_train, dataset_train,
        encoder1, initial_encoder1, func1, reducer1
    )

    # --- Fit regressors ---
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    def fit_and_predict(X_train, y_train, X_val, y_val):
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train, np.log(y_train))
        y_pred = rf.predict(X_val)
        return y_pred, r2_score(np.log(y_val), y_pred)

    # μ regressors
    ka_pred_mu, r2_ka_mu = fit_and_predict(X_train_mu, ka_train, X_val_mu, ka_val)
    cl_pred_mu, r2_cl_mu = fit_and_predict(X_train_mu, cl_train, X_val_mu, cl_val)

    # σ regressors
    ka_pred_sigma, r2_ka_sigma = fit_and_predict(X_train_sigma, ka_train, X_val_sigma, ka_val)
    cl_pred_sigma, r2_cl_sigma = fit_and_predict(X_train_sigma, cl_train, X_val_sigma, cl_val)

    log_ka_val = np.log(ka_val)
    log_cl_val = np.log(cl_val)

    # -------- Eight plots (2x4) --------
    marker_size = 200
    line_width = 8
    cmap = plt.get_cmap("tab10")

    # Row 0 = μ plots
    ax0, ax1, ax2, ax3 = axes[0]
    ax0.scatter(X_val_mu[:, 0], log_ka_val, c=cmap(0), s=marker_size, alpha=0.7, edgecolors="k", label="μ0")
    ax0.scatter(X_val_mu[:, 1], log_ka_val, c=cmap(1), s=marker_size, alpha=0.7, edgecolors="k", label="μ1")
    ax0.set_xlabel("Latent μ", fontsize=fontsize_labels)
    ax0.set_ylabel("log(ka)", fontsize=fontsize_labels)
    ax0.set_title("μ vs ka", fontsize=fontsize_labels)
    ax0.tick_params(axis="both", labelsize=fontsize_ticks)

    ax1.scatter(X_val_mu[:, 0], log_cl_val, c=cmap(0), s=marker_size, alpha=0.7, edgecolors="k", label="μ0")
    ax1.scatter(X_val_mu[:, 1], log_cl_val, c=cmap(1), s=marker_size, alpha=0.7, edgecolors="k", label="μ1")
    ax1.set_xlabel("Latent μ", fontsize=fontsize_labels)
    ax1.set_ylabel("log(CL)", fontsize=fontsize_labels)
    ax1.set_title("μ vs CL", fontsize=fontsize_labels)
    ax1.tick_params(axis="both", labelsize=fontsize_ticks)

    ax2.scatter(ka_pred_mu, log_ka_val, c="gray", s=marker_size, alpha=0.7, edgecolors="k")
    ax2.plot([ka_pred_mu.min(), ka_pred_mu.max()], [ka_pred_mu.min(), ka_pred_mu.max()], "r--", lw=line_width)
    ax2.set_xlabel("Pred log(ka)", fontsize=fontsize_labels)
    ax2.set_ylabel("True log(ka)", fontsize=fontsize_labels)
    ax2.text(0.05, 0.9, f"R² = {r2_ka_mu:.2f}", transform=ax2.transAxes,
             fontsize=fontsize_r2, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    ax3.scatter(cl_pred_mu, log_cl_val, c="gray", s=marker_size, alpha=0.7, edgecolors="k")
    ax3.plot([cl_pred_mu.min(), cl_pred_mu.max()], [cl_pred_mu.min(), cl_pred_mu.max()], "r--", lw=line_width)
    ax3.set_xlabel("Pred log(CL)", fontsize=fontsize_labels)
    ax3.set_ylabel("True log(CL)", fontsize=fontsize_labels)
    ax3.text(0.05, 0.9, f"R² = {r2_cl_mu:.2f}", transform=ax3.transAxes,
             fontsize=fontsize_r2, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    # Row 1 = σ plots
    ax4, ax5, ax6, ax7 = axes[1]
    ax4.scatter(X_val_sigma[:, 0], log_ka_val, c=cmap(2), s=marker_size, alpha=0.7, edgecolors="k", label="σ0")
    ax4.scatter(X_val_sigma[:, 1], log_ka_val, c=cmap(3), s=marker_size, alpha=0.7, edgecolors="k", label="σ1")
    ax4.set_xlabel("Latent σ", fontsize=fontsize_labels)
    ax4.set_ylabel("log(ka)", fontsize=fontsize_labels)
    ax4.set_title("σ vs ka", fontsize=fontsize_labels)
    ax4.tick_params(axis="both", labelsize=fontsize_ticks)

    ax5.scatter(X_val_sigma[:, 0], log_cl_val, c=cmap(2), s=marker_size, alpha=0.7, edgecolors="k", label="σ0")
    ax5.scatter(X_val_sigma[:, 1], log_cl_val, c=cmap(3), s=marker_size, alpha=0.7, edgecolors="k", label="σ1")
    ax5.set_xlabel("Latent σ", fontsize=fontsize_labels)
    ax5.set_ylabel("log(CL)", fontsize=fontsize_labels)
    ax5.set_title("σ vs CL", fontsize=fontsize_labels)
    ax5.tick_params(axis="both", labelsize=fontsize_ticks)

    ax6.scatter(ka_pred_sigma, log_ka_val, c="gray", s=marker_size, alpha=0.7, edgecolors="k")
    ax6.plot([ka_pred_sigma.min(), ka_pred_sigma.max()], [ka_pred_sigma.min(), ka_pred_sigma.max()], "r--", lw=line_width)
    ax6.set_xlabel("Pred log(ka)", fontsize=fontsize_labels)
    ax6.set_ylabel("True log(ka)", fontsize=fontsize_labels)
    ax6.text(0.05, 0.9, f"R² = {r2_ka_sigma:.2f}", transform=ax6.transAxes,
             fontsize=fontsize_r2, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    ax7.scatter(cl_pred_sigma, log_cl_val, c="gray", s=marker_size, alpha=0.7, edgecolors="k")
    ax7.plot([cl_pred_sigma.min(), cl_pred_sigma.max()], [cl_pred_sigma.min(), cl_pred_sigma.max()], "r--", lw=line_width)
    ax7.set_xlabel("Pred log(CL)", fontsize=fontsize_labels)
    ax7.set_ylabel("True log(CL)", fontsize=fontsize_labels)
    ax7.text(0.05, 0.9, f"R² = {r2_cl_sigma:.2f}", transform=ax7.transAxes,
             fontsize=fontsize_r2, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    # Add legends
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=4, fontsize=fontsize_legend, title="Encoders", title_fontsize=fontsize_legend)

    plt.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.05)
    plt.show()
    plt.close(fig)

    # ---------------- Restore original weights ----------------
    for module in applied_ema:
        if hasattr(module, "restore_original_weights"):
            module.restore_original_weights()

    torch.cuda.empty_cache()
    gc.collect()

   


def generate_plot_data(
    func_med, reducer_med, initial_encoder_med, encoder_med, noise_med,
    models, dataloader, dataset, t_dense, global_mean_time, global_mean_dose,
    global_mean, global_std, latent_dim, dim_parameters,
    initial_encoder, encoder, func, reducer, noise,
    ODEWrapper, num_simulated_total, add_noise_to_prediction,
    enable_onlymedian, enable_ae, enable_vae, normalization, truncation,
    use_ema_models=True
):
    import torch, gc, numpy as np
    from torch.nn.utils.rnn import pad_sequence

    plot_data_local = []
    device = next(func.parameters()).device

    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()

        # Wrap EMA if requested
        ema_contexts = []
        if use_ema_models:
            ema_contexts.extend([
                use_ema(encoder),
                use_ema(initial_encoder),
                use_ema(func),
                use_ema(reducer),
                use_ema(encoder_med),
                use_ema(initial_encoder_med),
                use_ema(func_med),
                use_ema(reducer_med)
            ])
        else:
            # Dummy context manager that does nothing
            @contextmanager
            def dummy_cm(): yield
            ema_contexts.extend([dummy_cm() for _ in range(8)])

        # Nest all EMA contexts
        from contextlib import ExitStack
        with ExitStack() as stack:
            for ctx in ema_contexts:
                stack.enter_context(ctx)

            # Set models to eval
            encoder.eval(); initial_encoder.eval()
            reducer.eval(); func.eval()
            encoder_med.eval(); initial_encoder_med.eval()
            reducer_med.eval(); func_med.eval()

            # Collect unique doses
            unique_doses = sorted(set(entry['amt'].item() for entry in dataset))
            num_doses = len(unique_doses)
            num_simulated_per_dose = max(1, num_simulated_total // num_doses)

            for dose_value in unique_doses:
                dose_filtered_dataset = [entry for entry in dataset if entry['amt'].item() == dose_value]
                if len(dose_filtered_dataset) == 0:
                    continue

                indices = np.random.choice(len(dose_filtered_dataset), num_simulated_per_dose, replace=True)
                batch_entries = [dose_filtered_dataset[i] for i in indices]

                max_len = max(len(entry['t']) for entry in batch_entries)
                masks = torch.zeros((len(batch_entries), max_len), dtype=torch.bool)
                for i, entry in enumerate(batch_entries):
                    masks[i, :len(entry['t'])] = 1

                batch = (
                    [entry['subject_id'] for entry in batch_entries],
                    pad_sequence([entry['t'] for entry in batch_entries], batch_first=True),
                    pad_sequence([entry['x_global'] for entry in batch_entries], batch_first=True),
                    masks,
                    torch.stack([entry['amt'] for entry in batch_entries]),
                    [entry['dose_times'] for entry in batch_entries]
                )

                # Preprocess batch
                id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list = preprocess_batch(
                    batch, device, truncation=truncation
                )

                # Encode latent
                k_param, mu_q, logvar_q,_ = encode_latent(
                    encoder, t_encoder, x_encoder,
                    enable_vae=enable_vae, enable_ae=enable_ae, enable_onlymedian=enable_onlymedian
                )

                # Prepare ODE input
                x0, ode_func,_,_ = prepare_ode_input(
                    initial_encoder, x_padded, k_param, func, dose_tensor, dose_times_list, enable_ae
                )
                

                # Make predictions
                pred_interp, pred_batch = make_predictions(
                    t_padded, t_dense, x0, ode_func, reducer, latent_dim, global_mean, global_std
                )

                if add_noise_to_prediction:
                    mask_pred = pred_batch > 0
                    pred_batch = torch.where(mask_pred, noise.sample(pred_batch, n_samples=1).squeeze(0), pred_batch)
                    pred_batch = torch.clamp(pred_batch, min=0)

                perc10_sim = torch.quantile(pred_batch, 0.10, dim=0)
                median_sim = torch.quantile(pred_batch, 0.50, dim=0)
                perc90_sim = torch.quantile(pred_batch, 0.90, dim=0)

                # Extract and destandardize data
                x_global_list = [entry['x_global'] for entry in dose_filtered_dataset]
                data_matrix = pad_sequence(x_global_list, batch_first=True)
                data_matrix_destd = destandardize_concentration(data_matrix, global_mean, global_std).T

                perc10_list, median_list, perc90_list = [], [], []
                for row in data_matrix_destd:
                    non_zero = row[row != 0]
                    if len(non_zero) > 0:
                        perc10_list.append(torch.quantile(non_zero, 0.10))
                        median_list.append(torch.quantile(non_zero, 0.50))
                        perc90_list.append(torch.quantile(non_zero, 0.90))
                    else:
                        perc10_list.append(torch.tensor(0., device=data_matrix_destd.device))
                        median_list.append(torch.tensor(0., device=data_matrix_destd.device))
                        perc90_list.append(torch.tensor(0., device=data_matrix_destd.device))

                plot_data_local.append({
                    "dose_value": dose_value * global_mean_dose,
                    "time_hours": t_dense.cpu().numpy() * global_mean_time,
                    "time_hours_data": torch.quantile(t_padded, 0.90, dim=0).cpu().numpy() * global_mean_time,
                    "perc10_sim": perc10_sim.detach().cpu().numpy(),
                    "median_sim": median_sim.detach().cpu().numpy(),
                    "perc90_sim": perc90_sim.detach().cpu().numpy(),
                    "perc10_data": torch.stack(perc10_list).detach().cpu().numpy(),
                    "median_data": torch.stack(median_list).detach().cpu().numpy(),
                    "perc90_data": torch.stack(perc90_list).detach().cpu().numpy()
                })

        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()

    return plot_data_local



def generate_dose_percentiles(
    dataset, t_dense, global_mean_time, global_mean_dose, global_mean, global_std,
    encoder_med, initial_encoder_med, func_med, reducer_med,
    encoder, initial_encoder, func, reducer,
    latent_dim, dim_parameters,
    enable_onlymedian, enable_ae, enable_vae,
    normalization, truncation,
    batch_size=32,
    save_csv_path="dose_data.csv"
):
    device = next(func.parameters()).device
    t_dense = t_dense.to(device)
    torch.cuda.empty_cache()
    gc.collect()

    # Set all models to eval
    for model in [encoder_med, initial_encoder_med, func_med, reducer_med,
                  encoder, initial_encoder, func, reducer]:
        model.eval()

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=None)  # adjust collate_fn if needed

    all_data = []
    plt.figure(figsize=(16, 8))

    # Collect unique doses across all batches
    unique_doses = set()
    for batch in dataloader:

        t_batch = batch['t'].to(device)
        x_batch = batch['x_global'].to(device)
        amt_batch = batch['amt'].to(device)
        dose_times_batch = batch['dose_times']
        x_dose_batch = batch['x_dose'].to(device)
        subject_id_batch = batch['subject_id']

        unique_doses.update(amt_batch.cpu().numpy().ravel())
    unique_doses = sorted(unique_doses)
    print("Unique doses:", unique_doses)
   
    # Loop over doses
    # Loop over doses
    for dose_value in unique_doses:
        batch_filtered_list = []
        for batch in dataloader:
            # Access batch as a dictionary
            t_batch = batch['t'].to(device)
            x_batch = batch['x_global'].to(device)
            amt_batch = batch['amt'].to(device)
            dose_times_batch = batch['dose_times']
            x_dose_batch = batch['x_dose'].to(device)
            subject_id_batch = batch['subject_id']
    
            mask = amt_batch == dose_value
            if mask.any():
                filtered_batch = {
                    "t": t_batch[mask],
                    "x_global": x_batch[mask],
                    "amt": amt_batch[mask],
                    "dose_times": [dose_times_batch[i] for i in range(len(dose_times_batch)) if mask[i]],
                    "x_dose": x_dose_batch[mask],
                    "subject_id": [subject_id_batch[i] for i in range(len(subject_id_batch)) if mask[i]]
                }
                batch_filtered_list.append(filtered_batch)

        if len(batch_filtered_list) == 0:
            continue

        # Concatenate filtered batches
        batch_filtered = {}
        for k in batch_filtered_list[0].keys():
            if isinstance(batch_filtered_list[0][k], list):
                batch_filtered[k] = sum([b[k] for b in batch_filtered_list], [])
            else:
                batch_filtered[k] = torch.cat([b[k] for b in batch_filtered_list], dim=0)

        # Pad sequences
        max_len = max([b.shape[0] for b in batch_filtered['t']]) if isinstance(batch_filtered['t'], list) else batch_filtered['t'].shape[1]
        masks = torch.zeros((len(batch_filtered['amt']), max_len), dtype=torch.bool, device=device)
        t_padded = pad_sequence(batch_filtered['t'], batch_first=True).to(device)
        x_padded = pad_sequence(batch_filtered['x_global'], batch_first=True).to(device)
        dose_tensor = batch_filtered['amt']
        x_dose = pad_sequence(batch_filtered['x_dose'], batch_first=True).to(device)

        batch = (
            batch_filtered['subject_id'],
            t_padded, x_padded, masks, dose_tensor,
            batch_filtered['dose_times'], x_dose, torch.zeros(len(batch_filtered['amt']), device=device)
        )

        # Preprocess
        _, _, _, t_encoder, x_encoder, _, _, _, dose_tensor, dose_times_list, _ = preprocess_batch(
            batch, device, truncation=truncation
        )

        x_encoder_raw = x_encoder.clone().detach()

        # Normalize if needed
        if normalization:
            k_param, _, _, _ = encode_latent(
                encoder_med, t_encoder, x_encoder,
                enable_vae=False, enable_ae=False, enable_onlymedian=True
            )
            x0, ode_func = prepare_ode_input(
                initial_encoder_med, x_padded, k_param, func_med,
                dose_tensor, dose_times_list, False
            )
            pred_used, _ = make_predictions(
                t_padded, t_dense, x0, ode_func, reducer_med, latent_dim, global_mean, global_std
            )
            x_encoder = destandardize_concentration(x_encoder, global_mean, global_std)
            x_encoder = x_encoder.detach()
        else:
            pred_used = torch.zeros_like(x_encoder_raw)

        # Save time tensor
        t_encoder_cpu = t_encoder.clone().detach().cpu()

        # Append to dataframe list
        for subj_idx in range(x_encoder.shape[0]):
            for time_idx in range(x_encoder.shape[1]):
                all_data.append({
                    "subject_id": batch[0][subj_idx],
                    "dose": dose_value * global_mean_dose,
                    "time": t_encoder_cpu[subj_idx, time_idx].item(),
                    "x_encoder_raw": x_encoder_raw[subj_idx, time_idx].item(),
                    "x_encoder_normalized": x_encoder[subj_idx, time_idx].item(),
                    "prediction_used": pred_used[subj_idx, time_idx].item()
                })

        # Compute percentiles
        perc10 = torch.quantile(x_encoder, 0.10, dim=0).detach().cpu().numpy()
        median = torch.quantile(x_encoder, 0.50, dim=0).detach().cpu().numpy()
        perc90 = torch.quantile(x_encoder, 0.90, dim=0).detach().cpu().numpy()
        t_encoder_plot = torch.quantile(t_encoder, 0.5, dim=0).detach().cpu().numpy()
        
        dose_label = f"{dose_value * global_mean_dose} mg"
        plt.plot(t_encoder_plot, perc10, linestyle='--', label=f"10th {dose_label}")
        plt.plot(t_encoder_plot, median, linestyle='-', label=f"50th {dose_label}")
        plt.plot(t_encoder_plot, perc90, linestyle='--', label=f"90th {dose_label}")

        # Cleanup
        del x_encoder, t_encoder, x_padded, t_padded, masks, dose_tensor, x_dose
        torch.cuda.empty_cache()
        gc.collect()

    plt.xlabel("Time index")
    plt.ylabel("Normalized x_encoder")
    plt.title("Percentiles of x_encoder per Dose")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Save CSV
    df = pd.DataFrame(all_data)
    df.to_csv(save_csv_path, index=False)

    torch.cuda.empty_cache()
    gc.collect()






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

        
            
    
       
   




          
           
           
def vpc(func_med,
       reducer_med,
       initial_encoder_med,
       encoder_med,
       noise_med,models, dataloader, global_mean_dose, global_mean_time,  global_mean, 
   global_std,
          dataset, latent_dim,
       dim_parameters, initial_encoder,encoder, func, reducer, noise, ODEWrapper,
     t_dense,compartment,onlymedian, enable_ae, enable_vae, add_noise_to_prediction,  
     num_simulated_total,normalization, truncation
 ):
         plot_data = generate_plot_data(func_med,
               reducer_med,
               initial_encoder_med,
               encoder_med,
               noise_med,models=models, dataloader=dataloader,
           dataset=dataset,
           t_dense=t_dense,
           global_mean_time=global_mean_time,
           global_mean_dose=global_mean_dose,
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
           enable_ae=enable_ae,
           enable_vae=enable_vae,
             enable_onlymedian=onlymedian,
           normalization=normalization,
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
         
             ax.plot(data["time_hours_data"], data["perc10_data"], label="Raw 10th", color="orange", linestyle="--")
             ax.plot(data["time_hours_data"], data["median_data"], label="Raw median", color="orange")
             ax.plot(data["time_hours_data"], data["perc90_data"], label="Raw 90th", color="orange", linestyle="--")
         
             ax.set_xlim(0, global_mean_time)
            # ax.set_ylim(0, 180)
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


def vpc_true(func_med,
             reducer_med,
             initial_encoder_med,
             encoder_med,
             noise_med,
             models,
             dataloader,
             dataset,
             t_dense,
             global_mean_time,
             global_mean_dose,
             global_mean,
             global_std,
             latent_dim,
             dim_parameters,
             initial_encoder,
             encoder,
             func,
             reducer,
             noise,
             ODEWrapper,
             compartment,
             onlymedian,
             enable_ae,
             enable_vae,
             add_noise_to_prediction,
             num_simulated_total,
             normalization,
             truncation=1,
             num_studies=100):

    all_study_medians = []

    for study_idx in range(num_studies):
        # Generate a single simulated study
        plot_data = generate_plot_data(
            func_med, reducer_med, initial_encoder_med, encoder_med, noise_med,
            models=models,
            dataloader=dataloader,
            dataset=dataset,
            t_dense=t_dense,
            global_mean_time=global_mean_time,
            global_mean_dose=global_mean_dose,
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
            enable_ae=enable_ae,
            enable_vae=enable_vae,
            enable_onlymedian=onlymedian,
            normalization=normalization,
            truncation=truncation
        )

        # Extract median per time point
        for dose_data in plot_data:
            all_study_medians.append(torch.tensor(dose_data['median_sim']))

    all_study_medians = torch.stack(all_study_medians)  # shape: [num_studies * doses, num_time_points]

    # Median of medians across studies
    median_of_medians = torch.median(all_study_medians, dim=0).values

    # 95% CI of the median using percentiles
    lower_95 = torch.quantile(all_study_medians, 0.025, dim=0)
    upper_95 = torch.quantile(all_study_medians, 0.975, dim=0)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.fill_between(t_dense.cpu().numpy() * global_mean_time,
                     lower_95.cpu().numpy(),
                     upper_95.cpu().numpy(),
                     color='blue', alpha=0.2, label='95% CI')
    plt.plot(t_dense.cpu().numpy() * global_mean_time,
             median_of_medians.cpu().numpy(),
             color='blue', marker='o', label='Median of medians')

    # Raw data
    for dose_value in sorted(set(entry['amt'].item() for entry in dataset)):
        dose_filtered_dataset = [entry for entry in dataset if entry['amt'].item() == dose_value]
        interp_all = [
            torch_linear_interpolate2(entry['t'].to(t_dense.device),
                                      entry['x_global'].to(t_dense.device),
                                      t_dense)
            for entry in dose_filtered_dataset
        ]
        data_matrix = torch.stack(interp_all)
        median_data = torch.quantile(data_matrix, 0.5, dim=0)
        plt.plot(t_dense.cpu().numpy() * global_mean_time,
                 destandardize_concentration(median_data, global_mean, global_std).cpu().numpy(),
                 color='orange', label='Raw median' if dose_value == 0 else None)

    plt.xlabel("Time (hours)")
    plt.ylabel(f"Concentration ({compartment})")
    plt.grid(True)
    plt.legend()
    plt.show()


