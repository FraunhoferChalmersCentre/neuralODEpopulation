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

from scipy.stats import norm

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, classification_report

from lib.utils.utils_preprocess import standardize_concentration, destandardize_concentration, collate_fn
from lib.utils.utils_shared import encode_latent, preprocess_batch, prepare_ode_input, make_predictions, ODEWrapper


def plot_individual_fits(
    latent_dim, t_dense, models, dataloader, device,
    global_mean, global_std, global_max_time,
    enable_nf, enable_ae, enable_onlymedian,
    truncation=0, max_plots=25, nr_row=5, nr_col=5, n_samples=2, ci_lower=0.05, ci_upper=0.95
):
    """
    Plots individual predictions with median and confidence intervals for a trained model.
    Also shows truncated (kept) vs removed time points in different colors.
    """
    encoder = models['encoder']
    initial_encoder = models['initial_encoder']
    func = models['func']
    reducer = models['reducer']
    noise = models.get('noise', None)

    fig, axes = plt.subplots(nr_row, nr_col, figsize=(5*nr_col, 4*nr_row))
    axes = axes.flatten() if nr_row*nr_col > 1 else [axes]
    #dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    plotted = 0

    for i, data in enumerate(dataloader):
        if i >= max_plots:
            break

        # Preprocess batch
        id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list, auc_tensor = preprocess_batch(
            data, device, truncation=truncation
        )

        # Encode latent
        k_param, mu_q, logvar_q, log_det = encode_latent(
            encoder,
            t_encoder,
            x_encoder,
            enable_nf=enable_nf,
            enable_ae=enable_ae,
            enable_onlymedian=enable_onlymedian
        )

        latent_dim = mu_q.size(1)
        eps = torch.randn(n_samples, latent_dim, device=mu_q.device)
        std_q = torch.exp(0.5 * logvar_q)
        k_param = mu_q + 0*std_q * eps

        # Expand tensors for ODE
        x_padded_exp = x_padded.expand(n_samples, *x_padded.shape[1:])
        t_padded_exp = t_padded.expand(n_samples, *t_padded.shape[1:])
        dose_tensor_exp = dose_tensor.expand(n_samples, *dose_tensor.shape[1:])
        x_encoder_exp = x_encoder.expand(n_samples, *x_encoder.shape[1:])
        
      
        
        
        # Prepare ODE input
        x0, ode_func = prepare_ode_input(
            initial_encoder,
            x_padded,
            k_param,
            func,
            dose_tensor,
            dose_times_list,
            enable_ae
        )
       
        # Make predictions
        
        pred_interp, pred_batch = make_predictions(
            t_padded_exp, t_dense, x0, ode_func, reducer, latent_dim, global_mean, global_std
        )

        # Denormalize for plotting
        t_encoder_np = t_encoder.squeeze(0).detach().cpu().numpy()
        x_encoder_np = (x_encoder.squeeze(0).detach().cpu().numpy() * global_std + global_mean)

        t_cut_np = t_cut.squeeze(0).detach().cpu().numpy()
        x_cut_np = (x_cut.squeeze(0).detach().cpu().numpy() * global_std + global_mean)

        # Prediction statistics
        pred_median = pred_batch.median(dim=0).values.detach().cpu().numpy()
        pred_lower = pred_batch.kthvalue(int(ci_lower*pred_batch.size(0)), dim=0).values.detach().cpu().numpy()
        pred_upper = pred_batch.kthvalue(int(ci_upper*pred_batch.size(0)), dim=0).values.detach().cpu().numpy()
        t_dense_np = t_dense.detach().cpu().numpy()

        # Plot
        ax = axes[i]
        # Kept/truncated values
        ax.plot(t_encoder_np, x_encoder_np, 'o', color='blue', label='Observed (used)')
        # Removed/cut values
        if len(t_cut_np) > 0:
            ax.plot(t_cut_np, x_cut_np, 'o', color='red', label='Removed')
        # Predictions
        ax.plot(t_dense_np, pred_median, '-', color='green', label='Predicted median')
        ax.fill_between(t_dense_np, pred_lower, pred_upper, color='green', alpha=0.3, label='95% CI')

        ax.set_title(f'Individual {i}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
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
            k_param, mu_q, logvar_q, log_det = encode_latent(
                encoder, t_encoder, x_encoder, enable_nf=False, enable_ae=True, enable_onlymedian=False
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


def plot_encoder_vs_parameters(df, dataset, encoder, encoder_med, initial_encoder_med, func_med,
                                          latent_dim, global_mean, global_std, t_dense, reducer_med,
                                          device, enable_nf, enable_ae, enable_onlymedian,normalization,
                                          truncation):
    if device is None:
        device = next(encoder.parameters()).device

    # Full dataset as one batch
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)

    encoder.eval()
    encoder_med.eval()
    initial_encoder_med.eval()
    func_med.eval()

    with torch.no_grad():
        for data in dataloader:
            # Preprocess
            id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list, auc = preprocess_batch(
                data, device, truncation=truncation
            )
            if normalization:
                # Encode with "med" encoder
                k_param, mu_q_med, logvar_q, log_det = encode_latent(
                    encoder_med, t_encoder, x_encoder,
                    enable_nf=False, enable_ae=False, enable_onlymedian=True
                )
    
               
                x0, ode_func = prepare_ode_input(
                    initial_encoder_med, x_padded, k_param, func_med, dose_tensor, dose_times_list, enable_ae
                )
    
                pred_interp, pred_batch = make_predictions(
                    t_padded, t_dense, x0, ode_func, reducer_med, latent_dim, global_mean, global_std
                )
    
                # Standardize for main encoder
                x_encoder = destandardize_concentration(x_encoder, global_mean, global_std) / pred_interp.detach()

            # Encode latent with main encoder
            k_param, mu_q, logvar_q, log_det = encode_latent(
                encoder, t_encoder, standardize_concentration(x_encoder, global_mean, global_std),
                enable_nf=enable_nf, enable_ae=enable_ae, enable_onlymedian=enable_onlymedian
            )

            mu_array = mu_q.cpu().numpy()[:, :2]  # first 2 latent dims

    # Lookup parameters for all IDs
    ka_array = np.array([df.loc[df["ID"] == int(sid), "ka"].values[0] for sid in id_list])
    cl_array = np.array([df.loc[df["ID"] == int(sid), "cl"].values[0] for sid in id_list])
    dose_array = np.array([df.loc[df["ID"] == int(sid), "AMT"].values[0] for sid in id_list])
    log_ka_array = np.log(ka_array)
    log_cl_array = np.log(cl_array)
    # Color mapping
    unique_doses = np.unique(dose_array)
    cmap = plt.get_cmap("tab10")
    dose_to_color = {d: cmap(i % 10) for i, d in enumerate(unique_doses)}
    colors = [dose_to_color[d] for d in dose_array]

    # 2x2 grid
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Panel 1: log(ka)
    axes[0].scatter(mu_array[:, 0], log_ka_array, c=colors, alpha=0.7, edgecolors='k', label='μ0')
    axes[0].scatter(mu_array[:, 1], log_ka_array, c=colors, alpha=0.7, edgecolors='k', marker='x', label='μ1')
    axes[0].set_xlabel("Latent μ", fontsize=14)
    axes[0].set_ylabel("log(ka)", fontsize=14)
    axes[0].set_title("Latent μ vs log(ka)", fontsize=16)
    axes[0].legend(fontsize=12)
    
    # Panel 2: log(CL)
    axes[1].scatter(mu_array[:, 0], log_cl_array, c=colors, alpha=0.7, edgecolors='k', label='μ0')
    axes[1].scatter(mu_array[:, 1], log_cl_array, c=colors, alpha=0.7, edgecolors='k', marker='x', label='μ1')
    axes[1].set_xlabel("Latent μ", fontsize=14)
    axes[1].set_ylabel("log(CL)", fontsize=14)
    axes[1].set_title("Latent μ vs log(CL)", fontsize=16)
    axes[1].legend(fontsize=12)
    
    # Legend for doses
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(d),
                          markerfacecolor=cmap(i % 10), markersize=8)
               for i, d in enumerate(unique_doses)]
    fig.legend(handles, [str(d) for d in unique_doses], title="Dose", loc="upper right")
    
    plt.tight_layout()
    plt.show()

 
def linear_regression_log_params_from_encoder_validation(
        df_train, dataset_train, df_val, dataset_val,
        encoder, encoder_med, initial_encoder_med, func_med,
        latent_dim, global_mean, global_std, t_dense, reducer_med,
        device=None, enable_nf=False, enable_ae=False, enable_onlymedian=False,
        truncation=1,normalization=False, dim_parameter_encoder=2):
    """
    Linear regression to predict log(ka) and log(CL) from encoder latent variables (μ0, μ1)
    using the same preprocessing and encoding pipeline as before.

    Parameters
    ----------
    df_train, df_val : pd.DataFrame
        DataFrames containing 'ID', 'ka', 'CL', 'AMT'.
    dataset_train, dataset_val : Dataset
        TrajectoryDataset instances.
    encoder, encoder_med, initial_encoder_med, func_med : torch.nn.Module
        Trained models for encoding and ODE.
    latent_dim : int
        Total latent space dimensionality.
    global_mean, global_std : np.array
        For destandardization.
    t_dense : np.array
        Dense time points for predictions.
    reducer_med : callable
        Function for reducing ODE outputs.
    device : torch.device
    enable_nf, enable_ae, enable_onlymedian : bool
    truncation : int
    dim_parameter_encoder : int
        Number of latent dimensions to extract (default 2)
    """

    if device is None:
        device = next(encoder.parameters()).device

    def extract_features(df, dataset):
        mus, ka_list, cl_list = [], [], []

        encoder.eval()
        encoder_med.eval()
        initial_encoder_med.eval()
        func_med.eval()

        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)

        with torch.no_grad():
            for data in dataloader:
                # Preprocess batch
                id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list, auc = preprocess_batch(
                    data, device, truncation=truncation
                )
                if normalization:
               
                    k_param, mu_q, logvar_q, log_det = encode_latent(
                        encoder_med,
                        t_encoder,
                        x_encoder,
                        enable_nf=False,
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
                        enable_ae
                    )
    
                    # Make predictions
                    pred_interp, pred_batch = make_predictions(
                        t_padded, t_dense, x0, ode_func, reducer_med, latent_dim, global_mean, global_std
                    )
    
                    # Destandardize encoder input
                    x_encoder = destandardize_concentration(x_encoder, global_mean, global_std) / pred_interp.detach()

                # Encode latent (final encoder)
                k_param, mu_q, logvar_q, log_det = encode_latent(
                    encoder,
                    t_encoder,
                    standardize_concentration(x_encoder, global_mean, global_std),
                    enable_nf=enable_nf,
                    enable_ae=enable_ae,
                    enable_onlymedian=enable_onlymedian
                )

                # Collect μ0, μ1
                for i, subj_id in enumerate(id_list):
                    mu = mu_q[i].cpu().numpy()[:dim_parameter_encoder]
                    mus.append(mu)
                    
                    row = df[df["ID"] == int(subj_id)]
                    if row.empty:
                        continue
                    ka_list.append(row["ka"].values[0])
                    cl_list.append(row["cl"].values[0])

        X = np.vstack(mus)
        y_ka = np.array(ka_list)
        y_cl = np.array(cl_list)

        return X, y_ka, y_cl

    # Extract features
    X_train, y_ka_train, y_cl_train = extract_features(df_train, dataset_train)
    X_val, y_ka_val, y_cl_val = extract_features(df_val, dataset_val)

    # Log-transform
    epsilon = 1e-8
    log_y_ka_train = np.log(y_ka_train + epsilon)
    log_y_cl_train = np.log(y_cl_train + epsilon)
    log_y_ka_val = np.log(y_ka_val + epsilon)
    log_y_cl_val = np.log(y_cl_val + epsilon)

    # Fit linear regression
    lr_ka = LinearRegression().fit(X_train, log_y_ka_train)
    lr_cl = LinearRegression().fit(X_train, log_y_cl_train)

    # Predict
    y_ka_pred = lr_ka.predict(X_val)
    y_cl_pred = lr_cl.predict(X_val)

    r2_ka = r2_score(log_y_ka_val, y_ka_pred)
    r2_cl = r2_score(log_y_cl_val, y_cl_pred)

    # ------------------ Plot ------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Linear regression ka
    axes[0].scatter(y_ka_pred, log_y_ka_val, c='blue', alpha=0.7, edgecolors='k')
    axes[0].plot([y_ka_pred.min(), y_ka_pred.max()],
                 [y_ka_pred.min(), y_ka_pred.max()], 'r--')
    axes[0].set_xlabel("Predicted log(ka)")
    axes[0].set_ylabel("log(True ka)")
    axes[0].set_title(f"Linear Regression ka (R²={r2_ka:.2f})")
    
    # Linear regression CL
    axes[1].scatter(y_cl_pred, log_y_cl_val, c='green', alpha=0.7, edgecolors='k')
    axes[1].plot([y_cl_pred.min(), y_cl_pred.max()],
                 [y_cl_pred.min(), y_cl_pred.max()], 'r--')
    axes[1].set_xlabel("Predicted log(CL)")
    axes[1].set_ylabel("log(True CL)")
    axes[1].set_title(f"Linear Regression CL (R²={r2_cl:.2f})")
    
    plt.tight_layout()
    plt.show()

def plot_encoder_and_regression(df_train, df_val, dataset_train, dataset_val,
                                encoder, encoder_med, initial_encoder_med, func_med,
                                latent_dim, global_mean, global_std, t_dense, reducer_med,
                                device, enable_nf, enable_ae, enable_onlymedian,
                                truncation,normalization, dim_parameter_encoder=2):


    if device is None:
        device = next(encoder.parameters()).device

    # ---------------- Extract latent variables for validation ----------------
    def extract_latents(df, dataset):
        mus, ka_list, cl_list, id_list_all = [], [], [], []
        encoder.eval()
        encoder_med.eval()
        initial_encoder_med.eval()
        func_med.eval()
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)

        with torch.no_grad():
            for data in dataloader:
                id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list, auc = preprocess_batch(
                    data, device, truncation=truncation
                )
                
                if normalization:
                    # Encode latent (median encoder)
                    k_param, mu_q, logvar_q, log_det = encode_latent(
                        encoder_med, t_encoder, x_encoder,
                        enable_nf=False, enable_ae=False, enable_onlymedian=True
                    )
       
    
                    # Prepare ODE input and predictions
                    x0, ode_func = prepare_ode_input(
                        initial_encoder_med, x_padded, k_param, func_med, dose_tensor, dose_times_list, enable_ae
                    )
                    pred_interp, pred_batch = make_predictions(
                        t_padded, t_dense, x0, ode_func, reducer_med, latent_dim, global_mean, global_std
                    )
    
                    # Encode latent with main encoder
                x_encoder = destandardize_concentration(x_encoder, global_mean, global_std) / torch.clamp(pred_interp, min=1.0).median()
                   
                k_param, mu_q, logvar_q, log_det = encode_latent(
                    encoder, t_encoder, x_encoder,
                    enable_nf=enable_nf, enable_ae=enable_ae, enable_onlymedian=enable_onlymedian
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

    # Validation features
    X_val, ka_val, cl_val, id_list_val = extract_latents(df_val, dataset_val)

    # Train regression on training set
    X_train, ka_train, cl_train, _ = extract_latents(df_train, dataset_train)
    lr_ka = LinearRegression().fit(X_train, np.log(ka_train))
    lr_cl = LinearRegression().fit(X_train, np.log(cl_train))

    ka_pred = lr_ka.predict(X_val)
    cl_pred = lr_cl.predict(X_val)

    r2_ka = r2_score(np.log(ka_val), ka_pred)
    r2_cl = r2_score(np.log(cl_val), cl_pred)

    # ------------------ Color mapping ------------------
    dose_array = np.array([df_val.loc[df_val["ID"] == int(sid), "AMT"].values[0] for sid in id_list_val])
    unique_doses = np.unique(dose_array)
    cmap = plt.get_cmap("tab10")
    dose_to_color = {d: cmap(i % 10) for i, d in enumerate(unique_doses)}
    colors = [dose_to_color[d] for d in dose_array]

    log_ka_val = np.log(ka_val)
    log_cl_val = np.log(cl_val)

    # ------------------ 2x2 Grid Plot ------------------
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))  # 1 row, 4 columns

    fontsize_labels = 16
    fontsize_ticks = 14
    fontsize_legend = 14
    fontsize_r2 = 14
    
    # Leftmost: μ0/μ1 vs log(ka)
    axes[0].scatter(X_val[:,0], log_ka_val, c=colors, alpha=0.7, edgecolors='k', label='μ0')
    axes[0].scatter(X_val[:,1], log_ka_val, c=colors, alpha=0.7, edgecolors='k', marker='x', label='μ1')
    axes[0].set_xlabel("Latent μ", fontsize=fontsize_labels)
    axes[0].set_ylabel("log(ka)", fontsize=fontsize_labels)
    axes[0].tick_params(axis='both', labelsize=fontsize_ticks)
    
    # Second: μ0/μ1 vs log(CL)
    axes[1].scatter(X_val[:,0], log_cl_val, c=colors, alpha=0.7, edgecolors='k', label='μ0')
    axes[1].scatter(X_val[:,1], log_cl_val, c=colors, alpha=0.7, edgecolors='k', marker='x', label='μ1')
    axes[1].set_xlabel("Latent μ", fontsize=fontsize_labels)
    axes[1].set_ylabel("log(ke)", fontsize=fontsize_labels)
    axes[1].tick_params(axis='both', labelsize=fontsize_ticks)
    
    # Third: regression ka (true on y-axis)
    axes[2].scatter(ka_pred, log_ka_val, c=colors, alpha=0.7, edgecolors='k')
    axes[2].plot([ka_pred.min(), ka_pred.max()],
                 [ka_pred.min(), ka_pred.max()], 'r--')
    axes[2].set_xlabel("Predicted log(ka)", fontsize=fontsize_labels)
    axes[2].set_ylabel("True log(ka)", fontsize=fontsize_labels)
    axes[2].tick_params(axis='both', labelsize=fontsize_ticks)
    axes[2].text(0.05, 0.9, f"R² = {r2_ka:.2f}", transform=axes[2].transAxes,
                 fontsize=fontsize_r2, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    
    # Fourth: regression CL (true on y-axis)
    axes[3].scatter(cl_pred, log_cl_val, c=colors, alpha=0.7, edgecolors='k')
    axes[3].plot([cl_pred.min(), cl_pred.max()],
                 [cl_pred.min(), cl_pred.max()], 'r--')
    axes[3].set_xlabel("Predicted log(ke)", fontsize=fontsize_labels)
    axes[3].set_ylabel("True log(ke)", fontsize=fontsize_labels)
    axes[3].tick_params(axis='both', labelsize=fontsize_ticks)
    axes[3].text(0.05, 0.9, f"R² = {r2_cl:.2f}", transform=axes[3].transAxes,
                 fontsize=fontsize_r2, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    
    # Unified legend in second panel
    from matplotlib.lines import Line2D
    marker_handles = [Line2D([0],[0], marker='o', color='w', label='μ0', markerfacecolor='gray', markersize=10, markeredgecolor='k'),
                      Line2D([0],[0], marker='x', color='w', label='μ1', markerfacecolor='gray', markersize=10, markeredgecolor='k')]
    dose_handles = [Line2D([0],[0], marker='o', color='w', label=str(d),
                            markerfacecolor=cmap(i%10), markersize=10) for i,d in enumerate(unique_doses)]
    
    axes[1].legend(handles=marker_handles + dose_handles, loc='upper left', bbox_to_anchor=(0.7, 1),
                   title="Latent & Dose", fontsize=fontsize_legend, title_fontsize=fontsize_legend)
    
    plt.tight_layout()
    plt.show()

    


def plot_two_models_encoders_and_regression(
    df_train, df_val, dataset_train, dataset_val,
    encoder1, initial_encoder1, func1, reducer1,
    encoder2, initial_encoder2, func2, reducer2,
    encoder_med, initial_encoder_med, func_med, reducer_med,
    latent_dim, global_mean, global_std, t_dense,
    device, enable_nf, enable_ae, enable_onlymedian, truncation, normalization,
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
    def extract_latents(df, dataset, encoder, init_enc, func, reducer, normalization=False):

        mus, sigmas, ka_list, cl_list, id_list_all = [], [], [], [], []
        encoder.eval()
        encoder_med.eval()
        initial_encoder_med.eval()
        func_med.eval()
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)

        with torch.no_grad():
            for data in dataloader:
                id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list, auc = preprocess_batch(
                    data, device, truncation=truncation
                )
                
                if normalization:
                    # Encode latent (median encoder)
                    k_param, _, _, _ = encode_latent(
                        encoder_med, t_encoder, x_encoder,
                        enable_nf=False, enable_ae=False, enable_onlymedian=True
                    )
            
                    # Prepare ODE input and predictions
                    x0, ode_func = prepare_ode_input(
                        initial_encoder_med, x_padded, k_param, func_med, dose_tensor, dose_times_list, False
                    )
                    pred_interp, pred_batch = make_predictions(
                        t_padded, t_dense, x0, ode_func, reducer_med, latent_dim, global_mean, global_std
                    )
    
                    # Normalize encoder inputs
                    pred_interp_clamped = torch.clamp(pred_interp, min=1.0)  # [num_subjects, seq_len]
                    subject_medians = pred_interp_clamped.median(dim=1, keepdim=True)[0]  # shape [num_subjects, 1]
                    
                    x_encoder = destandardize_concentration(x_encoder, global_mean, global_std) / subject_medians
                    
                # Encode latent with main encoder
                k_param, mu_q, logvar_q, log_det = encode_latent(
                    encoder, t_encoder, x_encoder,
                    enable_nf=enable_nf, enable_ae=enable_ae, enable_onlymedian=enable_onlymedian
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
        (encoder1, initial_encoder1, func1, reducer1, "Naïve", False),  # row 1
        (encoder2, initial_encoder2, func2, reducer2, "Baseline normalized", True),   # row 2
    ]
    
    for row_idx, (encoder, init_enc, func, reducer, model_name, normalization) in enumerate(models):
        # --- Extract latents (μ and σ) ---
        X_val, ka_val, cl_val, id_list_val = extract_latents(
            df_val, dataset_val, encoder, init_enc, func, reducer, normalization
        )
        X_train, ka_train, cl_train, _ = extract_latents(
            df_train, dataset_train, encoder, init_enc, func, reducer, normalization
        )
    
        # --- Fit regressors (using μ and σ) ---
        lr_ka = LinearRegression().fit(X_train, np.log(ka_train))
        lr_cl = LinearRegression().fit(X_train, np.log(cl_train))
        ka_pred = lr_ka.predict(X_val)
        cl_pred = lr_cl.predict(X_val)
        r2_ka = r2_score(np.log(ka_val), ka_pred)
        r2_cl = r2_score(np.log(cl_val), cl_pred)
    
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
    
    
    from matplotlib.lines import Line2D
    
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

    
    
def generate_plot_data(func_med,
       reducer_med,
       initial_encoder_med,
       encoder_med,
       noise_med,models, dataloader, dataset, t_dense, global_max_time, global_max_dose, global_mean, global_std,
                    latent_dim, dim_parameters, initial_encoder, encoder, func, reducer, noise,
                    ODEWrapper, num_simulated_total, add_noise_to_prediction,enable_onlymedian,
                    enable_ae, enable_nf, enable_vae,normalization, truncation=1):

               with torch.no_grad():
                     torch.cuda.empty_cache()
                     gc.collect()
                    # with ema_eval(models):
                      
                     encoder.eval()
                     initial_encoder.eval()
                     reducer.eval()
                     func.eval()
                     encoder_med.eval()
                     initial_encoder_med.eval()
                     reducer_med.eval()
                     func_med.eval()
                     device = next(func.parameters()).device
             
             
                     # Collect unique dose times
                     
            
                     unique_doses = sorted(set(entry['amt'].item() for entry in dataset))
                     
                     print(unique_doses)
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
                             auc_list = []
                             for entry in batch_entries:
                                 auc_val = torch.trapz(entry['x_global'], entry['t'])  # trapezoidal integration
                                 auc_list.append(auc_val)
                             auc_tensor = torch.stack(auc_list)  # shape [B]
                             batch = (
                                 [entry['subject_id'] for entry in batch_entries],                                  # id_list
                                 pad_sequence([entry['t'] for entry in batch_entries], batch_first=True),           # t_padded
                                 pad_sequence([entry['x_global'] for entry in batch_entries], batch_first=True),    # x_global_padded
                                 masks,                                                                             # mask
                                 torch.stack([entry['amt'] for entry in batch_entries]),                             # dose_tensor
                                 [entry['dose_times'] for entry in batch_entries],                                   # dose_times_list
                                 pad_sequence([entry['x_dose'] for entry in batch_entries], batch_first=True),
                                 auc_tensor                                                                        # x_dose_padded
                             )
                         
                     
                             # Preprocess
                             id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list, _ = preprocess_batch(
                                 batch, device, truncation=truncation
                             )
                             if normalization:
                                 k_param, mu_q, logvar_q, log_det = encode_latent(
                                     encoder_med,
                                     t_encoder,
                                     x_encoder,
                                     enable_nf=False,
                                     enable_ae=False,
                                     enable_onlymedian=True
                                 )
                                 
                                 
                                                               
                               
                                 
                                 x0, ode_func = prepare_ode_input(
                                     initial_encoder_med,
                                     x_padded,
                                     k_param,
                                     func_med,
                                     dose_tensor,
                                     dose_times_list,
                                     enable_ae
                                 )
                                 
                                 
                                 
                                 pred_interp, pred_batch = make_predictions(
                                     t_padded, t_dense, x0, ode_func, reducer_med, latent_dim,global_mean,global_std
                                 )
                                 
                               #  print(x_encoder)
                                 
                                 scaling = destandardize_concentration(x_encoder, global_mean, global_std) /  torch.clamp(pred_interp, min=1.0).median()
                                 x_encoder = scaling


                     
                             # Encode latent
                             k_param, mu_q, logvar_q, log_det = encode_latent(
                                 encoder, t_encoder, x_encoder, enable_nf=enable_nf, enable_ae=enable_ae, enable_onlymedian=enable_onlymedian
                             )
                            
                             
                             if enable_vae:
                                 eps = torch.randn(num_simulated_per_dose, dim_parameters, device=mu_q.device)
                                 k_param =  eps
                     
                       
                           
                             # Prepare ODE input
                             x0, ode_func = prepare_ode_input(
                                 initial_encoder,
                                 x_padded,
                                 k_param,
                                 func,
                                 dose_tensor,
                                 dose_times_list,
                                 enable_ae
                             )
                             
                          
                             # Make predictions
                             pred_interp, pred_batch = make_predictions(
                                 t_padded, t_dense, x0, ode_func, reducer, latent_dim, global_mean, global_std
                             )
                             
                        #     pred_batch=torch.exp(pred_batch)
                           #  print(pred_batch)
                         
                             if add_noise_to_prediction:
                                 mask = pred_batch > 0
                                 pred_batch = torch.where(mask, noise.sample(pred_batch, n_samples=1).squeeze(0), pred_batch)
                                 pred_batch = torch.clamp(pred_batch, min=0)
                 
                             perc10_sim = torch.quantile(pred_batch, 0.10, dim=0)  # shape [121]
                             median_sim = torch.quantile(pred_batch, 0.50, dim=0)
                             perc90_sim = torch.quantile(pred_batch, 0.90, dim=0)
                             del pred_batch, pred_interp  # free GPU memory
                             torch.cuda.empty_cache()

                                 
                     
                          #   print(median_sim)
                     
                             interp_all = [
                             torch_linear_interpolate2(entry['t'].to(t_dense.device),
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
             
                     # Clear cached memory
                     torch.cuda.empty_cache()    
                     # If you want to forcefully release all tensors
                     for obj in gc.get_objects():
                         try:
                             if torch.is_tensor(obj):
                                 del obj
                         except:
                             pass
                     
                   
                     gc.collect()
                     torch.cuda.empty_cache() 
                     
                     del encoder, initial_encoder, reducer, func
                     del encoder_med, initial_encoder_med, reducer_med, func_med
                     
                     return plot_data_local
           
def generate_dose_percentiles(
    dataset, t_dense, global_max_time, global_max_dose, global_mean, global_std,
    encoder_med, initial_encoder_med, func_med, reducer_med,
    encoder, initial_encoder, func, reducer,
    latent_dim, dim_parameters,
    enable_onlymedian=True, enable_ae=False, enable_nf=False,
    normalization=False, truncation=1
):
    import torch, gc
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.nn.utils.rnn import pad_sequence

    device = next(func.parameters()).device
    t_dense = t_dense.to(device)
    torch.cuda.empty_cache()
    gc.collect()

    # Set all models to eval
    for model in [encoder_med, initial_encoder_med, func_med, reducer_med,
                  encoder, initial_encoder, func, reducer]:
        model.eval()

    unique_doses = sorted({entry['amt'].item() for entry in dataset})

    plt.figure(figsize=(16, 8))

    for dose_value in unique_doses:
        # Filter full dataset for this dose
        dose_dataset = [entry for entry in dataset if entry['amt'].item() == dose_value]
        if len(dose_dataset) == 0:
            continue

        # Pad sequences and move to device
        max_len = max(len(entry['t']) for entry in dose_dataset)
        masks = torch.zeros((len(dose_dataset), max_len), dtype=torch.bool, device=device)
        t_padded = pad_sequence([entry['t'] for entry in dose_dataset], batch_first=True).to(device)
        x_padded = pad_sequence([entry['x_global'] for entry in dose_dataset], batch_first=True).to(device)
        dose_tensor = torch.stack([entry['amt'] for entry in dose_dataset]).to(device)
        x_dose = pad_sequence([entry['x_dose'] for entry in dose_dataset], batch_first=True).to(device)
        auc_tensor = torch.stack([torch.trapz(entry['x_global'], entry['t']) for entry in dose_dataset]).to(device)

        batch = (
            [entry['subject_id'] for entry in dose_dataset],
            t_padded, x_padded, masks, dose_tensor,
            [entry['dose_times'] for entry in dose_dataset],
            x_dose, auc_tensor
        )

        # Preprocess
        _, _, _, t_encoder, x_encoder, _, _, _, dose_tensor, dose_times_list, _ = preprocess_batch(
            batch, device, truncation=truncation
        )

        # Normalize if needed
        if normalization:
            k_param, _, _, _ = encode_latent(
                encoder_med, t_encoder, x_encoder,
                enable_nf=False, enable_ae=False, enable_onlymedian=True
            )
            x0_med, ode_func_med = prepare_ode_input(
                initial_encoder_med, t_encoder, k_param, func_med,
                dose_tensor, dose_times_list, enable_ae=False
            )
            pred_interp_med, _ = make_predictions(
                t_encoder, t_dense, x0_med, ode_func_med, reducer_med,
                latent_dim, global_mean, global_std
            )
            scaling = destandardize_concentration(x_encoder, global_mean, global_std) /  torch.clamp(pred_interp_med, min=1.0).median()
            x_encoder = scaling

            # Cleanup median encoder tensors
            del k_param, x0_med, ode_func_med, pred_interp_med
            torch.cuda.empty_cache()
            gc.collect()

        # Compute percentiles across subjects
        perc10 = torch.quantile(x_encoder, 0.10, dim=0).detach().cpu().numpy()
        median = torch.quantile(x_encoder, 0.50, dim=0).detach().cpu().numpy()
        perc90 = torch.quantile(x_encoder, 0.90, dim=0).detach().cpu().numpy()
        t_encoder_plot = torch.quantile(t_encoder, 0.5, dim=0).detach().cpu().numpy()
        indices_plot = np.arange(perc10.shape[0])

        dose_label = f"{dose_value * global_max_dose} mg"
        plt.plot(t_encoder_plot, perc10, linestyle='--', label=f"10th {dose_label}")
        plt.plot(t_encoder_plot, median, linestyle='-', label=f"50th {dose_label}")
        plt.plot(t_encoder_plot, perc90, linestyle='--', label=f"90th {dose_label}")

        # Cleanup per dose
        del x_encoder, t_encoder, x_padded, t_padded, masks, dose_tensor, x_dose, auc_tensor
        torch.cuda.empty_cache()
        gc.collect()

    plt.xlabel("Time index")
    plt.ylabel("Normalized x_encoder")
    plt.title("Percentiles of x_encoder per Dose")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Final cleanup
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
    global_max_time, global_max_dose, global_mean, global_std,
    compartment, ODEWrapper,
    add_noise_to_prediction, num_simulated_total,
    onlymedian, enable_nf, enable_ae, enable_vae, normalization,
    truncation
):
    fontsize_labels = 20
    fontsize_ticks = 15
    fontsize_legend = 15

    # ---- Generate plot data ----
    plot_data_1 = generate_plot_data(func_med=func_med, reducer_med=reducer_med,
                                     initial_encoder_med=initial_encoder_med, encoder_med=encoder_med,
                                     noise_med=noise_med, models=models_1, dataloader=dataloader,
                                     dataset=dataset, t_dense=t_dense, global_max_time=global_max_time,
                                     global_max_dose=global_max_dose, global_mean=global_mean, global_std=global_std,
                                     latent_dim=latent_dim, dim_parameters=dim_parameters,
                                     initial_encoder=initial_encoder_1, encoder=encoder_1, func=func_1,
                                     reducer=reducer_1, noise=noise_1, ODEWrapper=ODEWrapper,
                                     num_simulated_total=num_simulated_total,
                                     add_noise_to_prediction=add_noise_to_prediction,
                                     enable_ae=True, enable_nf=enable_nf, enable_vae=False,
                                     enable_onlymedian=onlymedian, normalization=normalization,
                                     truncation=truncation)

    plot_data_2 = generate_plot_data(func_med=func_med, reducer_med=reducer_med,
                                     initial_encoder_med=initial_encoder_med, encoder_med=encoder_med,
                                     noise_med=noise_med, models=models_2, dataloader=dataloader,
                                     dataset=dataset, t_dense=t_dense, global_max_time=global_max_time,
                                     global_max_dose=global_max_dose, global_mean=global_mean, global_std=global_std,
                                     latent_dim=latent_dim, dim_parameters=dim_parameters,
                                     initial_encoder=initial_encoder_2, encoder=encoder_2, func=func_2,
                                     reducer=reducer_2, noise=noise_2, ODEWrapper=ODEWrapper,
                                     num_simulated_total=num_simulated_total,
                                     add_noise_to_prediction=add_noise_to_prediction,
                                     enable_ae=enable_ae, enable_nf=enable_nf, enable_vae=enable_vae,
                                     enable_onlymedian=onlymedian, normalization=normalization,
                                     truncation=truncation)

    plot_data_3 = generate_plot_data(func_med=func_med, reducer_med=reducer_med,
                                     initial_encoder_med=initial_encoder_med, encoder_med=encoder_med,
                                     noise_med=noise_med, models=models_1, dataloader=val_loader,
                                     dataset=dataset_val, t_dense=t_dense, global_max_time=global_max_time,
                                     global_max_dose=global_max_dose, global_mean=global_mean, global_std=global_std,
                                     latent_dim=latent_dim, dim_parameters=dim_parameters,
                                     initial_encoder=initial_encoder_1, encoder=encoder_1, func=func_1,
                                     reducer=reducer_1, noise=noise_1, ODEWrapper=ODEWrapper,
                                     num_simulated_total=num_simulated_total,
                                     add_noise_to_prediction=add_noise_to_prediction,
                                     enable_ae=True, enable_nf=enable_nf, enable_vae=False,
                                     enable_onlymedian=onlymedian, normalization=normalization,
                                     truncation=truncation)

    plot_data_4 = generate_plot_data(func_med=func_med, reducer_med=reducer_med,
                                     initial_encoder_med=initial_encoder_med, encoder_med=encoder_med,
                                     noise_med=noise_med, models=models_2, dataloader=val_loader,
                                     dataset=dataset_val, t_dense=t_dense, global_max_time=global_max_time,
                                     global_max_dose=global_max_dose, global_mean=global_mean, global_std=global_std,
                                     latent_dim=latent_dim, dim_parameters=dim_parameters,
                                     initial_encoder=initial_encoder_2, encoder=encoder_2, func=func_2,
                                     reducer=reducer_2, noise=noise_2, ODEWrapper=ODEWrapper,
                                     num_simulated_total=num_simulated_total,
                                     add_noise_to_prediction=add_noise_to_prediction,
                                     enable_ae=enable_ae, enable_nf=enable_nf, enable_vae=enable_vae,
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
            ax.plot(data["time_hours"], data["perc10_data"], '--', color=raw_color, linewidth=lw_raw)
            ax.plot(data["time_hours"], data["median_data"], '-', color=raw_color, linewidth=lw_raw)
            ax.plot(data["time_hours"], data["perc90_data"], '--', color=raw_color, linewidth=lw_raw)
            
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
       noise_med,models, dataloader, global_max_dose, global_max_time,  global_mean, 
   global_std,
          dataset, latent_dim,
       dim_parameters, initial_encoder,encoder, func, reducer, noise, ODEWrapper,
     t_dense,compartment,onlymedian, enable_nf, enable_ae, enable_vae, add_noise_to_prediction,  
     num_simulated_total,normalization, truncation
 ):
         plot_data = generate_plot_data(func_med,
               reducer_med,
               initial_encoder_med,
               encoder_med,
               noise_med,models=models, dataloader=dataloader,
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
           enable_ae=enable_ae,
           enable_nf=enable_nf,
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
         
             ax.plot(data["time_hours"], data["perc10_data"], label="Raw 10th", color="orange", linestyle="--")
             ax.plot(data["time_hours"], data["median_data"], label="Raw median", color="orange")
             ax.plot(data["time_hours"], data["perc90_data"], label="Raw 90th", color="orange", linestyle="--")
         
             ax.set_xlim(0, global_max_time)
             ax.set_ylim(-10, 180)
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
             ODEWrapper,
             compartment,
             onlymedian,
             enable_nf,
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
            enable_ae=enable_ae,
            enable_nf=enable_nf,
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
    plt.fill_between(t_dense.cpu().numpy() * global_max_time,
                     lower_95.cpu().numpy(),
                     upper_95.cpu().numpy(),
                     color='blue', alpha=0.2, label='95% CI')
    plt.plot(t_dense.cpu().numpy() * global_max_time,
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
        plt.plot(t_dense.cpu().numpy() * global_max_time,
                 destandardize_concentration(median_data, global_mean, global_std).cpu().numpy(),
                 color='orange', label='Raw median' if dose_value == 0 else None)

    plt.xlabel("Time (hours)")
    plt.ylabel(f"Concentration ({compartment})")
    plt.grid(True)
    plt.legend()
    plt.show()




def build_dose_predictors_from_datasets(
    dataset_train, dataset_val,
    encoder_vae, initial_encoder_vae, reducer_vae, func_vae,
    encoder_med, initial_encoder_med, reducer_med, func_med,
    global_max_time, global_max_dose, global_mean, global_std,
    latent_dim, dim_parameter_encoder, t_dense,
    normalization=True, truncation=1, device="cuda"
):
    import torch, gc
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.nn.utils.rnn import pad_sequence
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, r2_score

    device = device
    t_dense = t_dense.to(device)

    def process_dataset(dataset, do_plot=False):
        # Set all models to eval
        for model in [encoder_vae, initial_encoder_vae, func_vae, reducer_vae,
                      encoder_med, initial_encoder_med, func_med, reducer_med]:
            model.eval()

        unique_doses = sorted({entry['amt'].item() for entry in dataset})
        all_x_encoder = []
        all_t_encoder = []
        all_dose_tensor = []

        plt.figure(figsize=(16, 8)) if do_plot else None

        for dose_value in unique_doses:
            dose_dataset = [entry for entry in dataset if entry['amt'].item() == dose_value]
            if len(dose_dataset) == 0:
                continue

            # Pad sequences
            max_len = max(len(entry['t']) for entry in dose_dataset)
            masks = torch.zeros((len(dose_dataset), max_len), dtype=torch.bool, device=device)
            t_padded = pad_sequence([entry['t'] for entry in dose_dataset], batch_first=True).to(device)
            x_padded = pad_sequence([entry['x_global'] for entry in dose_dataset], batch_first=True).to(device)
            dose_tensor_batch = torch.stack([entry['amt'] for entry in dose_dataset]).to(device)
            x_dose = pad_sequence([entry['x_dose'] for entry in dose_dataset], batch_first=True).to(device)
            auc_tensor = torch.stack([torch.trapz(entry['x_global'], entry['t']) for entry in dose_dataset]).to(device)

            batch = (
                [entry['subject_id'] for entry in dose_dataset],
                t_padded, x_padded, masks, dose_tensor_batch,
                [entry['dose_times'] for entry in dose_dataset],
                x_dose, auc_tensor
            )

            # Preprocess batch
            _, _, _, t_encoder, x_encoder, _, _, _, dose_tensor_batch, dose_times_list, _ = preprocess_batch(
                batch, device, truncation=truncation
            )

            # Normalize if needed
            if normalization:
                k_param_med, _, _, _ = encode_latent(
                    encoder_med, t_encoder, x_encoder,
                    enable_nf=False, enable_ae=False, enable_onlymedian=True
                )
                x0_med, ode_func_med = prepare_ode_input(
                    initial_encoder_med, x_padded, k_param_med, func_med,
                    dose_tensor_batch, dose_times_list, enable_ae=False
                )
                pred_interp_med, _ = make_predictions(
                    t_encoder, t_dense, x0_med, ode_func_med, reducer_med,
                    latent_dim, global_mean, global_std
                )
                scaling = destandardize_concentration(x_encoder, global_mean, global_std) / \
                          torch.clamp(pred_interp_med, min=1.0).median()
                x_encoder = scaling

                # Cleanup
                del k_param_med, x0_med, ode_func_med, pred_interp_med
                torch.cuda.empty_cache()
                gc.collect()

            # Store per dose
            all_x_encoder.append(x_encoder)
            all_t_encoder.append(t_encoder)
            all_dose_tensor.append(dose_tensor_batch)

            # Plot percentiles per dose
            if do_plot:
                perc10 = torch.quantile(x_encoder, 0.10, dim=0).detach().cpu().numpy()
                median = torch.quantile(x_encoder, 0.50, dim=0).detach().cpu().numpy()
                perc90 = torch.quantile(x_encoder, 0.90, dim=0).detach().cpu().numpy()
                t_plot = torch.quantile(t_encoder, 0.5, dim=0).detach().cpu().numpy()
                plt.plot(t_plot, perc10, linestyle='--', label=f"10th Dose {dose_value*global_max_dose}")
                plt.plot(t_plot, median, linestyle='-', label=f"50th Dose {dose_value*global_max_dose}")
                plt.plot(t_plot, perc90, linestyle='--', label=f"90th Dose {dose_value*global_max_dose}")

            # Cleanup per dose
            del x_encoder, t_encoder, x_padded, t_padded, masks, dose_tensor_batch, x_dose, auc_tensor
            torch.cuda.empty_cache()
            gc.collect()

        # Concatenate all for classifier input
        x_encoder_all = torch.cat(all_x_encoder, dim=0)
        t_encoder_all = torch.cat(all_t_encoder, dim=0)
        dose_tensor_all = torch.cat(all_dose_tensor, dim=0)

        # Encode with main VAE
        k_param, mu_q, logvar_q, log_det = encode_latent(
            encoder_vae, t_encoder_all, x_encoder_all,
            enable_nf=False, enable_ae=False, enable_onlymedian=False
        )

        # Flatten for classifier input
        X_raw = x_encoder_all.detach().cpu().numpy().reshape(len(dataset), -1)
        X_latent = mu_q.detach().cpu().numpy().reshape(len(dataset), -1)
        y_labels = (dose_tensor_all * global_max_dose).cpu().numpy().astype(int)

        if do_plot:
            plt.xlabel("Time index")
            plt.ylabel("Normalized x_encoder")
            plt.title("Percentiles of x_encoder per dose")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

        return X_raw, X_latent, y_labels

    # === Process datasets ===
    X_raw_train, X_latent_train, y_train = process_dataset(dataset_train, do_plot=True)
    X_raw_val, X_latent_val, y_val = process_dataset(dataset_val, do_plot=True)

    # === Train classifiers ===
    model_raw = LogisticRegression(max_iter=2000).fit(X_raw_train, y_train)
    model_latent = LogisticRegression(max_iter=2000).fit(X_latent_train, y_train)

    # === Predict ===
    y_pred_raw = model_raw.predict(X_raw_val)
    y_pred_latent = model_latent.predict(X_latent_val)

    # === Compute metrics ===
    metrics = {
        "raw_accuracy": accuracy_score(y_val, y_pred_raw),
        "latent_accuracy": accuracy_score(y_val, y_pred_latent),
        "raw_r2": r2_score(y_val, y_pred_raw),
        "latent_r2": r2_score(y_val, y_pred_latent)
    }

    return {
        "metrics": metrics,
        "model_raw": model_raw,
        "model_latent": model_latent
    }
