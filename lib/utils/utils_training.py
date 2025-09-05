import copy
import gc
import math
import time
import ast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchdiffeq import odeint as odeint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm

from lib.utils.utils_preprocess import destandardize_concentration, collate_fn, truncate_time_series


import torch
import matplotlib.pyplot as plt


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

def generate_plot_data(dataloader, dataset, t_dense, global_max_time, global_max_dose, global_mean, global_std,
                   latent_dim, dim_parameters, initial_encoder, encoder, func, reducer, noise,
                   ODEWrapper, num_simulated_total=500, add_noise_to_prediction=False,
                   enable_ae_training=False, enable_nf_training=False,truncation=1):

              with torch.no_grad():
                encoder.eval()
                initial_encoder.eval()
                reducer.eval()
                func.eval()
                device = next(func.parameters()).device
        
        
                # Collect unique dose times
                
       
                unique_doses = sorted(set(entry['amt'].item() for entry in dataset))
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
                                    [entry['subject_id'] for entry in batch_entries],         # id_list
                                    pad_sequence([entry['t'] for entry in batch_entries], batch_first=True),  # t_padded
                                    pad_sequence([entry['x_global'] for entry in batch_entries], batch_first=True),  # x_global_padded
                                    masks,                                                     # mask
                                    torch.stack([entry['amt'] for entry in batch_entries]),    # dose_tensor
                                    [entry['dose_times'] for entry in batch_entries]          # dose_times_list
                                )

                    
                
                        # Preprocess
                        id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list = preprocess_batch(
                            batch, device, truncation=truncation
                        )
                        
                   
                
                        # Encode latent
                        z_refined, mu_q, logvar_q, log_det = encode_latent(
                            encoder, t_encoder, x_encoder, enable_nf=enable_nf_training, enable_ae=enable_ae_training, enable_onlymedian=False
                        )
                  
                        if not enable_ae_training:
                            eps = torch.randn(num_simulated_per_dose, latent_dim, device=mu_q.device)
                            std_q = torch.exp(0.5 * logvar_q)
                            z_refined =  eps
                        
                        # eps = torch.randn(num_simulated_total, latent_dim, device=mu_q.device)
                        # std_q = torch.exp(0.5 * logvar_q)
                        # z_refined = mu_q + 0*std_q * eps  # right now deterministic
                        
                  
                      
                        # Prepare ODE input
                        x0, ode_func = prepare_ode_input(
                            initial_encoder,
                            x_padded,
                            z_refined,
                            func,
                            dose_tensor,
                            dose_times_list,
                            enable_ae_training
                        )
                        
            
                      
                        # Make predictions
                        pred_interp, pred_batch = make_predictions(
                            t_padded, t_dense, x0, ode_func, reducer, latent_dim, global_mean, global_std
                        )
                        
                       
                      #  print(pred_batch)
                    
                        # if add_noise_to_prediction:
                        #     mask = pred_batch > 0
                        #     pred_batch = torch.where(mask, noise.sample(pred_batch, n_samples=1).squeeze(0), pred_batch)
                        #     pred_batch = torch.clamp(pred_batch, min=0)
            
                        perc10_sim = torch.quantile(pred_batch, 0.10, dim=0)  # shape [121]
                        median_sim = torch.quantile(pred_batch, 0.50, dim=0)
                        perc90_sim = torch.quantile(pred_batch, 0.90, dim=0)
                                    
                            
                
                     #   print(median_sim)
                
                        interp_all = [
                        torch_linear_interpolate2(entry['t'].to(t_dense.device),
                                                  entry['x_global'].to(t_dense.device),
                                                  t_dense)
                        for entry in dataset
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
          
            
          
def vpc(dataloader, global_max_dose, global_max_time,  global_mean, 
  global_std,
         dataset, latent_dim,
      dim_parameters, initial_encoder,encoder, func, reducer, noise, ODEWrapper,
    t_dense,compartment,onlymedian, enable_nf_training, enable_ae_training, add_noise_to_prediction=False,  
    num_simulated_total=500,truncation=1
):
        plot_data = generate_plot_data(dataloader=dataloader,
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
          enable_nf_training=enable_nf_training,
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
            z_refined, mu_q, logvar_q, log_det = encode_latent(
                encoder, t_encoder, x_encoder, enable_nf=False, enable_ae=True, enable_onlymedian=False
            )
    
            # Prepare ODE input
            x0, ode_func = prepare_ode_input(
                initial_encoder,
                x_padded,
                z_refined,
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


    
         

def plot_individual_predictions(latent_dim, t_dense, models, dataset, device, global_mean, global_std, global_max_time,
                                enable_nf_training, enable_ae_training, enable_onlymedian_training, truncation=0, max_plots=25, nr_row=5, nr_col=5):
    """
    Plots individual predictions against actual data for a trained model.
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
        
        # Unpack batch
      #  id_list, t_padded, x_padded, t_encoder, x_encoder, mask,dose_tensor, dose_times_list, \
       #      = data
        
        
        # Move to device
        id_list, t_padded, x_padded, t_encoder, x_encoder,t_cut,x_cut, mask,dose_tensor, dose_times_list = preprocess_batch(
            data, device, truncation=truncation
        )
        
        # --- Encode latent ---
        z_refined, mu_q, logvar_q, log_det = encode_latent(
            encoder,
            t_encoder,
            x_encoder,
            enable_nf=enable_nf_training,
            enable_ae=enable_ae_training,
            enable_onlymedian=enable_onlymedian_training
        )
        
        # --- Prepare ODE input ---
        x0, ode_func = prepare_ode_input(
            initial_encoder,
            x_padded,
            z_refined,
            func,
            dose_tensor,
            dose_times_list,
            enable_ae_training
        )
        
        
        pred_interp, pred_batch = make_predictions(
            t_padded, t_dense, x0, ode_func, reducer, latent_dim,global_mean,global_std
        )
        
      
        # Denormalize and squeeze to 1D
        # Denormalize and squeeze to 1D
        pred_plot = (pred_interp.squeeze(0).detach().cpu()).numpy()
        x_true_plot = (x_padded.squeeze(0)[:mask.sum()].detach().cpu() * global_std + global_mean).numpy()
        t_plot = t_padded.squeeze(0)[:mask.sum()].detach().cpu().numpy()
        
        # Ensure chronological order
        sort_idx = np.argsort(t_plot)
        t_sorted = t_plot[sort_idx]
        x_true_plot = x_true_plot[sort_idx]
        pred_plot = pred_plot[sort_idx]
        
       

        # Plot
        ax = axes[i]
        ax.plot(t_sorted, x_true_plot, 'o', label='Actual')
        ax.plot(t_sorted, pred_plot, '-', label='Predicted')
        ax.set_title(f'Individual {i}')

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()



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


class ODEWrapper(nn.Module):
    def __init__(self, func, dose_times, transfusion_times, dose_amounts, dose_mask, keep_mask=None):
        super().__init__()
        self.func = func
        self.transfusion_times = transfusion_times          # [batch, max_len]
        self.dose_times = dose_times          # [batch, max_len]
        self.dose_amounts = dose_amounts      # [batch, max_len]
        self.dose_mask = dose_mask            # [batch, max_len]
        self.keep_mask = keep_mask            # [batch, 1] — binary mask (optional)

    def forward(self, t, x):
        # Pass keep_mask if provided
        if self.keep_mask is not None:
            return self.func(t, x, self.dose_times,self.transfusion_times,  self.dose_amounts, self.dose_mask, self.keep_mask)
        else:
            return self.func(t, x, self.dose_times,self.transfusion_times,  self.dose_amounts, self.dose_mask)

    
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

    



from torch.nn.utils.rnn import pad_sequence



def preprocess_batch(batch, device, truncation=1, dose_pad_value=-1.0):
    """
    Preprocess a batch from TrajectoryDataset, truncating time series if needed.

    Args:
        batch: output from collate_fn
        device: torch device
        truncation: optional truncation for encoder
        dose_pad_value: padding value for doses (unused here but kept for compatibility)

    Returns:
        occ_list       : list of subject IDs
        t_padded       : padded time sequences [B, T]
        x_padded       : padded DV sequences [B, T]
        t_encoder      : encoder time sequences [B, T_max]
        x_encoder      : encoder DV sequences [B, T_max]
        t_cut          : truncated times that were removed [B, T_cut]
        x_cut          : truncated values that were removed [B, T_cut]
        masks          : mask for valid encoder points [B, T_max]
        dose_tensor    : per-subject dose amounts
        dose_times_list: per-subject dose times
    """

    # Unpack batch
    occ_list, t_padded, x_padded, mask, dose_tensor, dose_times_list = batch

    # Move tensors to device
    t_padded = t_padded.to(device)
    x_padded = x_padded.to(device)
    mask = mask.to(device)
    dose_tensor = dose_tensor.to(device)
    dose_times_list = [dt.to(device) for dt in dose_times_list]

    if truncation > 0:
        # Truncate time series
        t_encoder_list, x_encoder_list, masks_list, t_cut_list, x_cut_list = truncate_time_series(
            t_padded, x_padded, truncation
        )

        # Pad sequences to max length in batch
        t_encoder = pad_sequence(t_encoder_list, batch_first=True)       # [B, T_max]
        x_encoder = pad_sequence(x_encoder_list, batch_first=True)       # [B, T_max]
       # masks = pad_sequence(masks_list, batch_first=True)               # [B, T_max], bool

        # Also pad the removed/cut values for consistency
        t_cut = pad_sequence(t_cut_list, batch_first=True)               # [B, T_cut_max]
        x_cut = pad_sequence(x_cut_list, batch_first=True)               # [B, T_cut_max]

    else:
        t_encoder = t_padded
        x_encoder = x_padded
        t_cut, x_cut = t_padded, x_padded
       # masks = mask

    return (
        occ_list,
        t_padded,
        x_padded,
        t_encoder,
        x_encoder,
        t_cut,
        x_cut,
        mask,
        dose_tensor,
        dose_times_list
    )








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
    t_encoder_trimmed = t_encoder #[:, 4:]
    x_normalized_trimmed = x_normalized #[:, 4:]

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





def prepare_ode_input(
    initial_encoder,
    x_padded,
    z_refined,
    func,
    dose_tensor,
    dose_times_list,
    ae_training=False
):
    """
    Prepare initial state and ODE function for the new TrajectoryDataset.

    Args:
        initial_encoder : nn.Module, encoder for initial state
        x_padded        : (batch, max_len) DV sequences
        z_refined       : latent refinement tensor
        func            : ODE function module
        dose_tensor     : (batch,) per-subject doses
        dose_times_list : list of per-subject dose times (tensors)

    Returns:
        x0       : initial latent state
        ode_func : ODEWrapper instance
    """

    # Encode initial state
    x0_1 = initial_encoder(x_padded[:, 0].unsqueeze(1))
    if ae_training:
        x0 = torch.cat([
            x0_1 + 0.01 * torch.randn_like(x0_1),
            z_refined + 0.01 * torch.randn_like(z_refined) 
        ], dim=1)
    else:    
        x0 = torch.cat([
            x0_1 + 0.01 * torch.randn_like(x0_1),
            z_refined 
        ], dim=1)
        
    # Dose mask: 1 where dose exists, 0 otherwise
    dose_mask = (dose_tensor != 0).float().unsqueeze(-1)  # [batch_size, 1]

    # Pad dose_times to a tensor
    max_doses = max([len(dt) for dt in dose_times_list])
    dose_times_padded = torch.zeros((len(dose_times_list), max_doses), device=x_padded.device)
    dose_times_mask = torch.zeros_like(dose_times_padded)
    for i, dt in enumerate(dose_times_list):
        if len(dt) > 0:
            dose_times_padded[i, :len(dt)] = dt
            dose_times_mask[i, :len(dt)] = 1

    # Expand dose_tensor to match dose_times
    dose_tensor_expanded = dose_tensor.unsqueeze(1).repeat(1, max_doses).unsqueeze(-1)
    


    # Create ODEWrapper with dose info
    ode_func = ODEWrapper(
        func,
        dose_times_padded,
        dose_tensor_expanded,
        dose_mask,
        dose_times_mask
    )

    return x0, ode_func






def make_predictions(t_padded, t_dense, x0, ode_func, reducer, latent_dim, global_mean, global_std):
    """
    Runs ODE forward and interpolates predictions at given time points.
    Returns:
        pred_interp: interpolated predictions at t_padded
        pred_batch: full ODE trajectory at dense time points
    """
    batch_size = t_padded.size(0)

    # Solve ODE
    pred = odeint(ode_func, x0, t_dense, method="rk4")
    
    pred_batch = pred.permute(1, 0, 2)  # [batch, time, latent_dim]
    
    
    
    # Reduce dimension and interpolate back to t_padded
    t_dense_exp = t_dense.unsqueeze(0).repeat(batch_size, 1)
    reduced = reducer(pred_batch[:, :, :latent_dim])
    
    reduced= destandardize_concentration(reduced, global_mean, global_std)
    
    pred_interp = batch_linear_interpolate_1d(reduced, t_dense_exp, t_padded)
    
    return pred_interp, reduced



def compute_loss_and_metrics(
    x_padded, mask, pred_interp,
    mu_q, logvar_q, log_det,
    epoch, warmup_epochs_iiv, free_bits,
    enable_ae_training, enable_nf_training,
    noise, global_mean, global_std
):
    """
    Computes reconstruction + KL/NF losses and metrics.
    Returns:
        loss, recon_loss, mse, KL_loss, log_det_sum, log_det_penalty, kl_gauss
    """
    # --- Reconstruction loss ---
    recon_loss_noise, mse = noise.nll(
        destandardize_concentration(x_padded, global_mean, global_std),
        pred_interp,
        mask
    )

    mu_std = torch.zeros_like(mu_q)
    logvar_std = torch.zeros_like(logvar_q)

    if enable_ae_training:
        KL_loss = torch.tensor(0.0)
        kl_gauss = torch.tensor(0.0)
        log_det_sum = torch.tensor(0.0)
        log_det_penalty = torch.tensor(0.0)
        kl_weight = 0.0
    else:
        free_bits_on = (
            0 if epoch + 1 >= warmup_epochs_iiv
            else free_bits * (1 - min(1.0, epoch / warmup_epochs_iiv))
        )
        kl_weight = (
            1.0 if epoch + 1 >= warmup_epochs_iiv
            else min(1.0, epoch / warmup_epochs_iiv)
        )

        if enable_nf_training:
            KL_loss, log_det_sum, kl_gauss, log_det_penalty = kl_divergence_NF(
                epoch, warmup_epochs_iiv, mu_q, logvar_q, mu_std, logvar_std, log_det,
                free_bits=free_bits_on, log_det_penalty_lambda=1,
            )
        else:
            KL_loss = kl_divergence_gaussians(mu_q, logvar_q, mu_std, logvar_std, free_bits_on)
            kl_gauss = KL_loss
            log_det_sum = torch.tensor(0.0)
            log_det_penalty = torch.tensor(0.0)

    # --- Total loss ---
    loss = recon_loss_noise + kl_weight * KL_loss + log_det_penalty

    return loss, recon_loss_noise, mse, KL_loss, log_det_sum, log_det_penalty, kl_gauss




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

def log_training_epoch(end_time, epoch, total_mse, total_loss, total_recon, total_kl, total_transform,
                       optimizer, noise, dataset, batch_size,num_batches, val_mse=None, val_LL=None,
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
                  f"MSE {total_mse/num_batches:.4f} "
                  f"-LL: {total_recon/num_batches:.4f}, "
                  f"Add. error: {add_error:.2f}, "
            #      f"val_mse: {val_mse:.3f}, "
            #      f"val_LL: {val_LL:.3f}, "
                  f"lr: {main_lr:.3f}, "
                  f"{end_time:.2f} seconds")
        elif enable_nf_training:
            print(f"Epoch {epoch}, "
                  f"MSE {total_mse/num_batches:.1f} "
                  f"loss: {total_loss/num_batches:.1f}, "
                  f"-LL: {batch_size/len(dataset)*total_recon:.1f}, "
                  f"KL: {batch_size/len(dataset)*total_kl:.8f}, "
                  f"log_det: {batch_size/len(dataset)*total_transform:.8f}, "
                  f"Add. error: {add_error:.2f}, "
                  f"lr: {main_lr:.3f}, "
                  f"{end_time:.1f} seconds")
        else:
            print(f"Epoch {epoch}, "
                  f"MSE {total_mse/num_batches:.1f} "
                  f"loss: {total_loss/num_batches:.1f}, "
                  f"-LL: {total_recon/num_batches:.2f}, "
                  f"KL: {total_kl/20:.4f}, "
                  f"Add. error: {add_error:.2f}, "
                  f"lr: {main_lr:.4f}, "
                  f"{end_time:.1f} seconds")

def plot_individual_fits(
    latent_dim, t_dense, models, dataloader, device,
    global_mean, global_std, global_max_time,
    enable_nf_training, enable_ae_training, enable_onlymedian_training,
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
        id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list = preprocess_batch(
            data, device, truncation=truncation
        )

        # Encode latent
        z_refined, mu_q, logvar_q, log_det = encode_latent(
            encoder,
            t_encoder,
            x_encoder,
            enable_nf=enable_nf_training,
            enable_ae=enable_ae_training,
            enable_onlymedian=enable_onlymedian_training
        )

        latent_dim = mu_q.size(1)
        eps = torch.randn(n_samples, latent_dim, device=mu_q.device)
        std_q = torch.exp(0.5 * logvar_q)
        z_refined = mu_q + 0*std_q * eps

        # Expand tensors for ODE
        x_padded_exp = x_padded.expand(n_samples, *x_padded.shape[1:])
        t_padded_exp = t_padded.expand(n_samples, *t_padded.shape[1:])
        dose_tensor_exp = dose_tensor.expand(n_samples, *dose_tensor.shape[1:])
        x_encoder_exp = x_encoder.expand(n_samples, *x_encoder.shape[1:])
        
      
        
        
        # Prepare ODE input
        x0, ode_func = prepare_ode_input(
            initial_encoder,
            x_padded,
            z_refined,
            func,
            dose_tensor,
            dose_times_list,
            enable_ae_training
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





def evaluate_on_val(
    global_mean,
    global_std,
    global_max_time,
    global_max_dose,
    enable_nf_training,
    enable_ae_training,
    enable_onlymedian_training,
    truncation,
    func,
    noise,
    reducer,
    dataloader_val,
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
    batch_size = dataloader_val.batch_size
    num_samples = len(dataloader_val.dataset)
    num_batches=len(dataloader_val)
    
    for batch in dataloader_val:
        id_list, t_padded, x_padded,t_encoder, x_encoder,t_cut, x_cut,  mask, dose_tensor, dose_times_list = preprocess_batch(
    batch, device, truncation=truncation
)
   
        # --- Encode latent ---
        z_refined, mu_q, logvar_q, log_det = encode_latent(
        encoder,
        t_encoder,
        x_encoder,
        enable_nf=enable_nf_training,
        enable_ae=True,
        enable_onlymedian=enable_onlymedian_training
        )
        
        # --- Prepare ODE input ---
        x0, ode_func = prepare_ode_input(
            initial_encoder,
            x_padded,
            z_refined,
            func,
            dose_tensor,
            dose_times_list,
            enable_ae_training
        )
        
        # --- Compute predictions ---
    

        # --- Compute loss and metrics ---
        pred_interp, pred_batch = make_predictions(
            t_padded, t_dense, x0, ode_func, reducer, latent_dim,global_mean,global_std
        )
        
        
        # --- Debug prints ---
       
        # --- Compute reconstruction loss ---
        recon_loss_noise, mse = noise.nll(
        destandardize_concentration(x_padded, global_mean, global_std),
        pred_interp
        )
        
        # --- Accumulate metrics ---
        total_mse += mse.item()
        total_LL += recon_loss_noise.item()

    return total_mse/num_batches, total_LL/num_batches


def train_loop_model(
    dataset, dataset_val, global_max_time, global_max_dose,
    global_mean, global_std, main_params, dataloader_val, dataloader, models,
    optimizer, scheduler, func, reducer, initial_encoder, encoder, noise, t_dense,
    n_epochs, warmup_epochs_noise, warmup_epochs_iiv, smoothing_start_epoch,
    traing_against_validation, enable_ae_training, enable_nf_training,
    enable_onlymedian_training, plot_from_training_records_enable, free_bits,
    truncation, print_epoch=1, plot_epoch=1, max_plots=4,
    nr_col=1, nr_row=5
):
    # === Initialize training ===
    device, latent_dim, dim_parameter_encoder, best_val_mse, best_val_LL, \
    epochs_no_improve, patience, batch_size, t_dense, mse_history = initialize_training(
        models, func, dataloader, t_dense, global_max_time
    )
    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
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
            (
             occ_list,          # id_list
             t_padded,          # padded time
             x_global_padded,   # padded global features
             mask,              # time mask
             dose_tensor,       # dose amounts
             dose_times_list,   # dose start times
         ) = batch
            

            
            # --- Preprocess batch ---
            # --- Preprocess batch ---
            id_list, t_padded, x_padded,  t_encoder, x_encoder,t_cut, x_cut, mask, dose_tensor, dose_times_list = preprocess_batch(
                batch, device, truncation=truncation
            )
            
         
            # --- Encode latent ---
            z_refined, mu_q, logvar_q, log_det = encode_latent(
                encoder,
                t_encoder,
                x_encoder,
                enable_nf=enable_nf_training,
                enable_ae=enable_ae_training,
                enable_onlymedian=enable_onlymedian_training
            )
            
            
        
            
            # --- Prepare ODE input ---
            x0, ode_func = prepare_ode_input(
                initial_encoder,
                x_padded,
                z_refined,
                func,
                dose_tensor,
                dose_times_list,
                enable_ae_training
            )


            # --- Compute loss and metrics ---
            pred_interp, pred_batch = make_predictions(
                t_padded, t_dense, x0, ode_func, reducer, latent_dim,global_mean,global_std
            )
    
            
            # 2. Compute losses
            loss, recon_loss_noise, mse, KL_loss, log_det_sum, log_det_penalty, kl_gauss = compute_loss_and_metrics(
                x_padded, mask, pred_interp,
                mu_q, logvar_q, log_det,
                epoch, warmup_epochs_iiv, free_bits,
                enable_ae_training, enable_nf_training,
                noise, global_mean, global_std
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
                enable_nf_training, enable_ae_training,enable_onlymedian_training, truncation,
                func, noise, reducer, dataloader_val,
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
        else:
            val_mse=0
            val_LL=0
        # --- Logging ---
        
        log_training_epoch(epoch_duration,
            epoch, total_mse, total_loss, total_recon, total_kl, total_transform,
            optimizer, noise, dataset, batch_size,num_batches,
            val_mse=val_mse, val_LL=val_LL,
            enable_ae_training=enable_ae_training,
            enable_nf_training=enable_nf_training,
            print_epoch=print_epoch
        )

        #--- Plot training records ---
        # if plot_from_training_records_enable and epoch % plot_epoch == 0:
        #     plot_from_training_records(
        #         batch_size, device, global_max_time, global_max_dose, global_mean, global_std,
        #         latent_dim, records=trajectory_records, func=func, reducer=reducer,
        #         initial_encoder=initial_encoder, ODEWrapper=ODEWrapper, t_dense=t_dense,
        #         max_plots=max_plots, nr_row=nr_row, nr_col=nr_col
        #     )
        # if plot_from_training_records_enable and epoch % plot_epoch == 0:
            
        #     plot_individual_predictions(t_dense, models, dataset, device, global_mean, global_std, global_max_time,
        #                                 truncation=100, max_plots=25, nr_row=5, nr_col=5)

        torch.cuda.empty_cache()
        gc.collect()

    return min(mse_history)/num_batches, best_val_mse


def run_model_variant(variant_name, train_dataset, val_dataset, test_dataset,
                      models, main_params, optimizer, scheduler, combined,
                      global_max_dose, global_max_time, global_mean, global_std,
                      latent_dim, noise, encoder, func, reducer, initial_encoder,
                      ODEWrapper, device, metrics, residuals, iteration,
                      n_epochs, train_loader, val_loader,test_loader,  base_dir,
                      warmup_noise, warmup_iiv, enable_ae, enable_nf,enable_onlymedian, plot_from_training_records_enable=False,
                      free_bits=0, truncation=1):
    """
    Train, evaluate, append metrics and residuals for a given model variant.

    Args:
        base_dir (str): directory to save metrics/residuals
        ...
        other args as before
    """
    # Run training loop
    mse_train, mse_validation = train_loop_model(
    train_dataset,                  # Training dataset
    val_dataset,                    # Validation dataset
    global_max_time,                # Maximum time value for normalization/scaling
    global_max_dose,                # Maximum dose value for normalization/scaling
    global_mean,                    # Global mean of observations (for destandardization)
    global_std,                     # Global standard deviation of observations
    main_params,                    # Dictionary of main training parameters (e.g., learning rate)
    val_loader,                     # Validation dataloader
    train_loader,                  # Training dataloader
    models,                         # Dictionary of model components (encoder, ODEFunc, reducer, etc.)
    optimizer,                      # Optimizer for training
    scheduler,                      # Learning rate scheduler
    func,                           # ODE function defining dynamics
    reducer,                        # Function to reduce latent trajectories
    initial_encoder,                # Initial condition encoder model
    encoder,                        # Encoder model for latent space
    noise,                          # Noise model for likelihood computation
    t_dense=combined,               # Dense time points to evaluate ODE predictions
    n_epochs=n_epochs,              # Number of training epochs
    warmup_epochs_noise=warmup_noise,  # Epochs for noise warmup (gradual training)
    warmup_epochs_iiv=warmup_iiv,      # Epochs for inter-individual variability warmup
    smoothing_start_epoch=1000,         # Epoch to start smoothing loss (optional)
    traing_against_validation=True,     # Whether to compute validation loss during training
    enable_ae_training=enable_ae,      # Whether to train autoencoder components
    enable_nf_training=enable_nf,      # Whether to train normalizing flows
    enable_onlymedian_training=enable_onlymedian,  # Whether to train only median predictions
    plot_from_training_records_enable=plot_from_training_records_enable,  # Plotting during training
    free_bits=free_bits,                # Free bits parameter for KL loss
    truncation=truncation,              # Fraction of sequence to truncate for encoder
    print_epoch=1,                      # Frequency (in epochs) to print training progress
    plot_epoch=1,                       # Frequency to generate plots during training
    max_plots=30,                       # Maximum number of individuals to plot
    nr_col=10,                          # Number of columns in plot grid
    nr_row=3                            # Number of rows in plot grid
)


   #  # Evaluate predictions
   # # mse_mean, r2_mean, mse_median, r2_median, res_df = #
    plot_individual_fits(
       latent_dim, combined, models, test_loader, device,
       global_mean, global_std, global_max_time,
       enable_nf, enable_ae, enable_onlymedian,
       truncation=truncation, max_plots=3, nr_row=1, nr_col=3, n_samples=1000, ci_lower=0.025, ci_upper=0.975)
       
    
    total_mse, total_LL = evaluate_on_val(
        global_mean,
        global_std,
        global_max_time,
        global_max_dose,
        enable_nf,
        enable_ae,
        enable_onlymedian,
        truncation,
        func,
        noise,
        reducer,
        test_loader,
        device,
        encoder,
        initial_encoder,
        combined,
    )
    print(total_mse)
    compute_residuals(latent_dim, global_mean, global_std, func, encoder, reducer, initial_encoder, noise, train_loader, combined, device, 0.35)

   #  r2, mse=compute_test_loss(
   #     latent_dim, combined, models, test_dataset, device,
   #     global_mean, global_std, global_max_time,
   #     enable_nf, enable_nf, enable_onlymedian,
   #     truncation=truncation, max_plots=25, nr_row=5, nr_col=5, n_samples=2, ci_lower=0.05, ci_upper=0.95
   # )
    
    # Append metrics and residuals
  #  append_metrics(metrics, residuals, variant_name, mse_mean, r2_mean,
   #                mse_median, r2_median, mse_validation, res_df)

    # Save metrics/residuals
  #  export_all_metrics_and_residuals(metrics, residuals, base_dir)
  #  return mse_train 
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




    