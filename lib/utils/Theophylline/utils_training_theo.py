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

from lib.utils.Theophylline.utils_preprocess_theo import destandardize_concentration, collate_fn, append_metrics, export_all_metrics_and_residuals
from lib.utils.Theophylline.utils_post_processing_theo import vpc_true, vpc, plot_individual_fits, compute_residuals
from lib.utils.Theophylline.utils_shared_theo import encode_latent, preprocess_batch, prepare_ode_input, make_predictions, prepare_ode_input_eval, truncate_time_series

from torch.nn.utils.rnn import pad_sequence


from contextlib import contextmanager
import pandas as pd





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
    best_val_loss = float("inf")
    best_model_state = {k: v.state_dict() for k, v in models.items()}  # initial model state

    epochs_no_improve = 0
    patience = 1  # stop if no improvement for 30 epochs

    batch_size = dataloader.batch_size
    t_dense = t_dense.to(device)

    mse_history = []

    return (device, latent_dim, dim_parameter_encoder, best_val_mse, best_val_LL, best_val_loss, 
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

    










def compute_loss_and_metrics(
    x_padded, mask, pred_interp,
    mu_q, logvar_q, log_det, mu_IC, logvar_IC,
    epoch, warmup_epochs_iiv, free_bits,
    enable_ae, enable_nf,
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
    
    mu_std_IC = torch.zeros_like(mu_IC)
    logvar_std_IC = torch.zeros_like(logvar_IC)
    
    
    if enable_ae:
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

        if enable_nf:
            KL_loss, log_det_sum, kl_gauss, log_det_penalty = kl_divergence_NF(
                epoch, warmup_epochs_iiv, mu_q, logvar_q, mu_std, logvar_std, log_det,
                free_bits=free_bits_on, log_det_penalty_lambda=1,
            )
        else:
            KL_loss =kl_divergence_gaussians(mu_q, logvar_q, mu_std, logvar_std, free_bits_on) + kl_divergence_gaussians(mu_IC, logvar_IC, mu_std_IC, logvar_std_IC, free_bits_on)
            kl_gauss = KL_loss
            log_det_sum = torch.tensor(0.0)
            log_det_penalty = torch.tensor(0.0)

    # --- Total loss ---
    loss = recon_loss_noise + kl_weight * KL_loss + log_det_penalty

    
    return loss, recon_loss_noise, mse, KL_loss, log_det_sum, log_det_penalty, kl_gauss




def accumulate_epoch_metrics(
    total_transform, total_loss, total_mse, total_kl, total_recon,
    log_det_sum, loss, mse, kl_gauss, recon_loss_noise, enable_ae
):
    total_transform += log_det_sum.item()
    total_loss += loss.item()
    total_mse += mse.item()
    total_kl += kl_gauss.item() if not enable_ae else 0.0
    total_recon += recon_loss_noise.item()
    return total_transform, total_loss, total_mse, total_kl, total_recon


def run_backprop_step(loss, main_params, optimizer):
    optimizer.zero_grad()
    loss.backward()
 
        
    for param_group in main_params:
        torch.nn.utils.clip_grad_norm_(param_group["params"], max_norm=0.5)
    optimizer.step()
    
    
    
def validate_and_update_early_stop(
    epoch, warmup_epochs_noise, traing_against_validation,
    val_mse, val_LL,val_loss, best_val_mse, best_val_LL,best_val_loss, epochs_no_improve,
    patience, models, enable_vae
):
    """
    Handles validation metrics, early stopping, and best model state update.

    Returns:
        best_val_mse, best_val_LL, epochs_no_improve, stop_training (bool), best_model_state
    """
    stop_training = False
    best_model_state = {k: v.state_dict() for k, v in models.items()}  # <<< initialize here

    if enable_vae:
        if val_loss < best_val_loss:
            best_val_mse = val_mse
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = {k: v.state_dict() for k, v in models.items()}
        else:
            epochs_no_improve += 1
         #   print(f"[Early Stop Monitor] No improvement for {epochs_no_improve} epochs, current val loss {val_loss:.1f}, best val loss {best_val_loss:.1f}")

        if epochs_no_improve >= patience and epoch > 200:
            print(f"Early stopping triggered after {epoch + 1} epochs due to loss not increasing")
            for k, v in models.items():
                v.load_state_dict(best_model_state[k])
            print(f"Best validation loss achieved: {best_val_loss:.4f}") 
            stop_training = True


    elif epoch < warmup_epochs_noise:
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            epochs_no_improve = 0
            best_model_state = {k: v.state_dict() for k, v in models.items()}
        else:
            epochs_no_improve += 1
        #    print(f"[Early Stop Monitor] No improvement for {epochs_no_improve} epochs, current val MSE {val_mse:.1f}, best val MSE {best_val_mse:.1f}")

        if epochs_no_improve >= patience and epoch > 100:
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
           # print(f"[Early Stop Monitor] No improvement for {epochs_no_improve} epochs, current val LL {val_LL:.1f}, best val LL {best_val_LL:.1f}")

        if epochs_no_improve >= patience and epoch > 200:
            print(f"Early stopping triggered after {epoch + 1} epochs due to LL not improving")
            for k, v in models.items():
                v.load_state_dict(best_model_state[k])
            print(f"Best validation MSE and LL achieved: {best_val_mse:.4f}, {best_val_LL:.4f}") 
            stop_training = True

    return best_val_mse, best_val_LL, best_val_loss, epochs_no_improve, stop_training, best_model_state

def log_training_epoch(end_time, epoch, total_mse, total_loss, total_recon, total_kl, total_transform,
                       optimizer, noise, dataset, batch_size,num_batches, val_mse=None, val_LL=None,
                       enable_ae=False, enable_nf=False, print_epoch=1):
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
        enable_ae: bool flag
        enable_nf: bool flag
        print_epoch: frequency of printing
    """
    if epoch % print_epoch != 0:
        return

    with torch.no_grad():
        main_lr = optimizer.param_groups[0]['lr']
       # end_time = time.time()
        add_error = float(noise.sigma_add)
        prop_error = float(noise.sigma_prop)

        if enable_ae:
            print(f"Epoch {epoch}, "
                  f"MSE {total_mse/num_batches:.4f} "
                  f"-LL: {total_recon/num_batches:.4f}, "
                  f"Add. error: {add_error:.2f}, "
                  f"Prop. error: {prop_error:.2f}, "
            #      f"val_mse: {val_mse:.3f}, "
            #      f"val_LL: {val_LL:.3f}, "
                  f"lr: {main_lr:.3f}, "
                  f"{end_time:.2f} seconds")
        elif enable_nf:
            print(f"Epoch {epoch}, "
                  f"MSE {total_mse/num_batches:.1f} "
                  f"loss: {total_loss/num_batches:.1f}, "
                  f"-LL: {total_recon/num_batches:.1f}, "
                  f"KL: {total_kl/num_batches:.8f}, "
                  f"log_det: {total_transform/num_batches:.8f}, "
                  f"Add. error: {add_error:.2f}, "
                  f"lr: {main_lr:.3f}, "
                  f"{end_time:.1f} seconds")
        else:
            print(f"Epoch {epoch}, "
                  f"MSE {total_mse/num_batches:.1f} "
                  f"loss: {total_loss/num_batches:.1f}, "
                  f"-LL: {total_recon/num_batches:.2f}, "
                  f"KL: {total_kl/num_batches:.4f}, "
                  f"Add. error: {add_error:.2f}, "
                  f"lr: {main_lr:.4f}, "
                  f"{end_time:.1f} seconds")





def evaluate_on_val(
    global_mean,
    global_std,
    global_max_time,
    global_max_dose,
    enable_nf,
    enable_ae,
    enable_vae,
    enable_onlymedian,
    truncation,
    func,
    noise,
    reducer,
    dataloader_val,
    device,
    encoder,
    initial_encoder,
    t_dense
):
    """
    Evaluate model on a validation dataset by computing total MSE and NLL.
    Reuses preprocess, encode_latent, and prepare_ode_input functions.
    """
   


    device = next(func.parameters()).device
    latent_dim = func.dim_latent

    total_mse = 0.0
    total_LL = 0.0
    total_loss = 0.0

    num_batches=len(dataloader_val)
    
    for batch in dataloader_val:
        id_list, t_padded, x_padded,t_encoder, x_encoder,t_cut, x_cut,  mask, dose_tensor, dose_times_list, masks_list = preprocess_batch(
    batch, device, truncation=truncation
)
   
        # --- Encode latent ---
        z_refined, mu_q, logvar_q, log_det = encode_latent(
        encoder,
        t_encoder,
        x_encoder,
        enable_nf=enable_nf,
        enable_ae=enable_ae,
        enable_onlymedian=enable_onlymedian
        )
        
        # --- Prepare ODE input ---
        x0, ode_func, mu_IC, logvar_IC = prepare_ode_input_eval(
            initial_encoder,
            x_padded,
            z_refined,
            func,
            dose_tensor,
            dose_times_list

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
    
        inverted = torch.stack([~m for m in masks_list])


  
        loss, recon_loss_noise, mse, KL_loss, log_det_sum, log_det_penalty, kl_gauss = compute_loss_and_metrics(
            x_padded, mask, pred_interp,
            mu_q, logvar_q, log_det,mu_IC, logvar_IC,
            10, 0, 0,
            enable_ae, enable_nf,
            noise, global_mean, global_std
        )
        
        
        # --- Accumulate metrics ---
        total_mse += mse.item()
        total_LL += recon_loss_noise.item()
        total_loss += loss.item()

    return total_mse/num_batches, total_LL/num_batches, total_loss/num_batches



def train_loop_model(
    dataset, dataset_val, global_max_time, global_max_dose,
    global_mean, global_std, main_params, dataloader_val, dataloader, models,
    optimizer, scheduler, func, reducer, initial_encoder, encoder, noise, device, t_dense,
    n_epochs, warmup_epochs_noise, warmup_epochs_iiv, smoothing_start_epoch,
    traing_against_validation, enable_ae, enable_vae, enable_nf,
    enable_onlymedian, plot_from_training_records_enable, free_bits,
    truncation, print_epoch=1, plot_epoch=1, max_plots=4,
    nr_col=1, nr_row=5
):
    # === Initialize training ===
    device, latent_dim, dim_parameter_encoder, best_val_mse, best_val_LL, best_val_loss, \
    epochs_no_improve, patience, batch_size, t_dense, mse_history = initialize_training(
        models, func, dataloader, t_dense, global_max_time
    )

    num_batches = len(dataloader)
    for epoch in range(n_epochs):
        # After loss.backward()


     
            
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
            id_list, t_padded, x_padded,  t_encoder, x_encoder,t_cut, x_cut, mask, dose_tensor, dose_times_list, masks_list = preprocess_batch(
                batch, device, truncation=truncation
            )
            
            
            
        
         
      
            z_refined, mu_q, logvar_q, log_det = encode_latent(
                encoder,
                t_encoder,
                x_encoder,
                enable_nf=enable_nf,
                enable_ae=enable_ae,
                enable_onlymedian=enable_onlymedian
            
            )
            
            
      
          
          
            
            x01, ode_func1, mu_IC, logvar_IC = prepare_ode_input(
                initial_encoder,
                x_padded,
                z_refined,
                func,
                dose_tensor,
                dose_times_list,
                enable_vae=enable_vae
    
            )
            
            
            
            pred_interp, pred_batch = make_predictions(
                t_padded, t_dense, x01, ode_func1, reducer, latent_dim,global_mean,global_std
            )
       
        
        
            
            # 2. Compute losses
            loss, recon_loss_noise, mse, KL_loss, log_det_sum, log_det_penalty, kl_gauss = compute_loss_and_metrics(
                x_padded, mask, pred_interp,
                mu_q, logvar_q, log_det,mu_IC, logvar_IC,
                epoch, warmup_epochs_iiv, free_bits,
                enable_ae, enable_nf,
                noise, global_mean, global_std
            )
       
            
             
            total_transform, total_loss, total_mse, total_kl, total_recon = accumulate_epoch_metrics(
                total_transform, total_loss, total_mse, total_kl, total_recon,
                log_det_sum, loss , mse, kl_gauss, recon_loss_noise, enable_ae
            )
     
            
     
         
            # if enable_ae and epoch < warmup_epochs_noise:
            #     run_backprop_step(mse, main_params, optimizer)

            # else:
            run_backprop_step(loss , main_params, optimizer)


            first_batch = False
            if epoch >= smoothing_start_epoch:
                for m in [models, encoder, initial_encoder, func, noise, reducer]:
                    if hasattr(m, "update_ema"):
                        m.update_ema(alpha=0.05)
            
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
        if traing_against_validation and epoch > warmup_epochs_iiv:
       
            
            val_mse, val_LL, val_loss = evaluate_on_val(
                global_mean, global_std, global_max_time, global_max_dose,
                enable_nf, enable_ae,enable_vae,enable_onlymedian, truncation,
                func, noise, reducer, dataloader_val,
                device, encoder, initial_encoder, t_dense
            )
            
            best_val_mse, best_val_LL,best_val_loss, epochs_no_improve, stop_training, best_model_state = \
                validate_and_update_early_stop(
                    epoch, warmup_epochs_noise, traing_against_validation,
                    val_mse, val_LL,val_loss, best_val_mse, best_val_LL,best_val_loss, epochs_no_improve,
                    patience, models, enable_vae
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
            enable_ae=enable_ae,
            enable_nf=enable_nf,
            print_epoch=print_epoch
        )

   

        torch.cuda.empty_cache()
        gc.collect()

    return min(mse_history)/num_batches, best_val_mse




def compute_test_metrics(
    latent_dim, dataset, models, device,
    global_mean, global_std, global_max_time,
    enable_nf, enable_ae, enable_onlymedian, t_dense,
    truncation=0,
    iteration=None  # allow passing iteration number
):
    """
    Computes mean/median R² and MSE across subjects,
    plus residuals on the cut (removed) values only.

    Returns:
        r2_mean, r2_median, mse_mean, mse_median, residuals_df
    """
    encoder = models['encoder']
    initial_encoder = models['initial_encoder']
    func = models['func']
    reducer = models['reducer']

    all_targets, all_predictions, all_ids, all_iterations = [], [], [], []
    per_subject_r2, per_subject_mse = [], []

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for data in dataloader:
        # Preprocess batch (single subject)
        id_list, t_padded, x_padded, t_encoder, x_encoder, t_cut, x_cut, mask, dose_tensor, dose_times_list, list123 = preprocess_batch(
            data, device, truncation=truncation
        )

        if t_cut.numel() == 0:
            continue  # skip if no cut values

        # Encode latent
        z_refined, mu_q, logvar_q, log_det = encode_latent(
            encoder,
            t_encoder,
            x_encoder,
            enable_nf=enable_nf,
            enable_ae=True,
            enable_onlymedian=enable_onlymedian
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

        # Interpolate predictions at cut times
        pred_interp, _ = make_predictions(
            t_cut, t_dense, x0, ode_func, reducer,
            latent_dim, global_mean, global_std
        )

        # Denormalize true and predicted cut values
        x_cut_true = (x_cut.squeeze(0).detach().cpu().numpy() * global_std + global_mean)
        x_cut_pred = pred_interp.squeeze(0).detach().cpu().numpy()

        # Collect global arrays
        all_targets.extend(x_cut_true.tolist())
        all_predictions.extend(x_cut_pred.tolist())
        all_ids.extend([id_list[0]] * len(x_cut_true))
        all_iterations.extend([iteration if iteration is not None else -1] * len(x_cut_true))

        # Per-subject metrics
        if len(x_cut_true) > 1:  # avoid error on single points
            per_subject_r2.append(r2_score(x_cut_true, x_cut_pred))
            per_subject_mse.append(mean_squared_error(x_cut_true, x_cut_pred))

    # Compute aggregated metrics
    if len(per_subject_r2) == 0:
        return None, None, None, None, pd.DataFrame(
            columns=["Prediction", "Observation", "Residual", "Iteration", "ID"]
        )

    r2_mean = float(np.mean(per_subject_r2))
    r2_median = float(np.median(per_subject_r2))
    mse_mean = float(np.mean(per_subject_mse))
    mse_median = float(np.median(per_subject_mse))

    # Build residuals DataFrame
    residuals_df = pd.DataFrame({
        "Prediction": all_predictions,
        "Observation": all_targets,
        "Residual": np.array(all_targets) - np.array(all_predictions),
        "Iteration": all_iterations,
        "ID": all_ids
    })

    return r2_mean, r2_median, mse_mean, mse_median, residuals_df




        
        
        

def run_model_variant(dim_parameter_encoder, variant_name, dataset_train, dataset_val, dataset_test,
                      models, main_params, optimizer, scheduler, t_dense,
                      global_max_dose, global_max_time, global_mean, global_std,
                      latent_dim, noise, encoder, func, reducer, initial_encoder,
                       metrics, residuals, iteration,
                      n_epochs, dataloader_val, train_loader,test_loader,  base_dir,
                      warmup_noise, warmup_iiv, enable_ae, enable_vae, enable_nf,enable_onlymedian, plot_from_training_records_enable,traing_against_validation,
                      free_bits, truncation):
    """
    Train, evaluate, append metrics and residuals for a given model variant.

    Args:
        base_dir (str): directory to save metrics/residuals
        ...
        other args as before
    """
    # Run training loop
    device = next(func.parameters()).device

    mse_train, mse_validation = train_loop_model(
    dataset_train,                  # Training dataset
    dataset_val,                    # Validation dataset
    global_max_time,                # Maximum time value for normalization/scaling
    global_max_dose,                # Maximum dose value for normalization/scaling
    global_mean,                    # Global mean of observations (for destandardization)
    global_std,                     # Global standard deviation of observations
    main_params,                    # Dictionary of main training parameters (e.g., learning rate)                  # Validation dataloader
    dataloader_val,
    train_loader,                  # Training dataloader
    models,                         # Dictionary of model components (encoder, ODEFunc, reducer, etc.)
    optimizer,                      # Optimizer for training
    scheduler,                      # Learning rate scheduler
    func,                           # ODE function defining dynamics
    reducer,                        # Function to reduce latent trajectories
    initial_encoder,                # Initial condition encoder model
    encoder,                        # Encoder model for latent space
    noise,  
    device,                        # Noise model for likelihood computation
    t_dense=t_dense,               # Dense time points to evaluate ODE predictions
    n_epochs=n_epochs,              # Number of training epochs
    warmup_epochs_noise=warmup_noise,  # Epochs for noise warmup (gradual training)
    warmup_epochs_iiv=warmup_iiv,      # Epochs for inter-individual variability warmup
    smoothing_start_epoch=1000,         # Epoch to start smoothing loss (optional)
    traing_against_validation=traing_against_validation,     # Whether to compute validation loss during training
    enable_ae=enable_ae,  
    enable_vae=enable_vae,    # Whether to train autoencoder components
    enable_nf=enable_nf,      # Whether to train normalizing flows
    enable_onlymedian=enable_onlymedian,  # Whether to train only median predictions
    plot_from_training_records_enable=plot_from_training_records_enable,  # Plotting during training
    free_bits=free_bits,                # Free bits parameter for KL loss
    truncation=truncation,              # Fraction of sequence to truncate for encoder
    print_epoch=10,                      # Frequency (in epochs) to print training progress
    plot_epoch=1,                       # Frequency to generate plots during training
    max_plots=30,                       # Maximum number of individuals to plot
    nr_col=10,                          # Number of columns in plot grid
    nr_row=3                            # Number of rows in plot grid
)

   #  # Evaluate predictions
   # # mse_mean, r2_mean, mse_median, r2_median, res_df = #
    plot_individual_fits(dataset_train,
        latent_dim, t_dense, models, train_loader, device,
        global_mean, global_std, global_max_time,
        enable_nf, enable_ae, enable_vae, enable_onlymedian,
        truncation=truncation, max_plots=8, nr_row=2, nr_col=4,
        num_simulated_total=100,  # instead of n_samples
        ci_lower=0.05, ci_upper=0.95,
        add_noise_to_prediction=False
    )
    
    
    vpc(models,
        train_loader,
        global_max_dose,
        global_max_time,
        global_mean,
        global_std,
        dataset_train,
        latent_dim,
        dim_parameter_encoder,
        initial_encoder,
        encoder,
        func,
        reducer,
        noise,
        t_dense,
        compartment="DV",
        onlymedian=False,
        enable_nf=False,
        enable_ae=True,
        enable_vae_training=False,
        add_noise_to_prediction=False,
        num_simulated_total=1000,
        truncation=truncation
    )
    
    compute_residuals(dataset_train, latent_dim, global_mean, global_std, global_max_time, func, encoder, reducer, initial_encoder, noise, train_loader, t_dense, device, 0.3)

    plot_individual_fits(dataset_test,
        latent_dim, t_dense, models, train_loader, device,
        global_mean, global_std, global_max_time,
        enable_nf, enable_ae, enable_vae, enable_onlymedian,
        truncation=truncation, max_plots=3, nr_row=1, nr_col=3,
        num_simulated_total=100,  # instead of n_samples
        ci_lower=0.05, ci_upper=0.95,
        add_noise_to_prediction=False
    )
       
    
    total_mse, total_LL, _  = evaluate_on_val(
        global_mean,
        global_std,
        global_max_time,
        global_max_dose,
        enable_nf,
        enable_ae,
        enable_vae,
        enable_onlymedian,
        truncation,
        func,
        noise,
        reducer,
        test_loader,
        device,
        encoder,
        initial_encoder,
        t_dense,
    )
    # print(total_mse)

    r2_mean, r2_median, mse_mean, mse_median, res_df=compute_test_metrics(
   latent_dim, dataset_test, models, device,
   global_mean, global_std, global_max_time,
   enable_nf, enable_ae, enable_onlymedian,t_dense,
   truncation=truncation
    )
    
    print(f"R² (mean):   {r2_mean:.4f}")
    print(f"R² (median): {r2_median:.4f}")
    print(f"MSE (mean):  {mse_mean:.4f}")
    print(f"MSE (median):{mse_median:.4f}")

    
    
    # Append metrics and residuals
    append_metrics(metrics, residuals, variant_name, mse_mean, r2_mean,
                   mse_median, r2_median, mse_validation, res_df)

    # Save metrics/residuals
    export_all_metrics_and_residuals(metrics, residuals, base_dir)
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




    