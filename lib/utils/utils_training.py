import copy
import gc
import time
import torch



from lib.utils.utils_preprocess import destandardize_concentration
from lib.utils.utils_post_processing import compute_residuals, plot_individual_fits
from lib.utils.utils_shared import encode_latent, preprocess_batch, prepare_ode_input, make_predictions, normalize_encoder_input






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
    dim_parameter_encoder = func.dim_parameters_dynamic
    latent_dim = func.dim_latent
       
    
    best_val_mse = float("inf")
    best_val_LL = float("inf")
    best_val_loss = float("inf")

    best_model_state = {k: v.state_dict() for k, v in models.items()}  # initial model state

    epochs_no_improve = 0
    patience = 30  # stop if no improvement for 30 epochs

    batch_size = dataloader.batch_size
    t_dense = t_dense.to(device)

    mse_history = []

    return (device, latent_dim, dim_parameter_encoder, best_val_mse, best_val_LL,best_val_loss, 
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
    encoder,
    x_padded, mask, pred_interp,
    mu_q, L_q,    # full-covariance outputs
    mu_p, L_p,    # from encoder.get_prior(batch_size)
    epoch, warmup_epochs_iiv, free_bits,
    enable_ae, enable_vae,
    noise, global_mean, global_std,
    only_median_training=False,
    replace_mask=None, repeat_factor=1
):
    """
    Computes reconstruction + KL losses and metrics for full-covariance encoder.
    Now uses a correlated prior p(z) = N(mu_p, L_p L_p^T)
    and posterior q(z|x) = N(mu_q, L_q L_q^T).
    """

    device = x_padded.device

    # === Repeat data if needed ===
    if repeat_factor > 1:
        mask = mask.repeat_interleave(repeat_factor, dim=0)
        x_padded = x_padded.repeat_interleave(repeat_factor, dim=0)

    # === Reconstruction loss ===
    recon_loss_noise, mse = noise.nll(
        destandardize_concentration(x_padded, global_mean, global_std),
        pred_interp, replace_mask, mask
    )

    # === KL warmup and free bits ===
    free_bits_on = (
        0 if epoch + 1 >= warmup_epochs_iiv
        else free_bits * (1 - min(1.0, epoch / warmup_epochs_iiv))
    )
    kl_weight = (
        1.0 if epoch + 1 >= warmup_epochs_iiv
        else min(1.0, epoch / warmup_epochs_iiv)
    )
    
  #  mu_p = torch.zeros_like(mu_q)
  #  logvar_p = torch.zeros_like(L_q)  # since exp(0) = 1 ⇒ σ² = 1

    # === Full-covariance KL ===
    KL_loss, denom = kl_divergence_gaussians(mu_q, L_q, mu_p, L_p, free_bits_on)
    #KL_loss=kl_divergence(mu_q,L_q,mu_p,logvar_p, free_bits_on)
    # === Combine losses ===
    loss = KL_loss + recon_loss_noise
    return loss, recon_loss_noise, mse, KL_loss








def accumulate_epoch_metrics(
    total_loss, total_mse, total_kl, total_recon,
     loss, mse, kl_gauss, recon_loss_noise, enable_ae
):
    total_loss += loss.item()
    total_mse += mse.item()
    total_kl += kl_gauss.item() if not enable_ae else 0.0
    total_recon += recon_loss_noise.item()
    return total_loss, total_mse, total_kl, total_recon


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
            print(f"[Early Stop Monitor] No improvement for {epochs_no_improve} epochs, current val loss {val_loss:.1f}, best val loss {best_val_loss:.1f}")

        if epochs_no_improve >= patience:
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

    return best_val_mse, best_val_LL, best_val_loss, epochs_no_improve, stop_training, best_model_state

def log_training_epoch(end_time, epoch, total_mse, total_loss, total_recon, total_kl, total_transform,
                       optimizer, noise, dataset, batch_size,num_batches, val_mse=None, val_LL=None,
                       enable_ae=False, enable_vae=False, print_epoch=1):
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
        enable_vae: bool flag
        print_epoch: frequency of printing
    """
    if epoch % print_epoch != 0:
        return

    with torch.no_grad():
        main_lr = optimizer.param_groups[0]['lr']
       # end_time = time.time()
        add_error = float(noise.sigma_add)

        if enable_ae:
            print(f"Epoch {epoch}, "
                  f"MSE {total_mse/num_batches:.4f} "
                  f"-LL: {total_recon/num_batches:.4f}, "
                  f"Add. error: {add_error:.2f}, "
            #      f"val_mse: {val_mse:.3f}, "
            #      f"val_LL: {val_LL:.3f}, "
                  f"lr: {main_lr:.3f}, "
                  f"{end_time:.2f} seconds")
        elif enable_vae:
            print(f"Epoch {epoch}, "
                  f"MSE {total_mse/num_batches:.1f} "
                  f"loss: {total_loss/num_batches:.1f}, "
                  f"-LL: {total_recon/num_batches:.1f}, "
                  f"KL: {total_kl/num_batches:.8f}, "
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
    global_min,
    global_max_time,
    global_max_dose,
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
        id_list, t_padded, x_padded,t_encoder, x_encoder,t_cut, x_cut,  mask, dose_tensor, dose_times_list, auc_tensor = preprocess_batch(
    batch, device, truncation=truncation
)
   
        # --- Encode latent ---
        k_param, mu_q, logvar_q, log_det = encode_latent(
        encoder,
        t_encoder,
        x_encoder,
        enable_vae=enable_vae,
        enable_ae=enable_ae,
        enable_onlymedian=enable_onlymedian
        )
        
        # --- Prepare ODE input ---
        x0, ode_func = prepare_ode_input(
            initial_encoder,
            x_padded,
            k_param,
            func,
            dose_tensor,
            dose_times_list,
            True
        )
        
        
        
        
        # --- Compute predictions ---
    

        # --- Compute loss and metrics ---
        pred_interp, pred_batch = make_predictions(
            t_padded, t_dense, x0, ode_func, reducer, latent_dim,global_mean,global_min
        )
        
        
        # --- Debug prints ---
       
        # --- Compute reconstruction loss ---
        recon_loss_noise, mse = noise.nll(
        destandardize_concentration(x_padded, global_mean, global_min),
        pred_interp
        )
        
        loss, recon_loss_noise, mse, KL_loss, log_det_sum, log_det_penalty, kl_gauss = compute_loss_and_metrics(
            x_padded, mask, pred_interp,
            mu_q, logvar_q, log_det,
            10, 0, 0,
            enable_ae, enable_vae,
            noise, global_mean, global_min
        )
        
        
        # --- Accumulate metrics ---
        total_mse += mse.item()
        total_LL += recon_loss_noise.item()
        total_loss += loss.item()

    return total_mse/num_batches, total_LL/num_batches, total_loss/num_batches


def replace_with_gaussian_noise(k_param: torch.Tensor, p: float):
    batch_size, latent_dim = k_param.shape
    device, dtype = k_param.device, k_param.dtype

    replace_mask = torch.rand(batch_size, device=device) < p
    noise = torch.randn(batch_size, latent_dim, device=device, dtype=dtype)
    out = torch.where(replace_mask[:, None], noise, k_param)

    return k_param, replace_mask  # return both

def set_all_train(models_dict, extra_models=None):
    """
    Ensures all models are set to training mode.
    
    Args:
        models_dict (dict): dictionary of models, e.g. {"func": func_vae, "encoder": encoder_vae, ...}
        extra_models (list): optional list of additional models not in models_dict
    """
    for m in models_dict.values():
        if m is not None:
            m.train()
    if extra_models is not None:
        for m in extra_models:
            if m is not None:
                m.train()
def freeze_all_modules(*modules):
    """
    Freezes the parameters of all modules passed as arguments.
    """
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False


def train_loop_model(func_med, reducer_med, encoder_med, 
    dataset, dataset_val, global_max_time, global_max_dose,
    global_mean, global_std, main_params, dataloader_val, dataloader, models,
    optimizer, scheduler, func, reducer, encoder, noise, t_dense,
    n_epochs, warmup_epochs_noise, warmup_epochs_iiv, smoothing_start_epoch,learn_prior,
     enable_ae,enable_vae,
    enable_onlymedian,normalization,
    truncation, print_epoch=1, plot_epoch=1, max_plots=4,
    nr_col=1, nr_row=5, free_bits=1
):
    # === Initialize training ===
    device, latent_dim, dim_parameter_encoder, best_val_mse, best_val_LL, best_val_loss, \
    epochs_no_improve, patience, batch_size, t_dense, mse_history = initialize_training(
        models, func, dataloader, t_dense, global_max_time
    )


    gc.collect()

     # Free unused GPU memory
    torch.cuda.empty_cache()
 
    
    encoder_med.eval()
    reducer_med.eval()
    func_med.eval()
    set_all_train(models)
    num_batches = len(dataloader)
    


    
    for epoch in range(n_epochs):
        # After loss.backward()
   #     encoder.freeze_prior_until(epoch, freeze_epochs=10)  # freeze first 10 epochs

        
            
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
             treatment_list,
             t_padded,          # padded time
             x_global_padded,   # padded global features
             mask,              # time mask
             dose_tensor,       # dose amounts
             dose_times_list,   # dose start times
             evid
       

         ) = batch
            
       
            
            #mask=mask.to(device)
            
            # --- Preprocess batch ---
            # --- Preprocess batch ---
            id_list,treatment_list, t_padded, x_padded,  t_encoder, x_encoder,t_cut, x_cut, mask, dose_tensor, dose_times_list, evid = preprocess_batch(
                batch, device, truncation=truncation
            )
           
            
            
            if normalization:
                x_encoder = normalize_encoder_input( x_encoder, t_encoder, x_padded, dose_tensor, dose_times_list, evid,
                 encoder_med, func_med, reducer_med,
                 t_dense, global_mean, global_std)
            
            k_param,_, mu_q, logvar_q,mu_p,logvar_p,mask_dropout, repeat_factor = encode_latent(
                encoder,
                t_encoder,
                x_encoder,
                dose_tensor,
                enable_vae=enable_vae,
                enable_ae=enable_ae,
                enable_onlymedian=enable_onlymedian, 
                warmup_epochs_iiv=warmup_epochs_iiv, 
                epoch=epoch,
                augment=True
                
            )
            
         
           # k_param, mask_dropout = replace_with_gaussian_noise(k_param,p_dropout)

          
            
            ode_func= prepare_ode_input(

                x_padded,
                k_param,
                func,
                dose_tensor,
                dose_times_list,
                evid,
                enable_ae,
                enable_onlymedian,
                repeat_factor
                
            )
            # if enable_onlymedian:
            #     k_param = k_param*0
            
            pred_interp, pred_batch = make_predictions(
                t_padded, t_dense, k_param, ode_func, reducer,global_mean,global_std,repeat_factor
            )
       

          #  print(x_padded, x_padded.size())
            #print(t_padded, t_padded.size())
         
            
            # 2. Compute losses
            loss, recon_loss_noise, mse, KL_loss = compute_loss_and_metrics(encoder,
                
                x_padded, mask, pred_interp,
                mu_q, logvar_q,
                mu_p,logvar_p,
                epoch, warmup_epochs_iiv, free_bits,
                enable_ae, enable_vae,
                noise, global_mean, global_std, enable_onlymedian, mask_dropout,repeat_factor
            )
       
            
             
            total_loss, total_mse, total_kl, total_recon = accumulate_epoch_metrics(
                 total_loss, total_mse, total_kl, total_recon,
                 loss , mse, KL_loss, recon_loss_noise, enable_ae
            )
     
            
     
         
       
           
            # --- Backpropagation ---
            run_backprop_step(loss , main_params, optimizer)



            first_batch = False
            if epoch >= smoothing_start_epoch:
                for m in [models, encoder, func, noise, reducer]:
                    if hasattr(m, "update_ema"):
                        m.update_ema(alpha=0.05)
            
        # --- Post-epoch operations ---
        mse_history.append(total_mse)
        scheduler.step(total_mse)
        end_time = time.time()
        epoch_duration = end_time - start_time
     
                
                
        # if plot_training and  epoch % print_epoch != 0:
        #     plot_individual_fits(
        #         latent_dim, t_dense, models, dataloader, device,
        #         global_mean, global_std, global_max_time,
        #         enable_vae, enable_ae, enable_onlymedian,
        #         truncation=truncation, max_plots=max_plots, nr_row=nr_row, nr_col=nr_col, n_samples=10, ci_lower=0.05, ci_upper=0.95
        #     )

        # --- Validation and early stopping ---
        # if traing_against_validation and epoch > warmup_epochs_iiv:
       
            
        #     val_mse, val_LL, val_loss = evaluate_on_val(
        #         global_mean, global_std, global_max_time, global_max_dose,
        #         enable_vae, enable_ae,enable_vae,enable_onlymedian, truncation,
        #         func, noise, reducer, dataloader_val,
        #         device, encoder, initial_encoder, t_dense
        #     )
            
        #     best_val_mse, best_val_LL,best_val_loss, epochs_no_improve, stop_training, best_model_state = \
        #         validate_and_update_early_stop(
        #             epoch, warmup_epochs_noise, traing_against_validation,
        #             val_mse, val_LL,val_loss, best_val_mse, best_val_LL,best_val_loss, epochs_no_improve,
        #             patience, models, enable_vae
        #         )
        #     if stop_training:
        #         break
        # else:
        val_mse=0
        val_LL=0
        # --- Logging ---
        
        log_training_epoch(epoch_duration,
            epoch, total_mse, total_loss, total_recon, total_kl, total_transform,
            optimizer, noise, dataset, batch_size,num_batches,
            val_mse=val_mse, val_LL=val_LL,
            enable_ae=enable_ae,
            enable_vae=enable_vae,
            print_epoch=print_epoch
        )

   

        torch.cuda.empty_cache()
        gc.collect()

    return min(mse_history)/num_batches, best_val_mse




def run_model_variant(variant_name, train_dataset, val_dataset, test_dataset,
                      models, main_params, optimizer, scheduler, combined,
                      global_max_dose, global_max_time, global_mean, global_min,
                      latent_dim, noise, encoder, func, reducer, initial_encoder,
                      ODEWrapper, device, metrics, residuals, iteration,
                      n_epochs, train_loader, val_loader,test_loader,  base_dir,
                      warmup_noise, warmup_iiv, enable_ae, enable_vae,enable_onlymedian, plot_training=False,
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
    global_min,                     # Global standard deviation of observations
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
    enable_ae=enable_ae,      # Whether to train autoencoder components
    enable_vae=enable_vae,      # Whether to train normalizing flows
    enable_onlymedian=enable_onlymedian,  # Whether to train only median predictions
    plot_training=plot_training,  # Plotting during training
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
       global_mean, global_min, global_max_time,
       enable_vae, enable_ae, enable_onlymedian,
       truncation=truncation, max_plots=3, nr_row=1, nr_col=3, n_samples=1000, ci_lower=0.025, ci_upper=0.975)
       
    
    total_mse, total_LL = evaluate_on_val(
        global_mean,
        global_min,
        global_max_time,
        global_max_dose,
        enable_vae,
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
  # otal_mse)
    compute_residuals(latent_dim, global_mean, global_min, func, encoder, reducer, initial_encoder, noise, train_loader, combined, device, 0.35)

   #  r2, mse=compute_test_loss(
   #     latent_dim, combined, models, test_dataset, device,
   #     global_mean, global_min, global_max_time,
   #     enable_vae, enable_vae, enable_onlymedian,
   #     truncation=truncation, max_plots=25, nr_row=5, nr_col=5, n_samples=2, ci_lower=0.05, ci_upper=0.95
   # )
    
    # Append metrics and residuals
  #  append_metrics(metrics, residuals, variant_name, mse_mean, r2_mean,
   #                mse_median, r2_median, mse_validation, res_df)

    # Save metrics/residuals
  #  export_all_metrics_and_residuals(metrics, residuals, base_dir)
  #  return mse_train 

def kl_divergence_gaussians(mu_q, L_q, mu_p, L_p, free_bits=0.0, eps=1e-6):
    """
    Compute KL[q||p] between two full-covariance Gaussians:
        q = N(mu_q, Σ_q = L_q L_q^T)
        p = N(mu_p, Σ_p = L_p L_p^T)

    Supports batch of size B:
        mu_q, mu_p: [B, D]
        L_q, L_p: [B, D, D]
    """
    B, D = mu_q.shape
    device = mu_q.device

    # Ensure L_q and L_p are batch 3D
    if L_q.dim() == 2:
        L_q = L_q.unsqueeze(0).expand(B, D, D)
    if L_p.dim() == 2:
        L_p = L_p.unsqueeze(0).expand(B, D, D)

    # Covariance matrices
    Sigma_q = torch.bmm(L_q, L_q.transpose(1, 2))  # [B, D, D]
    Sigma_p = torch.bmm(L_p, L_p.transpose(1, 2))  # [B, D, D]

    # Log-determinant terms
    logdet_q = 2 * torch.sum(torch.log(torch.diagonal(L_q, dim1=1, dim2=2) + eps), dim=1)
    logdet_p = 2 * torch.sum(torch.log(torch.diagonal(L_p, dim1=1, dim2=2) + eps), dim=1)

    # Inverse of prior covariance
    Sigma_p_inv = torch.linalg.inv(Sigma_p)

    # Trace term
    trace_term = torch.einsum('bij,bji->b', Sigma_p_inv, Sigma_q)

    # Quadratic term
    diff = (mu_p - mu_q).unsqueeze(-1)  # [B, D, 1]
    quad_term = torch.einsum('bik,bkj,bji->b', diff.transpose(1, 2), Sigma_p_inv, diff)
  #  print(Sigma_q)
    # KL
    kl = 0.5 * (trace_term + quad_term - D + (logdet_p - logdet_q))

    # Free bits
    if free_bits > 0:
        kl = torch.clamp(kl, min=free_bits * D)

    return kl.mean(), B




# def kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p, free_bits=0.0):
#     """
#     KL[q||p] between two diagonal Gaussians with optional mask and free bits.
    
#     Args:
#         mu_q, logvar_q: [B, D] approximate posterior
#         mu_p, logvar_p: [B, D] prior
#         free_bits: float, minimum KL per dim to prevent collapse
#         mask: optional tensor of shape [B], [B, 1], or [B, D] (1=keep, 0=drop)
    
#     Returns:
#         Scalar: mean KL divergence with masking and free bits
#     """
#     var_q = torch.exp(logvar_q)
#     var_p = torch.exp(logvar_p)
    
    
#     kl_per_dim = 0.5 * ((var_q + (mu_q - mu_p) ** 2) / var_p - 1 + logvar_p - logvar_q)  # [B, D]
#     kl=kl_per_dim.sum(dim=1)
#     # Apply free bits per dim
#     if free_bits > 0:
#         kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
   
#     denom = kl.shape[0]  # number of samples

#     return kl.mean(), denom


    