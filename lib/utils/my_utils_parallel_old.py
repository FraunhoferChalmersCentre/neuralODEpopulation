def map_doses_to_labels(df, dose_col='Dose'):
    """
    Args:
        df (pd.DataFrame): dataset containing a column with dose values
        dose_col (str): name of the dose column

    Returns:
        dose_to_label (dict): mapping from dose value to integer label
        df_labeled (pd.DataFrame): copy of df with dose column replaced by integer labels
    """
    unique_doses = sorted(df[dose_col].unique())
    dose_to_label = {dose: i for i, dose in enumerate(unique_doses)}

    df_labeled = df.copy()
    df_labeled[dose_col] = df[dose_col].map(dose_to_label)

    return dose_to_label, df_labeled

def check_gradients(module, label=""):
    print(f"\n🔍 Gradient Check for {label}:")
    for name, param in module.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"❌ {label}.{name}: No gradient")
            elif torch.all(param.grad == 0):
                print(f"⚠️ {label}.{name}: Zero gradient")
            else:
                print(f"✅ {label}.{name}: Grad norm = {param.grad.norm():.4e}")
def load_model(component, name):
    path = os.path.join(folder, f"final_{name}_{modelname}.pth")
    if os.path.exists(path):
        component.load_state_dict(torch.load(path))
        print(f"Loaded {name} from {path}")
    else:
        print(f"File not found, skipping: {path}")

modelname='MultipleDoseAddError'
def save_model(component, name):
    torch.save(component.state_dict(), os.path.join(folder, f"final_{name}_{modelname}.pth"))


def train_model(latent_dim,
    dataloader, func, reducer, initial_encoder,
    refiner1, refiner2, global_latent, noise,
    optimizer, scheduler, device, t_dense,
    kl_weight, alpha, n_epochs, warmup_epochs,
    smoothing_start_epoch, print_epoch, plot_epoch,
    ODEWrapper, conc_std, conc_mean,
    MAX_TIME, MAX_DOSE
):
    for epoch in range(n_epochs):
    
        total_loss = 0.0
        z_individual_list = []
        trajectory_records = []

        # === Enable or disable noise learning ===
        for param in noise.parameters():
            param.requires_grad = epoch >= warmup_epochs

        for id_list, t_padded, x_padded, mask, dose_tensor, dose_times_list in dataloader:
            t_padded, x_padded, mask = t_padded.to(device), x_padded.to(device), mask.to(device)
            dose_tensor = dose_tensor.to(device)
            dose_times_list = [dt.to(device) for dt in dose_times_list]
            batch_size = t_padded.size(0)

            dose_times_padded, dose_times_mask = pad_dose_times(dose_times_list)

            # === Create dose masks ===
            mask_low = dose_tensor == 0.5
            mask_high = dose_tensor == 1.0

            # === Prepare data for each group ===
            t_low, x_low = t_padded[mask_low], x_padded[mask_low]
            t_high, x_high = t_padded[mask_high], x_padded[mask_high]

            # === Refine latents for each group ===
            mu_q_low, logvar_q_low = refiner1(t_low, x_low, noise.log_sigma_add, noise.log_sigma_prop, global_latent.mu.detach(), global_latent.logvar.detach())
            mu_q_high, logvar_q_high = refiner2(t_high, x_high, noise.log_sigma_add, noise.log_sigma_prop, global_latent.mu.detach(), global_latent.logvar.detach())

            # === Merge low and high dose latent representations ===
            latent_dim = mu_q_low.shape[1]
            mu_q = torch.zeros(batch_size, latent_dim, device=device)
            logvar_q = torch.zeros(batch_size, latent_dim, device=device)
            mu_q[mask_low] = mu_q_low
            mu_q[mask_high] = mu_q_high
            logvar_q[mask_low] = logvar_q_low
            logvar_q[mask_high] = logvar_q_high

            std_q = torch.exp(0.5 * logvar_q)
            eps = torch.randn_like(std_q)
            z_refined = mu_q + eps * std_q
            mask_z = (torch.rand(batch_size) < alpha).to(device)
            z_input = torch.where(mask_z.unsqueeze(-1), z_refined, global_latent.mu.detach().unsqueeze(0).expand(batch_size, -1))

            # === ODE Prediction ===
            x0_latent = initial_encoder(torch.cat([x_padded[:, 0].unsqueeze(1)] * 2, dim=1))
            x0 = torch.cat([x0_latent, dose_tensor.unsqueeze(-1)], dim=-1)
            
            ode_func = ODEWrapper(func, dose_times_padded, dose_times_mask, z_input)
            pred = odeint(ode_func, x0, t_dense, method='dopri5')
            pred_batch = pred.permute(1, 0, 2)  # [batch, time, features]

            # === Interpolate and Calculate Loss ===
            t_dense_exp = t_dense.unsqueeze(0).repeat(batch_size, 1)
            test = reducer(pred_batch[:, :, :4])
            pred_interp = batch_linear_interpolate_1d(test, t_dense_exp, t_padded)
            recon_loss_noise = noise.nll(x_padded, pred_interp, mask)
            KL_loss = kl_divergence_gaussians(mu_q_low, logvar_q_low, global_latent.mu.detach(), global_latent.logvar.detach()) + \
                      kl_divergence_gaussians(mu_q_high, logvar_q_high, global_latent.mu.detach(), global_latent.logvar.detach())
            loss = recon_loss_noise + kl_weight * KL_loss

            # === Backprop ===


            loss.backward()
   
            optimizer.step()
  

            optimizer.zero_grad()
            residual = x_padded - pred_interp
    
            # === Store for analysis ===
            z_individual_list.append(z_refined.detach())
            for i in range(batch_size):
                trajectory_records.append((
                    id_list[i],
                    t_padded[i, mask[i]],
                    x_padded[i, mask[i]],
                    dose_tensor[i],
                    dose_times_list[i],
                    z_refined[i].detach()
                ))

        # === Global latent and EMA updates ===
        z_all = torch.cat(z_individual_list, dim=0)
        with torch.no_grad():
            if epoch < smoothing_start_epoch:
                global_latent.update(z_all, gamma=1.0)
            else:
                global_latent.update(z_all)
                func.update_ema(alpha=0.1)
                reducer.update_ema(alpha=0.1)
                initial_encoder.update_ema(alpha=0.1)
                refiner1.update_ema(alpha=0.1)
                refiner2.update_ema(alpha=0.1)

        scheduler.step()

        # === Logging ===
        if epoch % print_epoch == 0:
            with torch.no_grad():
                print(
                    f"Epoch {epoch}, "
                    f"-LL: {recon_loss_noise.item():.4f}, "
                    f"KL loss: {KL_loss.item():.4f}, "
                    f"Add. error: {(conc_std * torch.exp(noise.log_sigma_add)).item():.4f}, "
                    f"Prop. error: {torch.exp(noise.log_sigma_prop).item():.4f}, "
                    f"Global Latent Std: {np.round(torch.exp(0.5 * global_latent.logvar).cpu().numpy(), 4)}"
                    f"Residual std:, {residual.std().item()}, "
                    f"Estimated sigma:, {torch.exp(noise.log_sigma_add).item():4f}"
                    f"Residual std:, {residual.std().item():4f}"
                )

        if epoch % plot_epoch == 0:
            plot_from_training_records(
                records=trajectory_records,
                func=func,
                reducer=reducer,
                initial_encoder=initial_encoder,
                ODEWrapper=ODEWrapper,
                t_dense=t_dense,
                conc_mean=conc_mean,
                conc_std=conc_std,
                max_plots=6,
                MAX_TIME=MAX_TIME,
                MAX_DOSE=MAX_DOSE
            )
