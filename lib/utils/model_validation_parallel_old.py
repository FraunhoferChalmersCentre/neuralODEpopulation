# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 11:18:14 2025

@author: Baaz
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 11:16:15 2025

@author: Baaz
"""

def plot_individual_samples_vs_data_vectorized(
    individual_data,noise,global_latent,  # (id, t_real, x_real, dose, dose_times)
    refiner1, refiner2,
    func, reducer, initial_encoder, ODEWrapper,
    t_dense, conc_mean, conc_std, MAX_TIME, MAX_DOSE,
    n_samples=100,
    n_mcmc_samples=1000,
    burn_in=10,
    device='cpu'
):

    

    t_real, x_real, dose, dose_times, _id = individual_data
    t_np = t_real.cpu().numpy() * MAX_TIME
    y_obs = x_real.cpu().numpy() * conc_std + conc_mean  # de-standardize observed data

    # MCMC Sampling
    mu_prior = np.log(np.array([1.0, 3.0]))  # log(ka, cl)
    sigma_prior = np.array([0.5, 0.5])
    add_error = 2
    prop_error = 0.001

    mcmc_samples_log = metropolis_hastings_sampling(
        y_obs=y_obs,
        t_np=t_np,
        dose=dose * MAX_DOSE,
        dose_times=(dose_times.cpu().numpy() * MAX_TIME),
        add_error=add_error,
        prop_error=prop_error,
        mu_prior=mu_prior,
        sigma_prior=sigma_prior,
        n_samples=n_mcmc_samples,
        burn_in=burn_in,
        thinning=1
    )
    mcmc_samples = np.exp(mcmc_samples_log)  # shape (n_samples, 2)
    # === Plot correlation scatter of MCMC samples ===
    plt.figure(figsize=(6, 6))
    plt.scatter(mcmc_samples[:, 0], mcmc_samples[:, 1], alpha=0.3, s=10, color='purple')
    plt.xlabel('ka')
    plt.ylabel('cl')
    plt.title('Scatter plot of MCMC samples (ka vs cl)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Plot histogram of MCMC samples ===
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(mcmc_samples[:, 0], bins=30, color='skyblue', edgecolor='k')
    plt.xlabel('ka')
    plt.ylabel('Frequency')
    plt.title('Histogram of MCMC samples for ka')

    plt.subplot(1, 2, 2)
    plt.hist(mcmc_samples[:, 1], bins=30, color='salmon', edgecolor='k')
    plt.xlabel('cl')
    plt.ylabel('Frequency')
    plt.title('Histogram of MCMC samples for cl')

    plt.tight_layout()
    plt.show()

    # ... rest of your existing code for ODE simulation and plotting ...
    # (keep everything unchanged from here on)
    # ODE simulation for each MCMC sample
    ka = mcmc_samples[:, 0]
    cl = mcmc_samples[:, 1]
    v = np.log(5.0)
    dose_amount = dose * MAX_DOSE
    dose_times_np = MAX_TIME * dose_times.cpu().numpy() if torch.is_tensor(dose_times) else dose_times
    t_dense_np = MAX_TIME* t_dense.cpu().numpy() if torch.is_tensor(t_dense) else t_dense
    
    all_C2 = solve_individual_vectorized(ka, cl, v, dose_amount, dose_times_np, t_dense_np)


    
    all_C2_torch = torch.tensor(all_C2, device=device)  # move to torch tensor
    
    perc10_ode = torch.quantile(all_C2_torch, 0.10, dim=0).cpu().numpy()
    median_ode = torch.median(all_C2_torch, dim=0).values.cpu().numpy()
    perc90_ode = torch.quantile(all_C2_torch, 0.90, dim=0).cpu().numpy()

    # === Latent ODE prediction ===
    dose = dose.item()
    t_real, x_real, dose_times = t_real.to(device), x_real.to(device), dose_times.to(device)
    refiner = refiner1 if dose == 0.5 else refiner2

    t_real_rep = t_real.unsqueeze(0).repeat(n_samples, 1)
    x_real_rep = x_real.unsqueeze(0).repeat(n_samples, 1)

    mu, logvar = refiner(t_real_rep, x_real_rep, noise.log_sigma_add, noise.log_sigma_prop, global_latent.mu.detach(), global_latent.logvar.detach())
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z_samples = mu + eps * std

    x0_latent = initial_encoder(torch.cat([x_real[0].unsqueeze(0).unsqueeze(1)] * 2, dim=1))
    x0_latent_rep = x0_latent.repeat(n_samples, 1)

    dose_tensor = torch.full((n_samples, 1), dose, device=device)
    x0 = torch.cat([x0_latent_rep, dose_tensor], dim=-1)

    dose_times_padded = torch.nn.utils.rnn.pad_sequence([dose_times], batch_first=True).to(device)
    dose_mask = (dose_times_padded != 0).to(device)
    dose_times_padded_rep = dose_times_padded.repeat(n_samples, 1)
    dose_mask_rep = dose_mask.repeat(n_samples, 1)

    ode_func = ODEWrapper(func, dose_times_padded_rep, dose_mask_rep, z_samples)
    pred = odeint(ode_func, x0, t_dense.to(device), method='dopri5')  # [time, n_samples, latent_dim+1]

    x_pred = reducer(pred[:, :, :x0_latent.shape[-1]])
    x_preds = x_pred.squeeze(-1).permute(1, 0)

    perc10 = torch.quantile(x_preds, 0.10, dim=0)
    median = torch.quantile(x_preds, 0.50, dim=0)
    perc90 = torch.quantile(x_preds, 0.90, dim=0)

    x_real_interp = batch_linear_interpolate_1d(
        x_real.unsqueeze(0), t_real.unsqueeze(0), t_dense.unsqueeze(0)
    ).squeeze(0)

    perc10_real = destandardize_concentration(perc10, conc_mean, conc_std).detach().cpu().numpy()
    median_real = destandardize_concentration(median, conc_mean, conc_std).detach().cpu().numpy()
    perc90_real = destandardize_concentration(perc90, conc_mean, conc_std).detach().cpu().numpy()
    x_real_de = destandardize_concentration(x_real, conc_mean, conc_std).detach().cpu().numpy()

    time_hours = t_dense.cpu().numpy() * MAX_TIME
   
    # === Plot both MCMC and Latent ODE in one plot ===
    plt.figure(figsize=(10, 6))
 
    # MCMC (ODE) predictions in blue
    plt.fill_between(t_dense_np, perc10_ode, perc90_ode, color='blue', alpha=0.3, label='MCMC ODE 10-90% CI')
    plt.plot(t_dense_np, median_ode, color='blue', label='MCMC ODE Median')

    # Latent ODE predictions in red
    plt.fill_between(time_hours, perc10_real, perc90_real, color='red', alpha=0.3, label="Latent ODE 10-90% CI")
    plt.plot(time_hours, median_real, color='red', label='Latent ODE Median')

    # Observed data
    plt.scatter(t_real.cpu().numpy() * MAX_TIME, x_real_de, color='black', label='Observed data', zorder=5)

    plt.xlabel("Time (hours)")
    plt.ylabel("Concentration")
    plt.title(f"Individual {_id} - Dose {dose * MAX_DOSE:.2f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_individuals(ebes, num_plots, func, initial_encoder, reducer, torch_linear_interpolate,
                     destandardize_concentration, conc_mean, conc_std, t_dense,
                     cols=5, rows=None):
    n = min(num_plots, len(ebes))
    if rows is None:
        rows = (n + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(30, 20), squeeze=False)

    for idx, (individual_id, t, x_true, dose, dose_times, best_z) in enumerate(ebes[:n]):
        ax = axs[idx // cols, idx % cols]

        t = t.to('cpu')
        x_true = x_true.to('cpu')
        dose = dose if not isinstance(dose, torch.Tensor) else dose.to('cpu')
        dose_times = dose_times if not isinstance(dose_times, torch.Tensor) else dose_times.to('cpu')
        best_z = best_z.to('cpu')

        x0_latent = initial_encoder(torch.cat([x_true[0].unsqueeze(0)] * 2, dim=0))
        x0 = torch.cat([x0_latent, dose.unsqueeze(0)], dim=-1)

        t_dense_cpu = t_dense.to('cpu')

        pred = odeint(lambda t_, x_: func(t_, x_, dose_times, best_z),
                      x0, t_dense_cpu, method='rk4')
        x_pred = reducer(pred)

        x_pred_denorm = destandardize_concentration(x_pred, conc_mean, conc_std)
        x_pred_at_t = torch_linear_interpolate(t_dense_cpu, x_pred_denorm, t)

        ax.plot(t.numpy(), destandardize_concentration(x_true, conc_mean, conc_std).numpy(), 'o-', label='True')
        ax.plot(t.numpy(), x_pred_at_t.detach().numpy(), 'x--', label='Predicted')

        ax.set_title(f'Individual ID: {individual_id}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration')
        ax.legend()

    for i in range(n, rows * cols):
        fig.delaxes(axs[i // cols, i % cols])

    plt.tight_layout()
    plt.show()


def plot_ebes_vs_true_params(ebes, data_path=None, df=None, figsize=(15, 15)):
    """
    Plot EBEs vs true parameters ka and cl, colored by dose.
    
    Args:
        ebes: list of tuples, where each tuple has individual_id and best_z as last element.
        data_path: str, path to CSV data file. Optional if df is provided.
        df: pandas DataFrame, optional pre-loaded dataframe.
        figsize: tuple, size of the figure.
    """
    if df is None:
        if data_path is None:
            raise ValueError("Either data_path or df must be provided.")
        df = pd.read_csv(data_path)
    
    # Extract true params and dose for all subjects at time 0, indexed by ID
    df_zero = df[df['Time'] == 0].set_index("ID")
    true_params_df = df_zero[['ka', 'cl']]
    dose_values = df_zero['Dose']

    # Extract IDs and EBE vectors from ebes
    ids = [record[0] for record in ebes]
    ebe_vectors = np.stack([record[-1].numpy() for record in ebes])  # best_z is last element

    # Align true parameters and doses to EBEs by ID
    true_params = true_params_df.loc[ids].values
    dose_values_aligned = dose_values.loc[ids].values

    # Normalize doses for coloring
    unique_doses = np.unique(dose_values_aligned)
    dose_to_color = {dose: i for i, dose in enumerate(unique_doses)}
    colors = [dose_to_color[d] for d in dose_values_aligned]

    # Create color map
    cmap = plt.get_cmap("viridis", len(unique_doses))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    true_params_cols = ['ka', 'cl']

    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            scatter = ax.scatter(
                ebe_vectors[:, i],
                true_params[:, j],
                c=colors,
                cmap=cmap,
                alpha=0.7,
                edgecolor='k',
                linewidth=0.5
            )
            ax.set_xlabel(f"EBE dim {i+1}")
            ax.set_ylabel(f"True param {true_params_cols[j]}")
            corr = np.corrcoef(ebe_vectors[:, i], true_params[:, j])[0, 1]
            ax.set_title(f"Corr = {corr:.3f}")

    # Color legend
    if len(unique_doses) > 1:
        legend_handles = [
            mpatches.Patch(color=cmap(dose_to_color[d] / (len(unique_doses) - 1)), label=f"Dose: {int(d)}")
            for d in unique_doses
        ]
    else:
        legend_handles = [
            mpatches.Patch(color=cmap(0.5), label=f"Dose: {int(unique_doses[0])}")
        ]

    fig.legend(handles=legend_handles, title="Dose", loc='upper right')

    plt.suptitle("EBEs vs True Parameters (Colored by Dose)", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



def rf_regression_and_plot(X, data_path, true_params_cols=None, test_size=0.3, random_state=42):
    """
    Train Random Forest regressors to predict true parameters from EBE vectors.
    True parameters are extracted from a CSV file based on given IDs.
    Extracts EBE vectors from last element of each record in X.

    Args:
        X (list or array-like): List of records where the last element is EBE tensor.
        ids (list or array-like): IDs corresponding to each record in X.
        data_path (str): CSV path containing true parameter data.
        true_params_cols (list of str, optional): Parameter column names in CSV (default ['ka', 'cl']).
        test_size (float): Fraction for test split.
        random_state (int): Random seed.

    Returns:
        rf, cv_scores, y_test, y_pred
    """
    if true_params_cols is None:
        true_params_cols = ['ka', 'cl']
        
    # Load true params at time 0, index by ID
    df = pd.read_csv(data_path)
    df_zero = df[df['Time'] == 0].set_index("ID")
    true_params_df = df_zero[true_params_cols]
    ids = [record[0] for record in X]
    # Extract true param values for the given IDs
    y = true_params_df.loc[ids].values

    # Extract EBE vectors (last element, assumed torch.Tensor) and convert to numpy
    ebe_vectors = np.stack([record[-1].cpu().numpy() if hasattr(record[-1], 'cpu') else record[-1] for record in X])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        ebe_vectors, y, test_size=test_size, random_state=random_state
    )

    # Setup RF and CV
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    cv = KFold(n_splits=10, shuffle=True, random_state=random_state)

    # Cross-validate on training data
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='r2')
    print(f"Mean CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit on full training data
    rf.fit(X_train, y_train)

    # Predict on test set
    y_pred = rf.predict(X_test)

    # Plot results
    fig, axes = plt.subplots(1, y.shape[1], figsize=(6 * y.shape[1], 5))
    if y.shape[1] == 1:
        axes = [axes]  # ensure iterable

    for i, ax in enumerate(axes):
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.7)
        ax.plot(
            [y_test[:, i].min(), y_test[:, i].max()],
            [y_test[:, i].min(), y_test[:, i].max()],
            'r--'
        )
        ax.set_xlabel(f"True {true_params_cols[i]}")
        ax.set_ylabel(f"Predicted {true_params_cols[i]}")
        ax.set_title(f"{true_params_cols[i]}: $R^2$ = {r2:.2f}")

    plt.suptitle("Random Forest Regression: Test Set True vs Predicted", fontsize=18)
    plt.tight_layout()
    plt.show()

    return rf, cv_scores, y_test, y_pred

def estimate_ebes_for_all(dataset, trajectory_records, func, global_latent, refiner, initial_encoder,
                          reducer, torch_linear_interpolate, t_dense, n_steps=10, lr=1e-2, device='cpu'):
    func.to(device)
    global_latent.to(device)
    refiner.to(device)

    results = []  # List of tuples (individual_id, t, x_true, dose, dose_times, best_z)
    mse_loss = torch.nn.MSELoss()

    for idx in range(len(dataset)):
        individual_id, t, x_true, dose, dose_times, z_refined_saved, z_sampled = trajectory_records[idx]

        t = t.to(device)
        x_true = x_true.to(device)
        dose = dose.to(device) if isinstance(dose, torch.Tensor) else dose
        dose_times = dose_times.to(device) if isinstance(dose_times, torch.Tensor) else dose_times

        z = z_refined_saved.clone().detach().to(device).requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)

        best_loss = float('inf')
        best_z = None

        for step in range(n_steps):
            optimizer.zero_grad()

            x0_latent = initial_encoder(torch.cat([x_true[0].unsqueeze(0)] * 2, dim=0))
            x0 = torch.cat([x0_latent, dose.unsqueeze(0)], dim=-1)

            pred = odeint(lambda t_, x_: func(t_, x_, dose_times, z),
                          x0, t_dense.to(device), method='rk4')
            x_pred = reducer(pred[:, :2])

            x_pred_at_t = torch_linear_interpolate(t_dense.to(device), x_pred, t)

            loss = mse_loss(x_pred_at_t, x_true)
            loss.backward()

            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_z = z.detach().clone()

        results.append((individual_id, t.cpu(), x_true.cpu(),
                        dose if not isinstance(dose, torch.Tensor) else dose.cpu(),
                        dose_times if not isinstance(dose_times, torch.Tensor) else dose_times.cpu(),
                        best_z.cpu()))

    torch.save(results, "ebes_with_data.pt")
    return results



def simulate_and_plot_by_dose_refined(
    dataset, global_latent, refiner, initial_encoder, func, reducer, noise, ODEWrapper,
    t_dense, conc_mean, conc_std, MAX_TIME, compartment, num_simulated_total=500
):
    import matplotlib.pyplot as plt
    import numpy as np
    from torchdiffeq import odeint

    device = next(func.parameters()).device
    unique_doses = sorted(set(entry[2].item() for entry in dataset))
    num_doses = len(unique_doses)
    num_simulated_per_dose = max(1, num_simulated_total // num_doses)
    t_dense2 = torch.linspace(0, 1, steps=120).to(device)

    for dose_value in unique_doses:
        # Filter dataset for current dose
        dose_filtered_dataset = [entry for entry in dataset if entry[2].item() == dose_value]
        if len(dose_filtered_dataset) == 0:
            print(f"⚠️ No data found for dose {dose_value}. Skipping plot.")
            continue

        # Sample batch indices with replacement
        indices = np.random.choice(len(dose_filtered_dataset), num_simulated_per_dose, replace=True)
        batch_entries = [dose_filtered_dataset[i] for i in indices]

        t_reals = []
        x_trues = []
        doses = []
        dose_times_list = []

        # Collect data for batch
        for entry in batch_entries:
            t_real, x_true, dose, dose_times = entry[:4]
            t_reals.append(t_real.to(device))
            x_trues.append(x_true.to(device))
            doses.append(dose.to(device))
            dose_times_list.append(dose_times.to(device))

        # Prepare batch tensors
        doses_tensor = torch.stack(doses)                            # [batch]
        dose_times_padded = torch.nn.utils.rnn.pad_sequence(dose_times_list, batch_first=True)  # [batch, max_len]
        dose_mask = (dose_times_padded != 0)                         # boolean mask

     
        # Refiner requires t_real and x_true batch: pad sequences for batch processing
        # Pad t_reals and x_trues to max length for batch processing in refiner
        t_real_padded = torch.nn.utils.rnn.pad_sequence(t_reals, batch_first=True).to(device)  # [batch, max_len_t]
        x_true_padded = torch.nn.utils.rnn.pad_sequence(x_trues, batch_first=True).to(device)  # [batch, max_len_t, features]

        log_sigma_prop = noise.log_sigma_prop
        log_sigma_add = noise.log_sigma_add
        global_mu = global_latent.mu.detach()
        global_logvar = global_latent.logvar.detach()

        with torch.no_grad():
            mu_q, logvar_q = refiner(t_real_padded, x_true_padded, log_sigma_add, log_sigma_prop, global_mu, global_logvar)
            std_q = torch.exp(0.5 * logvar_q)
            eps = torch.randn_like(std_q)
            z_refined = mu_q + eps * std_q  # [batch, latent_dim]

            # Initial conditions for batch
            x0_init = torch.stack([x[0] for x in x_trues])               # [batch, features]
            x0_latent = initial_encoder(torch.cat([x0_init.unsqueeze(1)] * 2, dim=1))  # [batch, latent_dim]
            x0 = torch.cat([x0_latent, doses_tensor.unsqueeze(-1)], dim=-1)             # [batch, latent_dim+1]

            # ODE func and integration
            ode_func = ODEWrapper(func, dose_times_padded.to(device), dose_mask.to(device), z_refined)
            pred = odeint(ode_func, x0, t_dense.to(device), method='dopri5')            # [time, batch, latent_dim+1]

            # Reduce to observed dimension
            x_pred = reducer(pred[:, :, :x0_latent.shape[-1]])                           # [time, batch, 1]

            # Add noise sample to predictions (single sample)
            x_pred_noisy = noise.sample(x_pred, n_samples=1).squeeze(0)                 # [time, batch, 1]

        # Compute quantiles over batch
        perc10_sim = torch.quantile(x_pred_noisy.squeeze(-1), 0.10, dim=1)  # [time]
        median_sim = torch.quantile(x_pred_noisy.squeeze(-1), 0.50, dim=1)  # [time]
        perc90_sim = torch.quantile(x_pred_noisy.squeeze(-1), 0.90, dim=1)  # [time]

        # Interpolate real data for same dose to t_dense2
        interp_all = []
        for t_real, x_true in zip(t_reals, x_trues):
            x_interp = torch_linear_interpolate(t_real, x_true, t_dense2)
            interp_all.append(x_interp)
        data_matrix = torch.stack(interp_all)  # [batch, time]

        perc10_data = torch.quantile(data_matrix, 0.10, dim=0)  # [time]
        median_data = torch.quantile(data_matrix, 0.50, dim=0)  # [time]
        perc90_data = torch.quantile(data_matrix, 0.90, dim=0)  # [time]

        # De-standardize concentrations
        perc10_sim_real = destandardize_concentration(perc10_sim, conc_mean, conc_std)
        median_sim_real = destandardize_concentration(median_sim, conc_mean, conc_std)
        perc90_sim_real = destandardize_concentration(perc90_sim, conc_mean, conc_std)

        perc10_data_real = destandardize_concentration(perc10_data, conc_mean, conc_std)
        median_data_real = destandardize_concentration(median_data, conc_mean, conc_std)
        perc90_data_real = destandardize_concentration(perc90_data, conc_mean, conc_std)

        # Plot
        plt.figure(figsize=(12, 6))
        time_hours = t_dense.cpu().numpy() * MAX_TIME

        plt.plot(time_hours, perc10_sim_real.detach().cpu().numpy(), label="Simulated 10th percentile", color="blue", linestyle="--")
        plt.plot(time_hours, median_sim_real.detach().cpu().numpy(), label="Simulated median", color="blue", marker="o")
        plt.plot(time_hours, perc90_sim_real.detach().cpu().numpy(), label="Simulated 90th percentile", color="blue", linestyle="--")

        plt.plot(time_hours, perc10_data_real.detach().cpu().numpy(), label="Raw 10th percentile", color="orange", linestyle="--")
        plt.plot(time_hours, median_data_real.detach().cpu().numpy(), label="Raw median", color="orange")
        plt.plot(time_hours, perc90_data_real.detach().cpu().numpy(), label="Raw 90th percentile", color="orange", linestyle="--")

        plt.title(f"Dose {dose_value * MAX_DOSE:.0f} (Simulated vs Raw)")
        plt.xlabel("Time (hours)")
        plt.ylabel(f"Concentration ({compartment})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

   
def simulate_and_plot_by_dose(
    dataset, global_latent, initial_encoder, func, reducer, noise, ODEWrapper,
    t_dense, conc_mean, conc_std, MAX_TIME, MAX_DOSE,compartment, num_simulated_total=500,
    add_noise_to_prediction: bool = False  # 🔧 NEW ARGUMENT
):
    device = next(func.parameters()).device
    unique_doses = sorted(set(entry[2].item() for entry in dataset))
    num_doses = len(unique_doses)
    num_simulated_per_dose = max(1, num_simulated_total // num_doses)

    for dose_value in unique_doses:
        dose_filtered_dataset = [entry for entry in dataset if entry[2].item() == dose_value]
        if len(dose_filtered_dataset) == 0:
            print(f"⚠️ No data found for dose {dose_value}. Skipping plot.")
            continue

        indices = np.random.choice(len(dose_filtered_dataset), num_simulated_per_dose, replace=True)
        batch_entries = [dose_filtered_dataset[i] for i in indices]
        t_dense2 = torch.linspace(0, 1, steps=120).to(device)

        t_reals, x_trues, doses, dose_times_list, z_samples = [], [], [], [], []

        for entry in batch_entries:
            t_real, x_true, dose, dose_times = entry[:4]
            t_reals.append(t_real)
            x_trues.append(x_true)
            doses.append(dose)
            dose_times_list.append(dose_times)
            z_samples.append(global_latent.sample())

        doses_tensor = torch.stack(doses).to(device)
        dose_times_padded = torch.nn.utils.rnn.pad_sequence(dose_times_list, batch_first=True).to(device)
        dose_mask = (dose_times_padded != 0).to(device)
        z_tensor = torch.stack(z_samples).to(device)

        x0_list = []
        for x in xs:
            out = initial_encoder(x[0].unsqueeze(0))  # expect shape [1, 4]
            if out.dim() == 1:
                out = out.unsqueeze(0)  # convert [4] -> [1, 4]
            x0_list.append(out)
        x0_tensor = torch.cat(x0_list, dim=0)  # now shape [6, 4]

        

        x0 = torch.cat([x0_tensor, z_refined], dim=1)  # shape [batch_size, 6]
        ode_func = ODEWrapper(func, dose_times_padded,doses, dose_mask, z_tensor)
        pred = odeint(ode_func, x0, t_dense.to(device), method='dopri5')  # [time, batch, latent_dim+1]

        x_pred = reducer(pred[:, :, :x0_latent.shape[-1]])  # [time, batch, state_dim]

        # ✅ Optionally add noise
        if add_noise_to_prediction:
            x_pred = noise.sample(x_pred, n_samples=1).squeeze(0)

        # Compute quantiles
        perc10_sim = torch.quantile(x_pred.squeeze(-1), 0.10, dim=1)
        median_sim = torch.quantile(x_pred.squeeze(-1), 0.50, dim=1)
        perc90_sim = torch.quantile(x_pred.squeeze(-1), 0.90, dim=1)

        interp_all = [
            torch_linear_interpolate(t_real, x_true, t_dense2)
            for t_real, x_true in zip(t_reals, x_trues)
        ]
        data_matrix = torch.stack(interp_all)

        perc10_data = torch.quantile(data_matrix, 0.10, dim=0)
        median_data = torch.quantile(data_matrix, 0.50, dim=0)
        perc90_data = torch.quantile(data_matrix, 0.90, dim=0)

        # De-standardize
        perc10_sim_real = destandardize_concentration(perc10_sim, conc_mean, conc_std)
        median_sim_real = destandardize_concentration(median_sim, conc_mean, conc_std)
        perc90_sim_real = destandardize_concentration(perc90_sim, conc_mean, conc_std)

        perc10_data_real = destandardize_concentration(perc10_data, conc_mean, conc_std)
        median_data_real = destandardize_concentration(median_data, conc_mean, conc_std)
        perc90_data_real = destandardize_concentration(perc90_data, conc_mean, conc_std)

        # Plot
        plt.figure(figsize=(12, 6))
        time_hours = t_dense.cpu().numpy() * MAX_TIME

        plt.plot(time_hours, perc10_sim_real.detach().cpu().numpy(), label="Simulated 10th percentile", color="blue", linestyle="--")
        plt.plot(time_hours, median_sim_real.detach().cpu().numpy(), label="Simulated median", color="blue", marker="o")
        plt.plot(time_hours, perc90_sim_real.detach().cpu().numpy(), label="Simulated 90th percentile", color="blue", linestyle="--")
        
        plt.plot(time_hours, perc10_data_real.detach().cpu().numpy(), label="Raw 10th percentile", color="orange", linestyle="--")
        plt.plot(time_hours, median_data_real.detach().cpu().numpy(), label="Raw median", color="orange")
        plt.plot(time_hours, perc90_data_real.detach().cpu().numpy(), label="Raw 90th percentile", color="orange", linestyle="--")


        plt.title(f"Dose {dose_value * MAX_DOSE:.0f} (Simulated vs Raw)")
        plt.xlabel("Time (hours)")
        plt.ylabel(f"Concentration ({compartment})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()