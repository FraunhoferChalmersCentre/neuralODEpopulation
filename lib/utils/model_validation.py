"""
Created on Fri Jun 27 10:21:33 2025

@author: Baaz
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchdiffeq import odeint

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score

from lib.utils.my_utils import *



def simulate_and_plot(
    dataset,
    global_latent,
    refiner,
    initial_encoder,
    func,
    reducer,
    noise,
    t_dense,
    conc_mean,
    conc_std,
    max_time,
    compartment,
    num_simulated_total=500
):
    unique_doses = sorted(set(entry[2].item() for entry in dataset))
    num_doses = len(unique_doses)
    num_simulated_per_dose = num_simulated_total // num_doses

    all_preds = []
    with torch.no_grad():
        for dose_value in unique_doses:
            dose_filtered_dataset = [entry for entry in dataset if entry[2].item() == dose_value]
            for _ in range(num_simulated_per_dose):
                data_entry = dose_filtered_dataset[np.random.randint(len(dose_filtered_dataset))]
                t_real, x_real, dose, dose_times = data_entry[:4]

                z_sampled = global_latent.sample()
                mu_q, logvar_q = refiner(t_real, x_real, z_sampled)
                std_q = torch.exp(0.5 * logvar_q)
                eps = torch.randn_like(std_q)
                z_refined = mu_q + eps * std_q

                x0_input = torch.cat([x_real[0].unsqueeze(0)] * 2, dim=0)
                x0 = initial_encoder(x0_input)

                pred = odeint(lambda t, x: func(t, x, dose, dose_times, z_sampled), x0, t_dense, method='rk4')
                x_pred = reducer(pred)

                x_pred_noisy = noise.sample(x_pred, n_samples=1).squeeze(0)
                all_preds.append(x_pred_noisy)

    sim_matrix = torch.stack(all_preds)

    perc10_sim = torch.quantile(sim_matrix, 0.10, dim=0)
    median_sim = torch.quantile(sim_matrix, 0.50, dim=0)
    perc90_sim = torch.quantile(sim_matrix, 0.90, dim=0)

    interp_all = []
    for data_entry in dataset:
        t_real, x_real, _, _ = data_entry[:4]
        x_interp = torch_linear_interpolate(t_real, x_real, t_dense)
        interp_all.append(x_interp)
    data_matrix = torch.stack(interp_all)

    perc10_data = torch.quantile(data_matrix, 0.10, dim=0)
    median_data = torch.quantile(data_matrix, 0.50, dim=0)
    perc90_data = torch.quantile(data_matrix, 0.90, dim=0)

    perc10_sim_real = perc10_sim * conc_std + conc_mean
    median_sim_real = median_sim * conc_std + conc_mean
    perc90_sim_real = perc90_sim * conc_std + conc_mean

    perc10_data_real = perc10_data * conc_std + conc_mean
    median_data_real = median_data * conc_std + conc_mean
    perc90_data_real = perc90_data * conc_std + conc_mean

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    plt.plot(t_dense.numpy() * max_time, perc10_sim_real.numpy(), label="Simulated 10th percentile", color="blue", linestyle="--")
    plt.plot(t_dense.numpy() * max_time, median_sim_real.numpy(), label="Simulated median", color="blue", marker="o")
    plt.plot(t_dense.numpy() * max_time, perc90_sim_real.numpy(), label="Simulated 90th percentile", color="blue", linestyle="--")

    plt.plot(t_dense.numpy() * max_time, perc10_data_real.numpy(), label="Raw data 10th percentile", color="orange", linestyle="--")
    plt.plot(t_dense.numpy() * max_time, median_data_real.numpy(), label="Raw data median", color="orange")
    plt.plot(t_dense.numpy() * max_time, perc90_data_real.numpy(), label="Raw data 90th percentile", color="orange", linestyle="--")

    plt.title("All Doses Combined (Balanced Simulation with Learned Noise)")
    plt.xlabel("Time (hours)")
    plt.ylabel(f"Concentration ({compartment})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


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

            x0_input = torch.cat([x_true[0].unsqueeze(0)] * 2, dim=0).to(device)
            x0 = initial_encoder(x0_input)

            pred = odeint(lambda t_, x_: func(t_, x_, dose, dose_times, z),
                          x0, t_dense.to(device), method='rk4')
            x_pred = reducer(pred)

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

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    save_path = os.path.join(results_dir, "ebes_with_data.pt")
    torch.save(results, save_path)

    return results


def plot_individuals(ebes, num_plots, func, initial_encoder, reducer, torch_linear_interpolate,
                     conc_mean, conc_std, t_dense, cols=5, rows=None):
    n = min(num_plots, len(ebes))
    if rows is None:
        rows = (n + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(30, 20), squeeze=False)

    for idx, (individual_id, t, x_true, dose, dose_times, best_z) in enumerate(ebes[:n]):
        ax = axs[idx // cols, idx % cols]

        t = t.to('cpu')
        x_true = x_true.to('cpu')
        dose = dose.to('cpu') if isinstance(dose, torch.Tensor) else dose
        dose_times = dose_times.to('cpu') if isinstance(dose_times, torch.Tensor) else dose_times
        best_z = best_z.to('cpu')

        x0_input = torch.cat([x_true[0].unsqueeze(0)] * 2, dim=0)
        x0 = initial_encoder(x0_input)

        t_dense_cpu = t_dense.to('cpu')

        pred = odeint(lambda t_, x_: func(t_, x_, dose, dose_times, best_z),
                      x0, t_dense_cpu, method='rk4')
        x_pred = reducer(pred)

        # Denormalize
        x_pred_denorm = x_pred * conc_std + conc_mean
        x_true_denorm = x_true * conc_std + conc_mean

        x_pred_at_t = torch_linear_interpolate(t_dense_cpu, x_pred_denorm, t)

        ax.plot(t.numpy(), x_true_denorm.numpy(), 'o-', label='True')
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