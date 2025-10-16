
# libs/utils/TTE.py

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import torch
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from lib.utils.utils_data_generation import simulate_tumor_volume_with_event

def plot_kaplan_meier(tte_csv=None, df_tte=None, time_col="TIME_TO_EVENT", event_col="EVENT", title="Kaplan-Meier Curve", save_path=None):
    """
    Extract TTE data and plot Kaplan-Meier survival curve.

    Parameters
    ----------
    tte_csv : str, optional
        Path to TTE CSV file. Either `tte_csv` or `df_tte` must be provided.
    df_tte : pd.DataFrame, optional
        DataFrame containing TTE data.
    time_col : str
        Column name for time-to-event.
    event_col : str
        Column name for event indicator (1=event, 0=censored).
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure. If None, figure is shown.
    """

    if df_tte is None and tte_csv is None:
        raise ValueError("Either `tte_csv` or `df_tte` must be provided.")
    
    # Load data if CSV path is given
    if df_tte is None:
        df_tte = pd.read_csv(tte_csv, sep=';')
    
    # Ensure column names are consistent
    df_tte.columns = df_tte.columns.str.strip().str.upper()
    time_col = time_col.upper()
    event_col = event_col.upper()

    kmf = KaplanMeierFitter()
    kmf.fit(durations=df_tte[time_col], event_observed=df_tte[event_col])

    plt.figure(figsize=(6,4))
    kmf.plot_survival_function()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()




def compute_hazard(tumor_volume, alpha=None, beta=None):
    """
    Compute hazard function from tumor volume predictions.

    Parameters
    ----------
    tumor_volume : torch.Tensor
        Tensor of shape (T,) or (batch, T) with tumor volume predictions over time.
    alpha : torch.Tensor, optional
        Learnable baseline hazard parameter. If None, initialized as 0.01.
    beta : torch.Tensor, optional
        Learnable slope parameter. If None, initialized as 0.01.

    Returns
    -------
    hazard : torch.Tensor
        Hazard function values of the same shape as `tumor_volume`.
    alpha, beta : torch.Tensor
        Hazard parameters (useful if learnable)
    """

    
    
    # Linear hazard directly from tumor volume
    hazard = torch.clamp(alpha + beta * tumor_volume, min=0.0)

    return hazard, alpha, beta



# lib/utils/utils_hazard.py
import torch

import torch

def population_survival(
    hazards,
    dt=None,
    tumor_vol_pred=None,
    deterministic_threshold=1.2
):
    """
    Compute population and individual survival functions from hazard values,
    optionally enforcing deterministic death (survival=0) when predicted
    tumor volume exceeds a threshold.

    Parameters
    ----------
    hazards : torch.Tensor
        Tensor of shape (n_individuals, n_timepoints), hazard values.
    dt : float, optional
        Time step between hazard measurements (default 1.0).
    observed_mask : torch.Tensor, optional
        Mask tensor for observed timepoints (same shape as hazards).
    tumor_vol_pred : torch.Tensor, optional
        Predicted tumor volumes (same shape as hazards).
    deterministic_threshold : float, default=1.2
        Threshold multiplier of baseline tumor volume that triggers deterministic death.

    Returns
    -------
    S_pop : torch.Tensor
        Population survival function (mean of individual survival).
    S_ind : torch.Tensor
        Individual survival trajectories.
    """
    if dt is None:
        dt = 1.0

    hazards = torch.nan_to_num(hazards)
    new_dt = 0.01
    hazards_fine = hazards * (dt / new_dt)  # scale hazard
    cum_hazard = torch.cumsum(hazards_fine * new_dt, dim=1)

    # Compute cumulative hazard and survival
    #cum_hazard = torch.cumsum(hazards * dt, dim=1)
    
    # Prepend zero and trim last value so length stays the same as hazards
  #  zero_col = torch.zeros((cum_hazard.shape[0], 1), dtype=cum_hazard.dtype, device=cum_hazard.device)
   # cum_hazard = torch.cat([zero_col, cum_hazard[:, :-1]], dim=1)

    S_ind = torch.exp(-cum_hazard)

    # Ensure first column = 1.0 (initial survival)
    first_col = torch.ones((S_ind.shape[0], 1), dtype=S_ind.dtype, device=S_ind.device)
    S_ind = torch.cat([first_col, S_ind[:, 1:]], dim=1)
    
        # === Apply deterministic censoring instead of forced death ===


    # === Apply deterministic survival override ===
    if tumor_vol_pred is not None:
        # Ensure same shape
        assert tumor_vol_pred.shape == hazards.shape, \
            f"tumor_vol_pred must match hazards shape: {hazards.shape}, got {tumor_vol_pred.shape}"

        # Determine baseline per individual
        baseline = tumor_vol_pred[:, 0:1]  # shape (N, 1)
        threshold_values = deterministic_threshold * baseline

        # Create mask where tumor volume exceeds threshold
        exceed_mask = tumor_vol_pred >= threshold_values

        # For each individual, find first exceedance index
        exceed_idx = exceed_mask.float().argmax(dim=1)
        ever_exceeded = exceed_mask.any(dim=1)

        # Apply deterministic survival = 0 from exceedance onward
        for i in range(S_ind.shape[0]):
            if ever_exceeded[i]:
                idx = exceed_idx[i].item()
                # Set all subsequent survival to zero
                S_ind[i, idx:] = 0.0

    # Compute population survival as mean of individuals
    S_pop = S_ind.mean(dim=0)

    # Ensure first element = 1
    first_val = torch.tensor([1.0], dtype=S_pop.dtype, device=S_pop.device)
    S_pop = torch.cat([first_val, S_pop[1:]])

    return S_pop, S_ind


def survival_loss(alpha,beta, V_pred, S_KM, dt):

    hazard, _, _ = compute_hazard(V_pred, alpha=alpha, beta=beta)

    S_pop,_ = population_survival(hazard, dt, V_pred)
    loss = torch.mean((S_pop - S_KM)**2)
    return loss



# lib/utils/utils_TTE.py

import pandas as pd
import torch

def extract_predictions(df_or_csv, id_col="ID", time_col="TIME", dv_col="DV", evid_col="EVID"):
    import pandas as pd
    import torch
    import numpy as np

    if isinstance(df_or_csv, str):
        df = pd.read_csv(df_or_csv, sep=';')
    else:
        df = df_or_csv.copy()

    df.columns = df.columns.str.strip().str.upper()
    id_col = id_col.upper()
    time_col = time_col.upper()
    dv_col = dv_col.upper()
    evid_col = evid_col.upper()

    # ---- Only keep observations (EVID == 0) ----
    df = df[df[evid_col] == 0]

    # ---- Make sure all time points are included ----
    all_times = np.sort(df[time_col].unique())
    
    # ---- Pivot with fill_value=0 to avoid NaNs ----
    df_pivot = df.pivot_table(
        index=id_col,
        columns=time_col,
        values=dv_col,
        aggfunc='mean',
        fill_value=0.0  # missing tumor volumes set to 0
    )

    # ---- Reindex columns to include all times (fills missing with 0) ----
    df_pivot = df_pivot.reindex(columns=all_times, fill_value=0.0)

    # ---- Convert to torch tensors ----
    V_pred = torch.tensor(df_pivot.values, dtype=torch.float32)
    time_points = torch.tensor(df_pivot.columns.values, dtype=torch.float32)

    return V_pred, time_points




def fit_alpha_beta_to_KM(V_pred, S_KM, dt=0.5, lr=0.01, n_epochs=500, verbose=True, plot_progress=True):
    """
    Fit alpha and beta to Kaplan-Meier survival using tumor volume predictions.

    Parameters
    ----------
    V_pred : torch.Tensor
        Tumor volume predictions, shape (n_individuals, n_times)
    S_KM : torch.Tensor
        Kaplan-Meier survival curve at corresponding times, shape (n_times,)
    dt : float
        Time interval for cumulative hazard
    lr : float
        Learning rate for optimizer
    n_epochs : int
        Number of optimization steps
    verbose : bool
        Print loss every 50 epochs
    plot_progress : bool
        Plot predicted vs KM survival at the end

    Returns
    -------
    alpha : torch.Tensor
        Fitted alpha parameter
    beta : torch.Tensor
        Fitted beta parameter
    S_pred : torch.Tensor
        Predicted population survival curve
    """

    # Initialize alpha and beta as learnable parameters
    alpha = torch.tensor(0.005, dtype=torch.float32, requires_grad=True)
    beta = torch.tensor(0.005, dtype=torch.float32, requires_grad=True)

    optimizer = optim.Adam([alpha, beta], lr=lr)

    S_KM = S_KM.clone().float()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Compute hazards
        hazards, _, _ = compute_hazard(V_pred, alpha, beta)

        # Compute population survival
        S_pred, S_ind = population_survival(hazards, dt, V_pred)

        # Compute MSE loss
        loss = torch.mean((S_pred - S_KM) ** 2)
        
        # Compute individual MSE
      #  ind_loss = torch.mean((S_ind - S_KM[None, :])**2, dim=1)
        
        # Weight by time to event
      #  event_times = (S_ind < 1.0).float().argmax(dim=1).float() + 1e-6  # avoid zero
     #   weights = event_times / event_times.max()
     #   loss = torch.mean(weights * ind_loss)

        
        
        loss.backward()
        optimizer.step()

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:03d}: loss = {loss.item():.6f}, alpha={alpha.item():.4f}, beta={beta.item():.4f}")

    if plot_progress:
        plt.figure(figsize=(6, 4))
        plt.plot(S_KM.numpy(), label="Kaplan-Meier")
        plt.plot(S_pred.detach().numpy(), label="Predicted Survival")
        plt.xlabel("Time index")
        plt.ylabel("Survival Probability")
        plt.title("Fitted Survival vs Kaplan-Meier")
        plt.legend()
        plt.grid(True)
        plt.show()

    return alpha.detach(), beta.detach(), S_pred.detach()

def simulate_and_fit_iteration(iter_idx, save_dir, true_alpha, true_beta, plot_fit=False):
    """Simulate one dataset, fit alpha/beta, and return results."""
    print(f"\n--- Iteration {iter_idx+1} ---")

    os.makedirs(save_dir, exist_ok=True)
    tumor_path = os.path.join(save_dir, f"tumor_data_iter_{iter_idx+1}.csv")
    tte_path = os.path.splitext(tumor_path)[0] + "_tte.csv"

    # ----- PK/PD parameters -----
    a_drugs = [0.0005]
    add_e = 0.00001
    prop_e = 0.00001
    ka_mean = [0.6]
    ke_mean = [0.6]
    v_mean = [0]
    ka_sd = [0]
    ke_sd = [0]
    v_sd = [0]

    # ----- Tumor parameters -----
    k_growth_mean = 0.1
    k_growth_sd = 0.05
    V0_mean = 100.0
    V0_sd = 0.1

    # ----- Groups -----
    groups = [
        {
            'n_individuals': 1000,
            'dose_amounts_list': [[200, 200, 200, 200]],
            'dose_times_list': [[1, 6, 11, 16]]
        }
    ]

    # Simulate and save
    simulate_tumor_volume_with_event(
        n_individuals=groups[0]['n_individuals'],
        dose_amounts_list=groups[0]['dose_amounts_list'],
        dose_times_list=groups[0]['dose_times_list'],
        a_drugs=a_drugs,
        alpha=true_alpha,
        beta=true_beta,
        add_e=add_e,
        prop_e=prop_e,
        save_path=tumor_path,
        t_interval=(0, 16),
        sample_frequency=0.1,
        ka_mean=ka_mean,
        ke_mean=ke_mean,
        v_mean=v_mean,
        ka_sd=ka_sd,
        ke_sd=ke_sd,
        v_sd=v_sd,
        k_growth_mean=k_growth_mean,
        k_growth_sd=k_growth_sd,
        V0_mean=V0_mean,
        V0_sd=V0_sd,
        max_tumor_size=2000,
        plot=False
    )

    # ---- Load data ----
    df_tumor = pd.read_csv(tumor_path, sep=';')
    df_tumor.columns = df_tumor.columns.str.strip().str.upper()
    df_tte = pd.read_csv(tte_path, sep=';')
    df_tte.columns = df_tte.columns.str.strip().str.upper()

    # ---- Extract tumor predictions ----
    V_pred, time_points = extract_predictions(df_tumor, id_col="ID", time_col="TIME", dv_col="DV")

    # ---- Compute survival curve (KM) ----
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df_tte['TIME_TO_EVENT'], event_observed=df_tte['EVENT'])
    S_KM = torch.tensor(kmf.survival_function_.values.flatten(), dtype=torch.float32)
    times = torch.tensor(kmf.survival_function_.index.values, dtype=torch.float32)
    S_KM_interp = torch.tensor(
        np.interp(time_points.numpy(), times.numpy(), S_KM.numpy()),
        dtype=torch.float32
    )

    # ---- Fit alpha/beta ----
    dt = 0.1
    alpha_fit, beta_fit, S_pred = fit_alpha_beta_to_KM(V_pred, S_KM_interp, dt=dt, lr=0.001, n_epochs=1000)

    print(f"Fitted alpha: {alpha_fit.item():.6f}, beta: {beta_fit.item():.6f}")

    # ---- Optional plot ----
    if plot_fit:
        plt.figure(figsize=(6, 4))
        plt.plot(time_points, S_KM_interp, label="Kaplan-Meier (KM)", color='blue')
        plt.plot(time_points, S_pred.detach(), label="Fitted Model", color='red', linestyle='--')
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.title(f"Iteration {iter_idx+1} - Survival Fit")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ---- Return fitted parameters ----
    return alpha_fit.item(), beta_fit.item()
