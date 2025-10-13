# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 22:20:00 2025

@author: Baaz
"""
# libs/utils/TTE.py

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import torch
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

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

def population_survival(hazards, dt=None, observed_mask=None):
    if dt is None:
        dt = 1.0

    hazards = torch.nan_to_num(hazards)

    # Compute cumulative hazard
    cum_hazard = torch.cumsum(hazards * dt, dim=1)
    S_ind = torch.exp(-cum_hazard)

    # Avoid inplace: set first column to 1 by concatenation
    first_col = torch.ones((S_ind.shape[0], 1), dtype=S_ind.dtype, device=S_ind.device)
    S_ind = torch.cat([first_col, S_ind[:, 1:]], dim=1)

    # Population survival
    S_pop = S_ind.mean(dim=0)

    # Avoid inplace: first element to 1
    S_pop = torch.cat([torch.tensor([1.0], dtype=S_pop.dtype, device=S_pop.device), S_pop[1:]])

    return S_pop, S_ind


def survival_loss(params, V_pred, S_KM, dt=0.5):
    alpha, beta = params
    S_pop = population_survival(V_pred, alpha, beta, dt)
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
    alpha = torch.tensor(0.001, dtype=torch.float32, requires_grad=True)
    beta = torch.tensor(0.001, dtype=torch.float32, requires_grad=True)

    optimizer = optim.Adam([alpha, beta], lr=lr)

    S_KM = S_KM.clone().float()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Compute hazards
        hazards, _, _ = compute_hazard(V_pred, alpha, beta)

        # Compute population survival
        S_pred, _ = population_survival(hazards, dt=dt)

        # Compute MSE loss
        loss = torch.mean((S_pred - S_KM)**2)
        loss.backward()
        optimizer.step()

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:03d}: loss = {loss.item():.6f}, alpha={alpha.item():.4f}, beta={beta.item():.4f}")

    if plot_progress:
        plt.figure(figsize=(6,4))
        plt.plot(S_KM.numpy(), label="Kaplan-Meier")
        plt.plot(S_pred.detach().numpy(), label="Predicted Survival")
        plt.xlabel("Time index")
        plt.ylabel("Survival Probability")
        plt.title("Fitted Survival vs Kaplan-Meier")
        plt.legend()
        plt.grid(True)
        plt.show()

    return alpha.detach(), beta.detach(), S_pred.detach()
