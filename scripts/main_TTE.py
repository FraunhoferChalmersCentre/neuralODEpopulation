# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 22:50:00 2025

@author: Baaz
"""

import os
import sys
import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import numpy as np


# ---- Add project root to path ----
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# ---- Import helper functions ----
from lib.utils.utils_TTE import (
    plot_kaplan_meier,
    compute_hazard,
    population_survival,
    extract_predictions,
    fit_alpha_beta_to_KM,
    survival_loss
)

# Optional: Import lifelines KaplanMeierFitter for plotting alongside population survival
try:
    from lifelines import KaplanMeierFitter
except ImportError:
    print("lifelines not installed. Install via `pip install lifelines` to enable KM plotting.")
    KaplanMeierFitter = None

if __name__ == "__main__":
    # ---- Simulate command line arguments (optional) ----
    sys.argv = [
        'script_name',
        '--tumor_csv', 'lib/data/tumor_data.csv',
        '--tte_csv', 'lib/data/tumor_data_tte.csv',
        '--plot', 'True'
    ]

    # ---- Parse arguments ----
    parser = argparse.ArgumentParser(description="Load tumor and TTE data, compute population survival, and plot.")
    parser.add_argument("--tumor_csv", type=str, required=True, help="Path to tumor volume CSV")
    parser.add_argument("--tte_csv", type=str, required=True, help="Path to time-to-event CSV")
    parser.add_argument("--plot", type=bool, default=True, help="Whether to show plots")
    parser.add_argument("--save_plot", type=str, default=None, help="Optional path to save figures")
    args = parser.parse_args()

    # ---- Load tumor and TTE data ----
    df_tumor = pd.read_csv(args.tumor_csv, sep=';')
    df_tumor.columns = df_tumor.columns.str.strip().str.upper()
    df_tte = pd.read_csv(args.tte_csv, sep=';')
    df_tte.columns = df_tte.columns.str.strip().str.upper()
    print(f"Tumor data: {df_tumor.shape}, TTE data: {df_tte.shape}")

    # ---- Extract tumor predictions from DV column ----
    V_pred, time_points = extract_predictions(df_tumor, id_col="ID", time_col="TIME", dv_col="DV")
    print(f"Extracted tumor predictions: {V_pred.shape}, time points: {time_points}")

    # ---- Compute hazards using true simulation parameters ----
    alpha_true = torch.tensor(0.001)
    beta_true = torch.tensor(0.001)
    hazards, _, _ = compute_hazard(V_pred, alpha=alpha_true, beta=beta_true)
    print(f"Computed hazard tensor: {hazards.shape}")

    # ---- Build observed mask from TTE data ----
    n_individuals, n_times = V_pred.shape
    observed_mask = torch.zeros((n_individuals, n_times), dtype=torch.float32)
    sorted_ids = sorted(df_tte['ID'].unique())

    for i, id_ in enumerate(sorted_ids):
        tte_row = df_tte[df_tte['ID'] == id_].iloc[0]
        tte_time = tte_row['TIME_TO_EVENT']
        observed_mask[i, :] = (time_points <= tte_time).float()

    # ---- Compute population survival with censoring
    dt = 0.1  # spacing between time points
    S_pop, S_ind = population_survival(hazards, dt, V_pred)
    print("Population survival computed.")

    # ---- Plot Kaplan-Meier and population survival ----
    if args.plot:
        plt.figure(figsize=(6,4))

        # Plot Kaplan-Meier from TTE
        if KaplanMeierFitter is not None:
            kmf = KaplanMeierFitter()
            kmf.fit(durations=df_tte["TIME_TO_EVENT"], event_observed=df_tte["EVENT"])
            kmf.plot_survival_function(label="Kaplan-Meier (TTE)", color='blue')

        # Plot population survival from tumor predictions
        plt.plot(
            time_points.numpy(),
            S_pop.detach().numpy(),
            label="Population Survival (Predictions)",
            color='red',
            linestyle='--',
            linewidth=2
        )

        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.title("Kaplan-Meier vs Population Survival")
        plt.grid(True)
        plt.legend()
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {args.save_plot}")
        plt.show()

    # ---- Print population survival values ----
    print("\nPopulation survival (predictions) at each time point:")
    for t, s in zip(time_points.numpy(), S_pop.detach().numpy()):
        print(f"Time {t:.2f}: Survival {s:.4f}")


    kmf = KaplanMeierFitter()
    kmf.fit(durations=df_tte['TIME_TO_EVENT'], event_observed=df_tte['EVENT'])
    
    # Survival probabilities at the recorded times
    S_KM = torch.tensor(kmf.survival_function_.values.flatten(), dtype=torch.float32)
    
    # Corresponding times
    times = torch.tensor(kmf.survival_function_.index.values, dtype=torch.float32)
    S_KM_interp = torch.tensor(
    np.interp(time_points.numpy(), times.numpy(), S_KM.numpy()),
    dtype=torch.float32
)



    # Suppose V_pred, S_KM are torch tensors
    alpha_fit, beta_fit, S_pred = fit_alpha_beta_to_KM(V_pred, S_KM_interp, dt=dt, lr=0.001, n_epochs=1000)
    print("Fitted alpha:", alpha_fit.item())
    print("Fitted beta:", beta_fit.item())
    
    
    
    survival_loss(0.001,0.001, V_pred, S_KM_interp, dt=0.1)
    
    