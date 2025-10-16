# -*- coding: utf-8 -*-
"""
Merged simulation + alpha/beta fitting script
Author: Baaz
Date: 2025-10-15
Description:
Simulate tumor and TTE data multiple times, fit alpha/beta each time,
and summarize the estimated parameters.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ---- Add project root to path ----
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# ---- Import simulation and analysis functions ----
from lib.utils.utils_data_generation import simulate_tumor_volume_with_event
from lib.utils.utils_TTE import (
    compute_hazard,
    population_survival,
    extract_predictions,
    fit_alpha_beta_to_KM,
)

try:
    from lifelines import KaplanMeierFitter
except ImportError:
    KaplanMeierFitter = None


def simulate_and_fit_iteration(iter_idx, save_dir, true_alpha, true_beta, plot_fit=False):
    """Simulate one dataset, fit alpha/beta, and return results."""
    print(f"\n--- Iteration {iter_idx+1} ---")

    os.makedirs(save_dir, exist_ok=True)
    tumor_path = os.path.join(save_dir, f"tumor_data_iter_{iter_idx+1}.csv")
    tte_path = os.path.join(save_dir, f"tte_tumor_data_iter_{iter_idx+1}.csv")


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
        sample_frequency=0.01,
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
    dt = 0.01
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


def main():
    parser = argparse.ArgumentParser(description="Simulate tumor/TTE data multiple times and estimate alpha/beta.")
    parser.add_argument("--n_iter", type=int, default=10, help="Number of simulation iterations.")
    parser.add_argument("--save_dir", type=str, default="lib/data/multi_runs", help="Directory to save datasets.")
    parser.add_argument("--plot_fit", action="store_true", help="Plot fit for each iteration.")
    args = parser.parse_args()

    true_alpha = 0.001
    true_beta = 0.001

    results = []
    for i in range(args.n_iter):
        alpha_fit, beta_fit = simulate_and_fit_iteration(
            i, args.save_dir, true_alpha, true_beta, plot_fit=args.plot_fit
        )
        results.append((alpha_fit, beta_fit))

    df_results = pd.DataFrame(results, columns=["alpha_fit", "beta_fit"])
    stats = df_results.agg(["mean", "median", "std"])

    print("\n===== Summary of Estimated Parameters =====")
    print(stats)

    # Save summary
    summary_path = os.path.join(args.save_dir, "alpha_beta_summary.csv")
    df_results.to_csv(summary_path, index=False)
    print(f"\nSaved per-iteration results to: {summary_path}")


if __name__ == "__main__":
    main()
