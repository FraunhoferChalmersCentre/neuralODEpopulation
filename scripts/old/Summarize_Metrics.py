# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 15:59:10 2025

@author: Baaz
"""

import os
import argparse
import sys
import torch
import pandas as pd

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)
    metrics_dir = os.path.join(project_root, "results/run1")

    # Mock sys.argv for demo (remove when running from CLI)
    sys.argv = [
        'script_name',
        '--metrics_raw', 'metrics_raw.csv',
        '--metrics_raw_ae', 'metrics_raw_ae.csv',
        '--metrics_raw_ae_noise', 'metrics_raw_ae_noise.csv',
        '--metrics_saem_first', 'metrics_SAEM_First_Order.csv',
        '--metrics_saem_zero', 'metrics_SAEM_Zero_Order.csv'
    ]

    # Define parser
    parser = argparse.ArgumentParser(description="Load metrics CSV files and summarize them.")
    parser.add_argument("--metrics_raw", type=str, required=True, help="Path to VAE metrics CSV")
    parser.add_argument("--metrics_raw_ae", type=str, required=True, help="Path to AE metrics CSV")
    parser.add_argument("--metrics_raw_ae_noise", type=str, required=True, help="Path to AE with noise metrics CSV")
    parser.add_argument("--metrics_saem_first", type=str, required=True, help="Path to SAEM First Order CSV")
    parser.add_argument("--metrics_saem_zero", type=str, required=True, help="Path to SAEM Zero Order CSV")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Paths
    path_vae = os.path.join(metrics_dir, args.metrics_raw)
    path_ae = os.path.join(metrics_dir, args.metrics_raw_ae)
    path_ae_noise = os.path.join(metrics_dir, args.metrics_raw_ae_noise)
    path_saem_first = os.path.join(metrics_dir, args.metrics_saem_first)
    path_saem_zero = os.path.join(metrics_dir, args.metrics_saem_zero)

    # Load CSVs
    df_vae = pd.read_csv(path_vae)
    df_ae = pd.read_csv(path_ae)
    df_ae_noise = pd.read_csv(path_ae_noise)
    df_saem_first = pd.read_csv(path_saem_first)
    df_saem_zero = pd.read_csv(path_saem_zero)

    print("Loaded CSVs:")
    print(f" - VAE: {df_vae.shape}")
    print(f" - AE: {df_ae.shape}")
    print(f" - AE with noise: {df_ae_noise.shape}")
    print(f" - SAEM First Order: {df_saem_first.shape}")
    print(f" - SAEM Zero Order: {df_saem_zero.shape}")

    # Function to compute mean, std for raw metrics (VAE, AE, AE noise)
    def summarize_metrics_flat(df):
        metrics = ['mse_mean', 'r2_mean', 'mse_median', 'r2_median', 'mse_validation']
        summary = {}
        for m in metrics:
            if m in df.columns:
                mean_val = df[m].mean()
                std_val = df[m].std()
                summary[m] = f"{mean_val:.3f} ({std_val:.3f})"
        return summary

    # Function to transform SAEM-style summary tables into same format
    def import_saem_summary(df):
        summary = {}
        for _, row in df.iterrows():
            metric = row["Metric"].upper()
            stat = row["Statistic"].lower()
            val = row["Value"]
            sd = row["SD"] if "SD" in df.columns and not pd.isna(row["SD"]) else None

            if metric == "MSE" and stat == "mean":
                summary["mse_mean"] = f"{val:.3f}" if sd is None else f"{val:.3f} ({sd:.3f})"
            elif metric == "R2" and stat == "mean":
                summary["r2_mean"] = f"{val:.3f}" if sd is None else f"{val:.3f} ({sd:.3f})"
            elif metric == "MSE" and stat == "median":
                summary["mse_median"] = f"{val:.3f}" if sd is None else f"{val:.3f} ({sd:.3f})"
            elif metric == "R2" and stat == "median":
                summary["r2_median"] = f"{val:.3f}" if sd is None else f"{val:.3f} ({sd:.3f})"
            # optional: handle validation metric if present
        return summary

    # Compute summaries
    summary_vae = summarize_metrics_flat(df_vae)
    summary_ae = summarize_metrics_flat(df_ae)
    summary_ae_noise = summarize_metrics_flat(df_ae_noise)
    summary_saem_first = import_saem_summary(df_saem_first)
    summary_saem_zero = import_saem_summary(df_saem_zero)

    # Combine into a single DataFrame
    combined_summary = pd.DataFrame(
        [summary_vae, summary_ae, summary_ae_noise, summary_saem_first, summary_saem_zero],
        index=['VAE', 'AE', 'AE with noise', 'SAEM First Order', 'SAEM Zero Order']
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\nCombined Summary:")
    print(combined_summary)
