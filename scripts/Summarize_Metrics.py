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
    metrics_dir = os.path.join(project_root, "results/theo/run5")

    # Mock sys.argv for demo (remove when running from CLI)
    sys.argv = [
        'script_name',
        '--metrics_raw', 'metrics_raw.csv',
        '--metrics_raw_ae', 'metrics_raw_ae.csv',
        '--metrics_raw_ae_noise', 'metrics_raw_ae_noise.csv',
        '--metrics_monolix', 'metrics_monolix.csv',

    ]

    # Define parser
    parser = argparse.ArgumentParser(description="Load metrics CSV files and summarize them.")
    parser.add_argument("--metrics_raw", type=str, required=True, help="Path to VAE metrics CSV")
    parser.add_argument("--metrics_raw_ae", type=str, required=True, help="Path to AE metrics CSV")
    parser.add_argument("--metrics_raw_ae_noise", type=str, required=True, help="Path to AE with noise metrics CSV")
    parser.add_argument("--metrics_monolix", type=str, required=True, help="Path to SAEM First Order CSV")


    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Paths
    path_vae = os.path.join(metrics_dir, args.metrics_raw)
    path_ae = os.path.join(metrics_dir, args.metrics_raw_ae)
    path_ae_noise = os.path.join(metrics_dir, args.metrics_raw_ae_noise)
    path_saem_first = os.path.join(metrics_dir, args.metrics_monolix)

    # Load CSVs
    df_vae = pd.read_csv(path_vae)
    df_ae = pd.read_csv(path_ae)
    df_ae_noise = pd.read_csv(path_ae_noise)
    df_saem_first = pd.read_csv(path_saem_first)


    print("Loaded CSVs:")
    print(f" - VAE: {df_vae.shape}")
    print(f" - AE: {df_ae.shape}")
    print(f" - AE with noise: {df_ae_noise.shape}")
    print(f" - NODE monolix: {df_saem_first.shape}")

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
        # Compute mean and std for each metric if it exists
        if "mean_MSE" in df.columns:
            mean_val = df["mean_MSE"].mean()
            std_val = df["mean_MSE"].std()
            summary["mse_mean"] = f"{mean_val:.3f} ({std_val:.3f})"
        if "median_MSE" in df.columns:
            mean_val = df["median_MSE"].mean()
            std_val = df["median_MSE"].std()
            summary["mse_median"] = f"{mean_val:.3f} ({std_val:.3f})"
        if "mean_R2" in df.columns:
            mean_val = df["mean_R2"].mean()
            std_val = df["mean_R2"].std()
            summary["r2_mean"] = f"{mean_val:.3f} ({std_val:.3f})"
        if "median_R2" in df.columns:
            mean_val = df["median_R2"].mean()
            std_val = df["median_R2"].std()
            summary["r2_median"] = f"{mean_val:.3f} ({std_val:.3f})"
        return summary



    # Compute summaries
    summary_vae = summarize_metrics_flat(df_vae)
    summary_ae = summarize_metrics_flat(df_ae)
    summary_ae_noise = summarize_metrics_flat(df_ae_noise)
    summary_saem_first = import_saem_summary(df_saem_first)


    # Combine into a single DataFrame
    combined_summary = pd.DataFrame(
        [summary_vae, summary_ae, summary_ae_noise, summary_saem_first],
        index=['VAE', 'AE', 'AE with noise', 'Monolix NODE']
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\nCombined Summary:")
    print(combined_summary)
