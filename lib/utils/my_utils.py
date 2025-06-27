"""
Created on Fri Jun 27 10:09:11 2025

@author: Baaz
"""
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import os

class TrajectoryDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path)
        self.compartment = 'C2'  # Column name for the compartment to model

        # Estimate max values for dose and time (with upper limits)
        self.max_dose = min(self.df['Dose'].max(), 100.0)
        self.max_time = min(self.df['Time'].max(), 24.0)

        # Normalize dose and time columns
        self.df['Dose_norm'] = self.df['Dose'] / self.max_dose
        self.df['Time_norm'] = self.df['Time'] / self.max_time

        # Standardize concentration values in the specified compartment
        self.conc_mean = self.df[self.compartment].mean()
        self.conc_std = self.df[self.compartment].std()
        self.df['C2_norm'] = (self.df[self.compartment] - self.conc_mean) / self.conc_std

        # Parse and normalize dose time lists
        self.df['DoseTimesParsed'] = self.df['Dose times'].apply(ast.literal_eval)
        self.df['DoseTimesNorm'] = self.df['DoseTimesParsed'].apply(
            lambda lst: [t / self.max_time for t in lst]
        )

        # Extract individual trajectories from the dataframe
        self.trajectories = self._extract_trajectories()

    def _extract_trajectories(self):
        # Find row indices where new trajectories start (time == 0)
        start_idxs = self.df[self.df['Time'] == 0].index.tolist() + [len(self.df)]

        trajectories = []
        for i in range(len(start_idxs) - 1):
            group = self.df.iloc[start_idxs[i]:start_idxs[i+1]]

            t = torch.tensor(group['Time_norm'].values, dtype=torch.float32)
            x = torch.tensor(group['C2_norm'].values, dtype=torch.float32)
            dose = torch.tensor(group['Dose_norm'].values[0], dtype=torch.float32)
            dose_times = torch.tensor(group['DoseTimesNorm'].values[0], dtype=torch.float32)
            subject_id = group['ID'].iloc[0]

            trajectories.append((t, x, dose, dose_times, subject_id))

        return trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]



# ---- Collate ----
def collate_fn(batch):
    t_list, x_list, dose_list, dose_times_list, id_list = zip(*batch)  # unpack 5 elements
    t_padded = pad_sequence(t_list, batch_first=True)
    x_padded = pad_sequence(x_list, batch_first=True)
    mask = torch.stack([torch.ones_like(t) for t in t_list])
    dose_tensor = torch.stack(dose_list)
    return id_list, t_padded, x_padded, mask, dose_tensor, dose_times_list

    
def torch_linear_interpolate(x_dense, y_dense, x_target):
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

def kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p):
   """KL[q||p] between two diagonal Gaussians"""
   var_q = torch.exp(logvar_q)
   var_p = torch.exp(logvar_p)
   
   return 0.5 * torch.sum(
       (var_q + (mu_q - mu_p)**2) / var_p - 1 + logvar_p - logvar_q
   )


def load_model(component, name, load_dir, modelname):
    if load_dir is None:
        print(f"Skipping loading {name}, no load_dir specified.")
        return
    path = os.path.join(load_dir, f"final_{name}_{modelname}.pth")
    if os.path.exists(path):
        component.load_state_dict(torch.load(path))
        print(f"Loaded {name} from {path}")
    else:
        print(f"File not found, skipping: {path}")


def save_model(component, name, save_dir, modelname):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"final_{name}_{modelname}.pth")
    torch.save(component.state_dict(), path)
    print(f"Saved {name} to {path}")


def plot_from_training_records(
    records,
    func,
    t_dense,
    conc_mean,
    conc_std,
    max_plots,
    max_time,
    max_dose,
    compartment,
    initial_encoder,
    reducer,
    nr_row,
    nr_col
):
    with torch.no_grad():
        n = min(len(records), max_plots)
        fig, axs = plt.subplots(nr_row, nr_col, figsize=(30, 20), sharex=True)
        axs = axs.flatten()

        for i in range(n):
            id_list, t, x, dose, dose_times, z_refined, z_sampled = records[i]
            x_interp = torch.tensor(np.interp(t_dense.numpy(), t.numpy(), x.numpy()), dtype=torch.float32)
            x0_input = torch.cat([x[0].unsqueeze(0), x[0].unsqueeze(0)], dim=0)

            x0 = initial_encoder(x0_input)

            pred_1 = odeint(lambda t_, x_: func(t_, x_, dose, dose_times, z_refined), x0, t_dense, method='rk4')
            x_pred_1 = reducer(pred_1)

            pred_2 = odeint(lambda t_, x_: func(t_, x_, dose, dose_times, z_sampled), x0, t_dense, method='rk4')
            x_pred_2 = reducer(pred_2)

            ax = axs[i]
            ax.plot(t.numpy() * max_time, (x.numpy() * conc_std) + conc_mean, 'o-', label='Actual')
            ax.plot(t_dense.numpy() * max_time, (x_pred_1.numpy() * conc_std) + conc_mean, 'o-', label='Predicted_refined')
            # ax.plot(t_dense.numpy() * max_time, (x_pred_2.numpy() * conc_std) + conc_mean, 'o-', label='Predicted_sampled')

            ax.set_title(f"Individual {i+1} - Dose: {dose.item() * max_dose:.0f} mg")
            ax.set_ylabel(f"Concentration ({compartment})")
            ax.grid(True)
            ax.set_xlim(0, 24)
            ax.set_ylim(0, 20)
            ax.legend()

        axs[-1].set_xlabel("Time (hours)")
        plt.tight_layout()
        plt.show()
        plt.close()
