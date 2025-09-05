# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 13:28:36 2025

@author: MarcusBaaz
"""

import os
import random
import time
import math

# Third-party libraries
import numpy as np
import pandas as pd

# For plotting (optional, only if you use plotting later)
import matplotlib.pyplot as plt

# For PyTorch (if using torchdiffeq or neural nets later)
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint as odeint

def plot_batch_results(results, t_sample, dose_times):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract unique doses and create a colormap
    doses = [res['dose_amount'] for res in results]
    unique_doses = sorted(set(doses))
    cmap = plt.get_cmap('viridis', len(unique_doses))


    # Create a mapping dose -> color
    dose_to_color = {dose: cmap(i) for i, dose in enumerate(unique_doses)}

    for res in results:
        t = res['t_eval']
        c2 = res['C2_with_noise']
        dose = res['dose_amount']
        color = dose_to_color[dose]
        id_label = f'ID {res["ID"]} (dose: {dose})'
        sampled = np.interp(t_sample, t, c2)
        ax.plot(t, c2, label=id_label, color=color)
        ax.plot(t_sample, sampled, 'o', alpha=0.4, color=color)

    for t_dose in dose_times:
        ax.axvline(x=t_dose, color='gray', linestyle='--', alpha=0.3)

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('C2 Concentration (with noise)')
    # Optional: show legend with doses only
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=dose_to_color[d], lw=2, label=f'Dose {d}') for d in unique_doses]
    ax.legend(handles=legend_elements, title="Dose Amounts")

    ax.grid()
    plt.title("Simulated C2 Concentration Over Time (Batch)")
    plt.tight_layout()
    plt.show()



def save_results(sampled_data, save_path):
    df = pd.DataFrame(sampled_data)
    df.to_csv(save_path, index=False)


def simulate_2cpt_and_save_vectorized(n_individuals, add_e, prop_e, dose_amounts, dose_times, save_path, plot=False):
    t_interval = (0, 36)
    ka_mean = 1.0
    cl_mean = 1.5
    v_mean = 5.0

    ka_sd = 0.5
    cl_sd = 0.5

    add_error = add_e
    prop_error = prop_e

    extra_points = []
    window = 0.3  # time units around dose
    for dt in dose_times:
        extra_points.extend(np.linspace(dt - window, dt + window, 100))
        
    t_eval = np.unique(np.concatenate([np.linspace(t_interval[0], t_interval[1], 120), dose_times, extra_points]))
    
    t_sample = np.arange(t_interval[0], t_interval[1] + 1, 1)

    sampled_data = []
    all_results = []

    id_counter = 1  # to keep track of global ID

    for dose_amount in dose_amounts:
        # Vectorized parameter sampling
        ka_samples = np.random.lognormal(mean=np.log(ka_mean), sigma=ka_sd, size=n_individuals)
        cl_samples = np.random.lognormal(mean=np.log(cl_mean), sigma=cl_sd, size=n_individuals)
        v = np.log(v_mean)  # constant




        C2_simulated = solve_individual_vectorized(ka_samples, cl_samples, v, dose_amount, dose_times, t_eval)

        #C2_simulated = solve_individual_vectorized(ka_samples, cl_samples, v, dose_amount, dose_times, t_eval)

        # Add noise
        # Add noise only where C2_simulated > 0
        noise = np.random.normal(0, add_error, size=C2_simulated.shape)
        noise_prop = np.random.normal(0, prop_error, size=C2_simulated.shape)
        
        C2_with_noise = C2_simulated.copy()  # Avoid modifying original
        mask = C2_simulated > 0
        C2_with_noise[mask] = C2_simulated[mask] * (1 + noise_prop[mask]) + noise[mask]
        C2_with_noise[C2_with_noise < 0] = 0

        # Interpolate to sample times
        C2_sampled = np.array([np.interp(t_sample, t_eval, c2) for c2 in C2_with_noise])
        C1_dummy = np.zeros_like(C2_sampled)  # If you want C1, implement it similarly

        # Store results
        for i in range(n_individuals):
            for t, C1, C2 in zip(t_sample, C1_dummy[i], C2_sampled[i]):
                sampled_data.append({
                    'TIME': t,
                    'AMT': dose_amount,
                    'ID': id_counter + i,
                    'DV': C2,
                    'ka': ka_samples[i],
                    'cl': cl_samples[i],
                    'DOSE TIME': dose_times  # optionally dose_times[0] if scalar
                })

            if plot:
                all_results.append({
                    't_eval': t_eval,
                    'C2_with_noise': C2_with_noise[i],
                    'ID': id_counter + i,
                    'dose_amount': dose_amount
                })

        id_counter += n_individuals

    save_results(sampled_data, save_path)

    if plot:
        plot_batch_results(all_results, t_sample, dose_times)




def approx_dirac_delta_vectorized(t, dose_times, dt, scaling=0.02,normalize=1):
    epsilon = scaling
    return np.sum(np.exp(-((t - dose_times) / epsilon)**2) / (epsilon * np.sqrt(np.pi)))/normalize




def pk_2cpt_step(y, ka, cl, v, dose_amount, dose_times, t, dt):
    # y: shape (N_samples, 2) for C1, C2 at time t
    C1 = y[:, 0]
    C2 = y[:, 1]
    ka = np.asarray(ka)
    cl = np.asarray(cl)
    v = float(v)  # scalar
    dose_amount = float(dose_amount)  # scalar
    dose_times = np.asarray(dose_times)
    y = np.asarray(y)

    # Vectorized dose input (same for all samples since dose times & amount same)
    dose_input = dose_amount * approx_dirac_delta_vectorized(t, dose_times,dt)

    dC1dt = -ka * C1 + dose_input
    dC2dt = ka * C1 - (cl / v) * C2

    dy = np.stack([dC1dt, dC2dt], axis=1)
    y_next = y + dy * dt
    return y_next

def solve_individual_vectorized(ka, cl, v, dose_amount, dose_times, t_eval):
    # ka, cl shape: (N_samples,)
    N_samples = ka.shape[0]
    y = np.zeros((N_samples, 2))
    y[:, 0] = 0  # initial C1
    y[:, 1] = 0            # initial C2

    y_out = np.zeros((N_samples, len(t_eval)))
    y_out[:, 0] = y[:, 1]

    for i in range(1, len(t_eval)):
        dt = t_eval[i] - t_eval[i-1]
        y = pk_2cpt_step(y, ka, cl, v, dose_amount, dose_times, t_eval[i-1], dt)
        y_out[:, i] = y[:,1]  # store C2

    return y_out
