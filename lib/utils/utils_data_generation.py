# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 13:28:36 2025

@author: MarcusBaaz
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def plot_batch_results_truncated(results, dose_times):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract unique doses and create a colormap
    doses = [res['dose_amount'] for res in results]
    unique_doses = sorted(set(doses))
    cmap = plt.get_cmap('viridis', len(unique_doses))

    # Map dose -> color
    dose_to_color = {dose: cmap(i) for i, dose in enumerate(unique_doses)}

    for res in results:
        t = res['t_eval']
        c2 = res['C2_with_noise']
        t_sample = res['t_sample']   # <-- use per-individual samples
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

    # Legend (by dose only)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=dose_to_color[d], lw=2, label=f'Dose {d}') for d in unique_doses]
    ax.legend(handles=legend_elements, title="Dose Amounts")

    ax.grid()
    plt.title("Simulated C2 Concentration Over Time (Batch)")
    plt.tight_layout()
    plt.show()


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
    df.to_csv(save_path, index=False, sep=';')
    


def simulate_2cpt_and_save_vectorized(
    n_individuals, add_e, prop_e, dose_amounts, dose_times, save_path, plot=False
):
    t_interval = (0, 16)
    sample_frequency=0.5
    ka_mean = 0.3
    cl_mean = 0.2
    v_mean = 5.0

    ka_sd = 0.5
    cl_sd = 0.5

    add_error = add_e
    prop_error = prop_e

    extra_points = []
    window = 0.3
    for dt in dose_times:
        extra_points.extend(np.linspace(dt - window, dt + window, 100))

    t_eval = np.unique(np.concatenate([np.linspace(t_interval[0], t_interval[1], 120),
                                       dose_times, extra_points]))
    t_sample = np.arange(t_interval[0], t_interval[1] + sample_frequency, sample_frequency)

    sampled_data = []
    all_results = []
    id_counter = 1

    for dose_amount in dose_amounts:
        ka_samples = np.random.lognormal(mean=np.log(ka_mean), sigma=ka_sd, size=n_individuals)
        cl_samples = np.random.lognormal(mean=np.log(cl_mean), sigma=cl_sd, size=n_individuals)
        v = np.log(v_mean)

        # Simulate all individuals at once
        C2_simulated = solve_individual_vectorized(ka_samples, cl_samples, v, dose_amount, dose_times, t_eval)

        # Add noise
        noise = np.random.normal(0, add_error, size=C2_simulated.shape)
        noise_prop = np.random.normal(0, prop_error, size=C2_simulated.shape)
        C2_with_noise = C2_simulated.copy()
        mask = C2_simulated > 0
        C2_with_noise[mask] = C2_simulated[mask] * (1 + noise_prop[mask]) + noise[mask]
        C2_with_noise[C2_with_noise < 0] = 0

        C2_sampled = np.array([np.interp(t_sample, t_eval, c2) for c2 in C2_with_noise])
        C1_dummy = np.zeros_like(C2_sampled)

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
                    'DOSE TIME': dose_times
                })

            if plot:
                all_results.append({
                    't_eval': t_sample,
                    't_sample': t_sample,
                    'C2_with_noise': C2_sampled[i],
                    'ID': id_counter + i,
                    'dose_amount': dose_amount
                })

        id_counter += n_individuals

    save_results(sampled_data, save_path)

    if plot:
        plot_batch_results(all_results, t_sample, dose_times)




def simulate_2cpt_and_save_vectorized_truncated(
    n_individuals, add_e, prop_e, dose_amounts, dose_times, save_path,
    min_samples=4, plot=False
):
    """
    Simulates 2-compartment model, then truncates each individual's time series
    to mimic variable follow-up lengths.
    """
    # --- Step 1: Run the full simulation in memory ---
    sampled_data_full = []
    all_results_full = []

    def capture_results(data, results):
        nonlocal sampled_data_full, all_results_full
        sampled_data_full = data
        all_results_full = results

    # Override save_results temporarily to capture data
    orig_save_results = globals().get("save_results")
    def save_results_override(data, path):
        capture_results(data, all_results_full)
    globals()["save_results"] = save_results_override

    simulate_2cpt_and_save_vectorized(
        n_individuals, add_e, prop_e, dose_amounts, dose_times,
        save_path="__dummy__", plot=True
    )

    if orig_save_results:
        globals()["save_results"] = orig_save_results

    # --- Step 2: Apply truncation ---
    truncated_data = []
    truncated_results = []
    rng = np.random.default_rng()

    for subj_id in set(d["ID"] for d in sampled_data_full):
        subj_rows = [d for d in sampled_data_full if d["ID"] == subj_id]
        subj_rows_sorted = sorted(subj_rows, key=lambda x: x["TIME"])

        # choose random truncation length
        max_samples = len(subj_rows_sorted)
        n_keep = rng.integers(low=min_samples, high=max_samples + 1)

        kept_rows = subj_rows_sorted[:n_keep]
        truncated_data.extend(kept_rows)

        if plot:
            t_sample = [row["TIME"] for row in kept_rows]
            C2_with_noise = [row["DV"] for row in kept_rows]

            truncated_results.append({
                "t_eval": t_sample,
                "t_sample": t_sample,
                "C2_with_noise": np.array(C2_with_noise),
                "ID": subj_id,
                "dose_amount": kept_rows[0]["AMT"]
            })

    # --- Step 3: Save and/or plot ---
    save_results(truncated_data, save_path)

    if plot:
        plot_batch_results_truncated(truncated_results, dose_times)
        
def simulate_2cpt_sde_and_save_vectorized(
    n_individuals, add_e, prop_e, dose_amounts, dose_times,
    save_path, plot=False,
    sigma_ka=0.0, sigma_cl=0.0, sigma_v=0.0,    sigma_state1=0.0, sigma_state2=0.0
):
    """
    Simulate 2-compartment PK model with SDE structural noise
    for multiple individuals and save results.
    """

    t_interval = (0, 24)
    sample_frequency = 0.5
    ka_mean = 0.5
    cl_mean = 10.0
    v_mean = 200.0

    ka_sd = 0.3
    cl_sd = 0.3
    v_sd = 0.25   # <-- SD for lognormal variability in v

    add_error = add_e
    prop_error = prop_e

    # add denser points around dose events
    extra_points = []
    window = 0.3
    for dt in dose_times:
        extra_points.extend(np.linspace(dt - window, dt + window, 100))

    t_eval = np.unique(np.concatenate([
        np.linspace(t_interval[0], t_interval[1], 120),
        dose_times, extra_points
    ]))
    t_sample = np.arange(t_interval[0], t_interval[1] + sample_frequency, sample_frequency)

    sampled_data = []
    all_results = []
    id_counter = 1

    for dose_amount in dose_amounts:
        # sample inter-individual variability (log-normal distributions)
        ka_samples = np.random.lognormal(mean=np.log(ka_mean), sigma=ka_sd, size=n_individuals)
        cl_samples = np.random.lognormal(mean=np.log(cl_mean), sigma=cl_sd, size=n_individuals)
        v_samples  = np.random.lognormal(mean=np.log(v_mean),  sigma=v_sd,  size=n_individuals)

        # Simulate all individuals with SDE solver (parameter noise)
        C2_simulated = []
        for i in range(n_individuals):
            conc_i = solve_individual_vectorized_sde_param_noise(
                ka_samples[i:i+1], cl_samples[i:i+1], v_samples[i],
                dose_amount, dose_times, t_eval,
                sigma_ka=sigma_ka, sigma_cl=sigma_cl, sigma_v=sigma_v,    sigma_state1=sigma_state1, sigma_state2=sigma_state2
            )
            C2_simulated.append(conc_i[0])  # shape (len(t_eval),)
        C2_simulated = np.array(C2_simulated)

        # Add observation noise
        noise = np.random.normal(0, add_error, size=C2_simulated.shape)
        noise_prop = np.random.normal(0, prop_error, size=C2_simulated.shape)
        C2_with_noise = C2_simulated.copy()
        mask = C2_simulated > 0
        C2_with_noise[mask] = C2_simulated[mask] * (1 + noise_prop[mask]) + noise[mask]
        C2_with_noise[C2_with_noise < 0] = 0

        # Resample at uniform grid
        C2_sampled = np.array([np.interp(t_sample, t_eval, c2) for c2 in C2_with_noise])
        C1_dummy = np.zeros_like(C2_sampled)

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
                    'v': v_samples[i],
                    'DOSE TIME': dose_times
                })

            if plot:
                all_results.append({
                    't_eval': t_sample,
                    't_sample': t_sample,
                    'C2_with_noise': C2_sampled[i],
                    'ID': id_counter + i,
                    'dose_amount': dose_amount
                })

        id_counter += n_individuals

    save_results(sampled_data, save_path)

    if plot:
        plot_batch_results(all_results, t_sample, dose_times)



def pk_2cpt_sde_step_param_noise(
    y, ka, cl, v, dose_amount, dose_times, t, dt,
    sigma_ka=0.05, sigma_cl=0.05, sigma_v=0.0,
    sigma_state1=0.02, sigma_state2=0.02
):
    """
    Euler–Maruyama step for 2-compartment SDE with:
      - noise on parameters (ka, cl, v)
      - additive structural noise on state dynamics (C1, C2)

    y: shape (N_samples, 2) for C1, C2
    """
    C1 = y[:, 0]
    C2 = y[:, 1]

    ka = np.asarray(ka)
    cl = np.asarray(cl)
    v = float(v)
    dose_amount = float(dose_amount)
    dose_times = np.asarray(dose_times)

    # stochastic noise on parameters
    dW_ka = np.random.normal(0, np.sqrt(dt), size=ka.shape)
    dW_cl = np.random.normal(0, np.sqrt(dt), size=cl.shape)
    dW_v  = np.random.normal(0, np.sqrt(dt)) if sigma_v > 0 else 0.0

    ka_t = ka * (1 + sigma_ka * dW_ka)
    cl_t = cl * (1 + sigma_cl * dW_cl)
    v_t  = v * (1 + sigma_v * dW_v) if sigma_v > 0 else v

    # deterministic drift using noisy parameters
    dose_input = dose_amount * approx_dirac_delta_vectorized(t, dose_times, dt)
    dC1dt = -ka_t * C1 + dose_input
    dC2dt = ka_t * C1 - (cl_t / v_t) * C2

    # --- structural noise (Euler–Maruyama) ---
    dW1 = np.random.normal(0, np.sqrt(dt), size=C1.shape)
    dW2 = np.random.normal(0, np.sqrt(dt), size=C2.shape)

    dC1 = dC1dt * dt + sigma_state1 * C1 * dW1
    dC2 = dC2dt * dt + sigma_state2 * C2 * dW2

    y_next = y + np.stack([dC1, dC2], axis=1)
    return y_next


def solve_individual_vectorized_sde_param_noise(
        ka, cl, v, dose_amount, dose_times, t_eval,
        sigma_ka=0.05, sigma_cl=0.05, sigma_v=0.0,
        sigma_state1=0.02, sigma_state2=0.02):
    """
    Vectorized solver for 2-compartment SDE with parameter noise.
    ka, cl: arrays of size (N_samples,)
    v: scalar
    """
    N_samples = ka.shape[0]
    y = np.zeros((N_samples, 2))   # C1, C2
    y[:, 0] = 0
    y[:, 1] = 0

    y_out = np.zeros((N_samples, len(t_eval)))
    y_out[:, 0] = y[:, 1]  # plasma concentration (C2)

    for i in range(1, len(t_eval)):
        dt = t_eval[i] - t_eval[i - 1]
        y = pk_2cpt_sde_step_param_noise(
            y, ka, cl, v, dose_amount, dose_times,
            t_eval[i - 1], dt,
            sigma_ka=sigma_ka, sigma_cl=sigma_cl, sigma_v=sigma_v,
            sigma_state1=sigma_state1, sigma_state2=sigma_state2
        )
        y_out[:, i] = y[:, 1]  # plasma concentrations

    return y_out







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



def simulate_2cpt_and_save_vectorized_truncated_with_error(
    n_individuals, add_e, prop_e, dose_amounts, dose_times, save_path,
    N=1, prop_trunc=0.5, extra_add_error=4, plot=False
):
    """
    Simulates 2-compartment model, truncates last N time points for a subset of individuals,
    and applies extra additive error ONLY to truncated individuals.
    
    Args:
        n_individuals : int
            Number of individuals to simulate.
        add_e : float
            Base additive error (from original simulation).
        prop_e : float
            Proportional error (from original simulation).
        dose_amounts : list
            Doses administered.
        dose_times : list
            Times of administration.
        save_path : str
            File path to save simulated data.
        N : int
            Number of last time points to remove for truncated individuals.
        prop_trunc : float, optional
            Proportion of individuals to truncate (default 0.3).
        extra_add_error : float, optional
            Extra additive error applied ONLY to truncated individuals (default 0.2).
        plot : bool, optional
            Whether to plot truncated simulation results.
    """
    # --- Step 1: Run the full simulation in memory ---
    sampled_data_full = []
    all_results_full = []

    def capture_results(data, results):
        nonlocal sampled_data_full, all_results_full
        sampled_data_full = data
        all_results_full = results

    # Override save_results temporarily
    orig_save_results = globals().get("save_results")
    def save_results_override(data, path):
        capture_results(data, all_results_full)
    globals()["save_results"] = save_results_override

    simulate_2cpt_and_save_vectorized(
        n_individuals, add_e, prop_e, dose_amounts, dose_times,
        save_path="__dummy__", plot=True
    )

    if orig_save_results:
        globals()["save_results"] = orig_save_results

    # --- Step 2: Decide which individuals will be truncated ---
    unique_ids = list(set(d["ID"] for d in sampled_data_full))
    rng = np.random.default_rng()
    n_truncated = int(len(unique_ids) * prop_trunc)
    truncated_ids = set(rng.choice(unique_ids, size=n_truncated, replace=False))

    truncated_data = []
    truncated_results = []

    for subj_id in unique_ids:
        subj_rows = [d for d in sampled_data_full if d["ID"] == subj_id]
        subj_rows_sorted = sorted(subj_rows, key=lambda x: x["TIME"])

        if subj_id in truncated_ids:
            # Remove last N points
            kept_rows = subj_rows_sorted[:-N] if len(subj_rows_sorted) > N else []

            # add extra additive noise to DV
            for row in kept_rows:
                row = row.copy()
                row["DV"] = max(0, row["DV"] + rng.normal(0, extra_add_error))
                truncated_data.append(row)

            if plot and kept_rows:
                t_sample = [row["TIME"] for row in kept_rows]
                C2_with_noise = [row["DV"] for row in kept_rows]
                truncated_results.append({
                    "t_eval": t_sample,
                    "t_sample": t_sample,
                    "C2_with_noise": np.array(C2_with_noise),
                    "ID": subj_id,
                    "dose_amount": kept_rows[0]["AMT"]
                })

        else:
            # keep all rows, no extra error
            truncated_data.extend(subj_rows_sorted)

            if plot:
                t_sample = [row["TIME"] for row in subj_rows_sorted]
                C2_with_noise = [row["DV"] for row in subj_rows_sorted]
                truncated_results.append({
                    "t_eval": t_sample,
                    "t_sample": t_sample,
                    "C2_with_noise": np.array(C2_with_noise),
                    "ID": subj_id,
                    "dose_amount": subj_rows_sorted[0]["AMT"]
                })

    # --- Step 3: Save and/or plot ---
    save_results(truncated_data, save_path)

    if plot:
        plot_batch_results_truncated(truncated_results, dose_times)
        

def pk_2cpt_sde_step(y, ka, cl, v, dose_amount, dose_times, t, dt,
                     sigma1=0.05, sigma2=0.05):
    """
    Euler–Maruyama step for 2-compartment SDE.
    y: shape (N_samples, 2) for C1, C2
    """
    C1 = y[:, 0]
    C2 = y[:, 1]

    ka = np.asarray(ka)
    cl = np.asarray(cl)
    v = float(v)
    dose_amount = float(dose_amount)
    dose_times = np.asarray(dose_times)

    # deterministic drift (same as before)
    dose_input = dose_amount * approx_dirac_delta_vectorized(t, dose_times, dt)
    dC1dt = -ka * C1 + dose_input
    dC2dt = ka * C1 - (cl / v) * C2

    # stochastic diffusion terms
    dW1 = np.random.normal(0, np.sqrt(dt), size=C1.shape)
    dW2 = np.random.normal(0, np.sqrt(dt), size=C2.shape)

    dC1 = dC1dt * dt + sigma1 * C1 * dW1
    dC2 = dC2dt * dt + sigma2 * C2 * dW2

    y_next = y + np.stack([dC1, dC2], axis=1)
    return y_next



