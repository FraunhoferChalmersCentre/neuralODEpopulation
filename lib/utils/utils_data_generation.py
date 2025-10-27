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
    sigma_ka=0.0, sigma_ke=0.0, sigma_v=0.0,    sigma_state1=5, sigma_state2=5
):
    """
    Simulate 2-compartment PK model with SDE structural noise
    for multiple individuals and save results.
    """

    t_interval = (0, 24)
    sample_frequency = 0.5
    ka_mean = 0.6
    ke_mean = 0.6
    v_mean = 50.0

    ka_sd = 0.5
    ke_sd = 0.5
    v_sd = 0.1   # <-- SD for lognormal variability in v

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
        ke_samples = np.random.lognormal(mean=np.log(ke_mean), sigma=ke_sd, size=n_individuals)
        v_samples  = np.random.lognormal(mean=np.log(v_mean),  sigma=v_sd,  size=n_individuals)

        # Simulate all individuals with SDE solver (parameter noise)
        C2_simulated = []
        for i in range(n_individuals):
            conc_i = solve_individual_vectorized_sde_param_noise(
                ka_samples[i:i+1], ke_samples[i:i+1], v_samples[i],
                dose_amount, dose_times, t_eval,
                sigma_ka=sigma_ka, sigma_ke=sigma_ke, sigma_v=sigma_v,    sigma_state1=sigma_state1, sigma_state2=sigma_state2
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
                    'ke': ke_samples[i],
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
    y, ka, ke, v, dose_amount, dose_times, t, dt,
    sigma_ka=0.05, sigma_ke=0.05, sigma_v=0.0,
    sigma_state1=5, sigma_state2=5
):
    """
    Euler–Maruyama step for 2-compartment SDE with:
      - noise on parameters (ka, ke, v)
      - additive structural noise on state dynamics (C1, C2)

    y: shape (N_samples, 2) for C1, C2
    """
    C1 = y[:, 0]
    C2 = y[:, 1]

    ka = np.asarray(ka)
    ke = np.asarray(ke)
    v = float(v)
    dose_amount = float(dose_amount)
    dose_times = np.asarray(dose_times)

    # stochastic noise on parameters
    dW_ka = np.random.normal(0, np.sqrt(dt), size=ka.shape)
    dW_ke = np.random.normal(0, np.sqrt(dt), size=ke.shape)
    dW_v  = np.random.normal(0, np.sqrt(dt)) if sigma_v > 0 else 0.0

    ka_t = ka * (1 + sigma_ka * dW_ka)
    ke_t = ke * (1 + sigma_ke * dW_ke)
    v_t  = v * (1 + sigma_v * dW_v) if sigma_v > 0 else v

    # deterministic drift using noisy parameters
    dose_input = dose_amount * approx_dirac_delta_vectorized(t, dose_times, dt)
    dC1dt = -ka_t * C1 + dose_input
    dC2dt = ka_t * C1 - (ke_t) * C2

    # --- structural noise (Euler–Maruyama) ---
    dW1 = np.random.normal(0, np.sqrt(dt), size=C1.shape)
    dW2 = np.random.normal(0, np.sqrt(dt), size=C2.shape)

    dC1 = dC1dt * dt + sigma_state1  * dW1
    dC2 = dC2dt * dt + sigma_state2  * dW2

    y_next = y + np.stack([dC1, dC2], axis=1)
    return y_next


def solve_individual_vectorized_sde_param_noise(
        ka, ke, v, dose_amount, dose_times, t_eval,
        sigma_ka=0.05, sigma_ke=0.05, sigma_v=0.0,
        sigma_state1=0.02, sigma_state2=0.02):
    """
    Vectorized solver for 2-compartment SDE with parameter noise.
    ka, ke: arrays of size (N_samples,)
    v: scalar
    """
    N_samples = ka.shape[0]
    y = np.zeros((N_samples, 2))   # C1, C2
    y[:, 0] = 0
    y[:, 1] = v

    y_out = np.zeros((N_samples, len(t_eval)))
    y_out[:, 0] = y[:, 1]  # plasma concentration (C2)

    for i in range(1, len(t_eval)):
        dt = t_eval[i] - t_eval[i - 1]
        y = pk_2cpt_sde_step_param_noise(
            y, ka, ke, v, dose_amount, dose_times,
            t_eval[i - 1], dt,
            sigma_ka=sigma_ka, sigma_ke=sigma_ke, sigma_v=sigma_v,
            sigma_state1=sigma_state1, sigma_state2=sigma_state2
        )
        y_out[:, i] = y[:, 1]  # plasma concentrations

    return y_out





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_tumor_volume(
    n_individuals,
    dose_amounts_list,
    dose_times_list,
    a_drugs,
    k_growth_mean=0.1,
    k_growth_sd=0.02,
    V0_mean=100.0,
    V0_sd=10.0,
    add_e=0,
    prop_e=0,
    save_path="tumor_sim.csv",
    t_interval=(0,16),
    sample_frequency=0.5,
    ka_mean=0.6,
    ke_mean=0.6,
    v_mean=50,
    ka_sd=0.5,
    ke_sd=0.5,
    v_sd=0.25,
    max_tumor_size=200000,
    plot=False
):
    """
    Simulate tumor volume under multiple drugs using simplified PK model.
    Doses are now recorded at their actual scheduled times, including zeros.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    n_drugs = len(dose_amounts_list)
    sampled_data = []
    id_counter = 1

    # Time grid includes all dose times explicitly
    all_dose_times = np.unique(np.concatenate(dose_times_list))
    t_eval = np.unique(np.concatenate([
        np.linspace(t_interval[0], t_interval[1], 120),
        all_dose_times
    ]))
    t_eval.sort()
    t_sample = np.arange(t_interval[0], t_interval[1] + sample_frequency, sample_frequency)

    for ind in range(n_individuals):
        # Sample individual PK parameters
        ka = np.random.lognormal(mean=np.log(ka_mean), sigma=ka_sd, size=n_drugs)
        ke = np.random.lognormal(mean=np.log(ke_mean), sigma=ke_sd, size=n_drugs)
        v  = np.random.lognormal(mean=np.log(v_mean), sigma=v_sd, size=n_drugs)
        a  = np.array(a_drugs)

        # Tumor growth parameters
        k_growth = np.random.normal(k_growth_mean, k_growth_sd)
        V = np.random.normal(V0_mean, V0_sd)

        param_names = []
        param_values = []
        for d in range(n_drugs):
            param_names.extend([f'ka_drug{d+1}', f'ke_drug{d+1}', f'v_drug{d+1}', f'a_drug{d+1}'])
            param_values.extend([ka[d], ke[d], v[d], a[d]])
        param_names.extend(['k_growth', 'V0'])
        param_values.extend([k_growth, V])

        # PK compartments
        C1 = np.zeros(n_drugs)
        C2 = np.zeros(n_drugs)

        V_out = []
        stop_simulation = False

        # --- Simulation loop ---
        for i, t in enumerate(t_eval):
            if stop_simulation:
                break

            dt = t_eval[i] - t_eval[i - 1] if i > 0 else 0.01

            # Administer doses at the scheduled times
            # Administer doses at scheduled times
            for d in range(n_drugs):
                doses = np.array(dose_amounts_list[d])
                times = np.array(dose_times_list[d])
                mask = np.isclose(t, times, atol=1e-8)
                if mask.any():
                    amt = doses[mask][0]
                    C1[d] += amt
                    sampled_data.append({
                        'ID': id_counter,
                        'TIME': times[mask][0],
                        'DV': np.nan,
                        'AMT': amt,
                        'EVID': d+1,  # Always use drug ID for dosing events
                        'TREATMENT': f'Drug{d+1}',
                        'PARAM_NAMES': param_names,
                        'PARAM_VALUES': param_values
                    })


            # PK dynamics
            dC1 = -ka * C1
            dC2 = ka * C1 - ke * C2
            C1 += dC1 * dt
            C2 += dC2 * dt
            C2 = np.maximum(0, C2)

            # Tumor dynamics
            effect = np.sum(a * C2)
            dV = V * (k_growth - effect)
            V += dV * dt

            if V >= max_tumor_size:
                stop_simulation = True
                break

            V_out.append(V)

        # Sample tumor volume at observation times
        if V_out:
            t_eval_recorded = t_eval[:len(V_out)]
            t_sample_recorded = t_sample[t_sample <= t_eval_recorded[-1]]
            V_sampled = np.interp(t_sample_recorded, t_eval_recorded, V_out)

            noise_add = np.random.normal(0, add_e, size=V_sampled.shape)
            noise_prop = np.random.normal(0, prop_e, size=V_sampled.shape)
            V_noisy = np.maximum(0, V_sampled * (1 + noise_prop) + noise_add)

            for t_obs, dv in zip(t_sample_recorded, V_noisy):
                sampled_data.append({
                    'ID': id_counter,
                    'TIME': t_obs,
                    'DV': dv,
                    'AMT': 0,
                    'EVID': 0,
                    'TREATMENT': None,
                    'PARAM_NAMES': param_names,
                    'PARAM_VALUES': param_values
                })

        if plot and V_out:
            plt.plot(t_sample_recorded, V_noisy)

        id_counter += 1

    df = pd.DataFrame(sampled_data)
    df = df.sort_values(['ID', 'TIME'])
    df = df[~(df['DV'].isna() & (df['EVID'] == 0))]
    df = df[~((df['DV'].isna() & (df['EVID'] == 0)) | ((df['AMT'] == 0) & (df['EVID'] != 0)))]


    df.to_csv(save_path, index=False, sep=';')

    if plot:
        plt.xlabel("Time (hours)")
        plt.ylabel("Tumor Volume (DV)")
        plt.title("Simulated Tumor Volume under Drug Treatment")
        plt.show()

    print(f"Saved simulated tumor data to {save_path}")


import os
def simulate_tumor_volume_with_event(
    n_individuals,
    dose_amounts_list,
    dose_times_list,
    a_drugs,
    alpha=0.01,
    beta=0.01,
    k_growth_mean=0.1,
    k_growth_sd=0.02,
    V0_mean=100.0,
    V0_sd=10.0,
    add_e=0,
    prop_e=0,
    save_path="tumor_sim_with_event.csv",
    t_interval=(0,16),
    sample_frequency=0.5,
    ka_mean=0.6,
    ke_mean=0.6,
    v_mean=50,
    ka_sd=0.5,
    ke_sd=0.5,
    v_sd=0.25,
    max_tumor_size=2000,
    use_deterministic_event=True,
    plot=False
):
    """
    Simulate tumor volume under multiple drugs and generate time-to-event data.

    Event mechanisms:
      - Stochastic: hazard h(t) = max(0, alpha + beta * V)
      - Deterministic (optional): trigger when V >= 1.2 * V0

    Parameters
    ----------
    use_deterministic_event : bool, default=True
        Whether to include the deterministic event mechanism (V >= 1.2 * V0).
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    n_drugs = len(dose_amounts_list)
    sampled_data = []
    tte_data = []
    id_counter = 1
    deterministic_count = 0  # counter for deterministic events

    # Fine evaluation times
    extra_points = []
    window = 0.3
    for dt_list in dose_times_list:
        for dt in dt_list:
            extra_points.extend(np.linspace(dt - window, dt + window, 20))
    t_eval = np.unique(np.concatenate([
        np.linspace(t_interval[0], t_interval[1], 120),
        *dose_times_list,
        extra_points
    ]))
    t_sample = np.arange(t_interval[0], t_interval[1] + sample_frequency, sample_frequency)

    for ind in range(n_individuals):
        # Sample individual PK parameters
        ka = np.random.lognormal(mean=np.log(ka_mean), sigma=ka_sd, size=n_drugs)
        ke = np.random.lognormal(mean=np.log(ke_mean), sigma=ke_sd, size=n_drugs)
        v  = np.random.lognormal(mean=np.log(v_mean), sigma=v_sd, size=n_drugs)
        a  = np.array(a_drugs)

        # Sample tumor parameters
        k_growth = np.random.normal(k_growth_mean, k_growth_sd)
        V = np.random.normal(V0_mean, V0_sd)
        V_baseline = V  # store baseline tumor volume

        # Store parameters
        param_names = []
        param_values = []
        for d in range(n_drugs):
            param_names.extend([f'ka_drug{d+1}', f'ke_drug{d+1}', f'v_drug{d+1}', f'a_drug{d+1}'])
            param_values.extend([ka[d], ke[d], v[d], a[d]])
        param_names.extend(['k_growth', 'V0', 'alpha', 'beta'])
        param_values.extend([k_growth, V, alpha, beta])

        # Initialize PK compartments
        C1 = np.zeros(n_drugs)
        C2 = np.zeros(n_drugs)

        V_out = []
        hazard_cumsum = 0
        dt_prev = 0.01

        # Random number for stochastic event
        u_event = np.random.uniform()
        event_occurred = False
        event_type = "CENSORED"

        # Default TTE record (censored)
        tte_recorded = {
            'ID': id_counter,
            'TIME_TO_EVENT': t_sample[-1],
            'EVENT': 0,
            'EVENT_TYPE': event_type
        }

        for i, t in enumerate(t_eval):
            dt = t_eval[i] - t_eval[i-1] if i > 0 else dt_prev

            # Dosing
            for d in range(n_drugs):
                doses = np.array(dose_amounts_list[d])
                times = np.array(dose_times_list[d])
                mask = np.isclose(t, times, atol=1e-5)
                if mask.any():
                    C1[d] += doses[mask][0]
                    sampled_data.append({
                        'ID': id_counter,
                        'TIME': t,
                        'DV': np.nan,
                        'AMT': doses[mask][0],
                        'EVID': 1,
                        'TREATMENT': None,
                        'PARAM_NAMES': param_names,
                        'PARAM_VALUES': param_values
                    })

            # PK update
            dC1 = -ka * C1
            dC2 = ka * C1 - ke * C2
            C1 += dC1 * dt
            C2 += dC2 * dt
            C2 = np.maximum(0, C2)

            # Tumor dynamics
            effect = np.sum(a * C2)
            dV = V * (k_growth - effect)
            V += dV * dt
            V_out.append(V)

            # Prevent runaway growth
            if V > max_tumor_size:
                V = max_tumor_size

            # Hazard accumulation
            hazard = max(0, alpha + beta * V)
            hazard_cumsum += hazard * dt

            # --- Deterministic event (optional) ---
            if use_deterministic_event and (not event_occurred) and (V >= 1.2 * V_baseline):
                future_samples = t_sample[t_sample >= t]
                tte_recorded['TIME_TO_EVENT'] = (
                    future_samples[0] if len(future_samples) > 0 else t_sample[-1]
                )
                tte_recorded['EVENT'] = 1
                tte_recorded['EVENT_TYPE'] = "DETERMINISTIC"
                event_occurred = True
                deterministic_count += 1  # increment counter

            # --- Stochastic event ---
            elif (not event_occurred) and (1 - np.exp(-hazard_cumsum) >= u_event):
                future_samples = t_sample[t_sample >= t]
                tte_recorded['TIME_TO_EVENT'] = (
                    future_samples[0] if len(future_samples) > 0 else t_sample[-1]
                )
                tte_recorded['EVENT'] = 1
                tte_recorded['EVENT_TYPE'] = "STOCHASTIC"
                event_occurred = True

        # Save event data
        tte_data.append(tte_recorded)

        # Sample tumor volume data
        t_eval_recorded = t_eval[:len(V_out)]
        t_sample_recorded = t_sample[t_sample <= t_eval_recorded[-1]]
        V_sampled = np.interp(t_sample_recorded, t_eval_recorded, V_out)
        noise_add = np.random.normal(0, add_e, size=V_sampled.shape)
        noise_prop = np.random.normal(0, prop_e, size=V_sampled.shape)
        V_noisy = np.maximum(0, V_sampled * (1 + noise_prop) + noise_add)

        for t_obs, dv in zip(t_sample_recorded, V_noisy):
            sampled_data.append({
                'ID': id_counter,
                'TIME': t_obs,
                'DV': dv,
                'AMT': 0,
                'EVID': 0,
                'TREATMENT': None,
                'PARAM_NAMES': param_names,
                'PARAM_VALUES': param_values
            })

        id_counter += 1

        if plot and V_out:
            plt.plot(t_sample_recorded, V_noisy, alpha=0.3, color='gray')

    # Save outputs
    df_tumor = pd.DataFrame(sampled_data).sort_values(['ID', 'TIME'])
    df_tumor.to_csv(save_path, index=False, sep=';')

    df_tte = pd.DataFrame(tte_data)
    tte_save_path = os.path.join(os.path.dirname(save_path), "tte_" + os.path.basename(save_path))
    os.makedirs(os.path.dirname(tte_save_path), exist_ok=True)
    df_tte.to_csv(tte_save_path, index=False, sep=';')

    if plot:
        plt.xlabel("Time")
        plt.ylabel("Tumor Volume (DV)")
        plt.title("Simulated Tumor Volume")
        plt.show()

    print(f"Saved tumor volume data to {save_path}")
    print(f"Saved time-to-event data to {tte_save_path}")
    print(f"Number of individuals with deterministic events: {deterministic_count} "
          f"({deterministic_count / n_individuals * 100:.1f}%)")




# def simulate_tumor_volume_with_event(
#     n_individuals,
#     dose_amounts_list,
#     dose_times_list,
#     a_drugs,
#     alpha=0.01,
#     beta=0.01,
#     k_growth_mean=0.1,
#     k_growth_sd=0.02,
#     V0_mean=100.0,
#     V0_sd=10.0,
#     add_e=0,
#     prop_e=0,
#     save_path="tumor_sim_with_event.csv",
#     t_interval=(0,16),
#     sample_frequency=0.5,
#     ka_mean=0.6,
#     ke_mean=0.6,
#     v_mean=50,
#     ka_sd=0.5,
#     ke_sd=0.5,
#     v_sd=0.25,
#     max_tumor_size=2000,
#     plot=True,
#     use_deterministic_cutoff=True  # <-- new flag
# ):
#     """
#     Simulate tumor volume under multiple drugs and generate time-to-event data
#     with individual-specific hazard: h(t) = max(0, alpha + beta * V).
#     Deterministic event (tumor ≥ 1.2× baseline) is optional.
#     """
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import os

#     n_drugs = len(dose_amounts_list)
#     sampled_data = []
#     tte_data = []
#     id_counter = 1
#     n_triggered_1p2 = 0  # counter for deterministic events

#     # Fine evaluation times
#     extra_points = []
#     window = 0.3
#     for dt_list in dose_times_list:
#         for dt in dt_list:
#             extra_points.extend(np.linspace(dt - window, dt + window, 20))
#     t_eval = np.unique(np.concatenate([
#         np.linspace(t_interval[0], t_interval[1], 120),
#         *dose_times_list,
#         extra_points
#     ]))
#     t_sample = np.arange(t_interval[0], t_interval[1] + sample_frequency, sample_frequency)

#     for ind in range(n_individuals):
#         # Sample individual PK parameters
#         ka = np.random.lognormal(mean=np.log(ka_mean), sigma=ka_sd, size=n_drugs)
#         ke = np.random.lognormal(mean=np.log(ke_mean), sigma=ke_sd, size=n_drugs)
#         v  = np.random.lognormal(mean=np.log(v_mean), sigma=v_sd, size=n_drugs)
#         a  = np.array(a_drugs)

#         # Sample tumor parameters
#         k_growth = np.random.normal(k_growth_mean, k_growth_sd)
#         V = np.random.normal(V0_mean, V0_sd)
#         V0 = V  # store initial baseline

#         # Store parameters
#         param_names = []
#         param_values = []
#         for d in range(n_drugs):
#             param_names.extend([f'ka_drug{d+1}', f'ke_drug{d+1}', f'v_drug{d+1}', f'a_drug{d+1}'])
#             param_values.extend([ka[d], ke[d], v[d], a[d]])
#         param_names.extend(['k_growth', 'V0', 'alpha', 'beta'])
#         param_values.extend([k_growth, V0, alpha, beta])

#         # Initialize PK compartments
#         C1 = np.zeros(n_drugs)
#         C2 = np.zeros(n_drugs)

#         V_out = []
#         hazard_cumsum = 0
#         event_occurred = False
#         event_triggered_by_threshold = False
#         dt_prev = 0.01

#         u_event = np.random.uniform()

#         for i, t in enumerate(t_eval):
#             dt = t_eval[i] - t_eval[i-1] if i > 0 else dt_prev

#             # Dosing
#             for d in range(n_drugs):
#                 doses = np.array(dose_amounts_list[d])
#                 times = np.array(dose_times_list[d])
#                 mask = np.isclose(t, times, atol=1e-5)
#                 if mask.any():
#                     C1[d] += doses[mask][0]
#                     sampled_data.append({
#                         'ID': id_counter,
#                         'TIME': t,
#                         'DV': np.nan,
#                         'AMT': doses[mask][0],
#                         'EVID': 1,
#                         'TREATMENT': None,
#                         'PARAM_NAMES': param_names,
#                         'PARAM_VALUES': param_values
#                     })

#             # PK update
#             dC1 = -ka * C1
#             dC2 = ka * C1 - ke * C2
#             C1 += dC1 * dt
#             C2 += dC2 * dt
#             C2 = np.maximum(0, C2)

#             # Tumor growth
#             effect = np.sum(a * C2)
#             dV = V * (k_growth - effect)
#             V += dV * dt
#             V_out.append(V)

#             # Hazard process
#             hazard = max(0, alpha + beta * V)
#             hazard_cumsum += hazard * dt

#             # Event check
#             if not event_occurred:
#                 event_probability = 1 - np.exp(-hazard_cumsum)
#                 if event_probability >= u_event:
#                     tte = t
#                     event_occurred = True
#                     event_triggered_by_threshold = False
#                 elif use_deterministic_cutoff and V >= 1.2 * V0:
#                     tte = t
#                     event_occurred = True
#                     event_triggered_by_threshold = True

#                 if event_occurred:
#                     tte_data.append({
#                         'ID': id_counter,
#                         'TIME_TO_EVENT': tte,
#                         'EVENT': 1
#                     })

#         # Record if triggered by deterministic rule
#         if event_triggered_by_threshold:
#             n_triggered_1p2 += 1

#         # Interpolate tumor volumes
#         t_eval_recorded = t_eval[:len(V_out)]
#         t_sample_recorded = t_sample[t_sample <= t_eval_recorded[-1]]
#         V_sampled = np.interp(t_sample_recorded, t_eval_recorded, V_out)
#         noise_add = np.random.normal(0, add_e, size=V_sampled.shape)
#         noise_prop = np.random.normal(0, prop_e, size=V_sampled.shape)
#         V_noisy = np.maximum(0, V_sampled * (1 + noise_prop) + noise_add)

#         for t_obs, dv in zip(t_sample_recorded, V_noisy):
#             sampled_data.append({
#                 'ID': id_counter,
#                 'TIME': t_obs,
#                 'DV': dv,
#                 'AMT': 0,
#                 'EVID': 0,
#                 'TREATMENT': None,
#                 'PARAM_NAMES': param_names,
#                 'PARAM_VALUES': param_values
#             })

#         id_counter += 1

#         if plot and V_out:
#             plt.plot(t_sample_recorded, V_noisy)

#     # Save output files
#     df_tumor = pd.DataFrame(sampled_data).sort_values(['ID','TIME'])
#     df_tumor.to_csv(save_path, index=False, sep=';')
#     df_tte = pd.DataFrame(tte_data)
#     tte_save_path = os.path.join(os.path.dirname(save_path), "tte_" + os.path.basename(save_path))
#     os.makedirs(os.path.dirname(tte_save_path), exist_ok=True)
#     df_tte.to_csv(tte_save_path, index=False, sep=';')

#     # Plot if needed
#     if plot:
#         plt.xlabel("Time")
#         plt.ylabel("Tumor Volume (DV)")
#         plt.title("Simulated Tumor Volume and Event Times")
#         plt.show()

#     if use_deterministic_cutoff:
#         print(f"Number of individuals with deterministic (≥1.2×V0) events: {n_triggered_1p2} / {n_individuals}")
#         print(f"({100 * n_triggered_1p2 / n_individuals:.1f}% of all subjects)")

#     print(f"Saved tumor volume data to {save_path}")
#     print(f"Saved time-to-event data to {tte_save_path}")



def simulate_single_drug_concentration(
    n_individuals,
    dose_amounts,
    dose_times,
    ka_mean=0.6,
    ke_mean=0.6,
    v_mean=50.0,
    ka_sd=0.3,
    ke_sd=0.3,
    v_sd=0.25,
    corr_matrix=None,
    add_e=0,
    prop_e=0,
    t_interval=(0, 24),         # recording / sampling interval
    t_integration=None,          # integration interval (optional)
    sample_frequency=0.5,
    dt=0.01,
    save_path="single_drug_sim.csv",
    plot=False
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if corr_matrix is None:
        corr_matrix = np.eye(3)

    sd_vector = np.array([ka_sd, ke_sd, v_sd])
    cov_matrix = np.outer(sd_vector, sd_vector) * corr_matrix

    # Set integration interval if not provided
    if t_integration is None:
        t_integration = t_interval

    t0_int, t_end_int = t_integration
    t_eval = np.arange(t0_int, t_end_int + dt / 2, dt)

    t0_rec, t_end_rec = t_interval
    t_sample = np.arange(t0_rec, t_end_rec + sample_frequency / 2, sample_frequency)

    sampled_data = []
    id_counter = 1

    for ind in range(n_individuals):
        # Sample correlated PK parameters (lognormal)
        mean_vector = np.log([ka_mean, ke_mean, v_mean])
        normal_sample = np.random.multivariate_normal(mean_vector, cov_matrix)
        ka, ke, v = np.exp(normal_sample)
        param_names = ['ka', 'ke','v']
        param_values = [ka, ke, v]

        # Initialize compartments
        A_gut = 0.0
        A_central = v

        # Precompute indices for doses
        dose_indices = [int(round((t_dose - t0_int) / dt)) for t_dose in dose_times]
        # Prepare dose events
        doses = list(zip(dose_times, dose_amounts))  # [(time, amt), ...]
        dose_counter = 0
        n_doses = len(doses)
        
        # Integration loop
        conc_trace = np.zeros_like(t_eval)
        for i, t in enumerate(t_eval):
            # Administer all doses that occur at or before current time
            while dose_counter < n_doses and t >= doses[dose_counter][0]:
                amt = doses[dose_counter][1]
                A_gut += amt
                # Record dose only if within recording interval
                if t0_rec <= t <= t_end_rec:
                    sampled_data.append({
                        'ID': id_counter,
                        'TIME': t,
                        'DV': np.nan,
                        'AMT': amt,
                        'EVID': 1,
                        'PARAM_NAMES': param_names,
                        'PARAM_VALUES': param_values
                    })
                dose_counter += 1
        
            # Store current central compartment amount
            conc_trace[i] = A_central
        
            # 1-compartment PK derivatives
            dA_gut = -ka * A_gut
            dA_central = ka * A_gut - ke * A_central
        
            # Euler update
            A_gut += dA_gut * dt
            A_central += dA_central * dt
        
            # Avoid negatives
            A_gut = max(A_gut, 0)
            A_central = max(A_central, 0)


        # Sample concentrations only in recording interval
        C_sampled = np.interp(t_sample, t_eval, conc_trace)
        noise_add = np.random.normal(0, add_e, size=C_sampled.shape)
        noise_prop = np.random.normal(0, prop_e, size=C_sampled.shape)
        C_noisy = np.maximum(0, C_sampled * (1 + noise_prop) + noise_add)

        for t_obs, dv in zip(t_sample, C_noisy):
            sampled_data.append({
                'ID': id_counter,
                'TIME': t_obs,
                'DV': dv,
                'AMT': 0,
                'EVID': 0,
                'PARAM_NAMES': param_names,
                'PARAM_VALUES': param_values
            })

        if plot:
            plt.plot(t_sample, C_noisy, label=f"Ind {id_counter}")

        id_counter += 1

    df = pd.DataFrame(sampled_data).sort_values(['ID', 'TIME'])
    df.to_csv(save_path, index=False, sep=';')

    if plot:
        plt.xlabel("Time (hours)")
        plt.ylabel("Concentration (DV)")
        plt.title("Simulated Single-Drug Concentration-Time Profiles (Fixed-step)")
        plt.show()

    print(f"Saved simulated concentration data to {save_path}")
    return df



def simulate_single_drug_concentration_1comp(
    n_individuals,
    dose_amounts,
    dose_times,
    ke_mean=0.6,
    v_mean=50.0,
    ke_sd=0.3,
    v_sd=0.25,
    corr_matrix=None,
    add_e=0,
    prop_e=0,
    t_interval=(0, 24),         # recording / sampling interval
    t_integration=None,          # integration interval (optional)
    sample_frequency=0.5,
    dt=0.01,
    save_path="single_drug_sim_1comp.csv",
    plot=False
):
    """
    Simulate plasma concentration (DV) for a single IV bolus drug
    using a fixed-step Euler integration of a 1-compartment model.

    dA_central/dt = -ke * A_central

    Parameters:
        - dose_amounts: list of dose amounts
        - dose_times: list of dose times (can include pre-zero doses)
        - ke_mean, v_mean: typical values of elimination rate constant and volume
        - ke_sd, v_sd: variability (lognormal SDs)
        - corr_matrix: 2x2 correlation between ke and v (optional)
        - add_e, prop_e: additive and proportional noise
        - t_interval: time window for recording concentrations
        - t_integration: total simulation time (integration range)
        - sample_frequency: observation interval in hours
        - dt: integration step
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Default: no correlation
    if corr_matrix is None:
        corr_matrix = np.eye(2)

    # Convert SDs to covariance
    sd_vector = np.array([ke_sd, v_sd])
    cov_matrix = np.outer(sd_vector, sd_vector) * corr_matrix

    # Integration vs recording windows
    if t_integration is None:
        t_integration = t_interval

    t0_int, t_end_int = t_integration
    t_eval = np.arange(t0_int, t_end_int + dt / 2, dt)

    t0_rec, t_end_rec = t_interval
    t_sample = np.arange(t0_rec, t_end_rec + sample_frequency / 2, sample_frequency)

    sampled_data = []
    id_counter = 1

    for ind in range(n_individuals):
        # Sample correlated parameters (lognormal)
        mean_vector = np.log([ke_mean, v_mean])
        normal_sample = np.random.multivariate_normal(mean_vector, cov_matrix)
        ke, v = np.exp(normal_sample)
        ke = np.clip(ke, 1e-4, 10)
        v = np.clip(v, 1, 1e3)

        param_names = ['ke', 'v']
        param_values = [ke, v]

        # Initialize compartment
        A_central = 0.0

        # Prepare dose events
        doses = list(zip(dose_times, dose_amounts))
        dose_counter = 0
        n_doses = len(doses)

        conc_trace = np.zeros_like(t_eval)

        for i, t in enumerate(t_eval):
            # Administer dose(s)
            while dose_counter < n_doses and t + 1e-9 >= doses[dose_counter][0]:
                amt = doses[dose_counter][1]
                A_central += amt
                if t0_rec - 1e-9 <= t <= t_end_rec + 1e-9:
                    sampled_data.append({
                        'ID': id_counter,
                        'TIME': t,
                        'DV': np.nan,
                        'AMT': amt,
                        'EVID': 1,
                        'PARAM_NAMES': param_names,
                        'PARAM_VALUES': param_values
                    })
                dose_counter += 1

            # Store current central amount
            conc_trace[i] = A_central

            # Elimination
            dA_central = -ke * A_central

            # Euler update
            A_central += dA_central * dt
            A_central = max(A_central, 0)

        # Sample concentrations only within recording window
        C_sampled = np.interp(t_sample, t_eval, conc_trace / v)
        noise_add = np.random.normal(0, add_e, size=C_sampled.shape)
        noise_prop = np.random.normal(0, prop_e, size=C_sampled.shape)
        C_noisy = np.maximum(0, C_sampled * (1 + noise_prop) + noise_add)

        for t_obs, dv in zip(t_sample, C_noisy):
            sampled_data.append({
                'ID': id_counter,
                'TIME': t_obs,
                'DV': dv,
                'AMT': 0,
                'EVID': 0,
                'PARAM_NAMES': param_names,
                'PARAM_VALUES': param_values
            })

        if plot:
            plt.plot(t_sample, C_noisy, label=f"Ind {id_counter}")

        id_counter += 1

    df = pd.DataFrame(sampled_data).sort_values(['ID', 'TIME'])
    df.to_csv(save_path, index=False, sep=';')

    if plot:
        plt.xlabel("Time (hours)")
        plt.ylabel("Concentration (DV)")
        plt.title("Simulated 1-Compartment IV Bolus Concentration-Time Profiles")
        plt.show()

    print(f"Saved simulated 1-compartment concentration data to {save_path}")
    return df





def approx_dirac_delta_vectorized(t, dose_times, dt, tol=1e-3):
    """
    Approximate Dirac delta for discrete time steps.
    Returns 1/dt if t is within tol of any dose time, else 0.
    """
    return np.any(np.abs(t - np.array(dose_times)) < tol) / dt

def pk_2cpt_step(y, ka, cl, v, dose_amounts, dose_times_list, t, dt):
    """
    One step of vectorized 2-compartment PK simulation with multiple drugs.
    
    y: array (N_samples, 2) [C1, C2]
    ka, cl: arrays of shape (N_samples,)
    v: array or scalar
    dose_amounts: list of arrays (per drug)
    dose_times_list: list of arrays (per drug)
    t: current time
    dt: timestep
    """
    C1 = y[:, 0]
    C2 = y[:, 1]

    # Total dose input at this time for all drugs
    dose_input_total = np.zeros_like(C1)
    for doses, times in zip(dose_amounts, dose_times_list):
        doses_arr = np.array(doses)
        times_arr = np.array(times)
        mask = np.isclose(t, times_arr, atol=1e-5)
        if mask.any():
            dose_input_total += doses_arr[mask][0]

    dC1dt = -ka * C1 + dose_input_total
    dC2dt = ka * C1 - (cl / v) * C2

    dy = np.stack([dC1dt, dC2dt], axis=1)
    y_next = y + dy * dt
    y_next[:, 1] = np.clip(y_next[:, 1], 0, None)  # prevent negative conc
    return y_next

def solve_individual_vectorized(ka, cl, v, dose_amounts, dose_times_list, t_eval):
    N_samples = ka.shape[0]
    y = np.zeros((N_samples, 2))  # C1, C2 initial
    y_out = np.zeros((N_samples, len(t_eval)))
    y_out[:, 0] = y[:, 1]

    for i in range(1, len(t_eval)):
        dt = t_eval[i] - t_eval[i-1]
        y = pk_2cpt_step(y, ka, cl, v, dose_amounts, dose_times_list, t_eval[i-1], dt)
        y_out[:, i] = y[:, 1]
    return y_out

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def approx_dirac_delta_vectorized(t, dose_times, dt, tol=1e-3):
    """
    Approximate Dirac delta for discrete time steps.
    Returns 1/dt if t is within tol of any dose time, else 0.
    """
    return (np.any(np.abs(t - np.array(dose_times)) < tol) / dt)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def approx_dirac_delta_vectorized(t, dose_times, dt, tol=1e-3):
    """
    Approximate Dirac delta for discrete time steps.
    Returns 1/dt if t is within tol of any dose time, else 0.
    """
    return 1.0/dt if np.any(np.abs(t - np.array(dose_times)) < tol) else 0.0


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def approx_dirac_delta_vectorized(t, dose_times, dt, tol=1e-3):
    """Approximate Dirac delta for discrete time steps."""
    return np.any(np.abs(t - np.array(dose_times)) < tol) / dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_2cpt_and_save(
    n_individuals,
    add_e,
    prop_e,
    dose_amounts_list,
    dose_times_list,
    save_path,
    plot=False,
    t_interval=(0,16),
    sample_frequency=0.5,
    ka_mean=0.6,
    ke_mean=0.6,
    v_mean=50,
    ka_sd=0.5,
    ke_sd=0.5,
    v_sd=0.25
):
    n_drugs = len(dose_amounts_list)
    sampled_data = []
    id_counter = 1

    # Fine evaluation times
    extra_points = []
    window = 0.3
    for dt_list in dose_times_list:
        for dt in dt_list:
            extra_points.extend(np.linspace(dt - window, dt + window, 20))

    t_eval = np.unique(np.concatenate([np.linspace(t_interval[0], t_interval[1], 120),
                                       *dose_times_list,
                                       extra_points]))
    t_sample = np.arange(t_interval[0], t_interval[1] + sample_frequency, sample_frequency)

    for ind in range(n_individuals):
        # Individual parameters
        ka = np.random.lognormal(mean=np.log(ka_mean), sigma=ka_sd, size=n_drugs)
        ke = np.random.lognormal(mean=np.log(ke_mean), sigma=ke_sd, size=n_drugs)
        v  = np.random.lognormal(mean=np.log(v_mean), sigma=v_sd, size=n_drugs)

        # Initialize compartments
        C1 = np.zeros(n_drugs)
        C2 = np.zeros(n_drugs)
        y_out = np.zeros((len(t_eval), n_drugs))

        # Simulation loop
        for i, t in enumerate(t_eval):
            dt = t_eval[i] - t_eval[i-1] if i > 0 else 0.01

            # Add doses
            for d in range(n_drugs):
                times = np.array(dose_times_list[d])
                doses = np.array(dose_amounts_list[d])
                mask = np.isclose(t, times, atol=1e-5)
                if mask.any():
                    C1[d] += doses[mask][0]

            # Update compartments
            dC1 = -ka * C1
            dC2 = ka * C1 - (ke / v) * C2
            C1 += dC1 * dt
            C2 += dC2 * dt
            C2 = np.maximum(0, C2)
            y_out[i,:] = C2

        # Interpolate to observation times
        y_sampled_noisy = np.zeros((len(t_sample), n_drugs))
        for d in range(n_drugs):
            y_sampled = np.interp(t_sample, t_eval, y_out[:, d])
            # Add noise
            noise_add = np.random.normal(0, add_e, size=y_sampled.shape)
            noise_prop = np.random.normal(0, prop_e, size=y_sampled.shape)
            y_sampled_noisy[:, d] = np.maximum(0, y_sampled * (1 + noise_prop) + noise_add)

        # Store observation rows
        for i_time, t_obs in enumerate(t_sample):
            row = {'ID': id_counter, 'TIME': t_obs, 'EVID': 0}
            for d in range(n_drugs):
                row[f'DV_Drug{d+1}'] = y_sampled_noisy[i_time, d]
            sampled_data.append(row)

        # Store dosing rows
        for d in range(n_drugs):
            for amt, t_dose in zip(dose_amounts_list[d], dose_times_list[d]):
                row = {'ID': id_counter, 'TIME': t_dose, 'EVID': d+1}
                for j in range(n_drugs):
                    row[f'DV_Drug{j+1}'] = np.nan
                sampled_data.append(row)

        # Optional plotting
        if plot:
            for d in range(n_drugs):
                plt.plot(t_sample, y_sampled_noisy[:, d], label=f'ID {id_counter} Drug{d+1}')

        id_counter += 1

    # Convert to DataFrame
    df = pd.DataFrame(sampled_data)
    df = df.sort_values(['ID','TIME'])
    df.to_csv(save_path, index=False, sep=';')

    if plot:
        plt.xlabel("Time (hours)")
        plt.ylabel("Concentration (DV)")
        plt.title("Simulated 2CPT Profiles per Drug")
        plt.legend()
        plt.show()

    print(f"Saved simulated data to {save_path}")













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
        

def pk_2cpt_sde_step(y, ka, ke, v, dose_amount, dose_times, t, dt,
                     sigma1=0.05, sigma2=0.05):
    """
    Euler–Maruyama step for 2-compartment SDE.
    y: shape (N_samples, 2) for C1, C2
    """
    C1 = y[:, 0]
    C2 = y[:, 1]

    ka = np.asarray(ka)
    ke = np.asarray(ke)
    v = float(v)
    dose_amount = float(dose_amount)
    dose_times = np.asarray(dose_times)

    # deterministic drift (same as before)
    dose_input = dose_amount * approx_dirac_delta_vectorized(t, dose_times, dt)
    dC1dt = -ka * C1 + dose_input
    dC2dt = ka * C1 - (ke / v) * C2

    # stochastic diffusion terms
    dW1 = np.random.normal(0, np.sqrt(dt), size=C1.shape)
    dW2 = np.random.normal(0, np.sqrt(dt), size=C2.shape)

    dC1 = dC1dt * dt + sigma1 * C1 * dW1
    dC2 = dC2dt * dt + sigma2 * C2 * dW2

    y_next = y + np.stack([dC1, dC2], axis=1)
    return y_next



