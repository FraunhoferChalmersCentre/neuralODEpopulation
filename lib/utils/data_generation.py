
import numpy as np                  # for numerical operations
import matplotlib.pyplot as plt     # for plotting
import pandas as pd                 # for saving/loading data


from lib.utils.my_utils_parallel import *

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


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
    t_interval = (0, 24)
    ka_mean = 1.0
    cl_mean = 1.5
    v_mean = 5.0

    ka_sd = 0.5
    cl_sd = 0.2

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
        noise = np.random.normal(0, add_error, size=C2_simulated.shape)
        noise_prop = np.random.normal(0, prop_error, size=C2_simulated.shape)
        C2_with_noise = C2_simulated * (1 + noise_prop) + noise

        # Interpolate to sample times
        C2_sampled = np.array([np.interp(t_sample, t_eval, c2) for c2 in C2_with_noise])
        C1_dummy = np.zeros_like(C2_sampled)  # If you want C1, implement it similarly

        # Store results
        for i in range(n_individuals):
            for t, C1, C2 in zip(t_sample, C1_dummy[i], C2_sampled[i]):
                sampled_data.append({
                    'Time': t,
                    'Dose': dose_amount,
                    'ID': id_counter + i,
                    'C1': C1,
                    'C2': C2,
                    'ka': ka_samples[i],
                    'cl': cl_samples[i],
                    'Dose times': dose_times  # optionally dose_times[0] if scalar
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

