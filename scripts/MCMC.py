# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 23:19:17 2025

@author: Baaz
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Base PK Model ---
def approx_dirac_delta(t, t0, epsilon=0.05):
    return np.exp(-((t - t0) / epsilon)**2) / (epsilon * np.sqrt(np.pi))

def pk_2cpt_impulse(t, y, ka, cl, v, dose_amount, dose_times):
    C1, C2 = y
    dose_input = sum(dose_amount * approx_dirac_delta(t, t_dose) for t_dose in dose_times)
    dC1dt = -ka * C1 + dose_input
    dC2dt = ka * C1 - (cl / v) * C2
    return [dC1dt, dC2dt]

def solve_individual(ka, cl, v, dose_amount, dose_times, t_interval, t_eval):
    y0 = [dose_amount, 0]
    sol = solve_ivp(
        pk_2cpt_impulse, t_interval, y0,
        args=(ka, cl, v, dose_amount, dose_times),
        t_eval=t_eval,
        rtol=1e-6, atol=1e-8
    )
    return sol

# --- Simulate ONE individual ---
def simulate_single_individual(add_e=0.1, prop_e=0.1):
    np.random.seed(42)
    ka_pop, cl_pop, v_pop = 1.0, 3.0, 5.0
    ka_sd, cl_sd = 0.5, 0.5  # Log-normal SDs
    dose_amount = 100
    dose_times = [0, 4, 8,12]
    t_interval = (0, 24)
    t_sample = np.arange(0, 24.25, 0.25)

    ka_true = np.random.lognormal(np.log(ka_pop), ka_sd)
    cl_true = np.random.lognormal(np.log(cl_pop), cl_sd)
    v = np.log(v_pop)  # fixed for simplicity

    sol = solve_individual(ka_true, cl_true, v, dose_amount, dose_times, t_interval, t_sample)
    C2_true = sol.y[1]
    
    noise_add = np.random.normal(0, add_e, size=C2_true.shape)
    noise_prop = np.random.normal(0, prop_e, size=C2_true.shape)
    C2_obs = C2_true * (1 + noise_prop) + noise_add

    return t_sample, C2_obs, dose_amount, dose_times, ka_true, cl_true, v

# --- Log-posterior: Log-likelihood + log-prior ---
def log_posterior(params_log, t_sample, C2_obs, dose_amount, dose_times, v, add_e, prop_e, ka_mean=1.0, cl_mean=3.0, ka_sd=0.5, cl_sd=0.5):
    ka = np.exp(params_log[0])
    cl = np.exp(params_log[1])

    try:
        sol = solve_individual(ka, cl, v, dose_amount, dose_times, (0, 24), t_sample)
        C2_pred = sol.y[1]
    except Exception:
        return -np.inf  # failed integration

    sigma = np.sqrt((prop_e * C2_pred)**2 + add_e**2)
    log_likelihood = -0.5 * np.sum(((C2_obs - C2_pred) / sigma)**2 + np.log(2 * np.pi * sigma**2))

    log_prior_ka = -0.5 * ((params_log[0] - np.log(ka_mean)) / ka_sd)**2
    log_prior_cl = -0.5 * ((params_log[1] - np.log(cl_mean)) / cl_sd)**2

    return log_likelihood + log_prior_ka + log_prior_cl

# --- MH Sampler ---
def mh_sample(t_sample, C2_obs, dose_amount, dose_times, v, n_samples=5000, step_size=0.1, add_e=0.1, prop_e=0.1):
    samples = []
    current = np.log([1.0, 3.0])  # Start at population mean in log-space
    current_log_post = log_posterior(current, t_sample, C2_obs, dose_amount, dose_times, v, add_e, prop_e)

    for i in range(n_samples):
        proposal = current + np.random.normal(0, step_size, size=2)
        prop_log_post = log_posterior(proposal, t_sample, C2_obs, dose_amount, dose_times, v, add_e, prop_e)
        
        if np.log(np.random.rand()) < prop_log_post - current_log_post:
            current = proposal
            current_log_post = prop_log_post
        samples.append(np.exp(current))  # store in natural scale

    return np.array(samples)

# --- Main ---
if __name__ == "__main__":
    t_sample, C2_obs, dose_amount, dose_times, ka_true, cl_true, v = simulate_single_individual()

    samples = mh_sample(t_sample, C2_obs, dose_amount, dose_times, v, n_samples=2000, step_size=0.05)

    print(f"True ka: {ka_true:.3f}, cl: {cl_true:.3f}")
    print(f"Estimated ka (mean): {samples[:,0].mean():.3f}, cl (mean): {samples[:,1].mean():.3f}")
    print(f"Estimated ka (mean): {samples[:,0].std():.3f}, cl (mean): {samples[:,1].std():.3f}")
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(samples[:,0], bins=50, alpha=0.7, label='ka samples')
    axes[0].axvline(ka_true, color='r', linestyle='--', label='True ka')
    axes[0].legend()
    axes[0].set_title("Posterior samples for ka")

    axes[1].hist(samples[:,1], bins=50, alpha=0.7, label='cl samples')
    axes[1].axvline(cl_true, color='r', linestyle='--', label='True cl')
    axes[1].legend()
    axes[1].set_title("Posterior samples for cl")
    plt.tight_layout()
    plt.show()
# --- Simulate trajectories for 100 posterior samples ---
n_traj = 100
chosen_indices = np.random.choice(samples.shape[0], size=n_traj, replace=False)
t_dense = np.linspace(0, 24, 300)
trajectories = []

for idx in chosen_indices:
    ka_i, cl_i = samples[idx]
    sol = solve_individual(ka_i, cl_i, v, dose_amount, dose_times, (0, 24), t_dense)
    trajectories.append(sol.y[1])  # C2

# --- Plot all trajectories ---
plt.figure(figsize=(10, 6))
for traj in trajectories:
    plt.plot(t_dense, traj, color='blue', alpha=0.2)

# Add observed noisy data for comparison
plt.scatter(t_sample, C2_obs, color='red', s=10, label='Observed C2')

plt.title('Posterior Predictive C2 Trajectories (100 samples)')
plt.xlabel('Time (h)')
plt.ylabel('C2 Concentration')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
