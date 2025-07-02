import numpy as np
import matplotlib.pyplot as plt

def f(phi, t):
    x0, ke = phi
    return x0 * np.exp(-ke * t)

def log_prior(phi, mu, sigma_phi):
    # Independent normal priors on x0 and ke
    return -0.5 * np.sum(((phi - mu)**2) / sigma_phi)

def log_likelihood(phi, y, sigma, t):
    y_pred = f(phi, t)
    return -0.5 * ((y - y_pred)**2) / sigma

def log_posterior(phi, y, mu, sigma, sigma_phi, t):
    return log_likelihood(phi, y, sigma, t) + log_prior(phi, mu, sigma_phi)

def metropolis_hastings(y, mu, sigma, sigma_phi, t, n_samples=10000, proposal_std=0.1, burn_in=1000):
    samples = []
    current_phi = mu.copy()

    for i in range(n_samples + burn_in):
        # Propose new candidate: Gaussian random walk
        phi_candidate = np.random.normal(current_phi, proposal_std)

        # Calculate acceptance probability
        log_p_current = log_posterior(current_phi, y, mu, sigma, sigma_phi, t)
        log_p_candidate = log_posterior(phi_candidate, y, mu, sigma, sigma_phi, t)
        log_accept_ratio = log_p_candidate - log_p_current

        if np.log(np.random.rand()) < log_accept_ratio:
            current_phi = phi_candidate  # accept

        if i >= burn_in:
            samples.append(current_phi.copy())

    return np.array(samples)

if __name__ == "__main__":
    np.random.seed(42)

    # Observation at time t
    t = 5.0
    # True parameters (unknown)
    true_phi = np.array([3.0, 0.4])
    sigma_obs = 0.3  # observation noise variance

    # Generate synthetic observation
    y = f(true_phi, t) + np.random.normal(0, np.sqrt(sigma_obs))

    # Prior mean and variance for phi = (x0, ke)
    mu_prior = np.array([2.0, 0.5])
    sigma_phi_prior = np.array([1.0, 0.5])  # variance for each param

    # Run MCMC
    samples = metropolis_hastings(
        y=y,
        mu=mu_prior,
        sigma=sigma_obs,
        sigma_phi=sigma_phi_prior,
        t=t,
        n_samples=15000,
        proposal_std=np.array([0.05, 0.02]),
        burn_in=3000
    )

    # Plot results
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].hist(samples[:, 0], bins=50, density=True, color='skyblue', alpha=0.7)
    axs[0].axvline(true_phi[0], color='red', label='True x0')
    axs[0].set_title('Posterior samples for $x_0$')
    axs[0].legend()

    axs[1].hist(samples[:, 1], bins=50, density=True, color='lightgreen', alpha=0.7)
    axs[1].axvline(true_phi[1], color='red', label='True $k_e$')
    axs[1].set_title('Posterior samples for $k_e$')
    axs[1].legend()

    plt.suptitle(f"Observed y={y:.3f} at time t={t}")
    plt.show()
