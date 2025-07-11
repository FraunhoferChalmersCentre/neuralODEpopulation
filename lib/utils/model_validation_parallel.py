# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 16:24:41 2025

@author: Baaz
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 10:21:33 2025

@author: Baaz
"""

import os
import torch
    
import math
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from torchdiffeq import odeint as odeint

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score

from lib.utils.my_utils_parallel import *




def plotIndividualFits_MCMC(
    df, dataset, latent_dim,
    individual_data,  # (id, t_real, x_real, dose, dose_times)
    decoder1, decoder2,
    func, reducer, initial_encoder, ODEWrapper,t_dense,
    n_samples=100,
    n_mcmc_samples=1000,
    burn_in=10,
    device='cpu'
):
    MAX_TIME = estimate_max_time(df)
    MAX_DOSE = estimate_max_dose(df)
    conc_mean, conc_std = dataset.conc_mean, dataset.conc_std
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=collate_fn, num_workers=0)
    dose_times_lists = df['Dose times'].apply(ast.literal_eval).tolist()
    all_times = np.concatenate(dose_times_lists)
    
    unique_times_np = np.unique(all_times)
    dose_times = torch.from_numpy(unique_times_np).to(dtype=torch.float32, device=device)
    dose_times_tensor = dose_times / MAX_TIME
    
    t_dense = torch.unique(torch.cat([t_dense.to(device), dose_times_tensor]))
    
    
    t_dense_np = MAX_TIME* t_dense.cpu().numpy() if torch.is_tensor(t_dense) else t_dense
    t_dense_np2 = MAX_TIME* t_dense.cpu().numpy() if torch.is_tensor(t_dense) else t_dense
    
    dose_times_np = dose_times.cpu().numpy() if torch.is_tensor(dose_times) else dose_times
    
    

    extra_points = []
    window = 0.3  # time units around dose
    for dt in dose_times_np:
        extra_points.extend(np.linspace(dt - window, dt + window, 100))
    t_dense_np = np.unique(np.concatenate([t_dense_np, dose_times_np, extra_points]))

    t_real, x_real, dose, dose_times, _id = individual_data
    t_np = t_real.cpu().numpy() * MAX_TIME
    y_obs = x_real.cpu().numpy() * conc_std + conc_mean  # de-standardize observed data
    t_real, x_real, dose_times = t_real.to(device), x_real.to(device), dose_times.to(device)
    # MCMC Sampling
    mu_prior = np.log(np.array([0.8, 3.2]))  # log(ka, cl)
    sigma_prior = np.array([0.6, 0.4])
    add_error = 2
    prop_error = 0.001
   
    mcmc_samples_log = metropolis_hastings_sampling(
        y_obs=y_obs,
        t_np=t_dense_np,
        t_data=t_real*MAX_TIME,
        dose=dose * MAX_DOSE,
        dose_times=(dose_times.cpu().numpy() * MAX_TIME),
        add_error=add_error,
        prop_error=prop_error,
        mu_prior=mu_prior,
        sigma_prior=sigma_prior,
        n_samples=n_mcmc_samples,
        burn_in=burn_in,
        thinning=1
    )
    mcmc_samples = np.exp(mcmc_samples_log)  # shape (n_samples, 2)
    dose = dose.item()
    dose_amount = dose * MAX_DOSE
    # === Plot correlation scatter of MCMC samples ===
    plt.figure(figsize=(6, 6))
    plt.scatter(mcmc_samples[:, 0], mcmc_samples[:, 1], alpha=0.3, s=10, color='purple')
    plt.xlabel('ka')
    plt.ylabel('cl')
    plt.title('Scatter plot of MCMC samples (ka vs cl)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Plot histogram of MCMC samples ===
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(mcmc_samples[:, 0], bins=30, color='skyblue', edgecolor='k')
    plt.xlabel('ka')
    plt.ylabel('Frequency')
    plt.title('Histogram of MCMC samples for ka')

    plt.subplot(1, 2, 2)
    plt.hist(mcmc_samples[:, 1], bins=30, color='salmon', edgecolor='k')
    plt.xlabel('cl')
    plt.ylabel('Frequency')
    plt.title('Histogram of MCMC samples for cl')

    plt.tight_layout()
    plt.show()

    # ... rest of your existing code for ODE simulation and plotting ...
    # (keep everything unchanged from here on)
    # ODE simulation for each MCMC sample
    ka = mcmc_samples[:, 0]
    cl = mcmc_samples[:, 1]
    v = np.log(5.0)
   
    print(dose_times_np)

    all_C2 = solve_individual_vectorized(ka, cl, v, dose_amount, dose_times_np, t_dense_np)
    print("all_C2 shape:", all_C2.shape)


    
    all_C2_torch = torch.tensor(all_C2, device=device)  # move to torch tensor
    
    perc10_ode = torch.quantile(all_C2_torch, 0.10, dim=0).cpu().numpy()
    median_ode = torch.median(all_C2_torch, dim=0).values.cpu().numpy()
    perc90_ode = torch.quantile(all_C2_torch, 0.90, dim=0).cpu().numpy()

    # === Latent ODE prediction ===
   
    decoder = decoder1 if dose == 0.5 else decoder2

    t_real_rep = t_real.unsqueeze(0).repeat(n_samples, 1)
    x_real_rep = x_real.unsqueeze(0).repeat(n_samples, 1)

    mu, logvar = decoder(t_real_rep, x_real_rep)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z_samples = mu + eps * std

    x0_list = []
    for x in x_real_rep:
        out = initial_encoder(x[0].unsqueeze(0))  # expect shape [1, 4]
        if out.dim() == 1:
            out = out.unsqueeze(0)  # convert [4] -> [1, 4]
        x0_list.append(out)
    x0_tensor = torch.cat(x0_list, dim=0)  # now shape [6, 4]
    x0 = torch.cat([x0_tensor, z_samples], dim=-1)

    dose_tensor = torch.full((n_samples, 1), dose, device=device)

    dose_times_padded = torch.nn.utils.rnn.pad_sequence([dose_times], batch_first=True).to(device)
    dose_mask = (dose_times_padded != 0).to(device)
    dose_times_padded_rep = dose_times_padded.repeat(n_samples, 1)
    dose_mask_rep = dose_mask.repeat(n_samples, 1)
     
    t_dense_ode = torch.from_numpy(t_dense_np2)
    
    ode_func = ODEWrapper(func, dose_times_padded_rep,dose_tensor, dose_mask_rep)
    pred = odeint(ode_func, x0, t_dense.to(device), method='dopri5')  # [time, n_samples, latent_dim+1]
    
    x_pred = reducer(pred[:, :, :latent_dim])
    x_preds = x_pred.squeeze(-1).unsqueeze(0)  # Shape: [1, 120]

    print(x_pred.size())
    perc10 = torch.quantile(x_pred, 0.10, dim=1)
    median = torch.quantile(x_pred, 0.50, dim=1)
    perc90 = torch.quantile(x_pred, 0.90, dim=1)
    

    x_real_interp = batch_linear_interpolate_1d(
        x_real.unsqueeze(0), t_real.unsqueeze(0), t_dense.unsqueeze(0)
    ).squeeze(0)

    perc10_real = destandardize_concentration(perc10, conc_mean, conc_std).detach().cpu().numpy()
    median_real = destandardize_concentration(median, conc_mean, conc_std).detach().cpu().numpy()
    perc90_real = destandardize_concentration(perc90, conc_mean, conc_std).detach().cpu().numpy()
    x_real_de = destandardize_concentration(x_real, conc_mean, conc_std).detach().cpu().numpy()

    time_hours = t_dense_np
    print('t_dense:', t_dense.shape)
    print('perc10_real:', perc10_real.shape)
    print('perc90_real:', perc90_real.shape)

    # === Plot both MCMC and Latent ODE in one plot ===
    time_hours_mod = np.delete(time_hours, 1)
    perc10_ode_mod = np.delete(perc10_ode, 1)
    perc90_ode_mod = np.delete(perc90_ode, 1)
    median_ode_mod = np.delete(median_ode, 1)
    
    plt.figure(figsize=(10, 6))
    
    plt.fill_between(time_hours_mod, perc10_ode_mod, perc90_ode_mod, color='blue', alpha=0.3, label='MCMC ODE 10-90% CI')
    plt.plot(time_hours_mod, median_ode_mod, color='blue', label='MCMC ODE Median')
   
    print(perc10_real)
 

    # Latent ODE predictions in red
    plt.fill_between(MAX_TIME* t_dense, perc10_real, perc90_real, color='red', alpha=0.3, label="Latent ODE 10-90% CI")
    plt.plot(MAX_TIME* t_dense, median_real, color='red', label='Latent ODE Median')

    # Observed data
    plt.scatter(t_real.cpu().numpy() * MAX_TIME, x_real_de, color='black', label='Observed data', zorder=5)

    plt.xlabel("Time (hours)")
    plt.ylabel("Concentration")
    plt.title(f"Individual {_id} - Dose {dose * MAX_DOSE:.2f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    

def plotIndividualFits_test(ae,nf,
    test_dataset, df, latent_dim,noise,
    decoder1, decoder2,
    func, reducer, initial_encoder, ODEWrapper, t_dense,
    max_individuals=6, n_samples=50, device="cpu",
    truncation=0.5, add_noise=False  # <-- NEW ARGUMENT: normalized time (0-1)
):
    import math

    t_dense = t_dense.to(device)
    MAX_TIME = test_dataset.max_time
    MAX_DOSE = test_dataset.max_dose
    conc_mean, conc_std = test_dataset.conc_mean, test_dataset.conc_std

    # Separate indices by dose group
    dose_05_indices = []
    dose_10_indices = []

    for idx in range(len(test_dataset)):
        t, x, dose_tensor, dose_times_tensor, subject_id = test_dataset[idx]
        dose_val = dose_tensor.item()
        if abs(dose_val - 0.25) < 1e-3:
            dose_05_indices.append(idx)
        elif abs(dose_val - 1.0) < 1e-3:
            dose_10_indices.append(idx)

    n_each = max_individuals // 2
    selected_indices = dose_05_indices[:n_each] + dose_10_indices[:n_each]

    ncols = 3
    nrows = math.ceil(max_individuals / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for i, idx in enumerate(selected_indices):
        t, x, dose_tensor, dose_times_tensor, subject_id = test_dataset[idx]

        t, x = t.to(device), x.to(device)
        dose_tensor = dose_tensor.to(device).unsqueeze(0)
        dose = dose_tensor.item()
        dose_times_tensor = dose_times_tensor.to(device)

        # Normalize and truncate input for encoding
        t_full = t
        x_full = x
        t_norm = t_full
        x_std = x_full * conc_std + conc_mean

        # Mask for time <= truncation
        train_mask = t_norm <= truncation
        t_trunc = t_norm[train_mask].unsqueeze(0)
        x_trunc = x_full[train_mask].unsqueeze(0)


        decoder = decoder1 if abs(dose - 0.25) < 1e-3 else decoder2
        
        if ae: 
           # mu, _= decoder(t_trunc, x_trunc)
            z, mu, logvar,_ = decoder(t_trunc, x_trunc)
            z= mu.repeat(n_samples, 1)
            z= z + + torch.randn_like(z) * 0.01
        else:
            if nf:
                z_samples=[]
                for _ in range(n_samples):
                    zi, mu, logvar,_ = decoder(t_trunc, x_trunc)
                    z_samples.append(zi)
                    
                z = torch.stack(z_samples, dim=1).transpose(0,1)  # [B, n_samples, latent_dim]

            else:
                mu, logvar = decoder(t_trunc, x_trunc)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn(n_samples, *std.shape)

                z = mu + eps * std
                z=z.squeeze(dim=1)
 

        t_rep = t_trunc.repeat(n_samples, 1)
        x_rep = x_trunc.repeat(n_samples, 1)

     
            
        z = z.squeeze(1)  # If shape is [n_samples, 1, latent_dim]

        x0_latent = initial_encoder(x_trunc[:, 0].unsqueeze(1))
        x0_latent = x0_latent.repeat(n_samples, 1)  # shape: [1000, d]
      
        x0 = torch.cat([x0_latent,z], dim=1)

        #x0 = torch.cat([x0_1, z], dim=1)

        dose_times_padded = torch.nn.utils.rnn.pad_sequence([dose_times_tensor], batch_first=True).to(device)
        dose_mask = (dose_times_padded != 0).to(device)
        dose_times_padded = dose_times_padded.repeat(n_samples, 1)
        dose_mask = dose_mask.repeat(n_samples, 1)
        dose_rep = dose_tensor.repeat(n_samples, 1)

        ode_func = ODEWrapper(func, dose_times_padded, dose_rep, dose_mask)
        pred = odeint(ode_func, x0, t_dense, method='dopri5')
        pred = reducer(pred[:, :, :latent_dim])
        pred = destandardize_concentration(pred, conc_mean, conc_std)
        if add_noise:
            pred = noise.sample(pred, n_samples=1).squeeze(0)
            
        perc10 = torch.quantile(pred, 0.10, dim=1)
        perc90 = torch.quantile(pred, 0.90, dim=1)
        median = torch.median(pred, dim=1).values
        
        # Convert to numpy on CPU
        perc10 = perc10.detach().cpu().numpy()
        perc90 = perc90.detach().cpu().numpy()
        median = median.detach().cpu().numpy()



        time_dense_np = t_dense.cpu().numpy() * MAX_TIME

        # Define boolean masks on CPU numpy arrays for indexing:
        train_region = time_dense_np <= truncation * MAX_TIME
        test_region = time_dense_np > truncation * MAX_TIME
      

        # Then plot
        ax = axes[i // ncols][i % ncols]
        
        ax.fill_between(time_dense_np[train_region], perc10[train_region], perc90[train_region], color="blue", alpha=0.3, label="Pred 10–90% CI (Training)")
        ax.fill_between(time_dense_np[test_region], perc10[test_region], perc90[test_region], color="red", alpha=0.3, label="Pred 10–90% CI (Test)")
        
        ax.plot(time_dense_np[train_region], median[train_region], color="blue", label="Pred median (Training)")
        ax.plot(time_dense_np[test_region], median[test_region], color="red", label="Pred median (Test)")
        
        # Similarly for scatter actual points, separate by truncation
        train_points_mask = (t_norm.cpu().numpy() * MAX_TIME) <= (truncation * MAX_TIME)
        test_points_mask = (t_norm.cpu().numpy() * MAX_TIME) > (truncation * MAX_TIME)
        
        t_scaled = (t_norm.cpu().numpy()) * MAX_TIME
        x_scaled = (x_std.cpu().numpy())
        
        ax.scatter(t_scaled[train_points_mask], x_scaled[train_points_mask], color='blue', label='Training data')
        ax.scatter(t_scaled[test_points_mask], x_scaled[test_points_mask], color='red', label='Test data')
        
        ax.set_title(f"ID {subject_id} | Dose {dose * MAX_DOSE:.1f}")

        ax.set_title(f"ID {subject_id} | Dose {dose * MAX_DOSE:.1f}")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Concentration")
        ax.grid(True)
       # ax.set_xlim(0, MAX_TIME)
      #  ax.set_ylim(0, 20)
        ax.legend()

    plt.tight_layout()
    plt.show()

 
        
def vpc(onlymedian,
        df, dataset, latent_dim,
      dim_parameters, initial_encoder, func, reducer, noise, ODEWrapper,
    t_dense,compartment, num_simulated_total=500,
    add_noise_to_prediction: bool = False  # 🔧 NEW ARGUMENT
):
    
    
    device = next(func.parameters()).device
    t_dense=t_dense.to(device)
    MAX_TIME = estimate_max_time(df)
    MAX_DOSE = estimate_max_dose(df)
    conc_mean, conc_std = dataset.conc_mean, dataset.conc_std
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=collate_fn, num_workers=0)
    dose_times_lists = df['Dose times'].apply(ast.literal_eval).tolist()
    all_times = np.concatenate(dose_times_lists)
    
    unique_times_np = np.unique(all_times)
    dose_times_tensor = torch.from_numpy(unique_times_np).float().to(t_dense.device) / MAX_TIME
    t_dense = torch.unique(torch.cat([t_dense, dose_times_tensor]))

    
    t_dense = torch.unique(torch.cat([t_dense.to(device), dose_times_tensor]))
    
    unique_doses = sorted(set(entry[2].item() for entry in dataset))
    num_doses = len(unique_doses)
    num_simulated_per_dose = max(1, num_simulated_total // num_doses)

    for dose_value in unique_doses:
        dose_filtered_dataset = [entry for entry in dataset if entry[2].item() == dose_value]
        if len(dose_filtered_dataset) == 0:
            print(f"⚠️ No data found for dose {dose_value}. Skipping plot.")
            continue

        indices = np.random.choice(len(dose_filtered_dataset), num_simulated_per_dose, replace=True)
        batch_entries = [dose_filtered_dataset[i] for i in indices]
        t_dense2 = torch.linspace(0, 1, steps=120).to(device)
        

        t_reals, x_trues, doses, dose_times_list, z_samples = [], [], [], [], []
   
        for entry in batch_entries:
            t_real, x_true, dose, dose_times = entry[:4]

            t_reals.append(t_real)
            x_trues.append(x_true)
            doses.append(dose)
            dose_times_list.append(dose_times)
            sample_shape = torch.Size([dim_parameters])
            new_sample = torch.randn(sample_shape)
            z_samples.append(new_sample)
        
            
        dose_times=dose_times.to(device)
        t_dense2 = torch.unique(torch.cat([t_dense2, dose_times]))
        doses_tensor = torch.stack(doses).to(device)
        dose_times_padded = torch.nn.utils.rnn.pad_sequence(dose_times_list, batch_first=True).to(device)
        dose_mask = (dose_times_padded != 0).to(device)
        z_tensor = torch.stack(z_samples).to(device)
        if onlymedian:
            z_tensor=torch.zeros_like(z_tensor)
            
        x0_list = []
        x_trues=x_trues
        for x in x_trues:
            x=x.to(device)
            out = initial_encoder(x[0].unsqueeze(0))  # expect shape [1, 4]
            if out.dim() == 1:
                out = out.unsqueeze(0)  # convert [4] -> [1, 4]
            x0_list.append(out)
        x0_tensor = torch.cat(x0_list, dim=0)  # now shape [6, 4]

                # Expand doses_tensor to match the shape of dose_times_padded
        # dose_times_padded: [B, max_doses], so we want doses_tensor: [B, max_doses]
        if doses_tensor.dim() == 1:
            doses_tensor = doses_tensor.unsqueeze(1)  # [B] -> [B, 1]
        doses_tensor = doses_tensor.expand(-1, dose_times_padded.size(1))  # [B, max_doses]

        print(doses_tensor.size())
        ode_func = ODEWrapper(func, dose_times_padded,doses_tensor, dose_mask)
        x0 = torch.cat([x0_tensor, z_tensor], dim=1)  # shape [batch_size, 6]
        x0=x0.to(device)
        pred = odeint(ode_func, x0, t_dense.to(device), method='dopri5')  # [time, batch, latent_dim+1]
        
        x_pred = destandardize_concentration(reducer(pred[:, :, :latent_dim]), conc_mean,conc_std)  # [time, batch, state_dim]

        # ✅ Optionally add noise
        if add_noise_to_prediction:
            x_pred = noise.sample(x_pred, n_samples=1).squeeze(0)

        # Compute quantiles
        perc10_sim = torch.quantile(x_pred.squeeze(-1), 0.10, dim=1)
        median_sim = torch.quantile(x_pred.squeeze(-1), 0.50, dim=1)
        perc90_sim = torch.quantile(x_pred.squeeze(-1), 0.90, dim=1)

        interp_all = [
            torch_linear_interpolate2(t_real.to(t_dense2.device), x_true.to(t_dense2.device), t_dense2)
            for t_real, x_true in zip(t_reals, x_trues)
        ]

        data_matrix = torch.stack(interp_all)

        perc10_data = torch.quantile(data_matrix, 0.10, dim=0)
        median_data = torch.quantile(data_matrix, 0.50, dim=0)
        perc90_data = torch.quantile(data_matrix, 0.90, dim=0)

        # De-standardize
        perc10_sim_real = perc10_sim
        median_sim_real = median_sim
        perc90_sim_real = perc90_sim

        perc10_data_real = destandardize_concentration(perc10_data, conc_mean, conc_std)
        median_data_real = destandardize_concentration(median_data, conc_mean, conc_std)
        perc90_data_real = destandardize_concentration(perc90_data, conc_mean, conc_std)

        # Plot
        plt.figure(figsize=(12, 6))
        time_hours = t_dense.cpu().numpy() * MAX_TIME

        plt.plot(time_hours, perc10_sim_real.detach().cpu().numpy(), label="Simulated 10th percentile", color="blue", linestyle="--")
        plt.plot(time_hours, median_sim_real.detach().cpu().numpy(), label="Simulated median", color="blue", marker="o")
        plt.plot(time_hours, perc90_sim_real.detach().cpu().numpy(), label="Simulated 90th percentile", color="blue", linestyle="--")
        
        plt.plot(time_hours, perc10_data_real.detach().cpu().numpy(), label="Raw 10th percentile", color="orange", linestyle="--")
        plt.plot(time_hours, median_data_real.detach().cpu().numpy(), label="Raw median", color="orange")
        plt.plot(time_hours, perc90_data_real.detach().cpu().numpy(), label="Raw 90th percentile", color="orange", linestyle="--")

        plt.xlim(0, 24)
        plt.ylim(0, 120)

        plt.title(f"Dose {dose_value * MAX_DOSE:.0f} (Simulated vs Raw)")
        plt.xlabel("Time (hours)")
        plt.ylabel(f"Concentration ({compartment})")
        plt.grid(True)
        
        plt.legend()
        plt.tight_layout()
        plt.show()

def vpc_decoder(ae,nf,
        df, dataset, latent_dim,
      dim_parameters, initial_encoder,decoder1,decoder2, func, reducer, noise, ODEWrapper,
    t_dense,compartment, num_simulated_total=500,
    add_noise_to_prediction: bool = False  # 🔧 NEW ARGUMENT
):
    device = next(func.parameters()).device
    
    MAX_TIME = estimate_max_time(df)
    MAX_DOSE = estimate_max_dose(df)
   # print(MAX_DOSE)
    
    conc_mean, conc_std = dataset.conc_mean, dataset.conc_std
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=collate_fn, num_workers=0)
    dose_times_lists = df['Dose times'].apply(ast.literal_eval).tolist()
    all_times = np.concatenate(dose_times_lists)
    
    unique_times_np = np.unique(all_times)
    dose_times_tensor = torch.from_numpy(unique_times_np).float().to(t_dense.device) / MAX_TIME
    t_dense = torch.unique(torch.cat([t_dense, dose_times_tensor]))
    
   
    unique_doses = sorted(set(entry[2].item() for entry in dataset))
    num_doses = len(unique_doses)
    num_simulated_per_dose = max(1, num_simulated_total // num_doses)

    for dose_value in unique_doses:
        dose_filtered_dataset = [entry for entry in dataset if entry[2].item() == dose_value]
        if len(dose_filtered_dataset) == 0:
            print(f"⚠️ No data found for dose {dose_value}. Skipping plot.")
            continue

        indices = np.random.choice(len(dose_filtered_dataset), num_simulated_per_dose, replace=True)
        batch_entries = [dose_filtered_dataset[i] for i in indices]
        t_dense2 = torch.linspace(0, 1, steps=120).to(device)

        t_reals, x_trues, doses, dose_times_list = [], [], [], []
      #  print(dose_value)
        for entry in batch_entries:
            t_real, x_true, dose, dose_times = entry[:4]
            t_reals.append(t_real)
            x_trues.append(x_true)
            doses.append(dose)
            dose_times_list.append(dose_times)
            
        dose_times=dose_times.to(device)
        t_dense2 = torch.unique(torch.cat([t_dense2, dose_times]))    
      
        doses_tensor = torch.stack(doses).to(device)
        dose_times_padded = torch.nn.utils.rnn.pad_sequence(dose_times_list, batch_first=True).to(device)
        dose_mask = (dose_times_padded != 0).to(device)

        t_real_padded = torch.nn.utils.rnn.pad_sequence(t_reals, batch_first=True).to(device)
        x_true_padded = torch.nn.utils.rnn.pad_sequence(x_trues, batch_first=True).to(device)

        # 🧠 Use decoder to get posterior latent sample
        # 🧠 Choose decoder based on dose
        dose_value_float = float(dose_value)
        
        if nf:
            if dose_value_float == 0.25:
               z_tensor, mu_q, logvar_q,_ = decoder1(t_real_padded, x_true_padded)
            elif dose_value_float == 1.0:
               z_tensor, mu_q, logvar_q,_ = decoder2(t_real_padded, x_true_padded)
           
                
        else:
            
            if dose_value_float == 0.25:
               mu_q, logvar_q= decoder1(t_real_padded, x_true_padded)
               std_q = torch.exp(0.5 * logvar_q)
               eps = torch.randn_like(std_q)
               z_tensor = mu_q + eps * std_q
            elif dose_value_float == 1.0:
               mu_q, logvar_q = decoder2(t_real_padded, x_true_padded)
               std_q = torch.exp(0.5 * logvar_q)
               eps = torch.randn_like(std_q)
               z_tensor = mu_q + eps * std_q
           
           
            
            
          #  raise ValueError(f"Unsupported dose value: {dose_value_float}. Expected 0.5 or 1.0.")
        
        
       # print("mu mean:", mu_q, "logvar mean:", logvar_q)
       # std_q = torch.exp(0.5 * logvar_q)
       # eps = torch.randn_like(std_q)
       # z_tensor = mu_q + eps * std_q

        x0_list = []
        for x in x_trues:
            x=x.to(device)
            out = initial_encoder(x[0].unsqueeze(0))  # expect shape [1, 4]
            if out.dim() == 1:
                out = out.unsqueeze(0)  # convert [4] -> [1, 4]
            x0_list.append(out)
        x0_tensor = torch.cat(x0_list, dim=0)  # now shape [6, 4]

        if doses_tensor.dim() == 1:
             doses_tensor = doses_tensor.unsqueeze(1)  # [B] -> [B, 1]
             
        doses_tensor = doses_tensor.expand(-1, dose_times_padded.size(1))  # [B, max_doses]
        if ae:
            z_tensor=mu_q
    
       # print(dose_times_padded)
       # print(doses_tensor)
        ode_func = ODEWrapper(func, dose_times_padded,doses_tensor, dose_mask)
        x0 = torch.cat([x0_tensor, z_tensor], dim=1)  # shape [batch_size, 6]
        x0=x0.to(device)
        pred = odeint(ode_func, x0, t_dense.to(device), method='dopri5')  # [time, batch, latent_dim+1]
        

        x_pred = destandardize_concentration(reducer(pred[:, :, :latent_dim]), conc_mean, conc_std)  # Only reduce latent part

        # ✅ Optionally add noise
        if add_noise_to_prediction:
            x_pred = noise.sample(x_pred, n_samples=1).squeeze(0)

        # Compute quantiles
        perc10_sim = torch.quantile(x_pred.squeeze(-1), 0.10, dim=1)
        median_sim = torch.quantile(x_pred.squeeze(-1), 0.50, dim=1)
        perc90_sim = torch.quantile(x_pred.squeeze(-1), 0.90, dim=1)

        interp_all = [
              torch_linear_interpolate2(t_real.to(t_dense2.device), x_true.to(t_dense2.device), t_dense2)
              for t_real, x_true in zip(t_reals, x_trues)
          ]

        data_matrix = torch.stack(interp_all)

        perc10_data = torch.quantile(data_matrix, 0.10, dim=0)
        median_data = torch.quantile(data_matrix, 0.50, dim=0)
        perc90_data = torch.quantile(data_matrix, 0.90, dim=0)

        # De-standardize
        perc10_sim_real = perc10_sim
        median_sim_real = median_sim
        perc90_sim_real = perc90_sim

        perc10_data_real = destandardize_concentration(perc10_data, conc_mean, conc_std)
        median_data_real = destandardize_concentration(median_data, conc_mean, conc_std)
        perc90_data_real = destandardize_concentration(perc90_data, conc_mean, conc_std)

        # Plot
        plt.figure(figsize=(12, 6))
        time_hours = t_dense.cpu().numpy() * MAX_TIME

        plt.plot(time_hours, perc10_sim_real.detach().cpu().numpy(), label="Simulated 10th percentile", color="blue", linestyle="--")
        plt.plot(time_hours, median_sim_real.detach().cpu().numpy(), label="Simulated median", color="blue", marker="o")
        plt.plot(time_hours, perc90_sim_real.detach().cpu().numpy(), label="Simulated 90th percentile", color="blue", linestyle="--")

        plt.plot(time_hours, perc10_data_real.detach().cpu().numpy(), label="Raw 10th percentile", color="orange", linestyle="--")
        plt.plot(time_hours, median_data_real.detach().cpu().numpy(), label="Raw median", color="orange")
        plt.plot(time_hours, perc90_data_real.detach().cpu().numpy(), label="Raw 90th percentile", color="orange", linestyle="--")
        #plt.yscale('log')

       # plt.xlim(0, 24)
      #  plt.ylim(1, 120)

        plt.title(f"Dose {dose_value * MAX_DOSE:.0f} (Simulated vs Raw)")
        plt.xlabel("Time (hours)")
        plt.ylabel(f"Concentration ({compartment})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()






