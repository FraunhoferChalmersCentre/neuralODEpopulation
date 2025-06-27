"""
Created on Fri Jun 27 09:47:33 2025

@author: Baaz
"""

#import os

#os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"

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

import argparse
from lib.models.NNmodels import *
from lib.utils.model_validation import *

parser = argparse.ArgumentParser(description="Train Neural-ODE model on dataset.")
parser.add_argument("--data_path", type=str, required=True, help="Path to training CSV file")
parser.add_argument("--load_dir", type=str, default=None, help="Path to folder with saved models (optional)")
parser.add_argument("--save_dir", type=str, required=True, help="Directory where trained models will be saved")

if __name__ == "__main__":
    args = parser.parse_args()

    load_dir = args.load_dir
    save_dir = args.save_dir
    modelname = "MultipleDoseAddError"

    dataset = TrajectoryDataset(args.data_path)
    conc_mean, conc_std = dataset.conc_mean, dataset.conc_std
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, collate_fn=collate_fn, num_workers=0)

    func = ODEFunc()
    global_latent = GlobalLatent()
    refiner = IndividualRefiner()
    reducer = SimpleDecoder()
    initial_encoder = InitialConditionEncoder()
    noise = TrainableNoise()

    optimizer = torch.optim.Adam(
           list(func.parameters())
         + list(reducer.parameters()) 
         + list(initial_encoder.parameters())
         + list(refiner.parameters())
         + list(noise.parameters())
         ,
         lr=1e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.85)
    loss_fn = nn.MSELoss()
    ode_kwargs = {'method': 'rk4'}
    t_dense = torch.linspace(0, 1, steps=120)

    smoothing_start_epoch = 100
    n_epochs = 1
    warmup_epochs=100
    
    print_epoch=1
    plot_epoch=1
    max_plots=5
    nr_col=1
    nr_row=5    

    kl_weight = 0.8e-2 # You can anneal or tune this

    load_model(func, "func", load_dir, modelname)
    load_model(global_latent, "global_latent", load_dir, modelname)
    load_model(refiner, "refiner", load_dir, modelname)
    load_model(reducer, "reducer", load_dir, modelname)
    load_model(initial_encoder, "initial_encoder", load_dir, modelname)
    load_model(noise, "noise", load_dir, modelname)
    # load_model(dose_classifier, "dose_classifier", load_dir, modelname)  # Uncomment if used

    for epoch in range(n_epochs):
        total_loss = 0.0
        total_recon_loss_noise = 0
        total_recon_loss=0
        
        total_KL_loss = 0
        total_refiner_loss = 0.0
        z_individual_list = []
        trajectory_records = []
    
        for id_list, t_padded, x_padded, mask, dose_tensor, dose_times_list in dataloader:
            batch_size = t_padded.size(0)
    
            for i in range(batch_size):
                valid_len = mask[i].sum().int()
                t = t_padded[i, :valid_len]
                x_true = x_padded[i, :valid_len]
                dose = dose_tensor[i]
                dose_times = dose_times_list[i]
                subject_id = id_list[i]  # Correctly get subject ID

                # ==== 1. Global z (median modeling, full train) ====
                z_sampled =  global_latent.sample()  # global_latent.mu.detach() #
                mu_q, logvar_q = refiner(t, x_true, z_sampled)
                std_q = torch.exp(0.5 * logvar_q)
                eps = torch.randn_like(std_q)
                z_refined = mu_q + eps * std_q  # shape: (latent_dim,)
                
                x0 = initial_encoder(torch.cat([x_true[0].unsqueeze(0)] * 2, dim=0))  # size (2, latent_dim)
    
                pred = odeint(lambda t, x: func(t, x, dose, dose_times, z_refined), x0, t_dense, method='rk4')
                x_pred = reducer(pred)

                x_pred_at_t = torch_linear_interpolate(t_dense, x_pred, t)
                x_pred_at_t_orig = x_pred_at_t * conc_std + conc_mean
                
                kl_loss = kl_divergence_gaussians(mu_q, logvar_q, global_latent.mu, global_latent.logvar)
                recon_loss_noise = noise.nll(x_true, x_pred_at_t)
                mse_loss=loss_fn(x_true,x_pred_at_t)

                KL_loss=kl_weight * kl_loss
                
                loss = recon_loss_noise  + KL_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_recon_loss_noise += recon_loss_noise.item()
                total_recon_loss += mse_loss.item()
                total_KL_loss += KL_loss.item()
                total_loss += loss.item()
              
                z_individual_list.append(z_refined.detach())
                trajectory_records.append((subject_id, t, x_true, dose, dose_times, z_refined.detach(), z_sampled.detach()))
    
        # ==== SAEM Update ====
        z_all = torch.stack(z_individual_list, dim=0)
        
        if epoch < warmup_epochs:
            for param in noise.parameters():
                param.requires_grad = False
        else:
            for param in noise.parameters():
                param.requires_grad = True
        
        with torch.no_grad():
            if epoch < smoothing_start_epoch:
                global_latent.update(z_all, gamma=1.0)  # full update early
            else:
                global_latent.update(z_all)            # smoothed update later
    
        scheduler.step()

        if epoch % print_epoch == 0:
            with torch.no_grad():
                std = torch.exp(0.5 * global_latent.logvar)
                print(
                    f"Epoch {epoch}, "
                    f"MSE {total_recon_loss:.4f} "
                    f"-LL: {total_recon_loss_noise:.4f}, "
                    f"KL loss: {total_KL_loss:.4f}, "
                    f"Add. error: {conc_std*torch.exp(noise.log_sigma_add).item():.4f}, "
                    f"Prop. error: {torch.exp(noise.log_sigma_prop).item():.4f}, "
                    f"Global Latent Std: {std.cpu().numpy()}"
                )

        if epoch % plot_epoch == 0:
            plot_from_training_records(
                records=trajectory_records,
                func=func,
                t_dense=t_dense,
                conc_mean=conc_mean,
                conc_std=conc_std,
                max_plots=max_plots,
                max_time=dataset.max_time,
                max_dose=dataset.max_dose,
                compartment=dataset.compartment,
                initial_encoder=initial_encoder,
                reducer=reducer,
                nr_row=nr_row,
                nr_col=nr_col
            )

    save_model(noise, "noise", save_dir, modelname)
    save_model(func, "func", save_dir, modelname)
    save_model(global_latent, "global_latent", save_dir, modelname)
    save_model(refiner, "refiner", save_dir, modelname)
    save_model(reducer, "reducer", save_dir, modelname)
    # save_model(dose_classifier, "dose_classifier", save_dir, modelname)  # Uncomment if used
    save_model(initial_encoder, "initial_encoder", save_dir, modelname)


    # -----------------------
    # Post-training:
    # -----------------------

    #
    # Population prediction vs data
    #

    simulate_and_plot(
        dataset=dataset,
        global_latent=global_latent,
        refiner=refiner,
        initial_encoder=initial_encoder,
        func=func,
        reducer=reducer,
        noise=noise,
        t_dense=t_dense,
        conc_mean=dataset.conc_mean,
        conc_std=dataset.conc_std,
        max_time=dataset.max_time,
        compartment=dataset.compartment,
        num_simulated_total=1
    )

    #
    # EBE estimation (mode of individual posterior)
    #

    ebes = estimate_ebes_for_all(
        dataset=dataset,
        trajectory_records=trajectory_records,
        func=func,
        global_latent=global_latent,
        refiner=refiner,
        initial_encoder=initial_encoder,
        reducer=reducer,
        torch_linear_interpolate=torch_linear_interpolate,
        t_dense=t_dense,
        n_steps=1,
        lr=1e-2,
        device='cpu'
    )

    #
    # Individual predictions vs data
    #

    plot_individuals(
        ebes=ebes,
        num_plots=10,
        func=func,
        initial_encoder=initial_encoder,
        reducer=reducer,
        torch_linear_interpolate=torch_linear_interpolate,
        conc_mean=conc_mean,
        conc_std=conc_std,
        t_dense=t_dense,
        cols=5,
        rows=None
    )

    #
    # Correlation plot
    #
    plot_ebes_vs_true_params(ebes, data_path=args.data_path)

    # ---------------------------------------
    # Random forest to predict true params from EBEs
    # ---------------------------------------

    # rf_model, cv_scores, y_test, y_pred = rf_regression_and_plot(ebes, data_path=args.data_path)
