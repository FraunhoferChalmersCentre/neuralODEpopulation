# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 16:04:21 2025

@author: Baaz
"""

# SAFE SETTINGS TO AVOID OpenMP CRASHES


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint_adjoint as odeint
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import argparse
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Simulate command line arguments in Spyder
sys.argv = ['script_name',
            '--data_path', 'lib/data/simulated_data1.csv',
            '--save_dir', 'models',
            '--load_dir', 'models/old']


# Define your parser
parser = argparse.ArgumentParser(description="Train Neural-ODE model on dataset.")
parser.add_argument("--data_path", type=str, required=True, help="Path to training CSV file")
parser.add_argument("--load_dir", type=str, default=None, help="Path to folder with saved models (optional)")
parser.add_argument("--save_dir", type=str, required=True, help="Directory where trained models will be saved")

# Parse arguments
args = parser.parse_args()


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

print("Using device:", device)

from lib.utils.my_utils_parallel import *
from lib.models.NNmodels_parallel import *
from lib.utils.model_validation_parallel import *





# ---- Training ---- 
if __name__ == "__main__":
    args = parser.parse_args()

    load_dir = args.load_dir
    save_dir = args.save_dir
    modelname = "MultipleDoseAddError"
    df = pd.read_csv(args.data_path)
    dataset = TrajectoryDataset(args.data_path)
    conc_mean, conc_std = dataset.conc_mean, dataset.conc_std
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
   
   
    latent_dim=4
    dim_parameter_encoder = 2
    
    func = ODEFunc(latent_dim,dim_parameter_encoder, hidden_dim = 128).to(device)
    refiner1 = IndividualRefiner_VAE(input_dim=dim_parameter_encoder , hidden_dim=64, num_layers=1).to(device)
    refiner2 = IndividualRefiner_VAE(input_dim=dim_parameter_encoder , hidden_dim=64, num_layers=1).to(device)
    reducer = SimpleDecoder(latent_dim, hidden_dim=128).to(device)
    initial_encoder = InitialConditionEncoder(latent_dim=latent_dim, hidden_dim=32).to(device)

    noise = TrainableNoise(conc_std).to(device)

    
    main_params = [
    {"params": list(func.parameters()) + list(reducer.parameters()) + list(initial_encoder.parameters()) + list(refiner1.parameters()) + list(refiner2.parameters())},
    {"params": list(noise.parameters()), "lr": 1e-2},]


    optimizer = torch.optim.Adam(main_params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.85)
    
    t_dense = torch.linspace(0, 1, steps=120)
    t_dense = t_dense.to(device)
    dose_times_lists = df['Dose times'].apply(ast.literal_eval).tolist()
    all_times = np.concatenate(dose_times_lists)
    unique_times = np.unique(all_times)

    dose_times_tensor = torch.tensor(unique_times, dtype=torch.float32)/24


    t_dense = torch.unique(torch.cat([t_dense, dose_times_tensor]))



    smoothing_start_epoch=0
    n_epochs = 1000
    warmup_epochs=1000
    
    print_epoch=1
    plot_epoch=1
    max_plots=4
    nr_col=1
    nr_row=5    
    
    loss_fn = nn.MSELoss()

    kl_weight=0.1 # You can anneal or tune this
    alpha=1

   # load_model(func, "func")
   # load_model(global_latent, "global_latent")
   # load_model(refiner, "refiner")
   # load_model(reducer, "reducer")
   # load_model(initial_encoder, "initial_encoder")
   # load_model(noise, "noise")
    #load_model(dose_classifier, "dose_classifier")  # Uncomment if used

    #train_model(dataloader, func, reducer, initial_encoder, refiner1, refiner2, global_latent, noise, optimizer, scheduler, device, t_dense, kl_weight, alpha, n_epochs, warmup_epochs, smoothing_start_epoch, print_epoch, plot_epoch, ODEWrapper, conc_std, conc_mean, estimate_max_time(df),estimate_max_dose(df))

    train_model_vae(dim_parameter_encoder,latent_dim, dataloader, func, reducer, initial_encoder, refiner1, refiner2, noise, optimizer, scheduler, device, t_dense, kl_weight, alpha, n_epochs, warmup_epochs, smoothing_start_epoch, print_epoch, plot_epoch, ODEWrapper, conc_std, conc_mean, estimate_max_time(df),estimate_max_dose(df))

            



    
  #  save_model(noise, "noise")
  #  save_model(func, "func")
  #  save_model(global_latent, "global_latent")
  ##  save_model(refiner, "refiner")
  #  save_model(reducer, "reducer")
    #save_model(initial_encoder, "initial_encoder")


# -----------------------
# Post-training:
# -----------------------


#
# Population prediction vs data
#

### Predictions without encoder
#simulate_and_plot_by_dose(dataset, global_latent, initial_encoder, func, reducer, noise, ODEWrapper, t_dense, conc_mean, conc_std, estimate_max_time(df), estimate_max_dose(df), compartment="C2",num_simulated_total=1000,add_noise_to_prediction=False)
simulate_and_plot_by_dose_vae(dataset, dim_parameter_encoder, initial_encoder, func, reducer, noise, ODEWrapper, t_dense, conc_mean, conc_std, estimate_max_time(df), estimate_max_dose(df), compartment="C2",num_simulated_total=1000,add_noise_to_prediction=True)




#simulate_and_plot_by_dose_vae_refiner(dataset,global_latent, initial_encoder,refiner1,refiner2, func, reducer, noise, ODEWrapper, t_dense, conc_mean, conc_std, estimate_max_time(df), estimate_max_dose(df), compartment="C2",num_simulated_total=1000,add_noise_to_prediction=False)



individual_data = next(entry for entry in dataset if entry[2].item() == 0.5)  # or 1.0


from lib.utils.my_utils_parallel import *
from lib.models.NNmodels_parallel import *
from lib.utils.model_validation_parallel import *
plot_individual_samples_vs_data_vectorized_vae(dose_times_tensor,individual_data=dataset[99], refiner1=refiner1, refiner2=refiner2, func=func, reducer=reducer, initial_encoder=initial_encoder, ODEWrapper=ODEWrapper, t_dense=t_dense, conc_mean=conc_mean, conc_std=conc_std, MAX_TIME=estimate_max_time(df), MAX_DOSE=estimate_max_dose(df), n_samples=100,  n_mcmc_samples=1000, burn_in=50,device=device)


### Predictions with encoder
#simulate_and_plot_by_dose_refined(dataset, global_latent, refiner, initial_encoder, func, reducer, noise, ODEWrapper, t_dense, conc_mean, conc_std, MAX_TIME, compartment="C2",num_simulated_total=100)


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
    destandardize_concentration=destandardize_concentration,
    conc_mean=conc_mean,
    conc_std=conc_std,
    t_dense=t_dense,
    cols=5,
    rows=None
)


#
# Correlation plot
#
plot_ebes_vs_true_params(ebes, data_path=DATA_PATH)




# ---------------------------------------
# Random forest to predict true params from EBEs
# ---------------------------------------

rf_model, cv_scores, y_test, y_pred = rf_regression_and_plot(ebes, data_path=DATA_PATH)


