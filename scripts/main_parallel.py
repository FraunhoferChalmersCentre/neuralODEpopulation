# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 16:04:21 2025

@author: Baaz
"""

# SAFE SETTINGS TO AVOID OpenMP CRASHES



import os


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
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import argparse
import sys
import torch.optim as optim

# = optim.SGD(model.parameters(), lr=0.001)  # simple SGD, no momentum, no weight decay



# ---- Training ---- 
if __name__ == "__main__":
        
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)
    
    # Simulate command line arguments in Spyder
    sys.argv = ['script_name',
                '--data_test_path', 'lib/data/simulated_test_data1.csv', 
                '--data_path', 'lib/data/simulated_data1.csv',
                '--save_dir', 'models',
                '--load_dir', 'models/old']
    
    
    # Define your parser
    parser = argparse.ArgumentParser(description="Train Neural-ODE model on dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--data_test_path", type=str, required=True, help="Path to test CSV file")
    
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
    
    
    
    
    args = parser.parse_args()
    df = pd.read_csv(args.data_path)
    dataset = TrajectoryDataset(args.data_path)
    
    dataset_test=TrajectoryDataset(args.data_test_path)
    df_test=pd.read_csv(args.data_test_path)
    load_dir = args.load_dir
    save_dir = args.save_dir
    modelname = "MultipleDoseAddError_AE_NF2"

    

    latent_dim=2
    dim_parameter_encoder=3
    hid_dim=128
 
 

   ###### Normalizing Flow
    encoder1 = Encoder_Transformer_NF(dim_parameter_encoder, input_dim=2, model_dim=64,hidden_dim=64,hidden_flow_dim=32, num_heads=4, num_layers=2,num_flow_layers=2, dropout=0.1).to(device)
   
   ###### VAE / AE
  # encoder1 = Encoder_Transformer_VAE(dim_parameter_encoder, input_dim=2, model_dim=64,hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1)
 

    func = ODEFunc(latent_dim,dim_parameter_encoder,hid_dim ).to(device)
    reducer = SimpleDecoder(latent_dim, hidden_dim=8).to(device)
    initial_encoder = InitialConditionEncoder(latent_dim, hidden_dim=8).to(device)
    noise = TrainableNoise(dataset, size=1, init_add_std=1, init_prop_std=0).to(device)
    
    models = {
    "func": func,
    "encoder1": encoder1,
    "reducer": reducer,
    "initial_encoder": initial_encoder,
    "noise": noise,
    # add any other models...
    } 
   
    lr=0.001
   # load_models(models, save_dir, modelname,device)
    main_params = [
        {"params": list(func.parameters()) + list(reducer.parameters()) + list(initial_encoder.parameters()) +list(encoder1.parameters()) , "lr": lr},
        {"params": list(noise.parameters()), "lr": 1000*lr},
    ]
    
    optimizer = torch.optim.Adam(main_params, lr=lr)
   # optimizer = optim.SGD(main_params, lr=0.001)  # simple SGD, no momentum, no weight decay

   # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=130, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    for name, model in models.items():
          model.to(device)
          print(f"{name} is on {next(model.parameters()).device}")

    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=collate_fn, num_workers=0)

    train_model(dataloader,0.9,models, optimizer,scheduler, dim_parameter_encoder, latent_dim, func,
    reducer,
    initial_encoder,
    encoder1,
    noise,
    device,
    t_dense=torch.linspace(0, 1, steps=100),
    n_epochs=1000,
    warmup_epochs_noise=44400,
    warmup_epochs_iiv=100,
    smoothing_start_epoch=4000,
    remove_encoder=False,
    ae=True,
    nf=True,
    onlymedian=False, 
    plot_from_training_records_enable=True,              
    free_bits=0.1,                       
    batch_size=20,                        
    df=df,
    dataset=dataset,
    max_points_visible=0,  
    lr=lr,
    print_epoch=1,
    plot_epoch=1,
    max_plots=9,
    nr_col=3,
    nr_row=3)  


    #save_models(models, save_dir, "MultipleDoseAddError_NF_AE")


    #load_models(models, save_dir, "MultipleDoseAddError_NF_AE")
            
    #decoder = Encoder_Transformer_NF(...)  # same args as decoder1
 




# -----------------------
# Post-training:
# -----------------------


#
# Population prediction vs data
#

### Predictions without encoder
vpc(
    onlymedian=True,
    df=df,
    dataset=dataset,
    latent_dim=latent_dim,
    dim_parameter_encoder=dim_parameter_encoder,
    initial_encoder=initial_encoder,
    func=func,
    reducer=reducer,
    noise=noise,
    ODEWrapper=ODEWrapper,
    time_points=torch.linspace(0, 1, steps=120),
    compartment="C2",
    num_simulated_total=1000,
    add_noise_to_prediction=False
)


vpc_decoder(
    ae=True,
    nf=True,
    df=df,
    dataset=dataset,
    latent_dim=latent_dim,
    dim_parameters=dim_parameter_encoder,
    initial_encoder=initial_encoder,
    decoder1=encoder1,
    decoder2=encoder1,
    func=func,
    reducer=reducer,
    noise=noise,
    ODEWrapper=ODEWrapper,
    t_dense=torch.linspace(0, 1, steps=120),
    compartment="C2",
    num_simulated_total=1000,
    add_noise_to_prediction=False
)

plotIndividualFits_test(
    ae=True,
    nf=True,
    test_dataset=dataset_test,
    df=df_test,
    latent_dim=latent_dim,
    noise=noise,
    decoder1=decoder1,
    decoder2=decoder2,
    func=func,
    reducer=reducer,
    initial_encoder=initial_encoder,
    ODEWrapper=ODEWrapper,
    t_dense=torch.linspace(0, 1, steps=120),
    max_individuals=10,
    n_samples=2,
    device=device,
    truncation=0,
    add_noise=False
)







#plot_individual_samples_vs_data_vectorized_vae(df, dataset,latent_dim=latent_dim,individual_data=dataset[34], refiner1=refiner1, refiner2=refiner2, func=func, reducer=reducer, initial_encoder=initial_encoder, ODEWrapper=ODEWrapper, t_dense=t_dense,  n_samples=100,  n_mcmc_samples=1000, burn_in=50,device=device)

