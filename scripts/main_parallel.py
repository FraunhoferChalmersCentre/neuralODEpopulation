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
from torchdiffeq import odeint as odeint
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





# ---- Training ---- 
if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_csv(args.data_path)
    dataset = TrajectoryDataset(args.data_path)
    
    dataset_test=TrajectoryDataset(args.data_test_path)
    df_test=pd.read_csv(args.data_test_path)
    load_dir = args.load_dir
    save_dir = args.save_dir
    modelname = "MultipleDoseAddError_vae"

    

    latent_dim=4
    dim_parameter_encoder=2
    hid_dim=128
 
 

   ###### Normalizing Flow
   # refiner1 = Encoder_Transformer_NF(dim_parameter_encoder, input_dim=2, model_dim=64,hidden_dim=64,hidden_flow_dim=32, num_heads=4, num_layers=2,num_flow_layers=2, dropout=0.1).to(device)
   # refiner2 = Encoder_Transformer_NF(dim_parameter_encoder, input_dim=2, model_dim=64,hidden_dim=64, hidden_flow_dim=32, num_heads=4, num_layers=2,num_flow_layers=2, dropout=0.1).to(device)
   
   ###### VAE / AE
    refiner1 = Encoder_Transformer_VAE(dim_parameter_encoder, input_dim=2, model_dim=64,hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1)
    refiner2 = Encoder_Transformer_VAE(dim_parameter_encoder, input_dim=2, model_dim=64,hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1)
   

    func = ODEFunc(latent_dim,dim_parameter_encoder,hid_dim ).to(device)
    
    reducer = SimpleDecoder(latent_dim, hidden_dim=128).to(device)
    initial_encoder = InitialConditionEncoder(latent_dim, hidden_dim=32).to(device)
    noise = TrainableNoise(dataset).to(device)


   # load_models(models, save_dir, modelname,device)
    
    train_model(dim_parameter_encoder, latent_dim, func,
    reducer,
    initial_encoder,
    refiner1,
    refiner1,
    noise,
    device,
    t_dense=torch.linspace(0, 1, steps=480),
    n_epochs=400,
    warmup_epochs_noise=0,
    warmup_epochs_iiv=50,
    smoothing_start_epoch=300,
    ae=True,
    nf=False,                  
    free_bits=1,                       
    batch_size=10,                        
    df=df,
    dataset=dataset,
    max_points_visible=0,  
    lr=0.001,
    print_epoch=1,
    plot_epoch=10,
    max_plots=25,
    nr_col=5,
    nr_row=5)  


    save_models(models, save_dir, modelname)


    
            




# -----------------------
# Post-training:
# -----------------------


#
# Population prediction vs data
#

### Predictions without encoder
vpc(df, dataset, latent_dim,  dim_parameter_encoder, initial_encoder, func, reducer, noise, ODEWrapper, torch.linspace(0, 1, steps=120), compartment="C2",num_simulated_total=1000,add_noise_to_prediction=False)
vpc_refiner(True,df, dataset, latent_dim, dim_parameter_encoder, initial_encoder, refiner1, refiner2, func, reducer, noise, ODEWrapper, torch.linspace(0, 1, steps=120), compartment="C2",num_simulated_total=1000,add_noise_to_prediction=False)

plotIndividualFits_test(test_dataset=dataset, df=df, latent_dim=latent_dim, refiner1=refiner1, refiner2=refiner2, func=func, reducer=reducer, initial_encoder=initial_encoder, ODEWrapper=ODEWrapper, t_dense=torch.linspace(0, 1, steps=100), max_individuals=6, n_samples=1000, device=device, truncation=1)









#plot_individual_samples_vs_data_vectorized_vae(df, dataset,latent_dim=latent_dim,individual_data=dataset[34], refiner1=refiner1, refiner2=refiner2, func=func, reducer=reducer, initial_encoder=initial_encoder, ODEWrapper=ODEWrapper, t_dense=t_dense,  n_samples=100,  n_mcmc_samples=1000, burn_in=50,device=device)

