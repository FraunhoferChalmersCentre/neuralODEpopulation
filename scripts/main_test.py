# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 11:46:04 2025

@author: Baaz
"""
from lib.utils.my_utils_parallel import *
from lib.models.NNmodels_parallel import *
from lib.utils.model_validation_parallel import *
  
def main():
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

    # Simulate arguments (safe in Spyder)
    sys.argv = ['script_name',
                '--data_test_path', 'lib/data/simulated_test_data1.csv', 
                '--data_path', 'lib/data/simulated_data1.csv',
                '--save_dir', 'models',
                '--load_dir', 'models/old']

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_test_path", type=str, required=True)
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

   # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    device = torch.device("cpu")
    print("Using device:", device)
  

    df = pd.read_csv(args.data_path)
    dataset = TrajectoryDataset(args.data_path)
    dataset_test = TrajectoryDataset(args.data_test_path)
    df_test = pd.read_csv(args.data_test_path)
    load_dir = args.load_dir
    save_dir = args.save_dir
    modelname = "MultipleDoseAddError_AE_NF2"

    latent_dim = 4
    dim_parameter_encoder = 3
    hid_dim = 128

    encoder1 = Encoder_Transformer_NF(dim_parameter_encoder, input_dim=2, model_dim=64,
                                      hidden_dim=64, hidden_flow_dim=32, num_heads=4,
                                      num_layers=2, num_flow_layers=2, dropout=0.1).to(device)

    func = ODEFunc(latent_dim, dim_parameter_encoder, hid_dim).to(device)
    reducer = SimpleDecoder(latent_dim, hidden_dim=128).to(device)
    initial_encoder = InitialConditionEncoder(latent_dim, hidden_dim=32).to(device)
    noise = TrainableNoise(dataset, size=1, init_add_std=0.3, init_prop_std=0.01).to(device)

    models = {
        "func": func,
        "encoder1": encoder1,
        "reducer": reducer,
        "initial_encoder": initial_encoder,
        "noise": noise,
    }

    lr = 0.001
    main_params = [
        {"params": list(func.parameters()) +
                    list(reducer.parameters()) +
                    list(initial_encoder.parameters()) +
                    list(encoder1.parameters()), "lr": lr},
        {"params": list(noise.parameters()), "lr": 1000 * lr},
    ]
    optimizer = torch.optim.Adam(main_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )

    for name, model in models.items():
        model.to(device)
        print(f"{name} is on {next(model.parameters()).device}")

    dataloader = DataLoader(dataset, batch_size=5, shuffle=True,
                            collate_fn=collate_fn, num_workers=0)

    train_model(
        dataloader, 0.9, models, optimizer, scheduler,
        dim_parameter_encoder, latent_dim, func, reducer, initial_encoder, encoder1, noise, device,
        t_dense=torch.linspace(0, 1, steps=100),
        n_epochs=1000,
        warmup_epochs_noise=44400,
        warmup_epochs_iiv=100,
        smoothing_start_epoch=4000,
        remove_encoder=False,
        ae=False,
        nf=True,
        onlymedian=False,
        plot_from_training_records_enable=True,
        free_bits=0.1,
        batch_size=5,
        df=df,
        dataset=dataset,
        max_points_visible=0,
        lr=lr,
        print_epoch=1,
        plot_epoch=10,
        max_plots=9,
        nr_col=3,
        nr_row=3
    )

if __name__ == "__main__":
    main()
