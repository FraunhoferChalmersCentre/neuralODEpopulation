# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 22:00:24 2025

@author: Baaz
"""

#%%
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 22:03:26 2025

@author: Baaz
"""


import os
import sys
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import argparse


# ---- Training ---- 
if __name__ == "__main__":
        
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)
    
 
    sys.argv = ['script_name',
                '--data_validation_path', 'lib/data/simulated_validation_data6.csv', 
                '--data_test_path', 'lib/data/simulated_test_data6.csv', 
                '--data_path', 'lib/data/theophylline_data.csv',
                '--save_dir', 'models',
                '--load_dir', 'models/old',
                '--base_dir','results/theo/run5']


    from lib.utils.Theophylline.utils_preprocess_theo import save_models, load_models, prepare_datasets_and_loaders_train, prepare_optimizer, prepare_datasets_and_loaders, compute_global_stats, export_all_metrics_and_residuals, append_metrics, load_all_metrics_and_residuals_as_lists, TrajectoryDataset, collate_fn
    from lib.utils.Theophylline.utils_training_theo import run_model_variant, plot_individual_fits, vpc, compute_residuals, vpc_true
    from lib.models.NNmodels import Encoder_Transformer_NF, ODEFunc, SimpleDecoder, InitialConditionVAEEncoder, TrainableNoise
    from lib.utils.Theophylline.utils_shared_theo import ODEWrapper
    from lib.utils.Theophylline.utils_post_processing_theo import vpc_true, vpc, plot_individual_fits, compute_residuals, analyze_model_with_vpc

  
 
    # Define your parser
    parser = argparse.ArgumentParser(description="Train Neural-ODE model on dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--data_test_path", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--data_validation_path", type=str, required=True, help="Path to test CSV file")

    parser.add_argument("--load_dir", type=str, default=None, help="Path to folder with saved models (optional)")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory where trained models will be saved")
    parser.add_argument("--base_dir", type=str, required=True, help="Directory where metrics/residuals CSVs are stored")

    # Parse arguments
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    load_dir = args.load_dir
    save_dir = args.save_dir
   
    
    df = pd.read_csv(args.data_path, sep=";")
    all_ids = df['ID'].unique().tolist()

   

    n_epochs=2000
    lr=0.001

    
    
    latent_dim=2
    dim_parameter_encoder=2
    hid_dim=512



    dataset_train, train_loader, t_dense, global_max_dose, global_max_time, global_mean, global_std, global_max_value = prepare_datasets_and_loaders_train(args.data_path, args.base_dir, all_ids, device, batch_fraction=0.05,truncation=1)

    
   
    mse_train = float('inf')  # initialize mse_training high


    encoder = Encoder_Transformer_NF(
        dim_parameter_encoder, input_dim=2, model_dim=64, hidden_dim=64,
        hidden_flow_dim=16, num_heads=4, num_layers=2, num_flow_layers=3, dropout=0.1).to(device)

    func = ODEFunc(latent_dim, dim_parameter_encoder, hid_dim).to(device)
    reducer = SimpleDecoder(latent_dim, hidden_dim=16).to(device)
    initial_encoder = InitialConditionVAEEncoder(latent_dim, hidden_dim=32).to(device)
    noise = TrainableNoise(size=1, init_add_std=0.3, init_prop_std=0.02).to(device)

    models = {
        "func": func,
        "encoder": encoder,
        "reducer": reducer,
        "initial_encoder": initial_encoder,
        "noise": noise}

    #save_models(models, save_dir, "vae_theo")
 #   load_models(models, save_dir, "vae_theo")

    optimizer, scheduler, main_params=prepare_optimizer(models,device, lr=0.001)
    
 
  
   

    mse_train = run_model_variant(dim_parameter_encoder, 
        variant_name="",
        dataset_train=dataset_train,
        dataset_val=dataset_train,
        dataset_test=dataset_train,
        models=models,
        main_params=main_params,
        optimizer=optimizer,
        scheduler=scheduler,
        t_dense=t_dense,
        global_max_dose=global_max_dose,
        global_max_time=global_max_time,
        global_mean=global_mean,
        global_std=global_std,
        latent_dim=latent_dim,
        noise=noise,
        encoder=encoder,
        func=func,
        reducer=reducer,
        initial_encoder=initial_encoder,
        metrics=1,
        residuals=1,
        iteration=1,
        n_epochs=n_epochs,
        dataloader_val=train_loader,
        train_loader=train_loader,
        test_loader=train_loader,
        base_dir=args.base_dir,
        warmup_noise=1000,
        warmup_iiv=0,
        
        enable_ae=False,
        enable_vae=True,
        enable_nf=False,
        enable_onlymedian=False,
        plot_from_training_records_enable=False,
        traing_against_validation=False,
        free_bits=1,
        truncation=1
    )

   
    plot_individual_fits(dataset_train,
        latent_dim, t_dense, models, train_loader, device,
        global_mean, global_std, global_max_time,
        enable_nf=False, enable_ae=True, enable_vae=True, enable_onlymedian=False,
        truncation=1, max_plots=12, nr_row=3, nr_col=4,
        num_simulated_total=100,  # instead of n_samples
        ci_lower=0.05, ci_upper=0.95,
        add_noise_to_prediction=False
    )
    
    
    vpc(models,
        train_loader,
        global_max_dose,
        global_max_time,
        global_mean,
        global_std,
        dataset_train,
        latent_dim,
        dim_parameter_encoder,
        initial_encoder,
        encoder,
        func,
        reducer,
        noise,
        t_dense,
        compartment="DV",
        onlymedian=False,
        enable_nf=False,
        enable_ae=False,
        enable_vae_training=True,
        add_noise_to_prediction=False,
        num_simulated_total=1000,
        truncation=1
    )

    residuals_all, times_all, predictions_all, targets_all= compute_residuals(dataset_train, latent_dim, global_mean, global_std, global_max_time, func, encoder, reducer, initial_encoder, noise, train_loader, t_dense, device, 1)

    vpc_true(  models, dataset_train, t_dense, global_max_time, global_max_dose,
      global_mean, global_std, latent_dim, dim_parameter_encoder,
      initial_encoder, encoder, func, reducer, noise, add_noise_to_prediction=False,
      enable_onlymedian=False, enable_ae=False, enable_nf=False,
      enable_vae=True, truncation=1, num_repeats=100)


    analyze_model_with_vpc(
        models,
        dataset_train,
        t_dense,
        global_max_time,
        global_max_dose,
        global_mean,
        global_std,
        latent_dim,
        dim_parameter_encoder,
        initial_encoder,
        encoder,
        func,
        reducer,
        noise,
        device,
        add_noise_to_prediction=False,
        enable_onlymedian=False,
        enable_ae=False,
        enable_nf=False,
        enable_vae=True,
        truncation=1,
        num_repeats=100,
    )
