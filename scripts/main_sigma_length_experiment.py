# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 20:10:26 2025

@author: Baaz
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 16:04:21 2025

@author: Baaz
"""




import os
import argparse
import sys
import torch.optim as optim
import torch
import pandas as pd
import yaml



# ---- Training ---- 
if __name__ == "__main__":
        
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)
    
    # Simulate command line arguments in Spyder
    sys.argv = ['script_name',
                '--data_validation_path', 'lib/data/simulated_SDE_validation.csv', 
                '--data_test_path', 'lib/data/simulated_truncated.csv', 
                '--data_path', 'lib/data/simulated_SDE.csv',
                '--save_dir', 'models',
                '--load_dir', 'models']
    
    from lib.utils.utils_preprocess import load_models, save_models, prepare_optimizer, prepare_datasets_and_loaders_simulated, compute_global_stats, export_all_metrics_and_residuals, append_metrics, load_all_metrics_and_residuals_as_lists, TrajectoryDataset, collate_fn
    from lib.utils.utils_training import   train_loop_model
    from lib.utils.utils_shared import ODEWrapper
    from lib.utils.utils_post_processing import estimate_coverage, plot_single_model_encoders_and_regression_combined, plot_single_model_encoders_and_regression , plot_encoder_vs_samples, plot_encoder_histograms, plot_individual_fits, vpc
    from lib.models.NNmodels import InitialConditionVAEEncoder, Encoder_Transformer, ODEFunc, SimpleDecoder, InitialConditionEncoder, TrainableNoise
    
    parser = argparse.ArgumentParser(description="Train Neural-ODE model on dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--data_test_path", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--data_validation_path", type=str, required=True, help="Path to test CSV file")

    parser.add_argument("--load_dir", type=str, default=None, help="Path to folder with saved models (optional)")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory where trained models will be saved")
    
    # Parse arguments
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #  device = torch.device("cpu")
    
    print("Using device:", device)
        
    load_dir = args.load_dir
    save_dir = args.save_dir
    modelname = "MultipleDoseAddError_AE_NF2"

    

    
    args = parser.parse_args()
    df = pd.read_csv(args.data_path, sep=';')
    df_validation=pd.read_csv(args.data_validation_path, sep=';')
    df_test=pd.read_csv(args.data_test_path, sep=';')
    
    
    global_max_dose, global_max_time, global_mean, global_std, global_max_value, global_min_value=compute_global_stats(df)
    dataset_train, dataset_val, dataset_test, train_base_dataset, train_loader, val_loader, test_loader, t_dense, batch_size_train, batch_size_val, batch_size_test = prepare_datasets_and_loaders_simulated(args.data_path, args.data_validation_path, args.data_test_path, global_mean, global_max_dose, global_max_time, global_mean,global_std, global_std, device, batch_fraction=0.1,time_points=120)

    
    





    latent_dim=2
    dim_parameter_encoder=2
    hid_dim=512
 
    
 
  
    
    
  #  save_models(models, save_dir, "test1")
    encoder_ae = Encoder_Transformer(dim_parameter_encoder, input_dim=2, model_dim=64,hidden_dim=32, num_heads=4, dropout=0.001).to(device)
    func_ae = ODEFunc(latent_dim,dim_parameter_encoder,hid_dim ).to(device)
    reducer_ae = SimpleDecoder(latent_dim, hidden_dim=16).to(device)
    initial_encoder_ae = InitialConditionVAEEncoder(latent_dim, hidden_dim=8).to(device)
    noise_ae = TrainableNoise(size=1, init_add_std=10, init_prop_std=0).to(device)

    models = {
    "func": func_ae,
    "encoder": encoder_ae,
    "reducer": reducer_ae,
    "initial_encoder": initial_encoder_ae,
    "noise": noise_ae,
        }
        
    optimizer,scheduler, main_params = prepare_optimizer(models, device, lr=0.001)
    #save_models(models, save_dir, "ae")
    

   

    train_loop_model(
           p_dropout=0,
       func_med=func_ae,
       reducer_med=reducer_ae,
       initial_encoder_med=initial_encoder_ae,
       encoder_med=encoder_ae,
       dataset=dataset_train,
       dataset_val=dataset_val,
       global_max_time=global_max_time,
       global_max_dose=global_max_dose,
       global_mean=global_mean,
       global_min=global_std,
       main_params=main_params,
       dataloader_val=val_loader,
       dataloader=train_loader,
       models=models,
       optimizer=optimizer,
       scheduler=scheduler,
       func=func_ae,
       reducer=reducer_ae,
       initial_encoder=initial_encoder_ae,
       encoder=encoder_ae,
       noise=noise_ae,
       t_dense=t_dense,
       n_epochs=10000,
       warmup_epochs_noise=75,
       warmup_epochs_iiv=0,
       smoothing_start_epoch=0,
       traing_against_validation=False,
       enable_ae=False,
       enable_vae=True,
       enable_onlymedian=False,
       normalization=False,
       plot_training=False,
       free_bits=1,
       truncation=1,
       print_epoch=1,
       plot_epoch=1,
       max_plots=4,
       nr_col=1,
       nr_row=5
   )
               
    vpc(func_ae,
          reducer_ae,
          initial_encoder_ae,
          encoder_ae,
          noise_ae,models,
        train_loader,
        global_max_dose,
        global_max_time,
        global_mean,
        global_std,
        dataset_val,
        latent_dim,
        dim_parameter_encoder,
        initial_encoder_ae,
        encoder_ae,
        func_ae,
        reducer_ae,
        noise_ae,
        ODEWrapper,
        t_dense,
        compartment="DV",
        onlymedian=False,
        enable_ae=False,
        enable_vae=True,
        add_noise_to_prediction=True,
        num_simulated_total=1000,
        normalization=False,
        truncation=1
    )
    

  #  save_models(models, save_dir, "ae")
    plot_encoder_histograms(encoder_ae, initial_encoder_ae, func_ae, t_dense, reducer_ae, global_mean, global_std, dataset_train, encoder_ae, latent_dim, device=None, truncation=1, normalization=False)
    plot_encoder_vs_samples( encoder_ae, initial_encoder_ae, func_ae,
     t_dense, reducer_ae, global_mean, global_std,
     dataset_train, encoder_ae, latent_dim, device=None, truncation=1, normalization=False)
    
    plot_single_model_encoders_and_regression(
    df, df_validation, dataset_train, dataset_val,
    encoder_ae, initial_encoder_ae, func_ae, reducer_ae,
    encoder_ae, initial_encoder_ae, func_ae, reducer_ae,
    latent_dim, global_mean, global_std, t_dense,
    device, truncation=1, dim_parameter_encoder=2
)
    plot_single_model_encoders_and_regression_combined(
        df, df_validation, dataset_train, dataset_val,
        encoder_ae, initial_encoder_ae, func_ae, reducer_ae,
        encoder_ae, initial_encoder_ae, func_ae, reducer_ae,
        latent_dim, global_mean, global_std, t_dense,
        device, truncation=1, dim_parameter_encoder=2,   use_ema_models=True
)
 
    
  
    plot_individual_fits(
        models,
        dataset_train,
        device,
        t_dense,
        global_mean,
        global_std,
        truncation=1,
        max_plots=100,
        n_samples=100,
        ci_lower=0.1,
        ci_upper=0.9,
        nr_row=10,
        nr_col=10,
        enable_vae=True,
        enable_ae=False,
        enable_onlymedian=False,
        add_noise=True
    )
    
    
    estimate_coverage(
        models,
        dataset_train,
        device,
        t_dense,
        global_mean,
        global_std,
        truncation=1,
        max_individuals=50,
        n_samples=200,
        ci_lower=0.05,
        ci_upper=0.95,
        enable_vae=True,
        enable_ae=False,
        enable_onlymedian=False,
        add_noise=True,
        use_ema_models=True,
    )
    
