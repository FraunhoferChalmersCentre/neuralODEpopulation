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
import ast




from torch.utils.data import Dataset, DataLoader




# ---- Training ---- 
if __name__ == "__main__":
        
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)
    
    # Simulate command line arguments in Spyder
    sys.argv = ['script_name',
                '--data_validation_path', 'lib/data/simulated_validation_data5.csv', 
                '--data_test_path', 'lib/data/simulated_test_data5.csv', 
                '--data_path', 'lib/data/simulated_training_data5.csv',
                '--save_dir', 'models',
                '--load_dir', 'models/old']
    
    from lib.utils.my_utils import prepare_datasets_and_loaders_simulated, prepare_optimizer, plot_individual_fits, train_model, prepare_datasets_and_loaders, compute_global_stats, TrajectoryDataset, collate_fn_simulated, ODEWrapper
    from lib.models.NNmodels import Encoder_Transformer_NF, ODEFunc, SimpleDecoder, InitialConditionEncoder, TrainableNoise
    from lib.utils.model_validation import vpc_all, plot_node_latent_vs_reduced, vpc, rf_predict_params_from_encoder_validation, plot_encoder_mu_vs_params
    # Define your parser
    
    # Define your parser
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

    
     #   def __init__(self, path, compartment='C2', augment_with_prefixes=False, augment_dose_times=False, dose_jitter_std=0.01):

    
    args = parser.parse_args()
    df = pd.read_csv(args.data_path)
    df_validation=pd.read_csv(args.data_validation_path)
    df_test=pd.read_csv(args.data_test_path)
    
    
    global_max_dose, global_max_time, global_mean, global_std, global_max_value=compute_global_stats(df)
    dataset_train, dataset_validation, dataset_test, train_base_dataset, train_loader, val_loader, test_loader, combined, batch_size_train, batch_size_val, batch_size_test = prepare_datasets_and_loaders_simulated(args.data_path, args.data_validation_path, args.data_test_path, global_max_dose, global_max_time, global_mean, global_std, device, batch_fraction=0.05, trunctation=1)

    
    





    latent_dim=2
    dim_parameter_encoder=2
    hid_dim=512
 
    
    encoder_ae = Encoder_Transformer_NF(dim_parameter_encoder, input_dim=2, model_dim=64,hidden_dim=64,hidden_flow_dim=16, num_heads=4, num_layers=2,num_flow_layers=3, dropout=0.1).to(device)
    func_ae = ODEFunc(latent_dim,dim_parameter_encoder,hid_dim ).to(device)
    reducer_ae = SimpleDecoder(latent_dim, hidden_dim=16).to(device)
    initial_encoder_ae = InitialConditionEncoder(latent_dim, hidden_dim=8).to(device)
    noise_ae = TrainableNoise(size=1, init_add_std=2, init_prop_std=0).to(device)

    models = {
    "func": func_ae,
    "encoder": encoder_ae,
    "reducer": reducer_ae,
    "initial_encoder": initial_encoder_ae,
    "noise": noise_ae,
        }
        
    optimizer,scheduler, main_params = prepare_optimizer(models, device, lr=0.001)
    
    

   

    train_model(
          global_mean,
          global_std,
          global_max_time,
          global_max_dose,
          main_params,
          val_loader,
          train_loader,
          models, 
          optimizer, 
          scheduler, 
          func_ae, 
          reducer_ae,
          initial_encoder_ae, 
          encoder_ae,
          noise_ae,
          t_dense=combined,
          n_epochs=1000,
          warmup_epochs_noise=1000,
          warmup_epochs_iiv=0,
          smoothing_start_epoch=1000,
          traing_against_validation=False,
          enable_ae_training=True,
          enable_nf_training=False,
          enable_onlymedian_training=False, 
          plot_from_training_records_enable=False,              
          free_bits=1,                                            
          df=df,
          df_val=df_test,
          dataset=dataset_train,
          dataset_val=dataset_validation,
          max_points_visible=1,   
          print_epoch=1,
          plot_epoch=1,
          max_plots=50,
          nr_col=5,
          nr_row=10)  
   
    

    
    encoder = Encoder_Transformer_NF(dim_parameter_encoder, input_dim=2, model_dim=64,hidden_dim=64,hidden_flow_dim=16, num_heads=4, num_layers=2,num_flow_layers=3, dropout=0.1).to(device)
    func = ODEFunc(latent_dim,dim_parameter_encoder,hid_dim ).to(device)
    reducer = SimpleDecoder(latent_dim, hidden_dim=16).to(device)
    initial_encoder = InitialConditionEncoder(latent_dim, hidden_dim=8).to(device)
    noise = TrainableNoise(size=1, init_add_std=2, init_prop_std=0).to(device)
    #3.18
    models = {
    "func": func,
    "encoder": encoder,
    "reducer": reducer,
    "initial_encoder": initial_encoder,
    "noise": noise,
    # add any other models...
    } 
   
    optimizer,scheduler, main_params = prepare_optimizer(models, device, lr=0.001)



    train_model(
        global_mean,
        global_std,
        global_max_time,
        global_max_dose,
        main_params,
        val_loader,
        train_loader,
        models, 
        optimizer, 
        scheduler, 
        func, 
        reducer,
        initial_encoder, 
        encoder,
        noise,
        t_dense=combined,
        n_epochs=1000,
        warmup_epochs_noise=1000,
        warmup_epochs_iiv=0,
        smoothing_start_epoch=0,
        traing_against_validation=False,
        enable_ae_training=False,
        enable_nf_training=False,
        enable_onlymedian_training=False, 
        plot_from_training_records_enable=False,              
        free_bits=1,                                            
        df=df,
        df_val=df_test,
        dataset=dataset_train,
        dataset_val=dataset_validation,
        max_points_visible=1,   
        print_epoch=1,
        plot_epoch=1,
        max_plots=50,
        nr_col=5,
        nr_row=10)  
 






# -----------------------
# Post-training:
# -----------------------



#
# Population prediction vs data
#


### Predictions without encoder
vpc(global_max_dose,
    global_max_time,
    global_mean, 
    global_std,
    onlymedian=False,
    df_training=df_validation,
    df=df,
    dataset=train_base_dataset,
    latent_dim=latent_dim,
    dim_parameters=dim_parameter_encoder,
    initial_encoder=initial_encoder,
    encoder=encoder,
    func=func,
    reducer=reducer,
    noise=noise,
    ODEWrapper=ODEWrapper,
    t_dense=torch.linspace(0, 1, steps=120),
    compartment="C2",
    num_simulated_total=1000,
    enable_ae_training=True,
    enable_nf_training=False,
    add_noise_to_prediction=False
)


vpc(global_max_dose,
    global_max_time,
    global_mean, 
    global_std,
    onlymedian=False,
    df_training=df_test,
    df=df,
    dataset=dataset_test,
    latent_dim=latent_dim,
    dim_parameters=dim_parameter_encoder,
    initial_encoder=initial_encoder,
    encoder=encoder,
    func=func,
    reducer=reducer,
    noise=noise,
    ODEWrapper=ODEWrapper,
    t_dense=torch.linspace(0, 1, steps=120),
    compartment="C2",
    num_simulated_total=1000,
    enable_ae_training=True,
    enable_nf_training=False,
    add_noise_to_prediction=False
)

rf_predict_params_from_encoder_validation(df, train_base_dataset, df_test, dataset_test,encoder, latent_dim, dim_parameter_encoder,device=None, n_estimators=200, random_state=42)


vpc(global_max_dose,
    global_max_time,
    global_mean, 
    global_std,
    onlymedian=False,
    df_training=df_validation,
    df=df,
    dataset=train_base_dataset,
    latent_dim=latent_dim,
    dim_parameters=dim_parameter_encoder,
    initial_encoder=initial_encoder,
    encoder=encoder,
    func=func,
    reducer=reducer,
    noise=noise,
    ODEWrapper=ODEWrapper,
    t_dense=torch.linspace(0, 1, steps=120),
    compartment="C2",
    num_simulated_total=1000,
    enable_ae_training=False,
    enable_nf_training=False,
    add_noise_to_prediction=False
)


vpc(global_max_dose,
    global_max_time,
    global_mean, 
    global_std,
    onlymedian=False,
    df_training=df_test,
    df=df,
    dataset=dataset_test,
    latent_dim=latent_dim,
    dim_parameters=dim_parameter_encoder,
    initial_encoder=initial_encoder,
    encoder=encoder,
    func=func,
    reducer=reducer,
    noise=noise,
    ODEWrapper=ODEWrapper,
    t_dense=torch.linspace(0, 1, steps=120),
    compartment="C2",
    num_simulated_total=1000,
    enable_ae_training=False,
    enable_nf_training=False,
    add_noise_to_prediction=False
)


    

vpc_all(
    global_max_dose=global_max_dose,
    global_max_time=global_max_time,
    global_mean=global_mean,
    global_std=global_std,
    onlymedian=False,
    dataset_1=dataset_validation,
    dataset_2=dataset_test,  # test dataset
    df1=df_validation,
    df2=df_test,
    latent_dim=latent_dim,
    dim_parameters=dim_parameter_encoder,
    
    # AE models
    encoder_ae=encoder_ae,
    initial_encoder_ae=initial_encoder_ae,
    func_ae=func_ae,
    reducer_ae=reducer_ae,
    noise_ae=noise_ae,
    
    # VAE/NF models
    encoder_vae=encoder,
    initial_encoder_vae=initial_encoder,
    func_vae=func,
    reducer_vae=reducer,
    noise_vae=noise,
    
    ODEWrapper=ODEWrapper,
    t_dense=combined,
    compartment="C2",
    num_simulated_total=1000,
    add_noise_to_prediction=False
)




plot_encoder_mu_vs_params(df, train_base_dataset, encoder, latent_dim,dim_parameter_encoder, device=None)


rf_predict_params_from_encoder_validation(df, train_base_dataset, df_validation, dataset_validation,encoder, latent_dim, dim_parameter_encoder,device=None, n_estimators=200, random_state=42)


plot_individual_fits(global_max_dose,global_mean, global_std,False, False, train_base_dataset, df, latent_dim, noise,encoder, func, reducer, initial_encoder, ODEWrapper, combined,max_individuals=20, n_samples=100, device=device,truncation=0, add_noise=True)
