# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 21:30:52 2025

@author: Baaz
"""

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
    '--data_validation_path', 'lib/data/tumor_data_val.csv', 
    '--data_test_path', 'lib/data/tumor_data_test.csv', 
    '--data_path', 'lib/data/tumor_data_train.csv',
    '--save_dir', 'models',
    '--load_dir', 'models']
      
    from lib.utils.utils_preprocess import load_models, save_models, prepare_optimizer, prepare_datasets_and_loaders_simulated, compute_global_stats, export_all_metrics_and_residuals, append_metrics, load_all_metrics_and_residuals_as_lists
    from lib.utils.utils_training import   train_loop_model
    from lib.utils.utils_shared import ODEWrapper
    from lib.utils.utils_post_processing import vpc_true, estimate_coverage, plot_single_model_encoders_and_regression_combined, plot_single_model_encoders_and_regression , plot_encoder_vs_samples, plot_encoder_histograms, plot_individual_fits, vpc
    from lib.models.NNmodels import  Encoder_Transformer_Full, ODEFunc, SimpleDecoder, TrainableNoise
    
    parser = argparse.ArgumentParser(description="Train Neural-ODE model on dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--data_test_path", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--data_validation_path", type=str, required=True, help="Path to test CSV file")
    
    parser.add_argument("--load_dir", type=str, default=None, help="Path to folder with saved models (optional)")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory where trained models will be saved")
    
    # Parse arguments
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
      
    print("Using device:", device)
    
    load_dir = args.load_dir
    save_dir = args.save_dir
    modelname = "MultipleDoseAddError_AE_NF2"
    
      
       
    
    
    
    
    args = parser.parse_args()
    df = pd.read_csv(args.data_path, sep=';')
    df_validation=pd.read_csv(args.data_validation_path, sep=';')
    df_test=pd.read_csv(args.data_test_path, sep=';')
    
    
    global_max_dose, global_max_time, global_mean, global_std, global_max_value, global_min_value=compute_global_stats(df)
    dataset_train, dataset_val, dataset_test, train_base_dataset, train_loader, val_loader, test_loader, t_dense, batch_size_train, batch_size_val, batch_size_test = prepare_datasets_and_loaders_simulated(args.data_path, args.data_validation_path, args.data_test_path, global_mean, global_max_dose, global_max_time, global_mean,global_std, global_std, device, batch_fraction=0.05,time_points=120)
    
    
    
    
    
    
    
    
    dim_latent=2
    dim_parameters=2
    hid_dim=512
     
    number_drugs = df['EVID'].max()

 #  save_models(models, save_dir, "test1")
    encoder_med = Encoder_Transformer_Full(dim_latent,dim_parameters, input_dim=2, model_dim=128,hidden_dim=32, num_heads=4, dropout=0.001).to(device)
    func_med = ODEFunc(dim_latent,dim_parameters,hid_dim,number_drugs ).to(device)
    reducer_med = SimpleDecoder(dim_latent, hidden_dim=32).to(device)
    noise_med = TrainableNoise(size=1, init_add_std=1, init_prop_std=0).to(device)

    
    models = {
    "func": func_med,
    "encoder": encoder_med,
    "reducer": reducer_med,
    "noise": noise_med,
    }
    
    optimizer,scheduler, main_params = prepare_optimizer(models, device, lr=0.001,factor=0.5, patience=5, min_lr=1e-7, prior_lr_factor=1)
  #  save_models(models, save_dir, "med")
      
    
     
    
    train_loop_model(
           p_dropout=1,
       func_med=func_med,
       reducer_med=reducer_med,
       encoder_med=encoder_med,
       dataset=dataset_train,
       dataset_val=dataset_val,
       global_max_time=global_max_time,
       global_max_dose=global_max_dose,
       global_mean=global_mean,
       global_std=global_std,
       main_params=main_params,
       dataloader_val=val_loader,
       dataloader=train_loader,
       models=models,
       optimizer=optimizer,
       scheduler=scheduler,
       func=func_med,
       reducer=reducer_med,
       encoder=encoder_med,
       noise=noise_med,
       t_dense=t_dense,
       n_epochs=10000,
       warmup_epochs_noise=1000,
       warmup_epochs_iiv=0,
       smoothing_start_epoch=0,
       learn_prior=30,
       enable_ae=False,
       enable_vae=False,
       enable_onlymedian=True,
       normalization=False,
       truncation=1,
       print_epoch=1)
               
    vpc(func_med,
          reducer_med,
          encoder_med,
          noise_med,models,
        train_loader,
        global_max_dose,
        global_max_time,
        global_mean,
        global_std,
        dataset_train,
        encoder_med,
        func_med,
        reducer_med,
        noise_med,
        ODEWrapper,
        t_dense,
        onlymedian=True,
        enable_ae=False,
        enable_vae=False,
        add_noise_to_prediction=False,
        num_simulated_total=10,
        normalization=False,
        truncation=1
    )  

     
    dim_latent=2
    dim_parameters=2
    hid_dim=512
     
    truncation=1
    encoder_ae = Encoder_Transformer_Full(dim_latent,dim_parameters, input_dim=2, 
                                          model_dim=128,hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1,
                                          cov_diag_epsilon=1e-5, learn_prior_mean=True, learn_prior_covariance=True,
                                          diagonal_only=False).to(device)
    func_ae = ODEFunc(dim_latent,dim_parameters,hid_dim,number_drugs ).to(device)
    reducer_ae = SimpleDecoder(dim_latent, hidden_dim=32).to(device)
    noise_ae = TrainableNoise(size=1, init_add_std=1, init_prop_std=0).to(device)
    
 


    # # # Copy weights
    #func_ae.load_state_dict(func_med.state_dict())
    #reducer_ae.load_state_dict(reducer_med.state_dict())

    models2 = {
    "func": func_ae,
    "encoder": encoder_ae,
    "reducer": reducer_ae,
    "noise": noise_ae,
        }
        
    optimizer2,scheduler2, main_params2 = prepare_optimizer(models2, device,lr=0.001 ,factor=0.8, patience=10, min_lr=1e-5, prior_lr_factor=1)
    #load_models(models, save_dir, "vae")




    train_loop_model(
           p_dropout=0,
       func_med=func_med,
       reducer_med=reducer_med,
       encoder_med=encoder_med,
       dataset=dataset_train,
       dataset_val=dataset_val,
       global_max_time=global_max_time,
       global_max_dose=global_max_dose,
       global_mean=global_mean,
       global_std=global_std,
       main_params=main_params2,
       dataloader_val=val_loader,
       dataloader=train_loader,
       models=models2,
       optimizer=optimizer2,
       scheduler=scheduler2,
       func=func_ae,
       reducer=reducer_ae,
       encoder=encoder_ae,
       noise=noise_ae,
       t_dense=t_dense,
       n_epochs=1000,
       warmup_epochs_noise=0,
       warmup_epochs_iiv=0,
       smoothing_start_epoch=0,
       learn_prior=0,
       enable_ae=False,
       enable_vae=True,
       enable_onlymedian=False,
       normalization=True,
       truncation=truncation,
       print_epoch=1)
    
    plot_encoder_histograms(
        encoder_med, func_med, reducer_med,
       t_dense, global_mean, global_std,
       dataset_train, train_loader, encoder_ae, device=None, truncation=truncation, normalization=True
   )
    
  
    vpc_true(ODEWrapper,
    models2, dataset_test, t_dense, global_max_time, global_mean, global_std,global_max_dose,
    encoder_ae, func_ae, reducer_ae, noise_ae,
    func_med, reducer_med, encoder_med, noise_med,
    add_noise_to_prediction=True,
    enable_onlymedian=False, enable_ae=False,
    enable_vae=True, truncation=truncation, num_repeats=10,
    use_ema_models=True,
    fontsize=14,  show_confidence_intervals=True
)
    
    vpc(func_med,
          reducer_med,
          encoder_med,
          noise_med,models2,
        train_loader,
        global_max_dose,
        global_max_time,
        global_mean,
        global_std,
        dataset_train,
        encoder_ae,
        func_ae,
        reducer_ae,
        noise_ae,
        ODEWrapper,
        t_dense,
        onlymedian=False,
        enable_ae=True,
        enable_vae=False,
        add_noise_to_prediction=True,
        num_simulated_total=10,
        normalization=True,
        truncation=1
    )  
    
 
    plot_single_model_encoders_and_regression(
     df, df_validation, dataset_train, dataset_val,
     encoder_ae, func_ae, reducer_ae,
     encoder_med, func_med, reducer_med,
      global_mean, global_std, t_dense,
     device=None, truncation=truncation, use_ema_models=True, normalization=True)

    
    
    
    plot_individual_fits(
         models2,
         encoder_med, func_med, reducer_med,
         dataset_train,
         device,
         t_dense,
         global_mean,
         global_std,
         truncation=truncation,
         max_plots=10,
         n_samples=10,
         ci_lower=0.05,
         ci_upper=0.95,
         nr_row=1,
         nr_col=10,
         enable_vae=False,
         enable_ae=True,
         enable_onlymedian=False,
         add_noise=False,
         use_ema_models=True,
         normalization=True,
         fontsize=14  # Added a parameter to control font size
       ) 
    
    vpc_true(ODEWrapper,
    models2, dataset_train, t_dense, global_max_time, global_mean, global_std,global_max_dose,
    encoder_ae, func_ae, reducer_ae, noise_ae,
    func_med, reducer_med, encoder_med, noise_med,
    add_noise_to_prediction=True,
    enable_onlymedian=False, enable_ae=False,
    enable_vae=True, truncation=truncation, num_repeats=10,
    use_ema_models=True,
    fontsize=14,  show_confidence_intervals=True)
    
    vpc(func_med,
          reducer_med,
          encoder_med,
          noise_med,models,
        train_loader,
        global_max_dose,
        global_max_time,
        global_mean,
        global_std,
        dataset_train,
        latent_dim,
        dim_parameter_encoder,
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
        add_noise_to_prediction=False,
        num_simulated_total=10,
        normalization=False,
        truncation=1
    )  

    #   #  save_models(models, save_dir, "ae")
    # plot_encoder_histograms(encoder_ae, initial_encoder_ae, func_ae, t_dense, reducer_ae, global_mean, global_std, dataset_train, encoder_ae, latent_dim, device=None, truncation=0.3, normalization=True)
    # plot_encoder_vs_samples( encoder_ae, initial_encoder_ae, func_ae,
    #  t_dense, reducer_ae, global_mean, global_std,
    #  dataset_train, encoder_ae, latent_dim, device=None, truncation=0.3, normalization=True)
    
    # plot_single_model_encoders_and_regression(
    # df, df_validation, dataset_train, dataset_val,
    # encoder_ae, initial_encoder_ae, func_ae, reducer_ae,
    # encoder_ae, initial_encoder_ae, func_ae, reducer_ae,
    # latent_dim, global_mean, global_std, t_dense,
    # device, truncation=1, dim_parameter_encoder=2,     use_ema_models=True, normalization=False
    # )
    
    # plot_two_models_encoders_and_regression(
    #     df, df_validation, dataset_train, dataset_val,
    #     encoder_ae, initial_encoder_ae, func_ae, reducer_ae,
    #     encoder_ae, initial_encoder_ae, func_ae, reducer_ae,
    #     encoder_med, initial_encoder_med, func_med, reducer_med,
    #     latent_dim, global_mean, global_std, t_dense,
    #     device, enable_vae=False, enable_ae=True, enable_onlymedian=False, truncation=0.2, normalization=True,
    #     dim_parameter_encoder=2
    # )
    
    # plot_one_model_encoders_and_regression2(
    #     df, df_validation, dataset_train, dataset_val,
    #     encoder_ae, initial_encoder_ae, func_ae, reducer_ae,
    #     encoder_med, initial_encoder_med, func_med, reducer_med,
    #     latent_dim, global_mean, global_std, t_dense,
    #     device, enable_vae=False, enable_ae=True, enable_onlymedian=False, truncation=0.2, normalization=True,
    #     dim_parameter_encoder=2
    # )
    
    # plot_single_model_encoder_means(
    #     df, df_validation, dataset_train, dataset_val,
    #     encoder_ae, initial_encoder_ae, func_ae, reducer_ae,
    #     encoder_med, initial_encoder_med, func_med, reducer_med,
    #     latent_dim, global_mean, global_std, t_dense,
    #     device=None, truncation=0.2, dim_parameter_encoder=2,
    #     use_ema_models=True, normalization=True
    # )
     
      
    # plot_individual_fits(
    #   models,
    #   encoder_med, initial_encoder_med, func_med, reducer_med,
    #   dataset_train,
    #   device,
    #   t_dense,
    #   global_mean,
    #   global_std,
    #   truncation=0.3,
    #   max_plots=15,
    #   n_samples=10,
    #   ci_lower=0.05,
    #   ci_upper=0.95,
    #   nr_row=5,
    #   nr_col=3,
    #   enable_vae=True,
    #   enable_ae=False,
    #   enable_onlymedian=False,
    #   add_noise=True,
    #   use_ema_models=True,
    #   normalization=True,
    #   fontsize=14

    # )
    
    

    
    #   #  save_models(models, save_dir, "test1")
    # encoder_ae2 = Encoder_Transformer(dim_parameter_encoder, input_dim=2, model_dim=64,hidden_dim=32, num_heads=4, dropout=0.001).to(device)
    # func_ae2 = ODEFunc(latent_dim,dim_parameter_encoder,hid_dim ).to(device)
    # reducer_ae2 = SimpleDecoder(latent_dim, hidden_dim=16).to(device)
    # initial_encoder_ae2 = InitialConditionVAEEncoder(latent_dim, hidden_dim=8).to(device)
    # noise_ae2 = TrainableNoise(size=1, init_add_std=1, init_prop_std=0).to(device)
    
    # models2 = {
    # "func": func_ae2,
    # "encoder": encoder_ae2,
    # "reducer": reducer_ae2,
    # "initial_encoder": initial_encoder_ae2,
    # "noise": noise_ae2,
    # }
    
    # optimizer,scheduler, main_params = prepare_optimizer(models2, device, lr=0.001)
    # #load_models(models, save_dir, "vae")
    
    
       
    
    train_loop_model(
       p_dropout=0,
       func_med=func_med,
       reducer_med=reducer_med,
       initial_encoder_med=initial_encoder_med,
       encoder_med=encoder_med,
       dataset=dataset_train,
       dataset_val=dataset_val,
       global_max_time=global_max_time,
       global_max_dose=global_max_dose,
       global_mean=global_mean,
       global_std=global_std,
       main_params=main_params,
       dataloader_val=val_loader,
       dataloader=train_loader,
       models=models2,
       optimizer=optimizer,
       scheduler=scheduler,
       func=func_ae2,
       reducer=reducer_ae2,
       initial_encoder=initial_encoder_ae2,
       encoder=encoder_ae2,
       noise=noise_ae2,
       t_dense=t_dense,
       n_epochs=10000,
       warmup_epochs_noise=1000,
       warmup_epochs_iiv=0,
       smoothing_start_epoch=100,
       traing_against_validation=False,
       enable_ae=True,
       enable_vae=False,
       enable_onlymedian=False,
       normalization=False,
       plot_training=False,
       free_bits=1,
       truncation=0.4,
       print_epoch=1,
       plot_epoch=1,
       max_plots=4,
       nr_col=1,
       nr_row=5
       )
    
    vpc(func_med,
      reducer_med,
      initial_encoder_med,
      encoder_med,
      noise_med,models,
    train_loader,
    global_max_dose,
    global_max_time,
    global_mean,
    global_std,
    dataset_train,
    latent_dim,
    dim_parameter_encoder,
    initial_encoder_ae,
    encoder_ae2,
    func_ae2,
    reducer_ae2,
    noise_ae2,
    ODEWrapper,
    t_dense,
    compartment="DV",
    onlymedian=False,
    enable_ae=True,
    enable_vae=False,
    add_noise_to_prediction=True,
    num_simulated_total=1000,
    normalization=True,
    truncation=0.3
    )
    
    
    vpc_3(
        # Median model
        encoder_med, initial_encoder_med, func_med, reducer_med, noise_med,
        # Model 1
        encoder_ae, initial_encoder_ae, func_ae, reducer_ae, noise_ae, models1,
        encoder_ae, initial_encoder_ae, func_ae, reducer_ae, noise_ae, models1,
        # Model 2
  
        train_loader, dataset_train, val_loader, dataset_val, t_dense, latent_dim, dim_parameter_encoder,
        global_max_time, global_max_dose, global_mean, global_std,
        "DV", ODEWrapper,
        add_noise_to_prediction=True, num_simulated_total=1000, onlymedian=False, enable_ae=False, enable_vae=True,
        normalization=True,
        truncation=0.2
    )
    
    
    
    
    vpc_4(
        # Median model
        encoder_med, initial_encoder_med, func_med, reducer_med, noise_med,
        # Model 1
        encoder_ae, initial_encoder_ae, func_ae, reducer_ae, noise_ae, models1,
        encoder_ae, initial_encoder_ae, func_ae, reducer_ae, noise_ae, models1,
        # Model 2
  
        train_loader, dataset_train, test_loader, dataset_test, t_dense, latent_dim, dim_parameter_encoder,
        global_max_time, global_max_dose, global_mean, global_std,
        "DV", ODEWrapper,
        add_noise_to_prediction=True, num_simulated_total=1000, onlymedian=False, enable_ae=False, enable_vae=True,
        normalization=True,
        truncation=0.2
    )