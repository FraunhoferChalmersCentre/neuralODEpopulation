# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:54:49 2025

@author: Baaz
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 22:32:06 2025

@author: Baaz
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 21:45:47 2025

@author: Baaz
"""



import os
import argparse
import sys
import torch.optim as optim
import torch
import pandas as pd
import yaml
import ast



# ---- Training ---- 
if __name__ == "__main__":
        
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)
    
    # Simulate command line arguments in Spyder
    sys.argv = ['script_name',
    '--data_validation_path', 'lib/data/Simulated_ODE3_correlated_val.csv', 
    '--data_test_path', 'lib/data/Simulated_ODE3_correlated_test.csv', 

    '--data_path', 'lib/data/Simulated_ODE3_correlated_train.csv',
    '--save_dir', 'models',
    '--load_dir', 'models']
      
    from lib.utils.utils_preprocess import load_models, save_models, prepare_optimizer, prepare_datasets_and_loaders_simulated, compute_global_stats, export_all_metrics_and_residuals, append_metrics, load_all_metrics_and_residuals_as_lists
    from lib.utils.utils_training import   train_loop_model
    from lib.utils.utils_shared import ODEWrapper
    from lib.utils.utils_post_processing import plot_all_treatments_batch, vpc_true, estimate_coverage, plot_single_model_encoders_and_regression_combined, plot_single_model_encoders_and_regression , plot_encoder_histograms, plot_individual_fits, vpc, vpc_3
    from lib.models.NNmodels import  Encoder_Transformer, ODEFunc, SimpleDecoder, TrainableNoise
    
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
    
    
    
    

   # cov_list = ast.literal_eval(df['COV_IND'][0])
    
    
    
    dim_latent=2
    dim_parameters_IC=1
    dim_parameters_dynamic=2
    hidden_NODE_dim=512

    hidden_OBS_dim=16
    init_add_std=1
    init_prop_std=0
      
    number_drugs = df['EVID'].max()
    number_covariates=1
    hidden_Encoder_dim=36
    model_Encoder_dim=128
    heads_Encoder=4
    layers_Encoder=3
    dropout_Encoder=0.001
    learn_prior_mean=True
    learn_prior_covariance=True
    diagonal_only=False
    IC_dose_dependent=False

    lr = 0.001
    prior_lr_factor=50.00
    min_lr=1e-4
    reduction_factor=0.8
    patience=10

     
    truncation=1


    encoder_med = Encoder_Transformer(dim_latent=dim_latent,
    dim_parameters_IC=dim_parameters_IC,
    dim_parameters_dynamic=dim_parameters_dynamic,
    dim_cov=number_covariates, 
    model_dim=model_Encoder_dim,
    hidden_dim=hidden_Encoder_dim,
    num_heads=heads_Encoder,
    num_layers=layers_Encoder,
    dropout=0.1,
    cov_diag_epsilon=1e-5,
    learn_prior_mean=learn_prior_mean,
    learn_prior_covariance=learn_prior_covariance,
    diagonal_only=diagonal_only,
    IC_dose_dependent=IC_dose_dependent
    )
      
    func_med = ODEFunc(dim_latent=dim_latent,dim_parameters_IC=dim_parameters_IC,dim_parameters_dynamic=dim_parameters_dynamic,hid_dim=hidden_NODE_dim,drug_dim=number_drugs ).to(device)

    reducer_med = SimpleDecoder(dim_latent).to(device)

    noise_med = TrainableNoise(init_add_std=init_add_std, init_prop_std=init_prop_std).to(device)




    models_med = {
    "func": func_med,
    "encoder": encoder_med,
    "reducer": reducer_med,
    "noise": noise_med}
     


    optimizer1,scheduler1, main_params1 = prepare_optimizer(models_med, device,lr=lr,factor=reduction_factor ,  patience=patience,  min_lr=min_lr, prior_lr_factor=prior_lr_factor)
    #load_models(models, save_dir, "vae")


    train_loop_model(
       func_med=func_med,
       reducer_med=reducer_med,
       encoder_med=encoder_med,
       dataset=dataset_train,
       dataset_val=dataset_val,
       global_max_time=global_max_time,
       global_max_dose=global_max_dose,
       global_mean=global_mean,
       global_std=global_std,
       main_params=main_params1,
       dataloader_val=val_loader,
       dataloader=train_loader,
       models=models_med,
       optimizer=optimizer1,
       scheduler=scheduler1,
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
          noise_med,models_med,
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

 # %%
dim_latent=2
dim_parameters_IC=1
dim_parameters_dynamic=2
hidden_NODE_dim=512

hidden_OBS_dim=16
init_add_std=1
init_prop_std=0
  
hidden_Encoder_dim=36
model_Encoder_dim=128
heads_Encoder=4
layers_Encoder=3
dropout_Encoder=0.001
learn_prior_mean=True
learn_prior_covariance=True
diagonal_only=False
IC_dose_dependent=False

lr = 0.001
prior_lr_factor=1.00
min_lr=1e-5
reduction_factor=0.8
patience=20

 
truncation=1


encoder_ae = Encoder_Transformer(dim_latent=dim_latent,
dim_parameters_IC=dim_parameters_IC,
dim_parameters_dynamic=dim_parameters_dynamic,
dim_cov=number_covariates, 
model_dim=model_Encoder_dim,
hidden_dim=hidden_Encoder_dim,
num_heads=heads_Encoder,
num_layers=layers_Encoder,
dropout=0.1,
cov_diag_epsilon=1e-5,
learn_prior_mean=learn_prior_mean,
learn_prior_covariance=learn_prior_covariance,
diagonal_only=diagonal_only,
IC_dose_dependent=IC_dose_dependent
)
  
func_ae = ODEFunc(dim_latent=dim_latent,dim_parameters_IC=dim_parameters_IC,dim_parameters_dynamic=dim_parameters_dynamic,hid_dim=hidden_NODE_dim,drug_dim=number_drugs ).to(device)

reducer_ae = SimpleDecoder(dim_latent).to(device)

noise_ae = TrainableNoise(init_add_std=init_add_std, init_prop_std=init_prop_std).to(device)




models2 = {
"func": func_ae,
"encoder": encoder_ae,
"reducer": reducer_ae,
"noise": noise_ae}
 


optimizer2,scheduler2, main_params2 = prepare_optimizer(models2, device,lr=lr,factor=reduction_factor ,  patience=patience,  min_lr=min_lr, prior_lr_factor=prior_lr_factor)
#load_models(models, save_dir, "vae")




train_loop_model(
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
enable_vae=True, truncation=truncation, num_repeats=20,
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
    dataset_test,
    encoder_ae,
    func_ae,
    reducer_ae,
    noise_ae,
    ODEWrapper,
    t_dense,
    onlymedian=False,
    enable_ae=False,
    enable_vae=True,
    add_noise_to_prediction=False,
    num_simulated_total=10,
    normalization=True,
    truncation=truncation
)  

 
plot_single_model_encoders_and_regression(
 df, df_validation, dataset_train, dataset_val,
 encoder_ae, func_ae, reducer_ae,
 encoder_med, func_med, reducer_med,
  global_mean, global_std, t_dense,
 device=None, truncation=truncation, use_ema_models=True, normalization=True)




vpc_3(func_med,
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
    enable_ae=False,
    enable_vae=True,
    add_noise_to_prediction=False,
    num_simulated_total=10,
    normalization=True,
    truncation=truncation
)  

  #%% 
dim_latent=2
dim_parameters_IC=1
dim_parameters_dynamic=2
hidden_NODE_dim=512

hidden_OBS_dim=16
init_add_std=5
init_prop_std=0
  
hidden_Encoder_dim=36
model_Encoder_dim=128
heads_Encoder=4
layers_Encoder=3
dropout_Encoder=0.001
learn_prior_mean=False
learn_prior_covariance=False
diagonal_only=False
IC_dose_dependent=False

lr = 0.001
prior_lr_factor=1.00
min_lr=1e-5
reduction_factor=0.8
patience=20

 
truncation=1


encoder_vae = Encoder_Transformer(dim_latent=dim_latent,
dim_parameters_IC=dim_parameters_IC,
dim_parameters_dynamic=dim_parameters_dynamic,
dim_cov=number_covariates, 
model_dim=model_Encoder_dim,
hidden_dim=hidden_Encoder_dim,
num_heads=heads_Encoder,
num_layers=layers_Encoder,
dropout=0.1,
cov_diag_epsilon=1e-5,
learn_prior_mean=learn_prior_mean,
learn_prior_covariance=learn_prior_covariance,
diagonal_only=diagonal_only,
IC_dose_dependent=IC_dose_dependent
)
  
func_vae = ODEFunc(dim_latent=dim_latent,dim_parameters_IC=dim_parameters_IC,dim_parameters_dynamic=dim_parameters_dynamic,hid_dim=hidden_NODE_dim,drug_dim=number_drugs ).to(device)

reducer_vae = SimpleDecoder(dim_latent).to(device)

noise_vae = TrainableNoise(init_add_std=init_add_std, init_prop_std=init_prop_std).to(device)




models3 = {
"func": func_vae,
"encoder": encoder_vae,
"reducer": reducer_vae,
"noise": noise_vae}
 


optimizer3,scheduler3, main_params3 = prepare_optimizer(models3, device,lr=lr,factor=reduction_factor ,  patience=patience,  min_lr=min_lr, prior_lr_factor=prior_lr_factor)
#load_models(models, save_dir, "vae")




train_loop_model(
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
 models=models3,
 optimizer=optimizer3,
 scheduler=scheduler3,
 func=func_vae,
 reducer=reducer_vae,
 encoder=encoder_vae,
 noise=noise_vae,
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
models3, dataset_train, t_dense, global_max_time, global_mean, global_std,global_max_dose,
encoder_vae, func_vae, reducer_vae, noise_vae,
func_med, reducer_med, encoder_med, noise_med,
add_noise_to_prediction=True,
enable_onlymedian=False, enable_ae=False,
enable_vae=True, truncation=truncation, num_repeats=2,
use_ema_models=True,
fontsize=14,  show_confidence_intervals=True
)

vpc(func_med,
      reducer_med,
      encoder_med,
      noise_med,models3,
    train_loader,
    global_max_dose,
    global_max_time,
    global_mean,
    global_std,
    dataset_train,
    encoder_vae,
    func_vae,
    reducer_vae,
    noise_vae,
    ODEWrapper,
    t_dense,
    onlymedian=False,
    enable_ae=False,
    enable_vae=True,
    add_noise_to_prediction=False,
    num_simulated_total=10,
    normalization=True,
    truncation=truncation
)  

 
plot_single_model_encoders_and_regression(
 df, df_validation, dataset_train, dataset_val,
 encoder_ae, func_ae, reducer_ae,
 encoder_med, func_med, reducer_med,
  global_mean, global_std, t_dense,
 device=None, truncation=truncation, use_ema_models=True, normalization=True)




vpc_3(func_med,
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
    enable_ae=False,
    enable_vae=True,
    add_noise_to_prediction=False,
    num_simulated_total=10,
    normalization=True,
    truncation=truncation
)    
    
    
    
    
    
    