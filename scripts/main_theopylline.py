# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 20:09:11 2025

@author: Baaz
"""

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
              '--data_validation_path', 'lib/data/theophylline_data_formatted.csv', 
              '--data_test_path', 'lib/data/theophylline_data_formatted.csv', 
              '--data_path', 'lib/data/theophylline_data_formatted.csv',
     '--save_dir', 'models',
     '--load_dir', 'models',
     '--base_dir','results/theo/run7']
      
    from lib.utils.utils_preprocess import prepare_and_merge_datasets, prepare_datasets_and_loaders, load_models, save_models, prepare_optimizer, prepare_datasets_and_loaders_simulated, compute_global_stats, export_all_metrics_and_residuals, append_metrics, load_all_metrics_and_residuals_as_lists
    from lib.utils.utils_training import   train_loop_model, run_train_test
    from lib.utils.utils_post_processing import compute_test_metrics, compute_residuals, vpc_true, plot_single_model_encoders_and_regression , plot_encoder_histograms, plot_individual_fits, vpc
    from lib.models.NNmodels import  Encoder_Transformer, ODEFunc, SimpleDecoder, TrainableNoise
    
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
    modelname = "MultipleDoseAddError_AE_NF2"
    
      
       
    
    
    
    
    args = parser.parse_args()
    df = pd.read_csv(args.data_path, sep=';')
    df_validation=pd.read_csv(args.data_validation_path, sep=';')
    df_test=pd.read_csv(args.data_test_path, sep=';')
    
    
    global_max_dose, global_max_time, global_mean, global_std, global_max_value, global_min_value=compute_global_stats(df)
    df_train,dataset_train, dataset_val, dataset_test, train_base_dataset, train_loader, val_loader, test_loader, t_dense, batch_size_train, batch_size_val, batch_size_test = prepare_datasets_and_loaders(args.data_path, args.data_validation_path, args.data_test_path,  device, batch_fraction=0.2,time_points=120)
    
    
    
    

   # cov_list = ast.literal_eval(df['COV_IND'][0])
    metrics, already_done = load_all_metrics_and_residuals_as_lists(args.base_dir)
    start = already_done  # continue training or evaluation from here
    
    
    dim_latent=2
    dim_parameters_IC=1
    dim_parameters_dynamic=2
    hidden_NODE_dim=512

    hidden_OBS_dim=16
    init_add_std=1
    init_prop_std=0
      
    number_drugs = df['EVID'].max()
    number_covariates=1



 # %%
dim_latent=2
dim_parameters_IC=0
dim_parameters_dynamic=2
hidden_NODE_dim=256

hidden_OBS_dim=16
init_add_std=0.2
init_prop_std=0.02
  
hidden_Encoder_dim=36
model_Encoder_dim=128
heads_Encoder=4
layers_Encoder=2
dropout_Encoder=0.001
learn_prior_mean=True
learn_prior_covariance=True
diagonal_only=False
IC_dose_dependent=False

lr = 0.0001
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




models_ebvae = {
"func": func_ae,
"encoder": encoder_ae,
"reducer": reducer_ae,
"noise": noise_ae}
 


optimizer2,scheduler2, main_params2 = prepare_optimizer(models_ebvae, device,lr=lr,factor=reduction_factor ,  patience=patience,  min_lr=min_lr, prior_lr_factor=prior_lr_factor)
#load_models(models, save_dir, "vae")




train_loop_model(
 models_ebvae,
 models_ebvae,
 dataloader=train_loader,
 dataset_train=df,
 main_params=main_params2,
 optimizer=optimizer2,
 scheduler=scheduler2,
 t_dense=t_dense,
 n_epochs=1000,
 warmup_epochs_noise=0,
 warmup_epochs_iiv=0,
 smoothing_start_epoch=0,
 enable_ae=False,
 enable_vae=True,
 enable_onlymedian=False,
 normalization=False,
 truncation=truncation)

  
vpc_true(
models_ebvae,models_ebvae, dataset_train, t_dense, df,
add_noise_to_prediction=True,
enable_onlymedian=False, enable_ae=False,
enable_vae=True, truncation=truncation, num_repeats=2,
use_ema_models=True
)


compute_residuals(models_ebvae, models_ebvae, train_loader, t_dense,
                                                   df,
                                                   truncation=truncation, num_repeats=10, show_confidence_intervals=True)

plot_encoder_histograms(
    models_ebvae,models_ebvae,
   t_dense, df,
   dataset_train, train_loader, truncation=truncation, normalization=False
   )


plot_individual_fits(
     models_ebvae,
     models_ebvae,
     dataset_train,
     t_dense,
     df,
     truncation=truncation,
     max_plots=20,
     n_samples=100,
     ci_lower=0.05,
     ci_upper=0.95,
     nr_row=2,
     nr_col=10,
     enable_vae=False,
     enable_ae=True,
     enable_onlymedian=False,
     add_noise=True,
     use_ema_models=True,
     normalization=False,
     fontsize=14  # Added a parameter to control font size
   ) 
  #%% 
dim_latent=2
dim_parameters_IC=0
dim_parameters_dynamic=2
hidden_NODE_dim=256

hidden_OBS_dim=16
init_add_std=0.2
init_prop_std=0.02
  
hidden_Encoder_dim=36
model_Encoder_dim=128
heads_Encoder=4
layers_Encoder=2
dropout_Encoder=0.001
learn_prior_mean=True
learn_prior_covariance=True
diagonal_only=False
IC_dose_dependent=False

lr = 0.0001
prior_lr_factor=1.00
min_lr=1e-5
reduction_factor=0.8
patience=20

 
truncation=0.3

it=5
for it in range(0, it):
    
    df_test, df_train, dataset_train, dataset_test, train_loader, test_loader, t_dense= prepare_datasets_and_loaders(args.data_path, args.data_test_path, device, batch_fraction=0.2,time_points=120)
     
    prepare_and_merge_datasets(df_train, df_test,args.base_dir) 
      
    
    
    
    
    
    
    
    
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
    
    
    
    
    models_ebvae = {
    "func": func_ae,
    "encoder": encoder_ae,
    "reducer": reducer_ae,
    "noise": noise_ae}
     
    
    
    optimizer2,scheduler2, main_params2 = prepare_optimizer(models_ebvae, device,lr=lr,factor=reduction_factor ,  patience=patience,  min_lr=min_lr, prior_lr_factor=prior_lr_factor)
    #load_models(models, save_dir, "vae")
    
    
    n_epochs=1
    
        
    
    # cov_list = ast.literal_eval(df['COV_IND'][0])
    metrics, already_done = load_all_metrics_and_residuals_as_lists(args.base_dir)
    start = already_done  # continue training or evaluation from here
    
    
    run_train_test(models_ebvae, models_ebvae, dataset_train, dataset_test, dataset_test,df_train,
                          main_params2, optimizer2, scheduler2, t_dense, metrics,
                          n_epochs, train_loader, val_loader,test_loader,  args.base_dir,
                          warmup_noise=30, warmup_iiv=0,smoothing_start_epoch=0, enable_ae=False, enable_vae=True,enable_onlymedian=False,normalization=False,
                          free_bits=0, truncation=0.3)



