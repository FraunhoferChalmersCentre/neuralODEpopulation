#


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
      
    from lib.utils.utils_preprocess import load_models, save_models, prepare_optimizer, prepare_datasets_and_loaders_simulated, compute_global_stats
    from lib.utils.utils_training import   train_loop_model
    from lib.utils.utils_shared import ODEWrapper
    from lib.utils.utils_post_processing import vpc_dual, vpc, plot_single_model_encoders_and_regression , plot_encoder_histograms, plot_individual_fits
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
    
    
    
    


    
    
    number_covariates=1
    number_drugs=1
    
    
# %% Initialize and train median model    
    dim_latent=2
    dim_parameters_IC=1
    dim_parameters_dynamic=2
    hidden_NODE_dim=516

    init_add_std=0.2
    init_prop_std=0.01
      
    hidden_Encoder_dim=36
    model_Encoder_dim=128
    heads_Encoder=4
    layers_Encoder=3
    dropout_Encoder=0.001
    learn_prior_mean=True
    learn_prior_covariance=False
    diagonal_only=False
    IC_dose_dependent=False

    lr = 0.001
    prior_lr_factor=1.00
    min_lr=1e-5
    reduction_factor=0.8
    patience=20

     
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
     models_med,
     models_med,
     dataloader=train_loader,
     dataset_train=df,
     main_params=main_params1,
     optimizer=optimizer1,
     scheduler=scheduler1,
     t_dense=t_dense,
     n_epochs=1000,
     warmup_epochs_noise=1000,
     warmup_epochs_iiv=0,
     smoothing_start_epoch=0,
     enable_ae=False,
     enable_vae=False,
     enable_onlymedian=True,
     normalization=False,
     truncation=truncation)
    
   

      
    vpc(
    models_med,models_med, dataset_train, t_dense, df,
    add_noise_to_prediction=False,
    enable_onlymedian=True, enable_ae=False,
    enable_vae=False, normalization=False, truncation=truncation, num_repeats=1,
    use_ema_models=True
    )

# %% Initialize and train empirical bayes vae model
dim_latent=2
dim_parameters_IC=1
dim_parameters_dynamic=2
hidden_NODE_dim=516

hidden_OBS_dim=16
init_add_std=3
init_prop_std=0.1
  
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
prior_lr_factor=5.00
min_lr=1e-5
reduction_factor=0.8
patience=10

 
truncation=1


encoder_ebvae = Encoder_Transformer(dim_latent=dim_latent,
dim_parameters_IC=dim_parameters_IC,
dim_parameters_dynamic=dim_parameters_dynamic,
dim_cov=number_covariates, 
model_dim=model_Encoder_dim,
hidden_dim=hidden_Encoder_dim,
num_heads=heads_Encoder,
num_layers=layers_Encoder,
dropout=0.05,
cov_diag_epsilon=1e-5,
learn_prior_mean=learn_prior_mean,
learn_prior_covariance=learn_prior_covariance,
diagonal_only=diagonal_only,
IC_dose_dependent=IC_dose_dependent
)
  
func_ebvae = ODEFunc(dim_latent=dim_latent,dim_parameters_IC=dim_parameters_IC,dim_parameters_dynamic=dim_parameters_dynamic,hid_dim=hidden_NODE_dim,drug_dim=number_drugs ).to(device)

reducer_ebvae = SimpleDecoder(dim_latent).to(device)

noise_ebvae = TrainableNoise(init_add_std=init_add_std, init_prop_std=init_prop_std).to(device)




models_ebvae = {
"func": func_ebvae,
"encoder": encoder_ebvae,
"reducer": reducer_ebvae,
"noise": noise_ebvae}
 


optimizer2,scheduler2, main_params2 = prepare_optimizer(models_ebvae, device,lr=lr,factor=reduction_factor ,  patience=patience,  min_lr=min_lr, prior_lr_factor=prior_lr_factor)




train_loop_model(
 models_norm=models_med,
 models_eval=models_ebvae,
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
 normalization=True,
 truncation=truncation)

save_models(models_ebvae, save_dir, "ebvae")
  
vpc(
models_norm=models_med,models_eval=models_ebvae,dataset= dataset_test, t_dense=t_dense, df=df,
add_noise_to_prediction=True,
enable_onlymedian=False, enable_ae=False,
enable_vae=True, normalization=True, truncation=truncation, num_repeats=1,
use_ema_models=True,cols=3, export=True, pdf_filename="vpccorr.pdf"
)


plot_encoder_histograms(
  models_norm=models_med,  models_eval=models_ebvae,
     t_dense=t_dense, df=df,
     dataset=dataset_train, dataloader=train_loader,
  truncation=truncation, normalization=True
   )

  

plot_single_model_encoders_and_regression(
    df, df_validation, dataset_train, dataset_val,
   models_eval=models_ebvae, models_median=models_med,t_dense=t_dense,
   truncation=truncation, use_ema_models=True, normalization=True, single_figure=True, export=True
)

    
    
# %% Initialize and train vae model
dim_latent=2
dim_parameters_IC=1
dim_parameters_dynamic=2
hidden_NODE_dim=516

hidden_OBS_dim=16
init_add_std=1
init_prop_std=0.01
  
hidden_Encoder_dim=36
model_Encoder_dim=128
heads_Encoder=4
layers_Encoder=2
dropout_Encoder=0.001
learn_prior_mean=True
learn_prior_covariance=False
diagonal_only=False
IC_dose_dependent=False

lr = 0.001
prior_lr_factor=5.00
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
dropout=0.05,
cov_diag_epsilon=1e-5,
learn_prior_mean=learn_prior_mean,
learn_prior_covariance=learn_prior_covariance,
diagonal_only=diagonal_only,
IC_dose_dependent=IC_dose_dependent
)
  
func_vae = ODEFunc(dim_latent=dim_latent,dim_parameters_IC=dim_parameters_IC,dim_parameters_dynamic=dim_parameters_dynamic,hid_dim=hidden_NODE_dim,drug_dim=number_drugs ).to(device)

reducer_vae = SimpleDecoder(dim_latent).to(device)

noise_vae = TrainableNoise(init_add_std=init_add_std, init_prop_std=init_prop_std).to(device)




models_vae = {
"func": func_vae,
"encoder": encoder_vae,
"reducer": reducer_vae,
"noise": noise_vae}
 


optimizer3,scheduler3, main_params3 = prepare_optimizer(models_vae, device,lr=lr,factor=reduction_factor ,  patience=patience,  min_lr=min_lr, prior_lr_factor=prior_lr_factor)




train_loop_model(
 models_norm=models_med,
 models_eval=models_vae,
 dataloader=train_loader,
 dataset_train=df,
 main_params=main_params3,
 optimizer=optimizer3,
 scheduler=scheduler3,
 t_dense=t_dense,
 n_epochs=1000,
 warmup_epochs_noise=0,
 warmup_epochs_iiv=0,
 smoothing_start_epoch=0,
 enable_ae=False,
 enable_vae=True,
 enable_onlymedian=False,
 normalization=True,
 truncation=truncation)


plot_encoder_histograms(
  models_norm=models_med,  models_eval=models_vae,
     t_dense=t_dense, df=df,
     dataset=dataset_train, dataloader=train_loader,
  truncation=truncation, normalization=True
   )


  
vpc(
models_norm=models_med,models_eval=models_vae,dataset= dataset_train, t_dense=t_dense, df=df,
add_noise_to_prediction=True,
enable_onlymedian=False, enable_ae=False,
enable_vae=True, normalization=True, truncation=truncation, num_repeats=1,
use_ema_models=True,cols=2, export=True, pdf_filename="vpccorr.pdf"
)


# %% Initialize and train model

load_models(models_ebvae, load_dir, "ebvae", device=torch.device("cuda"))


models_eval_list = [models_vae, models_ebvae]

# Call the dual VPC plotting function
vpc_dual(
    models_med, models_vae, models_ebvae, dataset_val, t_dense, df,
    add_noise_to_prediction=False,
    enable_onlymedian=False, enable_ae=False,
    enable_vae=True, normalization=True, truncation=1, num_repeats=1,
    use_ema_models=False, cols=2, export=True, pdf_filename="vpc_dual.pdf",
    fontsize=14, show_confidence_intervals=True
)