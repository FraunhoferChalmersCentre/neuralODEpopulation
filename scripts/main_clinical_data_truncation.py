

import os
import argparse
import sys
import torch.optim as optim
import torch
import pandas as pd
import ast



# ---- Training ---- 
if __name__ == "__main__":
        
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)

    base_dir = os.environ.get("THEOPHYLLINE_BASE_DIR", "results/theo/run11")
    
    # Simulate command line arguments in Spyder
    sys.argv = ['script_name',
              '--data_validation_path', 'lib/data/theophylline_data_formatted.csv', 
              '--data_test_path', 'lib/data/theophylline_data_formatted.csv', 
              '--data_path', 'lib/data/theophylline_data_formatted.csv',
     '--save_dir', 'models',
     '--load_dir', 'models',
     '--base_dir', base_dir]
      
    from lib.utils.utils_preprocess import load_all_obsvspred, prepare_and_merge_datasets, prepare_datasets_and_loaders, load_models, save_models, prepare_optimizer, prepare_datasets_and_loaders_simulated, compute_global_stats, export_all_metrics_and_residuals, append_metrics, load_all_metrics_and_residuals_as_lists
    from lib.utils.utils_training import   train_loop_model, run_train_test
    from lib.utils.utils_post_processing import  plot_VPC_and_residuals, vpc, plot_single_model_encoders_and_regression , plot_encoder_histograms, plot_individual_fits
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

    
      
       
    
    
    
    
    args = parser.parse_args()
    df = pd.read_csv(args.data_path, sep=';')
    df_validation=pd.read_csv(args.data_validation_path, sep=';')
    df_test=pd.read_csv(args.data_test_path, sep=';')
      
    number_drugs = df['EVID'].max()
    number_covariates=1


  #%% 
dim_latent=2
dim_parameters_IC=1
dim_parameters_dynamic=2
hidden_NODE_dim=256

hidden_OBS_dim=16
init_add_std=0.2
init_prop_std=0.02
  
hidden_Encoder_dim=36
model_Encoder_dim=64
heads_Encoder=2
layers_Encoder=2

learn_prior_mean=True
learn_prior_covariance=True
diagonal_only=False
IC_dose_dependent=False

lr = 0.001
prior_lr_factor=1.00
min_lr=1e-5
reduction_factor=0.8
patience=20

 
truncation=0.3

it=1
for it in range(0, it):
    
    df_test, df_train, dataset_train, dataset_test, train_loader, test_loader, t_dense= prepare_datasets_and_loaders(args.data_path, args.data_test_path, device, batch_fraction=0.25,time_points=120)
     
    prepare_and_merge_datasets(df_train, df_test,args.base_dir, it) 
      
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
    
    
    n_epochs=int(os.environ.get("THEOPHYLLINE_TRUNCATION_EPOCHS", "10"))
    
        
    
    # cov_list = ast.literal_eval(df['COV_IND'][0])
    metrics, already_done = load_all_metrics_and_residuals_as_lists(args.base_dir)
    obsvspred = load_all_obsvspred(args.base_dir)
    start = already_done  # continue training or evaluation from here
    
    
    run_train_test(it, models_ebvae, models_ebvae, dataset_train, dataset_test, dataset_test,df_train,
                          main_params2, optimizer2, scheduler2, t_dense, metrics,obsvspred,
                          n_epochs, train_loader, test_loader,test_loader,  args.base_dir,
                          warmup_noise=100, warmup_iiv=100,smoothing_start_epoch=0, enable_ae=False, enable_vae=True,enable_onlymedian=False,normalization=False,
                          free_bits=0, truncation=0.3)


    plot_individual_fits(
     models_ebvae,
     models_ebvae,
     dataset_test,
     t_dense,
     df,
     truncation=0.3,
     max_plots=3,
     n_samples=100,
     ci_lower=0.05,
     ci_upper=0.95,
     nr_row=1,
     nr_col=3,
     enable_vae=False,
     enable_ae=False,
     enable_onlymedian=False,
     add_noise=True,
     use_ema_models=False,
     normalization=False,
     fontsize=14,
     export=True,
     pdf_filename=os.path.join(args.base_dir, "Figure7.pdf")# Added a parameter to control font size
 )
