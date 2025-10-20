# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 20:24:28 2025

@author: Baaz
"""

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
    

    sys.argv = ['script_name',
                '--data_validation_path', 'lib/data/theophylline_data_formatted.csv', 
                '--data_test_path', 'lib/data/theophylline_data_formatted.csv', 
                '--data_path', 'lib/data/theophylline_data_formatted.csv',
                '--save_dir', 'models',
                '--load_dir', 'models/old',
                '--base_dir','results/theo/run6']
      
    from lib.utils.utils_preprocess import load_models, save_models, prepare_optimizer, prepare_datasets_and_loaders, compute_global_stats, export_all_metrics_and_residuals, append_metrics, load_all_metrics_and_residuals_as_lists
    from lib.utils.utils_training import   run_model_variant
    from lib.utils.utils_shared import ODEWrapper
    from lib.models.NNmodels import  Encoder_Transformer_Full, ODEFunc, SimpleDecoder, TrainableNoise
    
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
  
   
    df = pd.read_csv(args.data_path, sep=";")
  
  

    variants = ["", "_ae", "_ae_noise"]
    metrics = {var: {"mse_mean": [], "r2_mean": [], "mse_median": [], "r2_median": [], "mse_validation": []} for var in variants}
    residuals = {var: [] for var in variants}
    
    n_repeats = 1000  # number of times to repeat evaluation
    n_epochs=200
    lr=0.001
    metrics, residuals, already_done = load_all_metrics_and_residuals_as_lists(args.base_dir)
    start = already_done  # continue training or evaluation from here
    
      
    latent_dim=2
    dim_parameter_encoder=2
    hid_dim=512
    p_dropout=0     
    number_drugs = df['EVID'].max()

 
       
    
    
    global_max_dose, global_max_time, global_mean, global_std, global_max_value, global_min_value=compute_global_stats(df)
    dataset_train, dataset_val, dataset_test, train_base_dataset, train_loader, val_loader, test_loader, t_dense, batch_size_train, batch_size_val, batch_size_test = prepare_datasets_and_loaders(args.data_path, args.data_validation_path, args.data_test_path, global_mean, global_max_dose, global_max_time, global_mean,global_std, global_std, device, batch_fraction=0.1,time_points=120)
    
    
    
    
    
    
    
  


    metrics, residuals, already_done = load_all_metrics_and_residuals_as_lists(args.base_dir)
    start = already_done  # continue training or evaluation from here

 #  save_models(models, save_dir, "test1")
    encoder_ae= Encoder_Transformer_Full(dim_parameter_encoder+latent_dim, input_dim=2, model_dim=128,hidden_dim=32, num_heads=4, dropout=0.001).to(device)
    func_ae = ODEFunc(latent_dim,dim_parameter_encoder,hid_dim,number_drugs ).to(device)
    reducer_ae = SimpleDecoder(latent_dim, hidden_dim=32).to(device)
   # initial_encoder_med = InitialConditionVAEEncoder(latent_dim, hidden_dim=8).to(device)
    noise_ae = TrainableNoise(size=1, init_add_std=0.1, init_prop_std=0).to(device)

    
    models = {
    "func": func_ae,
    "encoder": encoder_ae,
    "reducer": reducer_ae,
 #   "initial_encoder": initial_encoder_med,
    "noise": noise_ae,
    }
    
    optimizer,scheduler, main_params = prepare_optimizer(models, device, lr=0.001)
  #  save_models(models, save_dir, "med")
      
    
    mse_train = run_model_variant(
         p_dropout,
         variant_name="_ae",
         dataset_train=dataset_train,
         dataset_val=dataset_train,
         dataset_test=dataset_test,
         models=models,
         main_params=main_params,
         optimizer=optimizer,
         scheduler=scheduler,
         t_dense=t_dense,
         global_max_dose=global_max_dose,
         global_max_time=global_max_time,
         global_mean=global_mean,
         global_std=global_std,
         noise=noise_ae,
         encoder=encoder_ae,
         func=func_ae,
         reducer=reducer_ae,
         metrics=metrics,
         residuals=residuals,
         iteration=100,
         n_epochs=50,
         train_loader=train_loader,
         val_loader=train_loader,
         test_loader=test_loader,
         base_dir=args.base_dir,
         warmup_noise=1000,
         warmup_iiv=0,
         enable_ae=True,
         enable_vae=False,
         enable_onlymedian=False,
         truncation=0.3)
    
    