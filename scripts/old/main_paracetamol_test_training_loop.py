
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
                '--data_path', 'lib/data/paracetamol_data2.csv',
                '--save_dir', 'models',
                '--load_dir', 'models/old',
                '--base_dir','results/run1']
    
    
    from lib.utils.utils_paracetamol import prepare_optimizer, prepare_datasets_and_loaders, run_model_variant, compute_global_stats, export_all_metrics_and_residuals, append_metrics, load_all_metrics_and_residuals_as_lists, TrajectoryDataset_paracetamol, collate_fn, ODEWrapper
    from lib.models.NNmodels import Encoder_Transformer_NF, ODEFunc, SimpleDecoder, InitialConditionEncoder, TrainableNoise
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
  
   
    
    df = pd.read_csv(args.data_path, sep=";")
    global_max_dose, global_max_time, global_mean, global_std, global_max_value = compute_global_stats(df)
    base_dataset = TrajectoryDataset_paracetamol(args.data_path, augment_with_prefixes=False, augment_dose_times=False, max_dose=global_max_dose, max_time=global_max_time, conc_mean=global_mean, conc_std=global_std)
    all_ids = list(set(base_dataset.all_subject_ids))
    n_ids = len(all_ids)
  

    variants = ["", "_ae", "_ae_noise"]
    metrics = {var: {"mse_mean": [], "r2_mean": [], "mse_median": [], "r2_median": [], "mse_validation": []} for var in variants}
    residuals = {var: [] for var in variants}
    
    
   

    n_repeats = 20  # number of times to repeat evaluation
    n_epochs=1000
    lr=0.001
    metrics, residuals, already_done = load_all_metrics_and_residuals_as_lists(args.base_dir)
    start = already_done  # continue training or evaluation from here
    
    
    latent_dim=2
    dim_parameter_encoder=2
    hid_dim=512
 
    
   
   
    
    for i in range(n_repeats):    
        train_dataset, val_dataset, test_dataset, train_loader, val_loader, combined = prepare_datasets_and_loaders(
    args.data_path, args.base_dir, all_ids, i,already_done, global_max_dose, global_max_time, global_mean, global_std, device, batch_fraction=0.05, trunctation=0.6)

        
   
        mse_train = float('inf')  # initialize mse_training high

        
        max_attempts = 3
        attempt = 0
        
        while mse_train > 80 and attempt < max_attempts:
            attempt += 1
            print(f"\n===== Attempt {attempt} =====")
        
            encoder = Encoder_Transformer_NF(
                dim_parameter_encoder, input_dim=2, model_dim=64, hidden_dim=64,
                hidden_flow_dim=16, num_heads=4, num_layers=2, num_flow_layers=3, dropout=0.1
            ).to(device)
        
            func = ODEFunc(latent_dim, dim_parameter_encoder, hid_dim).to(device)
            reducer = SimpleDecoder(latent_dim, hidden_dim=16).to(device)
            initial_encoder = InitialConditionEncoder(latent_dim, hidden_dim=8).to(device)
            noise = TrainableNoise(size=1, init_add_std=1, init_prop_std=0).to(device)
        
            models = {
                "func": func,
                "encoder": encoder,
                "reducer": reducer,
                "initial_encoder": initial_encoder,
                "noise": noise
            }
        
            optimizer, scheduler,main_params = prepare_optimizer(models,device)
        
     
            # AE
            mse_train = run_model_variant(
                variant_name="_ae",
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                models=models,
                main_params=main_params,
                optimizer=optimizer,
                scheduler=scheduler,
                combined=combined,
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
                ODEWrapper=ODEWrapper,
                device=device,
                metrics=metrics,
                residuals=residuals,
                iteration=i,
                n_epochs=n_epochs,
                train_loader=train_loader,
                val_loader=val_loader,
                base_dir=args.base_dir,
                warmup_noise=1000,
                warmup_iiv=0,
                enable_ae=True,
                enable_nf=False,
                plot_from_training_records_enable=False,
                free_bits=1
            )
        
            print(f"Training MSE after attempt {attempt}: {mse_train}")
        
        if mse_train > 100:
            print("\n⚠️ Training did not reach desired MSE <= 100 after 3 attempts.")
        else:
            print("\n✅ Training succeeded with MSE <= 100.")
        
                
        # AE + Noise
        mse_train=run_model_variant(
            variant_name="_ae_noise",
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            models=models,
            main_params=main_params,
            optimizer=optimizer,
            scheduler=scheduler,
            combined=combined,
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
            ODEWrapper=ODEWrapper,
            device=device,
            metrics=metrics,
            residuals=residuals,
            iteration=i,
            n_epochs=n_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            base_dir=args.base_dir,
            warmup_noise=0,
            warmup_iiv=0,
            enable_ae=True,
            enable_nf=False,
            plot_from_training_records_enable=False,
            free_bits=1
        )
        
        # VAE / standard NF
        mse_train=run_model_variant(
            variant_name="",
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            models=models,
            main_params=main_params,
            optimizer=optimizer,
            scheduler=scheduler,
            combined=combined,
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
            ODEWrapper=ODEWrapper,
            device=device,
            metrics=metrics,
            residuals=residuals,
            iteration=i,
            n_epochs=n_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            base_dir=args.base_dir,
            warmup_noise=1000,
            warmup_iiv=20,
            enable_ae=False,
            enable_nf=False,
            plot_from_training_records_enable=False,
            free_bits=0
        )




