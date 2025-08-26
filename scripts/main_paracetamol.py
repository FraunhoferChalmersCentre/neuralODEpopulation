# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 16:55:49 2025

@author: Baaz
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 16:04:21 2025

@author: Baaz
"""

# SAFE SETTINGS TO AVOID OpenMP CRASHES



import os
import argparse
import sys
import torch.optim as optim
import torch
import pandas as pd
import ast




from torch.utils.data import Dataset, DataLoader, random_split


 


# ---- Training ---- 
if __name__ == "__main__":
        
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)
    
    # Simulate command line arguments in Spyder
    sys.argv = ['script_name',
                '--data_validation_path', 'lib/data/simulated_validation_data6.csv', 
                '--data_test_path', 'lib/data/simulated_test_data6.csv', 
                '--data_path', 'lib/data/paracetamol_data.csv',
                '--save_dir', 'models',
                '--load_dir', 'models/old']
    
    from lib.utils.my_utils import train_model_paracetamol, TrajectoryDataset_paracetamol, collate_fn, ODEWrapper,load_models, save_models
    from lib.models.NNmodels import Encoder_Transformer_NF, ODEFunc, SimpleDecoder, InitialConditionEncoder, TrainableNoise
    from lib.utils.model_validation import compute_residuals, predict_and_evaluate_mse, vpc, vpc_with_encoder, plot_individual_fits, plot_encoder_mu_vs_params, rf_predict_params_from_encoder_validation, rf_predict_params_from_encoderanddose_validation

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
    

    
     #   def __init__(self, path, compartment='C2', augment_with_prefixes=False, augment_dose_times=False, dose_jitter_std=0.01):

    
    args = parser.parse_args()
    df = pd.read_csv(args.data_path, sep=";")
    
    df_validation=pd.read_csv(args.data_validation_path)

    df_test=pd.read_csv(args.data_test_path)
    
    df_all = pd.concat([df], ignore_index=True)

    
    global_max_dose = df_all['Dose'].max() 
    global_max_time = df_all['Time'].max()
    global_mean = df_all['C2'].mean()
    global_std = df_all['C2'].std()
    global_max_value = df_all['C2'].max()
  
    base_dataset = TrajectoryDataset_paracetamol(
    args.data_path,
    augment_with_prefixes=False,
    augment_dose_times=False,
    max_dose=global_max_dose,
    max_time=global_max_time,
    conc_mean=global_mean,
    conc_std=global_std
)


    
    
    
    load_dir = args.load_dir
    save_dir = args.save_dir
    modelname = "MultipleDoseAddError_AE_NF2"


 


    latent_dim=4
    dim_parameter_encoder=2
    hid_dim=512
 
    

    encoder = Encoder_Transformer_NF(dim_parameter_encoder, input_dim=2, model_dim=64,hidden_dim=64,hidden_flow_dim=16, num_heads=4, num_layers=2,num_flow_layers=3, dropout=0.1).to(device)
   

    #conc_mean, conc_std = dataset.conc_mean, dataset.conc_std
    func = ODEFunc(latent_dim,dim_parameter_encoder,hid_dim ).to(device)
    reducer = SimpleDecoder(latent_dim, hidden_dim=16).to(device)
    initial_encoder = InitialConditionEncoder(latent_dim, hidden_dim=8).to(device)
    noise = TrainableNoise(size=1, init_add_std=2.9, init_prop_std=0).to(device)
    #2.9
    models = {
    "func": func,
    "encoder": encoder,
    "reducer": reducer,
    "initial_encoder": initial_encoder,
    "noise": noise,
    # add any other models...
    } 
   
    lr=0.001

    main_params = [
        {"params": list(func.parameters()) + list(reducer.parameters()) + list(initial_encoder.parameters()) +list(encoder.parameters()) , "lr": lr},
        {"params": list(noise.parameters()), "lr": 10*lr},
    ]
    
    optimizer = torch.optim.Adam(main_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=7,min_lr=1e-4)  # Set your desired minimum LR here)

    for name, model in models.items():
          model.to(device)
          print(f"{name} is on {next(model.parameters()).device}")

   
    train_size = int(0.8 * len(base_dataset))
    test_size  = len(base_dataset) - train_size
    train_dataset_raw, test_dataset = random_split(base_dataset, [train_size, test_size])
    train_indices = train_dataset_raw.indices if hasattr(train_dataset_raw, "indices") else range(len(train_dataset_raw))
    train_dataset = TrajectoryDataset_paracetamol(
        args.data_path,
        augment_with_prefixes=True,
        augment_dose_times=False,
        max_dose=global_max_dose,
        max_time=global_max_time,
        conc_mean=global_mean,
        conc_std=global_std,
        subset_indices=train_indices   # <- only apply augmentation on train subset
    )
    
    
    batch_size = max(1, int(len(train_dataset) * 0.05))
    full_loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    doses = torch.tensor([3], dtype=torch.float32) / 24
    time_points = torch.linspace(0, 1, steps=120)
    combined = torch.cat((time_points, doses)).to(device)
    combined, indices = torch.sort(combined)  # ensures ascending order
    val_dataset=train_dataset
    
    train_model_paracetamol(
        train_dataset,
        val_dataset,
        global_max_time,
        global_max_dose,
        global_mean, 
        global_std,
        main_params,
        train_loader,
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
        smoothing_start_epoch=1000,
        traing_against_validation=False,
        enable_ae_training=False,
        enable_nf_training=False,
        enable_onlymedian_training=False, 
        plot_from_training_records_enable=True,              
        free_bits=1,                                            
        max_points_visible=0.6,   
        print_epoch=1,
        plot_epoch=25,
        max_plots=30,
        nr_col=10,
        nr_row=3)  
 

  #  load_models(models, save_dir, "Paracetamol_VAE")
             

#    predict_and_evaluate_mse(
 #   enable_ae_training=False,
  #  enable_nf_training=True,
 #   noise=noise,
 #   df=df_test,
 #   models=models,
 #   dataloader=dataloader_test,
#    dataset=dataset_test,
 #   t_dense=torch.linspace(0, 1, steps=50),
#    max_points_visible=0.1)



  # save_models(models, save_dir, "Paracetamol_AE")

   # MSE 15.
   
    #decoder = Encoder_Transformer_NF(...)  # same args as decoder1
 




# -----------------------
# Post-training:
# -----------------------


#
# Population prediction vs data
#


### Predictions without encoder
vpc_with_encoder(global_max_value,global_max_dose,
    global_mean, 
    global_std,
    enable_ae_training=False,
    enable_nf_training=False,
    df_training=df,
    df=df,
    dataset=train_dataset_raw,
    latent_dim=latent_dim,
    dim_parameters=dim_parameter_encoder,
    initial_encoder=initial_encoder,
    encoder=encoder,
    func=func,
    reducer=reducer,
    noise=noise,
    ODEWrapper=ODEWrapper,
    t_dense=torch.linspace(0, 1, steps=240),
    compartment="C2",
    num_simulated_total=1000,
    add_noise_to_prediction=False
)

vpc_with_encoder(global_max_dose,
    global_mean, 
    global_std,
    enable_ae_training=False,
    enable_nf_training=False,
    df_training=df,
    df=df_test,
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
    add_noise_to_prediction=False
)

#rf_predict_params_from_encoder_validation(df, dataset_plot, df_test, dataset_test,encoder, latent_dim, dim_parameter_encoder,device=None, n_estimators=200, random_state=42)


vpc(global_max_dose,
    global_mean, 
    global_std,
    onlymedian=False,
    df_training=df,
    df=df,
    dataset=train_dataset_raw,
    latent_dim=latent_dim,
    dim_parameters=dim_parameter_encoder,
    initial_encoder=initial_encoder,
    func=func,
    reducer=reducer,
    noise=noise,
    ODEWrapper=ODEWrapper,
    t_dense=combined,
    compartment="C2",
    num_simulated_total=1000,
    add_noise_to_prediction=False
)




compute_residuals(global_mean, global_std,func,encoder,reducer,initial_encoder, noise,full_loader, combined, device)
  
plot_encoder_mu_vs_params(df, dataset_plot, encoder, latent_dim,dim_parameter_encoder, device=None)

#rf_predict_params_from_encoder_validation(df, dataset_plot, df_validation, dataset_validation,encoder, latent_dim, dim_parameter_encoder,device=None, n_estimators=200, random_state=42)
#rf_predict_params_from_encoderanddose_validation(df, dataset_plot, df_validation, dataset_validation,encoder, latent_dim, dim_parameter_encoder,device=None, n_estimators=200, random_state=42)


plot_individual_fits(global_max_dose,global_mean, global_std,False, False, test_dataset, latent_dim, noise,encoder, func, reducer, initial_encoder, ODEWrapper, combined,max_individuals=6, n_samples=100, device=device,truncation=0.6, add_noise=True)
