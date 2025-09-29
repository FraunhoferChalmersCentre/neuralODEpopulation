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

# # Load config
# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)




# ---- Training ---- 
if __name__ == "__main__":
        
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)
    
    # Simulate command line arguments in Spyder
    sys.argv = ['script_name',
                '--data_validation_path', 'lib/data/simulated_val_data8.csv', 
                '--data_test_path', 'lib/data/simulated_test_data5.csv', 
                '--data_path', 'lib/data/simulated_training_data_truncated.csv',
                '--save_dir', 'models',
                '--load_dir', 'models']
    
    from lib.utils.utils_preprocess import load_models, save_models, prepare_optimizer, prepare_datasets_and_loaders_simulated, compute_global_stats, export_all_metrics_and_residuals, append_metrics, load_all_metrics_and_residuals_as_lists, TrajectoryDataset, collate_fn
    from lib.utils.utils_training import   train_loop_model
    from lib.utils.utils_shared import ODEWrapper
    from lib.utils.utils_post_processing import plot_encoder_vs_samples, plot_encoder_histograms, generate_dose_percentiles2, build_dose_predictors_from_datasets, plot_individual_fits, plot_two_models_encoders_and_regression, vpc_2, vpc
    from lib.models.NNmodels import InitialConditionVAEEncoder, DoseClassifier, Encoder_Transformer_NF, ODEFunc, SimpleDecoder, InitialConditionEncoder, TrainableNoise
    
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
    dataset_train, dataset_val, dataset_test, train_base_dataset, train_loader, val_loader, test_loader, t_dense, batch_size_train, batch_size_val, batch_size_test = prepare_datasets_and_loaders_simulated(args.data_path, args.data_validation_path, args.data_test_path, global_max_dose, global_max_time, global_max_value,global_min_value, global_std, device, batch_fraction=0.1, trunctation=1,time_points=120)

    
    





    latent_dim=2
    dim_parameter_encoder=2
    hid_dim=512
 
    
 
    

    
    encoder_med = Encoder_Transformer_NF(dim_parameter_encoder, input_dim=2, model_dim=64,hidden_dim=64,hidden_flow_dim=16, num_heads=4, num_layers=2,num_flow_layers=3, dropout=0.1).to(device)
    func_med = ODEFunc(latent_dim,dim_parameter_encoder,hid_dim ).to(device)
    reducer_med = SimpleDecoder(latent_dim, hidden_dim=16).to(device)
    initial_encoder_med = InitialConditionEncoder(latent_dim, hidden_dim=8).to(device)
    noise_med = TrainableNoise(size=1, init_add_std=2, init_prop_std=0).to(device)
    #3.18
    models = {
    "func": func_med,
    "encoder": encoder_med,
    "reducer": reducer_med,
    "initial_encoder": initial_encoder_med,
    "noise": noise_med,
    # add any other models...
    } 
    
   # load_models(models, save_dir, "med")
   
    optimizer,scheduler, main_params = prepare_optimizer(models, device, lr=0.001)
 
    train_loop_model(
        p_dropout=1,
    func_med=func_med,
    reducer_med=reducer_med,
    initial_encoder_med=initial_encoder_med,
    encoder_med=encoder_med,
    dataset=dataset_train,
    dataset_val=dataset_val,
    global_max_time=global_max_time,
    global_max_dose=global_max_dose,
    global_max_value=global_max_value,
    global_min=global_min_value,
    main_params=main_params,
    dataloader_val=val_loader,
    dataloader=train_loader,
    models=models,
    optimizer=optimizer,
    scheduler=scheduler,
    func=func_med,
    reducer=reducer_med,
    initial_encoder=initial_encoder_med,
    encoder=encoder_med,
    noise=noise_med,
    t_dense=t_dense,
    n_epochs=1000,
    warmup_epochs_noise=1000,
    warmup_epochs_iiv=0,
    smoothing_start_epoch=1000,
    traing_against_validation=False,
    enable_ae=False,
    enable_vae=False,
    enable_nf=False,
    enable_onlymedian=True,
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
 
       
    vpc(func_med,
          reducer_med,
          initial_encoder_med,
          encoder_med,
          noise_med,models,
        train_loader,
        global_max_dose,
        global_max_time,
        global_max_value,
        global_min_value,
        dataset_train,
        latent_dim,
        dim_parameter_encoder,
        initial_encoder_med,
        encoder_med,
        func_med,
        reducer_med,
        noise_med,
        ODEWrapper,
        t_dense,
        compartment="DV",
        onlymedian=True,
        enable_nf=False,
        enable_ae=True,
        enable_vae=False,
        add_noise_to_prediction=False,
        num_simulated_total=1000,
        normalization=False,
        truncation=1
    )
    
    
  #  save_models(models, save_dir, "test1")
    encoder_ae = Encoder_Transformer_NF(dim_parameter_encoder, input_dim=2, model_dim=124,hidden_dim=64,hidden_flow_dim=16, num_heads=4, num_layers=2,num_flow_layers=3, dropout=0.1).to(device)
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
  #  save_models(models, save_dir, "ae")
    

   

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
       global_max_value=global_max_value,
       global_min=global_min_value,
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
       n_epochs=1000,
       warmup_epochs_noise=1000,
       warmup_epochs_iiv=0,
       smoothing_start_epoch=1000,
       traing_against_validation=False,
       enable_ae=True,
       enable_vae=False,
       enable_nf=False,
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
    

  #  save_models(models, save_dir, "ae")
    plot_encoder_histograms(encoder_ae, initial_encoder_ae, func_ae, t_dense, reducer_ae, global_max_value, global_min_value, dataset_train, encoder_ae, latent_dim, device=None, truncation=1, normalization=False)
    plot_encoder_vs_samples( encoder_med, initial_encoder_med, func_med,
     t_dense, reducer_med, global_max_value, global_min_value,
     dataset_train, encoder_ae, latent_dim, device=None, truncation=1, normalization=True)

    vpc(func_med,
              reducer_med,
              initial_encoder_med,
              encoder_med,
              noise_med,models,
            train_loader,
            global_max_dose,
            global_max_time,
            global_max_value,
            global_min_value,
            dataset_train,
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
            enable_nf=False,
            enable_ae=True,
            enable_vae=False,
            add_noise_to_prediction=False,
            num_simulated_total=1000,
            normalization=False,
            truncation=1
        )
 
    
    
    
 
    encoder_vae = Encoder_Transformer_NF(dim_parameter_encoder, input_dim=2, model_dim=124,hidden_dim=128,hidden_flow_dim=16, num_heads=4, num_layers=2,num_flow_layers=3, dropout=0.1).to(device)
    func_vae = ODEFunc(latent_dim,dim_parameter_encoder,hid_dim ).to(device)
    reducer_vae = SimpleDecoder(latent_dim, hidden_dim=16).to(device)
    initial_encoder_vae= InitialConditionEncoder(latent_dim, hidden_dim=8).to(device)
    noise_ae = TrainableNoise(size=1, init_add_std=2, init_prop_std=0).to(device)

    #3.18
    models = {
    "func": func_vae,
    "encoder": encoder_vae,
    "reducer": reducer_vae,
    "initial_encoder": initial_encoder_vae,
    "noise": noise_ae} 
    load_models(models, load_dir, "vae")
    
  
   
   
    import gc
    # Clear cached memory
    torch.cuda.empty_cache()    
    # If you want to forcefully release all tensors
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                del obj
        except:
            pass
    
  
    gc.collect()
    torch.cuda.empty_cache()

 
    optimizer,scheduler, main_params = prepare_optimizer(models, device, lr=0.001)
    
    
    train_loop_model( 0,     func_med,
          reducer_med,
          initial_encoder_med,
          encoder_med,
          dataset_train,
          dataset_val,
          global_max_time,
          global_max_dose,
          global_max_value,
          global_min,
          main_params,
          val_loader,
          train_loader,
          models,
          optimizer,
          scheduler,
          func_vae,
          reducer_vae,
          initial_encoder_vae,
          encoder_vae,
          noise_ae,
          t_dense,
          n_epochs=1000,
          warmup_epochs_noise=1000,
          warmup_epochs_iiv=0,
          smoothing_start_epoch=1000,
          traing_against_validation=False,
          enable_ae=False,
          enable_vae=True,
          enable_nf=False,
          enable_onlymedian=False,
          normalization=True,
          plot_training=True,
          free_bits=1,
          truncation=1,
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
            global_max_value,
            global_min,
            dataset_test,
            latent_dim,
            dim_parameter_encoder,
            initial_encoder_vae,
            encoder_vae,
            func_vae,
            reducer_vae,
            noise_ae,
            ODEWrapper,
            t_dense,
            compartment="DV",
            onlymedian=False,
            enable_nf=False,
            enable_ae=False,
            enable_vae=True,
            add_noise_to_prediction=False,
            num_simulated_total=1000,
            normalization=True,
            truncation=1
        )
    
    encoder_vae_norm = Encoder_Transformer_NF(dim_parameter_encoder, input_dim=2, model_dim=124,hidden_dim=64,hidden_flow_dim=16, num_heads=4, num_layers=2,num_flow_layers=3, dropout=0.1).to(device)
    func_vae_norm = ODEFunc(latent_dim,dim_parameter_encoder,hid_dim ).to(device)
    reducer_vae_norm = SimpleDecoder(latent_dim, hidden_dim=16).to(device)
    initial_encoder_vae_norm= InitialConditionEncoder(latent_dim, hidden_dim=8).to(device)
    #3.18
    models = {
    "func": func_vae_norm,
    "encoder": encoder_vae_norm,
    "reducer": reducer_vae_norm,
    "initial_encoder": initial_encoder_vae_norm,
    "noise": noise_ae} 
    # encoder_vae_norm.load_state_dict(encoder_vae.state_dict())
    # func_vae_norm.load_state_dict(func_vae.state_dict())
    # reducer_vae_norm.load_state_dict(reducer_vae.state_dict())
    # initial_encoder_vae_norm.load_state_dict(initial_encoder_vae.state_dict())
   
    optimizer,scheduler, main_params = prepare_optimizer(models, device, lr=0.001)
    
    
    
  

    train_loop_model(0,
      func_med,
      reducer_med,
      initial_encoder_med,
      encoder_med,
      dataset_train,
      dataset_val,
      global_max_time,
      global_max_dose,
      global_max_value,
      global_min,
      main_params,
      val_loader,
      train_loader,
      models,
      optimizer,
      scheduler,
      func_vae_norm,
      reducer_vae_norm,
      initial_encoder_vae_norm,
      encoder_vae_norm,
      noise_ae,
      t_dense,
      n_epochs=1000,
      warmup_epochs_noise=1000,
      warmup_epochs_iiv=0,
      smoothing_start_epoch=1000,
      traing_against_validation=False,
      enable_ae=False,
      enable_vae=True,
      enable_nf=False,
      enable_onlymedian=False,
      normalization=False,
      plot_training=True,
      free_bits=1,
      truncation=1,
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
        global_max_value,
        global_min,
        dataset_train,
        latent_dim,
        dim_parameter_encoder,
        initial_encoder_vae_norm,
        encoder_vae_norm,
        func_vae_norm,
        reducer_vae_norm,
        noise_ae,
        ODEWrapper,
        t_dense,
        compartment="DV",
        onlymedian=True,
        enable_nf=False,
        enable_ae=False,
        enable_vae=True,
        add_noise_to_prediction=False,
        num_simulated_total=10000,
        normalization=False,
        truncation=1
    )
    


    
    
    
    
    
    
    
    
    encoder_vae_norm2 = Encoder_Transformer_NF(dim_parameter_encoder, input_dim=2, model_dim=124,hidden_dim=64,hidden_flow_dim=16, num_heads=4, num_layers=2,num_flow_layers=3, dropout=0.1).to(device)
    func_vae_norm2 = ODEFunc(latent_dim,dim_parameter_encoder,hid_dim ).to(device)
    reducer_vae_norm2 = SimpleDecoder(latent_dim, hidden_dim=16).to(device)
    initial_encoder_vae_norm2= InitialConditionEncoder(latent_dim, hidden_dim=8).to(device)
    #3.18
    models = {
    "func": func_vae_norm2,
    "encoder": encoder_vae_norm2,
    "reducer": reducer_vae_norm2,
    "initial_encoder": initial_encoder_vae_norm2,
    "noise": noise_ae} 
    encoder_vae_norm2.load_state_dict(encoder_vae_norm.state_dict())
    func_vae_norm2.load_state_dict(func_vae_norm.state_dict())
    reducer_vae_norm2.load_state_dict(reducer_vae_norm.state_dict())
    initial_encoder_vae_norm2.load_state_dict(initial_encoder_vae_norm.state_dict())
   
    optimizer,scheduler, main_params = prepare_optimizer(models, device, lr=0.001)

    train_loop_model(0,
      func_vae_norm,
      reducer_vae_norm,
      initial_encoder_vae_norm,
      encoder_vae_norm,
      dataset_train,
      dataset_val,
      global_max_time,
      global_max_dose,
      global_max_value,
      global_min,
      main_params,
      val_loader,
      train_loader,
      models,
      optimizer,
      scheduler,
      func_vae_norm2,
      reducer_vae_norm2,
      initial_encoder_vae_norm2,
      encoder_vae_norm2,
      noise_ae,
      t_dense,
      n_epochs=1000,
      warmup_epochs_noise=1000,
      warmup_epochs_iiv=0,
      smoothing_start_epoch=1000,
      traing_against_validation=False,
      enable_ae=True,
      enable_vae=False,
      enable_nf=False,
      enable_onlymedian=False,
      normalization=True,
      plot_training=True,
      free_bits=1,
      truncation=1,
      print_epoch=1,
      plot_epoch=1,
      max_plots=4,
      nr_col=1,
      nr_row=5
  )
    
   


# -----------------------
# Post-training:
# -----------------------


vpc_2(
    # Median model
    encoder_med, initial_encoder_med, func_med, reducer_med, noise_med,
    # Model 1
    encoder_ae, initial_encoder_ae, func_ae, reducer_ae, noise_ae, models,
    # Model 2
    encoder_vae, initial_encoder_vae, func_vae, reducer_vae, noise_ae, models,
    test_loader, dataset_test, val_loader, dataset_val,t_dense, latent_dim, dim_parameter_encoder,
    global_max_time, global_max_dose, global_max_value, global_min,
    "DV", ODEWrapper,
    add_noise_to_prediction=False, num_simulated_total=1000,
    onlymedian=False, enable_nf=False, enable_ae=False, enable_vae=True,normalization=True,
    truncation=1
)



plot_two_models_encoders_and_regression(
    df, df_validation, dataset_train, dataset_val,
    encoder_ae, initial_encoder_ae, func_ae, reducer_ae,
    encoder_ae, initial_encoder_ae, func_ae, reducer_ae,
    encoder_ae, initial_encoder_ae, func_ae, reducer_ae,
    latent_dim, global_max_value, global_min, t_dense,
    device, enable_nf=False, enable_ae=True, enable_onlymedian=False, truncation=0.5, normalization=True,
    dim_parameter_encoder=2
)




build_dose_predictors_from_datasets(  dataset_train, dataset_val,
  encoder_vae_norm, initial_encoder_vae_norm, reducer_vae_norm, func_vae_norm,
  encoder_med, initial_encoder_med, reducer_med, func_med,
  global_max_time, global_max_dose, global_max_value, global_min,
  latent_dim, dim_parameter_encoder,t_dense,
  normalization=True, truncation=1, device=device)





generate_dose_percentiles2(
    dataset_train, t_dense, global_max_time, global_max_dose, global_max_value, global_min,
    encoder_med, initial_encoder_med, func_med, reducer_med,
    encoder_ae, latent_dim,
    batch_size=32,
    truncation=0.5,
    save_csv_path="dose_data.csv",
    device=None)

plot_encoder_histograms(dataset_train, encoder_ae, latent_dim, device=None, truncation=None)
