import ast
import os
import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

def pad_dose_times(dose_times_list, pad_value=-1.0):
    batch_size = len(dose_times_list)
    max_len = max([dt.size(0) for dt in dose_times_list])
    
    dose_times_padded = torch.full((batch_size, max_len), pad_value, dtype=dose_times_list[0].dtype, device=dose_times_list[0].device)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=dose_times_list[0].device)
    
    for i, dt in enumerate(dose_times_list):
        length = dt.size(0)
        dose_times_padded[i, :length] = dt
        mask[i, :length] = 1
    
    return dose_times_padded, mask



def truncate_time_series(t_batch, x_batch, truncation_time):
    """
    Truncate a batch of time series to a specified cutoff time, and also return
    the values that were removed.

    Args:
        t_batch (list[torch.Tensor]): List of time tensors, each shaped [T_i].
        x_batch (list[torch.Tensor]): List of value tensors corresponding to t_batch, each shaped [T_i, ...].
        truncation_time (float): Time cutoff. All entries with time > cutoff are removed.

    Returns:
        tuple:
            truncated_t (list[torch.Tensor]): Truncated time tensors.
            truncated_x (list[torch.Tensor]): Truncated value tensors.
            masks (list[torch.BoolTensor]): Boolean masks indicating kept indices.
            removed_t (list[torch.Tensor]): Time tensors that were removed.
            removed_x (list[torch.Tensor]): Value tensors that were removed.
    """
    truncated_t, truncated_x, masks = [], [], []
    removed_t, removed_x = [], []

    for t_i, x_i in zip(t_batch, x_batch):
        # Boolean mask: True where time ≤ cutoff
        mask = t_i <= truncation_time

        # Keep only values within cutoff
        truncated_t.append(t_i[mask])
        truncated_x.append(x_i[mask])
        masks.append(mask)

        # Keep values that were removed (inverse mask)
        inv_mask = ~mask
        removed_t.append(t_i[inv_mask])
        removed_x.append(x_i[inv_mask])

    return truncated_t, truncated_x, masks, removed_t, removed_x



        
def export_training_data(test_dataset, train_dataset_raw, global_mean, global_std, global_max_time, i,truncation, base_dir):
    def flatten_samples(times, values, ids, censoring_flag=0, truncate_mask=None):
        """Flatten time series data into lists for DataFrame export."""
        flat_t, flat_x, flat_ids, flat_time0, censoring = [], [], [], [], []
        
        for t_i, x_i, id_i in zip(times, values, ids):
            if truncate_mask is not None:
                # Apply mask to keep only selected points
                mask = truncate_mask.pop(0)
                t_i, x_i = t_i[mask], x_i[mask]

            t_list, x_list = t_i.tolist(), x_i.tolist()
            
            flat_t.extend(t_list)
            flat_x.extend(x_list)
            flat_ids.extend([id_i] * len(t_list))
            # Mark dosing event (example: time==3 after scaling)
            flat_time0.extend([1 if np.round(t * global_max_time, 4) == 3 else 0 for t in t_list])
            censoring.extend([censoring_flag] * len(t_list))
        
        return flat_ids, flat_t, flat_x, flat_time0, censoring

    # --- TRAINING DATA ---
    train_t, train_x, train_ids = zip(*[(s[0], s[1], s[6]) for s in train_dataset_raw])
    ids_train, t_train, x_train, time0_train, censor_train = flatten_samples(train_t, train_x, train_ids)

    # --- TEST DATA (after truncation) ---
    test_t, test_x, test_ids = zip(*[(s[0], s[1], s[6]) for s in test_dataset])
    truncation_time = 0.6
    truncated_t, truncated_x, masks = truncate_time_series(test_t, test_x, truncation_time)
    ids_test, t_test, x_test, time0_test, censor_test = flatten_samples(truncated_t, truncated_x, test_ids)

    # --- DISCARDED (censored) TEST POINTS ---
    disc_t, disc_x, disc_ids, time0_disc, censor_disc = [], [], [], [], []
    for t_i, x_i, mask, id_i in zip(test_t, test_x, masks, test_ids):
        inv_mask = ~mask.bool()
        ids_d, t_d, x_d, time0_d, censor_d = flatten_samples(
            [t_i[inv_mask]], [x_i[inv_mask]], [id_i],
            censoring_flag=1
        )
        disc_ids.extend(ids_d)
        disc_t.extend(t_d)
        disc_x.extend(x_d)
        time0_disc.extend(time0_d)
        censor_disc.extend(censor_d)

    # --- DESTANDARDIZE + SCALE TIME ---
    def process_values(times, values):
        times = np.round(np.array(times) * global_max_time, 4)
        values = np.round(destandardize_concentration(np.array(values), global_mean, global_std), 4)
        return times, values

    time_train, dv_train = process_values(t_train, x_train)
    time_test, dv_test = process_values(t_test, x_test)
    time_disc, dv_disc = process_values(disc_t, disc_x)

    # --- EXPORT ---
    df_export = pd.DataFrame({
        'ID': ids_train + ids_test + disc_ids,
        'Time': np.concatenate([time_train, time_test, time_disc]),
        'DV': np.concatenate([dv_train, dv_test, dv_disc]),
        'Amount': time0_train + time0_test + time0_disc,
        'Censoring': censor_train + censor_test + censor_disc
    })

    os.makedirs(base_dir, exist_ok=True)
    filename = os.path.join(base_dir, f"monolix_data_{i}.csv")
   #df_export.to_csv(filename, index=False)




   
    






class TrajectoryDataset(Dataset):
    def __init__(
        self,
        path,
        compartment='DV',
        use_dose_normalization=True,
        augment_with_prefixes=False,
        augment_dose_times=False,
        dose_jitter_std=0.01,
        max_dose=None,
        max_time=None,
        conc_mean=None,
        conc_std=None,
        dose_max_abs_dict=None,
        subset_ids=None
    ):
  

        # === Step 1: Load data ===
        self.df = pd.read_csv(path, sep=";")
        self.compartment = compartment
        self.augment = augment_with_prefixes
        self.augment_dose_times = augment_dose_times
        self.dose_jitter_std = dose_jitter_std
        self.use_dose_normalization = use_dose_normalization

        # Replace '.' with NaN and convert numeric
        self.df.replace('.', pd.NA, inplace=True)
        for col in ['AMT', 'TIME', compartment]:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Convert DOSE TIME to list of floats
        if 'DOSE TIME' in self.df.columns:
            self.df['DOSE TIME'] = self.df['DOSE TIME'].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) else []
            )
        else:
            self.df['DOSE TIME'] = [[] for _ in range(len(self.df))]

        # Filter by subset_ids if provided
        if subset_ids is not None:
            self.df = self.df[self.df['ID'].isin(subset_ids)].reset_index(drop=True)

        # === Step 2: Global normalization ===
        self.max_dose = max_dose if max_dose is not None else self.df['AMT'].max()
        self.max_time = max_time if max_time is not None else self.df['TIME'].max()
        self.conc_mean = conc_mean if conc_mean is not None else self.df[compartment].mean()
        self.conc_std = conc_std if conc_std is not None else self.df[compartment].std()

        self.df['AMT_norm'] = self.df['AMT'] / self.max_dose
        self.df['TIME_norm'] = self.df['TIME'] / self.max_time
        self.df[f'{compartment}_norm'] = (self.df[compartment] - self.conc_mean) / self.conc_std

        # === Step 3: Optional dose normalization ===
        if self.use_dose_normalization:
            if dose_max_abs_dict is None:
                dose_group_median = {}
                for dose, df_dose in self.df.groupby('AMT'):
                    median_val = float(df_dose[compartment].median())
                    dose_group_median[dose] = median_val if median_val != 0 else 1.0
                self.dose_max_abs = dose_group_median
            else:
                self.dose_max_abs = dose_max_abs_dict

            self.df[f'{compartment}_dose_norm'] = self.df.apply(
                lambda row: row[compartment] / (self.dose_max_abs[row['AMT']] + 1e-8),
                axis=1
            )

        # === Step 4: Extract trajectories ===
        self.trajectories = self._extract_trajectories()

        # === Step 5: Generate augmented samples ===
        self.samples = self._generate_samples(self.trajectories)

    def _extract_trajectories(self):
        trajectories = []
    
        for subject_id, group in self.df.groupby('ID'):
            t = torch.tensor(group['TIME_norm'].values, dtype=torch.float32)
            x_global = torch.tensor(group[f'{self.compartment}_norm'].values, dtype=torch.float32)
    
            if self.use_dose_normalization:
                x_dose = torch.tensor(group[f'{self.compartment}_dose_norm'].values, dtype=torch.float32)
            else:
                x_dose = None
    
            amt_val = group['AMT'].dropna().values
            amt = torch.tensor(amt_val[0], dtype=torch.float32) / self.max_dose if len(amt_val) > 0 else torch.tensor(0.0)
    
            dose_times_list = group['DOSE TIME'].values[0] if len(group['DOSE TIME'].values) > 0 else []
            dose_times = torch.tensor(dose_times_list, dtype=torch.float32) / self.max_time
    
            # === Compute MEAN of raw (unnormalized) concentration values ===
            mean_val = float(group[self.compartment].fillna(0).mean())
    
            trajectories.append({
                't': t,
                'x_global': x_global,
                'amt': amt,
                'dose_times': dose_times,
                'subject_id': subject_id,
                'x_dose': x_dose,
                'auc': torch.tensor(mean_val, dtype=torch.float32)  # <-- replace auc with mean
            })
        return trajectories
    
    
    def _generate_samples(self, trajectories):
        augmented = []
        for traj in trajectories:
            t = traj['t']
            x_global = traj['x_global']
            x_dose = traj['x_dose']
            amt = traj['amt']
            dose_times = traj['dose_times']
            auc = traj['auc']
            subject_id_str = str(traj['subject_id'])
    
            # Original sample
            sample = {
                't': t,
                'x_global': x_global,
                'amt': amt,
                'dose_times': dose_times,
                'subject_id': subject_id_str,
                'auc': auc
            }
            if self.use_dose_normalization:
                sample['x_dose'] = x_dose
            augmented.append(sample)
    
            # Optional prefix/suffix augmentation
            if self.augment:
                T = len(t)
                for end in range(1, T):
                    sample_end = {
                        't': t[:end],
                        'x_global': x_global[:end],
                        'amt': amt,
                        'dose_times': dose_times,
                        'subject_id': subject_id_str + f"_prefix{end}",
                        'auc': auc
                    }
                    if self.use_dose_normalization:
                        sample_end['x_dose'] = x_dose[:end]
                    augmented.append(sample_end)
    
                for start in range(1, T - 1):
                    sample_start = {
                        't': t[start:],
                        'x_global': x_global[start:],
                        'amt': amt,
                        'dose_times': dose_times,
                        'subject_id': subject_id_str + f"_suffix{start}",
                        'auc': auc
                    }
                    if self.use_dose_normalization:
                        sample_start['x_dose'] = x_dose[start:]
                    augmented.append(sample_start)
        return augmented

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
















def collate_fn(batch):
    t_list = [s['t'] for s in batch]
    x_global_list = [s['x_global'] for s in batch]
    dose_list = [s['amt'] for s in batch]
    dose_times_list = [s['dose_times'] for s in batch]
    id_list = [s['subject_id'] for s in batch]
    auc_list = [s['auc'] for s in batch]

    # Check if dose-normalized values exist
    x_dose_list = [s['x_dose'] for s in batch] if 'x_dose' in batch[0] else None

    # Pad time & x_global
    t_padded = pad_sequence(t_list, batch_first=True)
    x_global_padded = pad_sequence(x_global_list, batch_first=True)

    # Build mask
    max_len = t_padded.size(1)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, t in enumerate(t_list):
        mask[i, :len(t)] = 1

    dose_tensor = torch.stack(dose_list)
    auc_tensor = torch.stack(auc_list)

    if x_dose_list is not None:
        x_dose_padded = pad_sequence(x_dose_list, batch_first=True)
    else:
        x_dose_padded = None

    return id_list, t_padded, x_global_padded, mask, dose_tensor, dose_times_list, x_dose_padded, auc_tensor













def prepare_optimizer(models,device, lr=0.001):
    main_params = [
        {"params": list(models["func"].parameters()) +
                   list(models["reducer"].parameters()) +
                   list(models["initial_encoder"].parameters()) +
                   list(models["encoder"].parameters()), "lr": lr},
        {"params": list(models["noise"].parameters()), "lr": 10*lr}
    ]
    optimizer = torch.optim.Adam(main_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=7, min_lr=1e-4
    )
    
    for name, model in models.items():
          model.to(device)
          print(f"{name} is on {next(model.parameters()).device}")
    
    return optimizer, scheduler, main_params


def prepare_datasets_and_loaders(data_path,base_dir, all_ids, i,already_done, global_max_dose, global_max_time,
                                 global_mean, global_std, device, batch_fraction=0.05,truncation=1):

    """
    Splits the dataset into train/val/test, creates DataLoaders, and generates the combined time+dose tensor.

    Returns:
        train_dataset, val_dataset, test_dataset, train_loader, val_loader, combined
    """


    n_ids = len(all_ids)
    random.shuffle(all_ids)

   # all_ids = list(range(n_ids))

    test_frac = 0.3
    val_frac = 0.1
    train_frac = 1 - test_frac - val_frac  # 0.6
    
    # Compute exact counts
    n_test = 3 # int(12 * 0.3)   # 3
    n_val  = 2 #int(12 * 0.1)   # 1
    n_train =  7 #12 - n_test - n_val  # 8
    
    # Split IDs
    train_ids = all_ids[:n_train]                 # first 7
    val_ids   = all_ids[n_train:n_train + n_val] # next 2
    test_ids  = all_ids[n_train + n_val:]        # last 3

    train_dataset = TrajectoryDataset(
    data_path,  # <--- pass path, not df
    augment_with_prefixes=False,
    augment_dose_times=False,
    max_dose=global_max_dose,
    max_time=global_max_time,
    conc_mean=global_mean,
    conc_std=global_std,
    subset_ids=train_ids
    )
    
    
    train_export_dataset = TrajectoryDataset(
    data_path,  # <--- pass path, not df
    augment_with_prefixes=False,
    augment_dose_times=False,
    max_dose=global_max_dose,
    max_time=global_max_time,
    conc_mean=global_mean,
    conc_std=global_std,
    subset_ids=train_ids
    )
    
    val_dataset = TrajectoryDataset(
        data_path,
        augment_with_prefixes=False,
        augment_dose_times=False,
        max_dose=global_max_dose,
        max_time=global_max_time,
        conc_mean=global_mean,
        conc_std=global_std,
        subset_ids=val_ids
    )
    test_dataset = TrajectoryDataset(
        data_path,
        augment_with_prefixes=False,
        augment_dose_times=False,
        max_dose=global_max_dose,
        max_time=global_max_time,
        conc_mean=global_mean,
        conc_std=global_std,
        subset_ids=test_ids
    )


  #  export_training_data(test_dataset, train_export_dataset, global_mean, global_std, global_max_time, i+already_done, truncation, base_dir)

    # Create DataLoaders
    batch_size = max(1, int(len(train_dataset) * batch_fraction))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
    val_loader   = DataLoader(val_dataset, batch_size=max(1, int(len(val_dataset) * batch_fraction)), shuffle=False, collate_fn=collate_fn, drop_last=False)
    test_loader  = DataLoader(test_dataset, batch_size=max(1, int(len(test_dataset) * batch_fraction)), shuffle=False, collate_fn=collate_fn, drop_last=False)

    # Combined dose + time tensor
    #doses = torch.tensor([3], dtype=torch.float32) / 24
    time_points = torch.linspace(0, 1, steps=120)
  

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader,test_loader, time_points

def prepare_datasets_and_loaders_simulated(data_path_train, data_path_val, data_path_test,
                                        
                                           global_max_dose, global_max_time,
                                           global_mean, global_std, device,
                                           batch_fraction=0.05, trunctation=1,time_points=120):
    """
    Loads train/val/test datasets from CSV paths, creates DataLoaders, and generates
    the combined time+dose tensor.

    Returns:
        train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, combined
    """

    # --- Load datasets ---
    train_dataset = TrajectoryDataset(
        data_path_train,
        augment_with_prefixes=False,
        augment_dose_times=False,
        max_dose=global_max_dose,
        max_time=global_max_time,
        conc_mean=global_mean,
        conc_std=global_std
    )
    train_base_dataset = TrajectoryDataset(
        data_path_train,
        augment_with_prefixes=False,
        augment_dose_times=False,
        max_dose=global_max_dose,
        max_time=global_max_time,
        conc_mean=global_mean,
        conc_std=global_std
    )

    val_dataset = TrajectoryDataset(
        data_path_val,
        augment_with_prefixes=False,
        augment_dose_times=False,
        max_dose=global_max_dose,
        max_time=global_max_time,
        conc_mean=global_mean,
        conc_std=global_std
    )

    test_dataset = TrajectoryDataset(
        data_path_test,
        augment_with_prefixes=False,
        augment_dose_times=False,
        max_dose=global_max_dose,
        max_time=global_max_time,
        conc_mean=global_mean,
        conc_std=global_std
    )



    # --- Create DataLoaders ---
    batch_size_train = max(1, int(len(train_dataset) * batch_fraction))
    batch_size_val = max(1, int(len(val_dataset)))
    batch_size_test = max(1, int(len(test_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, collate_fn=collate_fn)

    # --- Combined dose + time tensor ---
    df = pd.read_csv(data_path_train, sep=';')
    dose_times_lists = df['DOSE TIME'].apply(ast.literal_eval).tolist()
    all_times = np.concatenate(dose_times_lists)
    unique_times_np = np.unique(all_times)
   

   # doses = torch.tensor([3], dtype=torch.float32) / 24
    time_points = torch.linspace(0, 1, steps=time_points)
    dose_times_tensor = torch.from_numpy(unique_times_np).float().to(time_points.device) / global_max_time
    combined = torch.cat((time_points, dose_times_tensor))
    combined, _ = torch.sort(combined)  # ensure ascending order

    return train_dataset, val_dataset, test_dataset,train_base_dataset,  train_loader, val_loader, test_loader, combined, batch_size_train, batch_size_val, batch_size_test


def compute_global_stats(df):
    """
    Compute global statistics for the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame with columns 'Dose', 'Time', 'C2'
    
    Returns:
        global_max_dose, global_max_time, global_mean, global_std, global_max_value
    """


    for col in ['AMT', 'DV']:
        df[col] = pd.to_numeric(df[col].replace('.', pd.NA), errors='coerce')
    
    # Replace '.' with NaN and convert to numeric

    global_max_dose = df['AMT'].max()
    global_max_time = df['TIME'].max()
    global_mean = df['DV'].mean()
    global_std = df['DV'].std()
    global_max_value = df['DV'].max()
    
    return global_max_dose, global_max_time, global_mean, global_std, global_max_value



def export_all_metrics_and_residuals(metrics, residuals, base_dir, variants=["", "_ae", "_ae_noise"]):
    os.makedirs(base_dir, exist_ok=True)
    
    for var in variants:
        # Export metrics
        metrics_df = pd.DataFrame(metrics[var])
        metrics_file = os.path.join(base_dir, f"metrics_raw{var}.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Saved metrics: {metrics_file}")

        # Export residuals
        residuals_file = os.path.join(base_dir, f"residuals_NODE{var}.csv")
        if residuals[var]:  # only concat if list is non-empty
            res_all = pd.concat(residuals[var], ignore_index=True)
            res_all.to_csv(residuals_file, index=False)
        else:
            # create empty CSV with standard columns if no residuals yet
            res_all = pd.DataFrame(columns=["Prediction", "Observation", "Residual", "Iteration", "ID"])
            res_all.to_csv(residuals_file, index=False)
        print(f"Saved residuals: {residuals_file}")
        
        
        
def append_metrics(metrics,residuals, variant, mse_mean, r2_mean, mse_median, r2_median, mse_validation, res_df):
    """
    Append metrics and residuals safely for a given variant.
    """
    metrics[variant]["mse_mean"].append(mse_mean)
    metrics[variant]["r2_mean"].append(r2_mean)
    metrics[variant]["mse_median"].append(mse_median)
    metrics[variant]["r2_median"].append(r2_median)
    metrics[variant]["mse_validation"].append(mse_validation)
    residuals[variant].append(res_df)


def load_all_metrics_and_residuals_as_lists(base_dir, variants=["", "_ae", "_ae_noise"]):
    """
    Loads metrics and residuals CSVs for all specified variants,
    and converts all metrics to Python lists so they can be appended.

    Returns:
        metrics: dict of dicts (all lists)
        residuals: dict of lists
        already_done: int, max length of mse_mean lists across variants
    """
    from lib.utils.my_utils import load_existing_metrics, load_existing_residuals

    metrics = {}
    residuals = {}

    ensure_result_files(base_dir)  # make sure all files exist

    for var in variants:
        metrics_file = os.path.join(base_dir, f"metrics_raw{var}.csv")
        loaded = load_existing_metrics(metrics_file, 
                                       ["mse_mean","r2_mean","mse_median","r2_median","mse_validation"])
        # Convert each series/list to pure Python list
        metrics[var] = {k: list(v) for k, v in loaded.items()}

        residuals_file = os.path.join(base_dir, f"residuals_NODE{var}.csv")
        res = load_existing_residuals(residuals_file)
        residuals[var] = list(res)  # ensure list

    # Compute already_done
    already_done = max(len(metrics[var]["mse_mean"]) for var in variants)

    return metrics, residuals, already_done







def ensure_result_files(base_dir: str):
    os.makedirs(base_dir, exist_ok=True)
    print(f"[INFO] Base dir: {os.path.abspath(base_dir)}")

    files_and_headers = {
        "metrics_raw.csv": ["mse_mean","r2_mean","mse_median","r2_median","mse_validation"],
        "residuals_NODE.csv": ["id","time","residual"],
        "metrics_raw_ae.csv": ["mse_mean","r2_mean","mse_median","r2_median","mse_validation"],
        "residuals_NODE_ae.csv": ["id","time","residual"],
        "metrics_raw_ae_noise.csv": ["mse_mean","r2_mean","mse_median","r2_median","mse_validation"],
        "residuals_NODE_ae_noise.csv": ["id","time","residual"],
    }

    for fname, headers in files_and_headers.items():
        path = os.path.join(base_dir, fname)
        if not os.path.exists(path):
            print(f"[INFO] Creating {path}")
            pd.DataFrame(columns=headers).to_csv(path, index=False)
        else:
            print(f"[INFO] Already exists: {path}")

def load_existing_residuals(filepath):
    """Load residuals from a CSV if it exists, else return an empty list."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return [df]   # we keep as list to match append/concat later
    else:
        return []
    
def load_existing_metrics(filepath, expected_columns):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Ensure expected columns exist
        for col in expected_columns:
            if col not in df.columns:
                raise ValueError(f"Missing column {col} in {filepath}")
        return {col: df[col].tolist() for col in expected_columns}
    else:
        return {col: [] for col in expected_columns}



def standardize_concentration(conc, mean, std): return (conc - mean) / std
def destandardize_concentration(norm_conc, mean, std): return norm_conc * std + mean




def save_models(models: dict, save_dir: str, model_name: str):
    """
    Save multiple models to a given directory.

    Args:
        models (dict): Dictionary with model names as keys and model instances as values.
        save_dir (str): Directory to save the models.
        model_name (str): Base name to prepend to each saved model file.
    """
    os.makedirs(save_dir, exist_ok=True)
    for key, model in models.items():
        path = os.path.join(save_dir, f"{model_name}_{key}.pt")
        torch.save(model.state_dict(), path)
    print(f"Saved {len(models)} models to {save_dir}")

def load_models(models: dict, load_dir: str, model_name: str, device=torch.device("cpu")):
    """
    Load model states into existing model instances from a directory.

    Args:
        models (dict): Dictionary with model names as keys and initialized model instances as values.
        load_dir (str): Directory where models are saved.
        model_name (str): Base name prepended to each saved model file.
        device (torch.device): Device to map the loaded model parameters.

    Returns:
        dict: Dictionary with loaded model instances.
    """
    for key, model in models.items():
        path = os.path.join(load_dir, f"{model_name}_{key}.pt")
        if os.path.isfile(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            print(f"Loaded {key} from {path}")
        else:
            print(f"Warning: Model file {path} not found. Skipping load for {key}.")
    
   # models=models.to(device)
    return models    



