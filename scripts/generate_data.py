import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from lib.utils.utils_data_generation import simulate_tumor_volume, simulate_single_drug_concentration

def main():
    base_dir = os.getcwd()
    default_save_dir = os.path.join(base_dir, "lib", "data")
    default_save_path = os.path.join(default_save_dir, "tumor_data_test.csv")
    os.makedirs(default_save_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Simulate tumor volume under drug treatments.")
    parser.add_argument(
        "--save_path",
        type=str,
        default=default_save_path,
        help="Path to save the simulated CSV file",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Include this flag to show plots of tumor volume",
    )
    args, _ = parser.parse_known_args()

    # ----- Define PK/PD parameters (same for all groups) -----
    ka_mean   = [0.6, 0.5]      # Drug A, Drug B
    ke_mean   = [0.6, 0.4]
    v_mean    = [0, 0]
    ka_sd     = [0, 0]
    ke_sd     = [0, 0]
    v_sd      = [0, 0]
    a_drugs   = [0.0003, 0.0012]     # Effect coefficients for each drug
    add_e     = 5
    prop_e    = 0.0001
    
    
    # ----- Define tumor parameters (mean and std) -----
    k_growth_mean = 0.1
    k_growth_sd   = 0.01
    V0_mean       = 100.0
    V0_sd         = 10.0
    n=100

    # ----- Define groups: only number of individuals and dosing info -----
    groups = [
        # Group 1: Drug A only
        {
            'n_individuals': n,
            'dose_amounts_list': [[0, 0, 0, 0], [0, 0, 0, 0]],
            'dose_times_list':   [[0, 0, 0, 0], [0, 0, 0, 0]]
        },
        
        {
            'n_individuals': n,
            'dose_amounts_list': [[200, 200, 200, 200], [0, 0, 0, 0]],
            'dose_times_list':   [[0, 5, 10, 15], [0, 0, 0, 0]]
        },
        # Group 2: Drug B only
        {
            'n_individuals': n,
            'dose_amounts_list': [[0, 0, 0, 0], [150, 150, 150, 150]],
            'dose_times_list':   [[0, 0, 0, 0], [1, 6, 11, 16]]
        },
        
        {
            'n_individuals': n,
            'dose_amounts_list': [[200, 200, 200, 200], [150, 150, 150, 150]],
            'dose_times_list':   [[0, 5, 10, 15], [1, 6, 11, 16]]
        }
    ]

    combined_data = []
    id_offset = 0

    for treatment_idx, group in enumerate(groups, start=1):
        temp_save_path = os.path.join(default_save_dir, f"temp_group_{treatment_idx}.csv")
    
        simulate_tumor_volume(
            n_individuals=group['n_individuals'],
            dose_amounts_list=group['dose_amounts_list'],
            dose_times_list=group['dose_times_list'],
            a_drugs=a_drugs,
            add_e=add_e,
            prop_e=prop_e,
            save_path=temp_save_path,
            plot=True,
            t_interval=(0, 16),
            sample_frequency=0.5,
            ka_mean=ka_mean,
            ke_mean=ke_mean,
            v_mean=v_mean,
            ka_sd=ka_sd,
            ke_sd=ke_sd,
            v_sd=v_sd,
            k_growth_mean=k_growth_mean,
            k_growth_sd=k_growth_sd,
            V0_mean=V0_mean,
            V0_sd=V0_sd
        )

        # Load simulated data
        df_group = pd.read_csv(temp_save_path, sep=';')
        df_group.columns = df_group.columns.str.strip().str.upper()

        # Update IDs
        df_group['ID'] += id_offset
        id_offset = df_group['ID'].max()

        # Set treatment column
        df_group['TREATMENT'] = treatment_idx

        combined_data.append(df_group)

    # Combine all groups and save final CSV
    df_all = pd.concat(combined_data, ignore_index=True)
    df_all = df_all.sort_values(['ID', 'TIME'])
    df_all.to_csv(args.save_path, index=False, sep=';')
    print(f"Saved combined dataset with {len(df_all)} rows to {args.save_path}")


if __name__ == "__main__":
    main()
# %%
# single_drug_simulation.py
import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from lib.utils.utils_data_generation import simulate_single_drug_concentration

def main():
    base_dir = os.getcwd()
    default_save_dir = os.path.join(base_dir, "lib", "data")
    default_save_path = os.path.join(default_save_dir, "Simulated_ODE3_corr_test.csv")
    os.makedirs(default_save_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Simulate single-drug concentration for 2 groups.")
    parser.add_argument(
        "--save_path",
        type=str,
        default=default_save_path,
        help="Path to save the simulated CSV file",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Include this flag to show concentration plots",
    )
    args, _ = parser.parse_known_args()

   #  # ----- Define dose times -----
   
    
   # ----- Define groups -----
    # dose_times = [0, 8, 16, 24]
    # groups = [
    #     {"name": "Low Dose",  "n_individuals": 100, "dose_amounts": [400, 400, 400,400]},
    #     {"name": "High Dose",  "n_individuals": 100, "dose_amounts": [800, 800, 800,800]}
    # ]
    
    dose_times = [0, 3, 8]
    groups = [
        {"name": "Dose 200",  "n_individuals": 100, "dose_amounts": [350, 350, 350]},
        {"name": "Dose 600", "n_individuals": 100, "dose_amounts": [600, 600, 600]},
        {"name": "Dose 400",  "n_individuals": 100, "dose_amounts": [400, 400, 400]},
        {"name": "Dose 800", "n_individuals": 100, "dose_amounts": [800, 800, 800]},
        {"name": "Dose 1000",  "n_individuals": 100, "dose_amounts": [850, 850, 850]}
    ]


    combined_data = []
    id_offset = 0
    corr_matrix = np.array([
        [1.0, 0.3, 0.2], #ka, ke, v
        [0.3, 1.0, 0.15],
        [0.2, 0.15, 1.0]
    ])
    
    
    
    for group_idx, group in enumerate(groups, start=1):
        temp_save_path = os.path.join(default_save_dir, f"temp_group_{group_idx}.csv")
    
        df_group = simulate_single_drug_concentration(
            n_individuals=group["n_individuals"],
            dose_amounts=group["dose_amounts"],
            dose_times=dose_times,
            ka_mean=0.4, ke_mean=0.6, v_mean=50,
            ka_sd=0.5, ke_sd=0.5, v_sd=0.5,
            add_e=5, prop_e=0.0001,
            t_interval=(0,48), sample_frequency=0.5,
            save_path=temp_save_path,
            plot=args.plot,
            corr_matrix=corr_matrix  # <-- Add correlation here

        )

        df_group.columns = df_group.columns.str.strip().str.upper()
        df_group["ID"] += id_offset
        id_offset = df_group["ID"].max()
        df_group["TREATMENT"] = group_idx
        combined_data.append(df_group)

    # Combine all groups and save final CSV
    df_all = pd.concat(combined_data, ignore_index=True)
    df_all = df_all.sort_values(["ID","TIME"])
    df_all.to_csv(args.save_path, index=False, sep=";")
    print(f"Saved single-drug concentration dataset with {len(df_all)} rows to {args.save_path}")


if __name__ == "__main__":
    main()
# %%

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from lib.utils.utils_data_generation import simulate_tumor_volume_with_event

def main():
    base_dir = os.getcwd()
    default_save_dir = os.path.join(base_dir, "lib", "data")
    default_save_path = os.path.join(default_save_dir, "tumor_data.csv")
    os.makedirs(default_save_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Simulate tumor volume and time-to-event data under drug treatments.")
    parser.add_argument(
        "--save_path",
        type=str,
        default=default_save_path,
        help="Path to save the simulated CSV file",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Include this flag to show plots of tumor volume",
    )
    args, _ = parser.parse_known_args()

    # ----- PK/PD parameters -----
    a_drugs = [0.0005]     # Effect coefficients for each drug
    add_e = 0.00001
    prop_e = 0.00001
    ka_mean   = [0.6]
    ke_mean   = [0.6]
    v_mean    = [0]
    ka_sd     = [0]
    ke_sd     = [0]
    v_sd      = [0]

    # ----- Tumor parameters -----
    k_growth_mean = 0.1
    k_growth_sd   = 0.05
    V0_mean       = 100.0
    V0_sd         = 0.1

    # ----- Event hazard parameters -----
    alpha = 0.001
    beta = 0.001

    # ----- Groups: individuals and dosing -----
    groups = [
    {
        'n_individuals': 1000,
        'dose_amounts_list': [[200, 200, 200, 200]],  # one inner list for the single drug
        'dose_times_list': [[1, 6, 11, 16]]           # one inner list for the single drug
    }
]


    combined_tumor_data = []
    combined_tte_data = []
    id_offset = 0

    for treatment_idx, group in enumerate(groups, start=1):
        temp_save_path = os.path.join(default_save_dir, f"temp_group_{treatment_idx}.csv")

        simulate_tumor_volume_with_event(
            n_individuals=group['n_individuals'],
            dose_amounts_list=group['dose_amounts_list'],
            dose_times_list=group['dose_times_list'],
            a_drugs=a_drugs,
            alpha=alpha,
            beta=beta,
            add_e=add_e,
            prop_e=prop_e,
            save_path=temp_save_path,
            t_interval=(0, 16),
            sample_frequency=0.1,
            ka_mean=ka_mean,
            ke_mean=ke_mean,
            v_mean=v_mean,
            ka_sd=ka_sd,
            ke_sd=ke_sd,
            v_sd=v_sd,
            k_growth_mean=k_growth_mean,
            k_growth_sd=k_growth_sd,
            V0_mean=V0_mean,
            V0_sd=V0_sd,
            max_tumor_size=2000,
            plot=False
        
        )

        # Load simulated tumor data
        df_tumor = pd.read_csv(temp_save_path, sep=';')
        df_tumor.columns = df_tumor.columns.str.strip().str.upper()
        df_tumor['ID'] += id_offset
        df_tumor['TREATMENT'] = treatment_idx
        combined_tumor_data.append(df_tumor)

        # Load simulated TTE data
        tte_path = os.path.join(os.path.dirname(temp_save_path), "tte_" + os.path.basename(temp_save_path))
        df_tte = pd.read_csv(tte_path, sep=';')
        df_tte.columns = df_tte.columns.str.strip().str.upper()
        df_tte['ID'] += id_offset
        df_tte['TREATMENT'] = treatment_idx
        combined_tte_data.append(df_tte)

        id_offset = df_tumor['ID'].max()

    # Combine all groups and save
    df_all_tumor = pd.concat(combined_tumor_data, ignore_index=True).sort_values(['ID','TIME'])
    df_all_tumor.to_csv(args.save_path, index=False, sep=';')
    print(f"Saved combined tumor volume dataset to {args.save_path}")

    tte_save_path = os.path.splitext(args.save_path)[0] + "_tte.csv"
    df_all_tte = pd.concat(combined_tte_data, ignore_index=True).sort_values(['ID'])
    df_all_tte.to_csv(tte_save_path, index=False, sep=';')
    print(f"Saved combined time-to-event dataset to {tte_save_path}")

if __name__ == "__main__":
    main()
