import os
import argparse
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from lib.data.data_generation import *


def main():
    # Get base directory (where this script is executed from)
    base_dir = os.getcwd()
    

    # Set default save directory and path
    default_save_dir = os.path.join(base_dir, "lib", "data")
    default_save_path = os.path.join(default_save_dir, "simulated_data1.csv")

    parser = argparse.ArgumentParser(description="Simulate and save 2CPT PK data.")
    parser.add_argument(
        "--save_path",
        type=str,
        default=default_save_path,
        help="Path to save the simulated CSV file (default: lib/data/simulated_data1.csv)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Include this flag to show plots of simulated concentration profiles",
    )

    # ✔️ Parse known args only, ignore extra ones from Spyder/runcell
    args, _ = parser.parse_known_args()

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Run simulation
    simulate_2cpt_and_save_vectorized(
        n_individuals=50,
        add_e=2,
        prop_e=0.001,
        dose_amounts=[50,200],
        dose_times=[1, 5, 9, 13],
        save_path=args.save_path,
        plot=True,
    )

if __name__ == "__main__":
    main()
