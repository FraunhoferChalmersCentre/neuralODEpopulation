from lib.data.data_generation import simulate_2cpt_and_save
import argparse


parser = argparse.ArgumentParser(description="Simulate and save 2CPT data.")
parser.add_argument("--save_path", type=str, required=True, help="Path to save the simulated CSV file")
if __name__ == "__main__":
    args = parser.parse_args()

    simulate_2cpt_and_save(
        n_individuals=2,                    # Number of individuals per dose group
        dose_amounts=[50],                 # Dose amount(s)
        dose_times=[4, 8, 12],             # Dose times
        save_path=args.save_path,          # Now passed via command-line
        plot=True
    )
