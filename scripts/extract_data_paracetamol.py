import pandas as pd
import ast

# Load Excel/CSV file
df = pd.read_csv("timecourses.csv")  # change to read_excel if .xlsx

# Parse list fields
df["time"] = df["time"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
df["value"] = df["value"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

rows = []
for _, row in df.iterrows():
    indiv_id = int(row["individual_pk"]) if not pd.isna(row["individual_pk"]) else 1
    subset_id = int(row["subset_pk"]) if not pd.isna(row["subset_pk"]) else 0
    
    # combine both IDs into one unique ID
    combined_id = f"{indiv_id}_{subset_id}"
    
    times = row["time"]
    values = row["value"]

    # example: two doses at 0 and 3 hr
    dose_times = [3]

    # --- Add 3 extra measurements at times 0, 1, 2 with value 0 ---
    extra_times = [0, 1, 2]
    extra_values = [0, 0, 0]
    for t, v in zip(extra_times, extra_values):
        rows.append({
            "Occasion": combined_id,
            "ID": indiv_id,
            "Time": t,
            "C1": v,
            "C2": v * 10000,
            "Dose": 1,
            "ka": 0,
            "cl": 0,
            "Dose times": dose_times
        })

    # --- Shift original times by +3 ---
    for t, v in zip(times, values):
        rows.append({
            "Occasion": combined_id,
            "ID": indiv_id,
            "Time": t + 3,   # shift by 3
            "C1": v,
            "C2": v * 10000,
            "Dose": 1,
            "ka": 0,
            "cl": 0,
            "Dose times": dose_times
        })

# Build final dataframe
final_df = pd.DataFrame(rows)

# Save
final_df.to_csv("paracetamol_data2.csv", index=False)

print(final_df.head(20))
