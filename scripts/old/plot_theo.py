import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file = "theophylline_data.csv"
df = pd.read_csv(file)

# Replace '.' with NaN and convert numeric columns properly
df = df.replace('.', pd.NA)
df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
df['CONC'] = pd.to_numeric(df['CONC'], errors='coerce')

# Drop rows without TIME or CONC values
df = df.dropna(subset=['TIME', 'CONC'])

# Plot all subjects
plt.figure(figsize=(8,6))

for subject_id, group in df.groupby('ID'):
    plt.plot(group['TIME'], group['CONC'], marker='o', linestyle='-', label=f"ID {subject_id}")

plt.xlabel("Time (hours)")
plt.ylabel("Concentration")
plt.title("Theophylline Concentration-Time Profiles")
plt.legend(title="Subject ID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
