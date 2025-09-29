# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 22:10:05 2025

@author: Baaz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------
# Load CSV
# -------------------------------
csv_path = "dose_data.csv"  # Adjust path if needed
df = pd.read_csv(csv_path)

# Ensure numeric columns
numeric_cols = ["time", "x_encoder_normalized", "dose", "prediction_used"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------------
# Optional: filter top 10% based on max of normalized value per individual per dose group
# -------------------------------
# Make sure there is an 'id' column identifying individuals
if "id" not in df.columns:
    df["id"] = df.index  # fallback if no ID exists

top_percentile_dfs = []
for dose_value, dose_df in df.groupby("dose"):
    # Compute max per individual
    max_per_individual = dose_df.groupby("id")["x_encoder_normalized"].max()
    threshold = np.percentile(max_per_individual, 99)  # top 10%
    # Keep only individuals whose max is above threshold
    top_ids = max_per_individual[max_per_individual >= threshold].index
    top_df = dose_df[dose_df["id"].isin(top_ids)]
    top_percentile_dfs.append(top_df)

df_top10 = pd.concat(top_percentile_dfs)

# -------------------------------
# Plot percentiles per dose group
# -------------------------------
def compute_percentiles(group, column):
    values = group[column].dropna()
    if len(values) == 0:
        return pd.Series({"perc10": np.nan, "median": np.nan, "perc90": np.nan})
    perc10 = np.percentile(values, 10)
    median = np.percentile(values, 50)
    perc90 = np.percentile(values, 90)
    return pd.Series({"perc10": perc10, "median": median, "perc90": perc90})

fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0, 1, df_top10["dose"].nunique()))

for i, dose_value in enumerate(sorted(df_top10["dose"].unique())):
    dose_df = df_top10[df_top10["dose"] == dose_value]
    percentiles_ratio = dose_df.groupby("time").apply(
        lambda g: compute_percentiles(g, "x_encoder_normalized")
    )
    label = f"{dose_value} mg"
    ax.plot(percentiles_ratio.index, percentiles_ratio["perc10"], linestyle='--', color=colors[i])
    ax.plot(percentiles_ratio.index, percentiles_ratio["median"], linestyle='-', color=colors[i], label=label)
    ax.plot(percentiles_ratio.index, percentiles_ratio["perc90"], linestyle='--', color=colors[i])

ax.set_title("Top 10% normalized values by dose group")
ax.set_xlabel("Time")
ax.set_ylabel("x_encoder_normalized")
ax.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# Classifier using normalized data
# -------------------------------
df_classifier = df_top10.dropna(subset=["x_encoder_normalized", "dose"])
X = df_classifier[["x_encoder_normalized"]]
y = df_classifier["dose"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
