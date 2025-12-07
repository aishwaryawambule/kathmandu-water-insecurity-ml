"""
Merge household data with aquifer cluster assignments
"""
import pandas as pd

# Load data
# We load the clustered aquifer data and the raw household data.
df_aq = pd.read_csv("../data/processed/kathmandu_aquifer_clustered.csv")
df_hh = pd.read_csv("../data/raw/kathmandu_household.csv")

# Merge household with aquifer cluster info
# We perform a left join on 'aquifer_id' to attach aquifer characteristics (including the cluster ID)
# to each household. This allows the model to learn the relationship between aquifer state and household water insecurity.
df_hh = df_hh.merge(
    df_aq[['aquifer_id', 'aquifer_cluster', 'water_table_depth_m', 
           'depletion_rate_m_per_year', 'tds_mg_per_L', 'nitrate_mg_per_L']], 
    on='aquifer_id', 
    how='left'
)

# Check for missing merges
# Ensure that every household was successfully linked to an aquifer well.
missing = df_hh['aquifer_cluster'].isna().sum()
if missing > 0:
    print(f"Warning: {missing} households without cluster assignment")

# Save merged dataset
# This final dataset contains all features needed for training the prediction model.
df_hh.to_csv("../data/processed/kathmandu_household_with_cluster.csv", index=False)
print(f"  Merged dataset saved: {len(df_hh)} households")
print(f"  Features: {len(df_hh.columns)} columns")
print(f"  Clusters: {df_hh['aquifer_cluster'].nunique()} unique clusters")
