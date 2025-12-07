"""
Synthetic data generation for Kathmandu Valley water insecurity study.
Generates realistic aquifer and household datasets based on hydrogeological patterns.
Part of: Predicting Household Water Insecurity Due to Urban Groundwater Depletion
"""
import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Ensure directories exist
os.makedirs('../data/raw', exist_ok=True)
os.makedirs('../data/processed', exist_ok=True)

print("=== Generating Synthetic Data for Kathmandu Valley ===\n")

# ============================================================================
# PART 1: AQUIFER DATA GENERATION
# ============================================================================
print("Generating aquifer dataset...")

# Number of aquifer wells to generate
n_aquifers = 500

# Kathmandu Valley geographic bounds (approximate)
lat_min, lat_max = 27.60, 27.80
lon_min, lon_max = 85.20, 85.46

# Generate aquifer data
aquifer_data = {
    'aquifer_id': [f'AQ{i:04d}' for i in range(1, n_aquifers + 1)],
    'lat': np.random.uniform(lat_min, lat_max, n_aquifers),
    'lon': np.random.uniform(lon_min, lon_max, n_aquifers),
    'district': np.random.choice(['Kathmandu', 'Lalitpur', 'Bhaktapur'], n_aquifers, p=[0.5, 0.3, 0.2]),
    'well_type': np.random.choice(['shallow_tube', 'deep_tube', 'dug'], n_aquifers, p=[0.4, 0.4, 0.2]),
    'season': np.random.choice(['wet', 'dry'], n_aquifers, p=[0.5, 0.5]),
    'land_use': np.random.choice(['urban', 'peri-urban', 'agriculture'], n_aquifers, p=[0.7, 0.2, 0.1])
}

# Generate well depths based on well type
well_depths = []
for well_type in aquifer_data['well_type']:
    if well_type == 'dug':
        depth = np.random.uniform(3, 10)
    elif well_type == 'shallow_tube':
        depth = np.random.uniform(8, 100)
    else:  # deep_tube
        depth = np.random.uniform(100, 220)
    well_depths.append(depth)

aquifer_data['well_depth_m'] = well_depths

# Generate water table depth (shallower wells have shallower water tables)
water_table_depths = []
for i, well_type in enumerate(aquifer_data['well_type']):
    if well_type == 'dug':
        wtd = np.random.uniform(0.2, 10)
    elif well_type == 'shallow_tube':
        wtd = np.random.uniform(2, 25)
    else:  # deep_tube
        wtd = np.random.uniform(4, 60)
    water_table_depths.append(wtd)

aquifer_data['water_table_depth_m'] = water_table_depths

# Aquifer thickness (geological layer)
aquifer_data['aquifer_thickness_m'] = np.random.uniform(5, 50, n_aquifers)

# Transmissivity (m²/day) - higher for shallow aquifers
transmissivity = []
for thickness in aquifer_data['aquifer_thickness_m']:
    # Transmissivity correlates with thickness
    base = np.random.uniform(30, 1000)
    trans = base * (thickness / 30)
    transmissivity.append(trans)

aquifer_data['transmissivity_m2_per_day'] = transmissivity

# Hydraulic conductivity (m/day)
aquifer_data['hydraulic_conductivity_m_per_day'] = [
    t / thick for t, thick in zip(aquifer_data['transmissivity_m2_per_day'], 
                                   aquifer_data['aquifer_thickness_m'])
]

# Water quality parameters
# TDS (Total Dissolved Solids) - higher in urban areas and deeper wells
tds_values = []
for i, (land_use, wtd) in enumerate(zip(aquifer_data['land_use'], aquifer_data['water_table_depth_m'])):
    base_tds = 200 if land_use == 'peri-urban' else 400
    # Add depth factor and random variation
    tds = base_tds + (wtd * 10) + np.random.normal(0, 200)
    tds = max(150, min(2000, tds))  # Clamp to realistic range
    tds_values.append(tds)

aquifer_data['tds_mg_per_L'] = tds_values

# Nitrate (mg/L) - higher in urban areas due to pollution
nitrate_values = []
for land_use in aquifer_data['land_use']:
    if land_use == 'urban':
        nitrate = np.random.uniform(2, 10)
    elif land_use == 'peri-urban':
        nitrate = np.random.uniform(1, 6)
    else:  # agriculture
        nitrate = np.random.uniform(2, 8)
    nitrate_values.append(nitrate)

aquifer_data['nitrate_mg_per_L'] = nitrate_values

# Fluoride (mg/L)
aquifer_data['fluoride_mg_per_L'] = np.random.uniform(0.01, 0.9, n_aquifers)

# pH (slightly alkaline to neutral)
aquifer_data['pH'] = np.random.uniform(6.2, 8.2, n_aquifers)

# Depletion rate (m/year) - KEY PARAMETER
# Higher depletion in urban areas with deep water tables
depletion_rates = []
for land_use, wtd in zip(aquifer_data['land_use'], aquifer_data['water_table_depth_m']):
    if land_use == 'urban':
        # Urban areas have higher depletion
        base_rate = np.random.uniform(0.8, 2.5)
        # Deep water tables indicate more stress
        if wtd > 30:
            base_rate += np.random.uniform(0.2, 0.8)
    elif land_use == 'peri-urban':
        base_rate = np.random.uniform(0.3, 1.5)
    else:  # agriculture
        base_rate = np.random.uniform(0.2, 1.2)
    
    depletion_rates.append(max(0.1, base_rate))

aquifer_data['depletion_rate_m_per_year'] = depletion_rates

# Years depleting (how long has depletion been occurring)
aquifer_data['years_depleting'] = np.random.randint(1, 15, n_aquifers)

# Create DataFrame
df_aquifer = pd.DataFrame(aquifer_data)

# Round numerical columns for readability
numerical_cols = ['lat', 'lon', 'well_depth_m', 'water_table_depth_m', 
                  'transmissivity_m2_per_day', 'hydraulic_conductivity_m_per_day',
                  'aquifer_thickness_m', 'tds_mg_per_L', 'nitrate_mg_per_L', 
                  'fluoride_mg_per_L', 'pH', 'depletion_rate_m_per_year']

for col in numerical_cols:
    df_aquifer[col] = df_aquifer[col].round(2)

# Save aquifer data
df_aquifer.to_csv('../data/raw/kathmandu_aquifer.csv', index=False)
print(f"✓ Generated {len(df_aquifer)} aquifer wells")
print(f"  - Shallow tube wells: {(df_aquifer['well_type'] == 'shallow_tube').sum()}")
print(f"  - Deep tube wells: {(df_aquifer['well_type'] == 'deep_tube').sum()}")
print(f"  - Dug wells: {(df_aquifer['well_type'] == 'dug').sum()}")
print(f"  - Average depletion rate: {df_aquifer['depletion_rate_m_per_year'].mean():.2f} m/year")

# ============================================================================
# PART 2: HOUSEHOLD DATA GENERATION
# ============================================================================
print("\nGenerating household dataset...")

# Number of households
n_households = 2000

# Generate household data
household_data = {
    'household_id': [f'HH{i:05d}' for i in range(1, n_households + 1)],
    'lat': np.random.uniform(lat_min, lat_max, n_households),
    'lon': np.random.uniform(lon_min, lon_max, n_households),
    'district': np.random.choice(['Kathmandu', 'Lalitpur', 'Bhaktapur'], n_households, p=[0.5, 0.3, 0.2]),
    'season': np.random.choice(['wet', 'dry'], n_households, p=[0.5, 0.5])
}

# Household size (number of members)
household_data['household_size'] = np.random.choice([3, 4, 5, 6, 7, 8], n_households, p=[0.15, 0.30, 0.25, 0.15, 0.10, 0.05])

# Water source access (binary indicators)
# Urban households more likely to have piped water
uses_piped = []
uses_well = []
uses_tanker = []

for district in household_data['district']:
    # Piped water access
    if district == 'Kathmandu':
        piped = np.random.choice([0, 1], p=[0.3, 0.7])
    else:
        piped = np.random.choice([0, 1], p=[0.4, 0.6])
    
    # Groundwater well access (complementary to piped)
    if piped == 1:
        well = np.random.choice([0, 1], p=[0.4, 0.6])
    else:
        well = np.random.choice([0, 1], p=[0.2, 0.8])
    
    # Tanker access (emergency source)
    if piped == 0 and well == 0:
        tanker = 1  # Must have at least one source
    else:
        tanker = np.random.choice([0, 1], p=[0.7, 0.3])
    
    uses_piped.append(piped)
    uses_well.append(well)
    uses_tanker.append(tanker)

household_data['uses_piped_water'] = uses_piped
household_data['uses_groundwater_well'] = uses_well
household_data['uses_tanker'] = uses_tanker

# Assign nearest aquifer to each household
# For simplicity, we randomly assign aquifer IDs (in reality, this would be spatial nearest neighbor)
household_data['aquifer_id'] = np.random.choice(df_aquifer['aquifer_id'].values, n_households)

# ============================================================================
# PART 3: WATER INSECURITY INDEX CALCULATION
# ============================================================================
print("Calculating Water Insecurity Index...")

# For each household, calculate insecurity based on:
# 1. Aquifer characteristics (depletion, depth, quality)
# 2. Household characteristics (size, water sources)
# 3. Seasonal factors

water_insecurity_scores = []

for i in range(n_households):
    hh_aquifer_id = household_data['aquifer_id'][i]
    aquifer_row = df_aquifer[df_aquifer['aquifer_id'] == hh_aquifer_id].iloc[0]
    
    # Base score from aquifer depletion (0-40 points)
    depletion_score = min(40, aquifer_row['depletion_rate_m_per_year'] * 15)
    
    # Water table depth contribution (0-25 points)
    depth_score = min(25, aquifer_row['water_table_depth_m'] * 0.5)
    
    # Water quality contribution (0-15 points)
    quality_score = 0
    if aquifer_row['tds_mg_per_L'] > 1000:
        quality_score += 8
    if aquifer_row['nitrate_mg_per_L'] > 5:
        quality_score += 7
    
    # Household size pressure (0-10 points)
    size_score = (household_data['household_size'][i] - 3) * 1.5
    
    # Water source diversity (negative points for more sources)
    source_diversity = (household_data['uses_piped_water'][i] + 
                       household_data['uses_groundwater_well'][i] + 
                       household_data['uses_tanker'][i])
    source_penalty = -5 * (source_diversity - 1)  # More sources = lower insecurity
    
    # Seasonal factor
    season_factor = 5 if household_data['season'][i] == 'dry' else 0
    
    # Total score
    total_score = depletion_score + depth_score + quality_score + size_score + source_penalty + season_factor
    
    # Add some random variation
    total_score += np.random.normal(0, 5)
    
    # Clamp to 0-100 range
    total_score = max(0, min(100, total_score))
    
    water_insecurity_scores.append(round(total_score, 2))

household_data['water_insecurity_index'] = water_insecurity_scores

# Create DataFrame
df_household = pd.DataFrame(household_data)

# Round coordinates
df_household['lat'] = df_household['lat'].round(6)
df_household['lon'] = df_household['lon'].round(6)

# Save household data
df_household.to_csv('../data/raw/kathmandu_household.csv', index=False)
print(f"✓ Generated {len(df_household)} households")
print(f"  - Average household size: {df_household['household_size'].mean():.1f}")
print(f"  - Households with piped water: {df_household['uses_piped_water'].sum()} ({df_household['uses_piped_water'].mean()*100:.1f}%)")
print(f"  - Households with well access: {df_household['uses_groundwater_well'].sum()} ({df_household['uses_groundwater_well'].mean()*100:.1f}%)")
print(f"  - Average water insecurity index: {df_household['water_insecurity_index'].mean():.2f}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n=== Data Generation Summary ===")
print(f"Aquifer wells: {len(df_aquifer)}")
print(f"Households: {len(df_household)}")
print(f"\nWater Insecurity Distribution:")
print(f"  Low Risk (0-33): {(df_household['water_insecurity_index'] < 33).sum()} households")
print(f"  Moderate Risk (33-66): {((df_household['water_insecurity_index'] >= 33) & (df_household['water_insecurity_index'] < 66)).sum()} households")
print(f"  High Risk (66-100): {(df_household['water_insecurity_index'] >= 66).sum()} households")

print("\n✓ Data generation complete!")
print("  Files saved:")
print("    - data/raw/kathmandu_aquifer.csv")
print("    - data/raw/kathmandu_household.csv")
