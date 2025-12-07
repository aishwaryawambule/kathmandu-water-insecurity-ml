# kathmandu_aquifer_synthetic_generator.py
import numpy as np
import pandas as pd
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N = 2000

# Kathmandu Valley approx bounding box for sampling (coarse)
LAT_MIN, LAT_MAX = 27.60, 27.80
LON_MIN, LON_MAX = 85.20, 85.45

districts = ['Kathmandu', 'Lalitpur', 'Bhaktapur']
well_types = ['dug', 'shallow_tube', 'deep_tube']
land_uses = ['urban', 'peri-urban', 'agriculture', 'green']
seasons = ['dry', 'wet']

rows = []
for i in range(N):
    lat = np.random.uniform(LAT_MIN, LAT_MAX)
    lon = np.random.uniform(LON_MIN, LON_MAX)

    # district by rough location weighting (not exact)
    district = np.random.choice(districts, p=[0.6, 0.25, 0.15])

    # well type probabilities: many shallow and dug wells, fewer deep tube wells
    wt = np.random.choice(well_types, p=[0.25, 0.55, 0.20])

    # well depth depends on well type
    if wt == 'dug':
        well_depth = np.random.normal(6, 2)  # small shallow dug well (m)
        well_depth = float(max(1.5, well_depth))
    elif wt == 'shallow_tube':
        well_depth = np.random.normal(40, 15)  # typical tube well depths
        well_depth = float(np.clip(well_depth, 8, 120))
    else:  # deep_tube
        well_depth = np.random.normal(150, 60)
        well_depth = float(np.clip(well_depth, 30, 400))

    # season affects water table: dry season typically lowers water table
    season = np.random.choice(seasons, p=[0.6, 0.4])
    # base water table depth (m): shallow to tens of meters depending on location & season
    if season == 'dry':
        wt_base = np.random.normal(15, 10)
    else:
        wt_base = np.random.normal(8, 6)

    # add dependence on well_type and random noise
    if wt == 'dug':
        water_table_depth = float(max(0.2, np.random.normal(wt_base * 0.6, 1.0)))
    elif wt == 'shallow_tube':
        water_table_depth = float(max(0.5, np.random.normal(wt_base * 1.0, 2.5)))
    else:
        water_table_depth = float(max(1.0, np.random.normal(wt_base * 1.6, 4.0)))

    # transmissivity: use log-normal style to cover ~160 - 1050 m2/day reported range
    # (we allow some outside-range tail for variability)
    transmissivity = float(np.exp(np.random.normal(np.log(300), 0.9)))  # m2/day
    transmissivity = float(np.clip(transmissivity, 20, 2000))

    # hydraulic conductivity = transmissivity / aquifer_thickness (assume screen/thickness)
    aquifer_thickness = np.random.uniform(5, 50)
    hydraulic_conductivity = float(transmissivity / aquifer_thickness)

    # specific capacity (m3/day/m) synthetic proxy correlated with transmissivity
    specific_capacity = float(transmissivity * np.random.uniform(0.002, 0.01))

    # pump capacity (L/min) depending on well type:
    if wt == 'dug':
        pump_capacity_lpm = float(np.clip(np.random.normal(5, 3), 1, 30))
    elif wt == 'shallow_tube':
        pump_capacity_lpm = float(np.clip(np.random.normal(50, 40), 5, 300))
    else:
        pump_capacity_lpm = float(np.clip(np.random.normal(400, 350), 50, 2000))

    pumping_rate_m3_per_day = pump_capacity_lpm * 60 * 24 / 1000.0  # L/min -> m3/day

    # Water quality: TDS (mg/L) - studies report values roughly 100-1000+ mg/L in KV
    # We use a mixture: some wells low, some high. Use skewed distribution.
    tds = float(np.clip(np.random.lognormal(mean=np.log(400), sigma=0.6), 50, 2000))

    # Nitrate (mg/L) - typically small but can be elevated in urban/periurban.
    # Use base and increase with urban landuse and shallow wells/heavy pumping (contamination).
    land_use = np.random.choice(land_uses, p=[0.6, 0.25, 0.08, 0.07])

    nitrate = np.random.normal(2.0, 1.5)  # base
    if land_use == 'urban':
        nitrate += np.random.uniform(0.5, 6.0)
    if water_table_depth < 5:
        nitrate += np.random.uniform(0.0, 4.0)
    nitrate = float(max(0.01, nitrate))

    # Fluoride (mg/L) - natural geogenic contributions: typical 0.2 - 2.0, some higher
    fluoride = float(np.clip(np.random.normal(0.6, 0.4) + (transmissivity/10000.0), 0.01, 5.0))

    # pH: roughly 6.5 - 8.5
    pH = float(np.clip(np.random.normal(7.2, 0.5), 5.5, 9.0))

    # household water insecurity index: synthetic target 0-100
    # make it higher if deep water table, high TDS, low pump capacity per household
    insecurity = 0.0
    # contribution from water table depth (more depth -> more insecurity)
    insecurity += (water_table_depth / 60.0) * 40.0
    # contribution from TDS (higher TDS -> more insecurity)
    insecurity += ((tds - 200.0) / 1800.0) * 30.0
    # contribution from pumping shortage: if pump capacity small relative to depth, increase insecurity
    pump_score = 1.0 - np.tanh(pump_capacity_lpm / (10.0 + water_table_depth))
    insecurity += pump_score * 30.0
    # add some noise and clip
    insecurity = float(np.clip(insecurity + np.random.normal(0, 6.0), 0, 100))

    rows.append({
        'id': i + 1,
        'lat': lat,
        'lon': lon,
        'district': district,
        'well_type': wt,
        'well_depth_m': round(well_depth, 2),
        'water_table_depth_m': round(water_table_depth, 2),
        'transmissivity_m2_per_day': round(transmissivity, 2),
        'hydraulic_conductivity_m_per_day': round(hydraulic_conductivity, 4),
        'aquifer_thickness_m': round(aquifer_thickness, 2),
        'specific_capacity_m3_per_day_per_m': round(specific_capacity, 4),
        'pump_capacity_lpm': round(pump_capacity_lpm, 2),
        'pumping_rate_m3_per_day': round(pumping_rate_m3_per_day, 3),
        'TDS_mg_per_L': round(tds, 1),
        'nitrate_mg_per_L': round(nitrate, 3),
        'fluoride_mg_per_L': round(fluoride, 3),
        'pH': round(pH, 2),
        'land_use': land_use,
        'season': season,
        'household_water_insecurity_index': round(insecurity, 2),
    })

df = pd.DataFrame(rows)

# quick sanity checks and save
print("Rows:", len(df))
print(df.head())

# Basic stats
print("\nSummary statistics for selected columns:")
print(df[['well_depth_m','water_table_depth_m','transmissivity_m2_per_day','TDS_mg_per_L','pump_capacity_lpm','household_water_insecurity_index']].describe())

# Save CSV
outname = "kathmandu_aquifer_2000.csv"
df.to_csv(outname, index=False)
print(f"\nSaved synthetic dataset to: {outname}")
