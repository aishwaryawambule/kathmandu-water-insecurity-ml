import numpy as np
import pandas as pd
import random
from scipy.spatial import cKDTree

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# --- PARAMETERS ---
NUM_AQUIFERS = 500       # number of synthetic wells / aquifer sampling points
NUM_HOUSEHOLDS = 2000    # number of households

# Approx bounding box (Kathmandu Valley)
LAT_MIN, LAT_MAX = 27.60, 27.80
LON_MIN, LON_MAX = 85.20, 85.45

well_types = ['dug', 'shallow_tube', 'deep_tube']
land_uses = ['urban', 'peri‑urban', 'agriculture', 'green']
seasons = ['dry', 'wet']

# Water‑quality / aquifer parameter distributions (loosely based on Kathmandu studies)
def sample_tds():
    # TDS often 100–1000 mg/L [some wells worse], we allow 50–2000 with skew
    return float(np.clip(np.random.lognormal(mean=np.log(400), sigma=0.7), 50, 2500))

def sample_nitrate(well_type, land_use, water_table_m):
    # nitrate tends to be elevated in shallow / contaminated / urban aquifers
    base = np.random.normal(5, 3)
    if well_type == 'dug' or well_type == 'shallow_tube':
        base += np.random.uniform(0, 5)
    if land_use == 'urban' and water_table_m < 10:
        base += np.random.uniform(0, 5)
    return float(max(base, 0.1))

def sample_pH():
    return float(np.clip(np.random.normal(7.2, 0.5), 6.0, 8.5))

aquifers = []
for i in range(NUM_AQUIFERS):
    lat = np.random.uniform(LAT_MIN, LAT_MAX)
    lon = np.random.uniform(LON_MIN, LON_MAX)
    wt = random.choices(well_types, weights=[0.2, 0.6, 0.2])[0]
    land_use = random.choices(land_uses, weights=[0.5,0.3,0.1,0.1])[0]
    # Well depth (m)
    if wt == 'dug':
        well_depth = float(max(1.5, np.random.normal(8, 3)))
    elif wt == 'shallow_tube':
        well_depth = float(np.clip(np.random.normal(40, 15), 8, 120))
    else:
        well_depth = float(np.clip(np.random.normal(150, 50), 30, 400))
    # Seasonal / base water table depth
    # assume water‑table fluctuates; we store a base water_table_depth
    water_table_depth = float(np.clip(np.random.normal(15, 7), 0.5, well_depth - 1.0))
    # Water quality
    tds = sample_tds()
    nitrate = sample_nitrate(wt, land_use, water_table_depth)
    pH = sample_pH()
    # Aquifer transmissivity / conductivity proxies
    transmissivity = float(np.exp(np.random.normal(np.log(250), 0.8)))
    transmissivity = float(np.clip(transmissivity, 20, 2000))
    thickness = float(np.random.uniform(5, 50))
    hydraulic_conductivity = transmissivity / thickness

    aquifers.append({
        'aquifer_id': f"AQ_{i+1:04d}",
        'lat': lat,
        'lon': lon,
        'well_type': wt,
        'land_use': land_use,
        'well_depth_m': round(well_depth,2),
        'water_table_depth_m': round(water_table_depth,2),
        'tds_mg_per_L': round(tds,1),
        'nitrate_mg_per_L': round(nitrate,3),
        'pH': round(pH,2),
        'transmissivity_m2_per_day': round(transmissivity,2),
        'hydraulic_conductivity_m_per_day': round(hydraulic_conductivity,4)
    })

df_aq = pd.DataFrame(aquifers)
df_aq.to_csv("kathmandu_aquifer.csv", index=False)

# Build spatial index to assign households to nearest aquifer
coords = np.array(df_aq[['lat','lon']])
tree = cKDTree(coords)

households = []
for i in range(NUM_HOUSEHOLDS):
    # Randomly sample household location
    lat = np.random.uniform(LAT_MIN, LAT_MAX)
    lon = np.random.uniform(LON_MIN, LON_MAX)
    # find nearest aquifer
    dist, idx = tree.query([lat,lon])
    aq = df_aq.iloc[idx]

    # Water‑use behavior
    uses_piped = random.random() < 0.7    # ~70% connected (from surveys) :contentReference[oaicite:5]{index=5}
    uses_well = random.random() < 0.4
    uses_tanker = random.random() < 0.2
    household_size = random.choice(range(1, 9))
    # sample per-capita water consumption (lc/d)
    base_lpcd = np.random.normal(80, 20) if (uses_piped or uses_well or uses_tanker) else np.random.normal(40,15)
    base_lpcd = float(max(5, base_lpcd))

    # Choose season
    season = random.choice(seasons)
    # adjust for season (less water usage/availability in dry season)
    lpcd = base_lpcd * (0.9 if season == 'dry' else 1.0)

    # Simple water insecurity index combining factors: well_quality, consumption, supply reliability
    # higher insecurity if water quality (tds / nitrate) is poor, consumption low, only tank or well
    quality_penalty = ((aq['tds_mg_per_L'] / 1000.0) + (aq['nitrate_mg_per_L'] / 50.0))
    use_penalty = 0 if uses_piped else 0.3
    consumption_penalty = max(0, (50 - lpcd)/50)  # 50 lpcd as minimal basic threshold
    insecurity = float(np.clip((quality_penalty * 0.4 + use_penalty * 0.3 + consumption_penalty * 0.3) * 100, 0, 100))

    households.append({
        'household_id': f"HH_{i+1:05d}",
        'lat': round(lat,6),
        'lon': round(lon,6),
        'aquifer_id': aq['aquifer_id'],
        'uses_piped_water': uses_piped,
        'uses_groundwater_well': uses_well,
        'uses_tanker': uses_tanker,
        'household_size': household_size,
        'season': season,
        'per_capita_lpcd': round(lpcd,2),
        'water_insecurity_index': round(insecurity,2)
    })

df_hh = pd.DataFrame(households)
df_hh.to_csv("kathmandu_household.csv", index=False)

print("Generated aquifer data ({} rows) and household data ({} rows)".format(len(df_aq), len(df_hh)))

