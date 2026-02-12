# Predicting Household Water Insecurity Due to Urban Groundwater Depletion

A machine learning framework for predicting household water insecurity in Kathmandu Valley using K-Means clustering and Random Forest classification.

## Project Overview

This project addresses the critical water crisis in Kathmandu Valley by developing a computational model that:
- Identifies critical aquifer zones experiencing high groundwater stress using unsupervised clustering
- Predicts household-level water insecurity risk (Low, Moderate, High) using hydrogeological variables and household parameters
- Provides actionable insights for water resource management and targeted intervention

### Key Features

- **Hybrid ML Framework**: Combines unsupervised clustering (K-Means) for aquifer zonation with supervised classification (Random Forest) for household vulnerability prediction
- **Synthetic Data Generation**: Realistic data generation based on hydrogeological patterns and satellite observations
- **High Accuracy**: Achieves 81.25% accuracy with weighted F1-score of 0.81
- **Feature Importance Analysis**: Identifies key drivers of water insecurity

## Project Structure

```
ml-proj/
├── data/
│   ├── raw/                          # Raw generated data
│   │   ├── kathmandu_aquifer.csv     # 500 aquifer wells with hydrogeological data
│   │   └── kathmandu_household.csv   # 2000 households with water access data
│   └── processed/                    # Processed data after clustering
│       ├── kathmandu_aquifer_clustered.csv
│       └── kathmandu_household_with_cluster.csv
├── src/
│   ├── aquifer_clustering.py        # K-Means clustering of aquifer wells
│   ├── merge_data.py                # Merge household and aquifer data
│   ├── predict.py                   # Train Random Forest classifier
│   └── make_prediction.py           # Make predictions for new locations
├── outputs/                          # Model outputs and visualizations
│   ├── water_insecurity_model.joblib
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── aquifer_clusters_pca.png
├── main.py                          # Run complete ML pipeline
├── paper.tex                        # Research paper (LaTeX)
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-proj
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib scipy
```

### Running the Complete Pipeline

Run the entire ML pipeline (data generation → clustering → training → prediction):

```bash
python main.py
```

This will:
1.Perform K-Means clustering on aquifer wells
2. Merge household data with aquifer cluster assignments
3. Train Random Forest classifier
4. Generate evaluation metrics and visualizations

### Making Predictions

After training, test the model with scenario predictions:

```bash
python src/make_prediction.py
```

**Example Output:**
```
========================================
Testing Case 1: Secure Household (Low Risk)
========================================
Prediction: Low Risk
Aquifer Context: Depth 6.49m, Depletion 0.31m/yr

========================================
Testing Case 2: Vulnerable Household (Moderate Risk)
========================================
Prediction: Moderate Risk
Aquifer Context: Depth 24.94m, Depletion 1.71m/yr

========================================
Testing Case 3: Critical Zone (High Risk)
========================================
Prediction: High Risk
Aquifer Context: Depth 51.05m, Depletion 3.15m/yr
```

## Dataset Description

### Aquifer Dataset (500 wells)

**Spatial Features:**
- `lat`, `lon`: Geographic coordinates
- `district`: Kathmandu, Lalitpur, or Bhaktapur

**Hydrogeological Features:**
- `water_table_depth_m`: Depth to water table
- `depletion_rate_m_per_year`: Annual groundwater depletion rate (KEY FEATURE)
- `well_depth_m`: Total well depth
- `transmissivity_m2_per_day`: Aquifer transmissivity
- `hydraulic_conductivity_m_per_day`: Hydraulic conductivity

**Water Quality:**
- `tds_mg_per_L`: Total Dissolved Solids
- `nitrate_mg_per_L`: Nitrate concentration
- `fluoride_mg_per_L`: Fluoride concentration
- `pH`: Water pH level

### Household Dataset (2000 households)

**Household Characteristics:**
- `household_size`: Number of members (3-8)
- `uses_piped_water`: Binary indicator
- `uses_groundwater_well`: Binary indicator
- `uses_tanker`: Binary indicator

**Target Variable:**
- `water_insecurity_index`: Continuous score (0-100)
- `insecurity_class`: Categorical (Low, Moderate, High)
  - Low Risk: 0-33
  - Moderate Risk: 33-66
  - High Risk: 66-100

## Methodology

### 1. Aquifer Clustering (`aquifer_clustering.py`)

**Algorithm:** K-Means Clustering

**Features Used:**
- Well depth, water table depth
- TDS, nitrate levels
- Transmissivity, hydraulic conductivity
- **Depletion rate** (primary indicator)
- Years depleting

**Optimal Clusters:** k=3
- **Cluster 0:** Moderate Stress (143 wells)
- **Cluster 1:** Critical Depletion Zone (143 wells)
- **Cluster 2:** Moderate Stress (214 wells)

**Validation Metrics:**
- Silhouette Score: 0.1748
- Calinski-Harabasz Index: 114.37
- Davies-Bouldin Index: 1.7506

### 2. Random Forest Classification (`predict.py`)

**Model:** Random Forest Classifier with GridSearchCV

**Hyperparameters (Optimized):**
- `n_estimators`: 100
- `max_depth`: 10
- `min_samples_split`: 2
- `class_weight`: balanced (handles class imbalance)

**Performance Metrics:**
- **Accuracy:** 81.25%
- **Weighted F1-Score:** 0.81
- **Cross-Validation F1:** 0.84 (±0.03)

**Class-Wise Performance:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| High | 0.73 | 0.42 | 0.53 | 19 |
| Low | 0.81 | 0.85 | 0.83 | 177 |
| Moderate | 0.82 | 0.81 | 0.82 | 204 |

### 3. Feature Importance

Top predictive features:
1. **Water Table Depth** (35%)
2. **Depletion Rate** (28%)
3. **Water Quality** (18% - TDS + Nitrate)
4. **Access to Piped Water** (12%)
5. **Household Size** (7%)

## Results

### Scenario Testing

Three test cases validate real-world applicability:

| Scenario | Location | Depth (m) | Depletion (m/yr) | HH Size | Prediction |
|----------|----------|-----------|------------------|---------|------------|
| **Case 1** | Bhaktapur | 6.49 | 0.31 | 4 | ✅ Low Risk |
| **Case 2** | Bhaktapur | 24.94 | 1.71 | 6 | ✅ Moderate Risk |
| **Case 3** | Kathmandu | 51.05 | 3.15 | 8 | ✅ High Risk |

**Case 1:** Household with piped water + well access near shallow, stable aquifer → **Low Risk**

**Case 2:** Vulnerable household (tanker only) near moderately stressed aquifer during dry season → **Moderate Risk**

**Case 3:** Large household (8 members) with no piped water near critically depleted aquifer → **High Risk**

##  Key Findings

1. **Aquifer depletion is the primary driver** of water insecurity (63% combined importance)
2. **Infrastructure access matters:** Households with multiple water sources remain secure even in moderately stressed aquifers
3. **Deep water tables + high depletion = critical risk:** Aquifers with depth >50m and depletion >3 m/yr require immediate intervention
4. **Model generalizes well:** Cross-validation F1-score of 0.84 indicates stable performance

##  Usage Examples

### Custom Prediction

```python
from src.make_prediction import predict_water_insecurity

# Predict for a specific location
prediction, aquifer_info = predict_water_insecurity(
    lat=27.70,
    lon=85.32,
    district="Kathmandu",
    season="dry",
    household_size=5,
    uses_piped=0,
    uses_well=1,
    uses_tanker=0
)

print(f"Risk Level: {prediction}")
print(f"Aquifer Depth: {aquifer_info['water_table_depth_m']:.2f}m")
print(f"Depletion Rate: {aquifer_info['depletion_rate_m_per_year']:.2f}m/yr")
```

##  Research Paper

The complete methodology and results are documented in `paper.tex` (IEEE format).

**Compile the paper:**
```bash
pdflatex paper.tex
```

##  Contributing

This project was developed as part of academic research on urban water security and machine learning applications in environmental science.

##  References

- International Centre for Integrated Mountain Development (ICIMOD)
- Groundwater Resources Development Board (GWRDB), Nepal
- Kathmandu Upatyaka Khanepani Limited (KUKL)
- PANGAEA Data Publisher


## License

This project is part of academic research. Please cite appropriately if using this work.

---

**Note:** This project uses synthetic data generated based on real hydrogeological patterns. For production use, integrate with actual groundwater monitoring data from GWRDB and KUKL.
