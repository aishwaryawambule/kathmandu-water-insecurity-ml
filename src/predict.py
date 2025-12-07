"""
Random Forest classification of household water insecurity.
Part of: Predicting Household Water Insecurity Due to Urban Groundwater Depletion
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Ensure outputs directory exists
os.makedirs('../outputs', exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv("../data/processed/kathmandu_household_with_cluster.csv")

# Feature Engineering
# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['season', 'district'], drop_first=False)

# Define features and target
exclude_cols = ['household_id', 'aquifer_id', 'lat', 'lon', 'water_insecurity_index']
X = df.drop(columns=exclude_cols)

# ---------------------------------------------------------
# CONVERT TARGET TO CLASSES
# ---------------------------------------------------------
# We bin the continuous index into 3 categories:
# 0-33: Low Insecurity
# 33-66: Moderate Insecurity
# 66-100: High Insecurity
def classify_insecurity(score):
    if score < 33:
        return 'Low'
    elif score < 66:
        return 'Moderate'
    else:
        return 'High'

df['insecurity_class'] = df['water_insecurity_index'].apply(classify_insecurity)
y = df['insecurity_class']

print("\n=== Class Distribution ===")
print(y.value_counts())

# Encode target labels (Low, Moderate, High -> 0, 1, 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nFeatures: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Random Forest Classifier with GridSearch
print("\n=== Training Random Forest Classifier ===")
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None] # Handle potential class imbalance
}

grid_search = GridSearchCV(
    rf, param_grid, 
    cv=5, 
    scoring='f1_weighted', # Optimize for F1 Score
    n_jobs=-1, 
    verbose=1
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")

# Cross-validation scores
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1)
print(f"Cross-validation F1 (Weighted): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Test set predictions
y_pred = best_rf.predict(X_test)

# Evaluation Metrics
print("\n=== Test Set Performance ===")
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score (Weighted): {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('../outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: outputs/confusion_matrix.png")

# 3. Feature importance (top 20)
feat_imp = pd.Series(
    best_rf.feature_importances_, 
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
top_n = min(20, len(feat_imp))
sns.barplot(x=feat_imp.head(top_n), y=feat_imp.head(top_n).index, palette='viridis')
plt.title(f'Top {top_n} Feature Importances', fontsize=14)
plt.xlabel('Importance Score', fontsize=12)
plt.tight_layout()
plt.savefig('../outputs/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: outputs/feature_importance.png")

# Save feature importance to CSV
feat_imp.to_csv('../outputs/feature_importances.csv', header=['importance'])

# 4. Distribution of predictions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(y_test, bins=30, alpha=0.7, label='Actual', edgecolor='black')
axes[0].hist(y_pred, bins=30, alpha=0.7, label='Predicted', edgecolor='black')
axes[0].set_xlabel('Water Insecurity Index')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution: Actual vs Predicted')
axes[0].legend()
fig, axes = plt.subplots(1, 1, figsize=(7, 5)) # Changed to 1 subplot

axes.hist(y_test, bins=30, alpha=0.7, label='Actual', edgecolor='black')
axes.hist(y_pred, bins=30, alpha=0.7, label='Predicted', edgecolor='black')
axes.set_xlabel('Water Insecurity Index')
axes.set_ylabel('Frequency')
axes.set_title('Distribution: Actual vs Predicted')
axes.legend()
axes.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: outputs/distribution_analysis.png")

# Save model and artifacts
joblib.dump(best_rf, '../outputs/water_insecurity_model.joblib')
joblib.dump(list(X.columns), '../outputs/model_features.joblib')
joblib.dump(le, '../outputs/label_encoder.joblib')

print("\n=== Prediction Pipeline Complete ===")
print("All outputs saved in 'outputs/' directory")
