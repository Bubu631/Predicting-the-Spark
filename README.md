# Predicting the Spark: An Investigation into the Drivers of US Wildfires



A comprehensive machine learning project that predicts wildfire risk across the continental United States by integrating meteorological data, historical wildfire records, and a novel vegetation fuel combustibility index (C-index) engineered from FBFM40 raster data.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Part 1: Data Preprocessing](#part-1-data-preprocessing)
- [Part 2: Feature Engineering — The C-index](#part-2-feature-engineering--the-c-index)
- [Part 3: Exploratory Data Analysis](#part-3-exploratory-data-analysis)
- [Part 4: Modeling Pipeline](#part-4-modeling-pipeline)
- [Part 5: Results](#part-5-results)
- [Part 6: Key Findings](#part-6-key-findings)
- [Part 7: Limitations & Future Work](#part-7-limitations--future-work)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)

---

## Project Overview

Wildfires cause billions of dollars in damage annually and pose severe threats to ecosystems and communities. This project builds a complete ML pipeline to predict daily wildfire occurrence using environmental, meteorological, and vegetation fuel data covering the entire continental US from 2014–2024.

**Key contributions:**
- Engineered a novel **C-index** (Composite Flammability Index) from 250m-resolution FBFM40 raster data
- Addressed severe **class imbalance** (94.4% No Fire vs 5.6% Fire) through strategic downsampling
- Compared 5 model families: Logistic Regression (Elastic Net), Decision Tree, XGBoost, Random Forest, and Neural Network
- Achieved **AUC 0.83** and **78% Recall** with the tuned Random Forest — correctly identifying nearly 4 out of 5 wildfires

---

## Datasets

### US Wildfire Dataset (Kaggle)
- **Source:** [Kaggle — US Wildfire Dataset](https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset)
- **Size:** ~8 million daily observations
- **Features:** Meteorological variables (temperature, humidity, precipitation, wind speed, solar radiation), fire risk indices (burning index, fuel moisture, energy release component), and official wildfire records from IRWIN

| Category | Parameters | Unit |
|---|---|---|
| Spatial Coordinates | Longitude, Latitude | degrees |
| Moisture Variables | Precipitation (pr), Relative humidity (rmax/rmin), Specific humidity (sph), VPD | mm/day, %, kg/kg, kPa |
| Energy/Radiation | Solar radiation (srad), Temperature (tmmn/tmmx) | W/m², K |
| Wind Variables | Wind speed (vs) | m/s |
| Fire Risk Indices | Fuel moisture (fm100/fm1000), ERC, Burning index (bi) | %, index |

### FBFM40 Vegetation Fuel Model
- **Source:** [LANDFIRE FBFM40](https://landfire.gov/fuel/fbfm40)
- **Resolution:** 250-meter raster data
- **Coverage:** Continental US vegetation classification and combustibility

---

## Part 1: Data Preprocessing

### 1.1 Cleaning Placeholder Values

A recurring placeholder value (32767) was found across **15 variables** in **25,725 rows**. This pattern indicated artificial entries, not extreme weather. These were replaced with column medians to ensure data integrity.

| | pr | rmax | erc |
|---|---|---|---|
| **Raw mean** | 90.3 | 164.5 | 130.0 |
| **Raw max** | 32767.0 | 32767.0 | 32767.0 |
| **Cleaned mean** | 1.7 | 76.1 | 47.9 |
| **Cleaned max** | 528.0 | 100.0 | 121.0 |

### 1.2 Class Imbalance Challenge

Wildfire events are extremely rare in the dataset — only **5.6%** of observations are fire events, creating severe class imbalance.

![Class Imbalance](figures/class_imbalance_bar.png)

A standard model trained on this data would achieve high accuracy by simply always predicting "No", making it useless for actual prediction.

---

## Part 2: Feature Engineering — The C-index

### 2.1 FBFM40 Vegetation Analysis

The FBFM40 raster provides dozens of vegetation fuel classes across the continental US. The dominant classes include Dry Grass (12.4%), Agricultural Land (9.5%), and Moderate Shrub (6.0%).

![FBFM40 Area Statistics](figures/fbfm40_area_statistics.png)

![FBFM40 Vegetation Map](figures/fbfm40_vegetation_map.png)

### 2.2 Flammability Classification

Each fuel type was mapped to a **1–5 flammability score**:
- **Level 1 — Very Low:** Almost non-flammable (water, urban, barren)
- **Level 2 — Low:** Slow fire spread
- **Level 3 — Moderate:** Noticeable burning & spread
- **Level 4 — High:** Strong & rapid fire behavior
- **Level 5 — Extreme:** Very fast & severe fire spread

![Flammability Level Distribution](figures/flammability_level_distribution.png)

### 2.3 Spatial Flammability Patterns

The western US shows many **high to extreme** flammability areas → higher wildfire risk. Central & Eastern regions are mostly **low to moderate** flammability.

![Flammability Spatial Map](figures/flammability_spatial_map.png)

### 2.4 Constructing the C-index

The C-index aggregates fine-scale 250m fuel data into 10km grid cells:

**C_index = Σ p_k(cell) · flammability(k)**

Where `p_k` is the proportion of area occupied by fuel type `k` within each 10km grid cell, and `flammability(k)` is the corresponding flammability level (1–5).

![C-index National Map](figures/c_index_national_map.png)

### 2.5 Validating the C-index

Overlaying 2018 wildfire ignition points on the C-index map shows wildfires concentrate in **higher C-index regions** (orange/red), while they are rare in low C-index areas (blue).

![Wildfire Overlay — Fire Locations](figures/wildfire_overlay_on_cindex_yes.png)

![Wildfire Overlay — Non-fire Locations](figures/wildfire_overlay_on_cindex_no.png)

However, the KDE comparison reveals strong overlap between fire and non-fire distributions — the C-index indicates **flammability potential** but is not a deterministic predictor of ignition by itself.

![C-index KDE: Fire vs No Fire](figures/cindex_kde_yes_vs_no.png)

![C-index Comparison](figures/cindex_fire_vs_nofire_comparison.png)

---

## Part 3: Exploratory Data Analysis

### 3.1 Feature Distributions

![Feature Distributions](figures/feature_distributions.png)

![Feature Distributions by Wildfire Class](figures/feature_distributions_by_wildfire.png)

### 3.2 Correlation Analysis

Many environmental variables are strongly correlated with each other (e.g., etr and pet at 0.99), but **no single variable** shows a strong linear correlation with Wildfire. Predicting wildfires requires capturing complex, non-linear interactions.

![Correlation Heatmap](figures/correlation_heatmap.png)

![Correlation with Wildfire](figures/correlation_with_wildfire.png)

### 3.3 Temporal Patterns

Monthly wildfire counts reveal a recurring seasonal pattern: lowest in winter, rising through spring, peaking in **June–August** every year from 2014–2024.

![Monthly Wildfire Time Series](figures/monthly_wildfire_timeseries.png)

![Seasonal Boxplot](figures/seasonal_boxplot_by_month.png)

![Year-Month Heatmap](figures/heatmap_year_month.png)

### 3.4 Time Series Decomposition

![Rolling Mean Trend](figures/rolling_mean_trend.png)

![Additive Decomposition](figures/seasonal_decomposition_additive.png)

![Multiplicative Decomposition](figures/seasonal_decomposition_multiplicative.png)

---

## Part 4: Modeling Pipeline

### 4.1 Strategy

1. **Feature Selection:** 21 numeric features + engineered interaction terms (expanded to 76 features)
2. **Class Imbalance Handling:** Random downsampling to create balanced 1:1 training set; test set kept in original imbalanced distribution
3. **Train/Test Split:** 80/20 stratified split
4. **Models:** Elastic Net → XGBoost → Random Forest → Neural Network
5. **Evaluation:** AUC, Recall (primary), Precision, F1

### 4.2 Step 1: Elastic Net Baseline

Class-weighted logistic regression with L1+L2 penalty for automatic feature selection:
- **AUC: 0.695** | Recall: 0.63 | Precision: 0.10
- Top predictors: `fm1000` (negative), `year` (positive trend), `erc`

### 4.3 Step 2: Interaction Terms

Pairwise interaction terms between 11 core variables (e.g., `vpd*tmmn`, `c_index*tmmn`):
- **AUC: 0.698** (+0.003) | Recall: 0.64
- Key insight: interaction features dominate the top-15 importance list, confirming non-linear relationships

### 4.4 Step 3: Tree Ensembles & Neural Network

After downsampling to balanced 1:1 training data:

| Model | AUC | Recall | Precision |
|---|---|---|---|
| Decision Tree | 0.743 | 0.71 | 0.110 |
| XGBoost | 0.762 | 0.71 | 0.115 |
| Random Forest | 0.758 | 0.71 | 0.114 |
| Neural Network | 0.681 | 0.69 | 0.086 |

![XGBoost Feature Importance](figures/xgboost_feature_importance.png)

![Random Forest Feature Importance (Baseline)](figures/random_forest_feature_importance_baseline.png)

![Neural Network Training](figures/neural_network_training.png)

### 4.5 Step 4: Hyperparameter Tuning

**XGBoost** — Tuned `learning_rate` → best at 0.10 (validation AUC: 0.7659)

![XGBoost LR Tuning](figures/xgboost_lr_tuning.png)

**Random Forest** — Tuned `max_features` → best at 0.9 (validation AUC: 0.8169)

![Random Forest max_features Tuning](figures/random_forest_max_features_tuning.png)

### 4.6 Step 5: PCA Analysis

PCA compressed 15 weather variables into 7 components explaining ~95% variance. However, **tree models performed better on raw features** — PCA removes structure useful for tree splits.

![PCA Variance Explained](figures/pca_variance_explained.png)

---

## Part 5: Results

### Final Model Comparison

| Metric | Tuned XGBoost | **Tuned Random Forest** |
|---|---|---|
| **AUC Score** | 0.7718 | **0.8308** |
| **Recall** | 0.7189 | **0.7844** |
| **Precision** | 0.1188 | **0.1443** |
| **F1 Score** | 0.2038 | **0.2438** |

### Champion: Tuned Random Forest

The final model achieves **AUC 0.83** and correctly identifies **78% of wildfires** on the test set.

**Key drivers of the model's predictions:**

![Final Feature Importance](figures/random_forest_final_feature_importance.png)

- **Spatial context:** `longitude` (0.17) & `latitude` (0.17) — where you are matters most
- **Long-term fuel conditions:** `fm1000` (0.10) — dead fuel moisture is critical
- **Key interaction:** `inter_c_tmmn` (0.06) — fuel flammability × temperature

---

## Part 6: Key Findings

1. **Tree ensembles work best** — Random Forest (AUC 0.83) outperforms logistic regression, neural networks, and even XGBoost for this tabular wildfire data

2. **High recall is achievable at the cost of precision** — The model catches ~78% of wildfires, but only ~14% of alerts are actual fires (acceptable for an early warning system where missing a fire is costlier than a false alarm)

3. **Non-linear interactions matter** — Temperature–dryness–fuel interactions (e.g., `c_index*tmmn`) provide clear gains over linear models alone

4. **PCA hurts tree performance** — Compressing weather variables loses structure useful for decision tree splits

5. **Key physical drivers are consistent** — Spatial location, dead fuel moisture, radiation, and temperature–fuel interactions repeatedly appear as top predictors across all models

---

## Part 7: Limitations & Future Work

### Limitations
- **Spatial homogeneity:** Fire and non-fire points overlap heavily, reducing class separability
- **Simple imbalance handling:** Random downsampling discards potentially useful majority-class information
- **Single train-test split:** No spatial or temporal cross-validation
- **Limited model scope:** Did not explore LightGBM, CatBoost, or advanced neural architectures
- **Incomplete drivers:** Missing ignition sources, suppression actions, and detailed land-use data

### Future Extensions
- **SMOTE/ADASYN/focal loss** for advanced class imbalance handling
- **Spatial cross-validation** + rolling time windows
- **Cost-sensitive optimization** for real-world early warning deployment
- **Feature enrichment:** lightning data, land-cover, human activity
- **Spatio-temporal panel modeling** with spatial neighborhoods and time lags

---

## Tech Stack

| Category | Tools |
|---|---|
| **Languages** | Python |
| **ML/DL** | scikit-learn, XGBoost, PyTorch |
| **Data** | pandas, NumPy, GeoPandas |
| **Visualization** | Matplotlib, Seaborn, Folium |
| **Geospatial** | rasterio, scipy.spatial.KDTree |
| **Imbalanced Learning** | imbalanced-learn (RandomUnderSampler) |
| **Platform** | Google Colab (GPU) |

---

## Project Structure

```
Predicting-the-Spark/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore rules
├── figures/                               # All visualizations (31 figures)
│   ├── fbfm40_area_statistics.png
│   ├── fbfm40_vegetation_map.png
│   ├── flammability_level_distribution.png
│   ├── flammability_spatial_map.png
│   ├── c_index_national_map.png
│   ├── wildfire_overlay_on_cindex_yes.png
│   ├── wildfire_overlay_on_cindex_no.png
│   ├── cindex_fire_vs_nofire_comparison.png
│   ├── cindex_kde_yes_vs_no.png
│   ├── correlation_heatmap.png
│   ├── monthly_wildfire_timeseries.png
│   ├── seasonal_boxplot_by_month.png
│   ├── xgboost_lr_tuning.png
│   ├── random_forest_max_features_tuning.png
│   ├── random_forest_final_feature_importance.png
│   └── ... (31 total)
├── notebooks/
│   └── Predict_the_spark.ipynb            # Full Jupyter notebook with outputs
└── src/
    └── predict_the_spark.py               # Python script version
```

---

## Setup & Usage

### Option 1: Google Colab (Recommended)

1. Upload `notebooks/Predict_the_spark.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells — data is automatically downloaded via `kagglehub`

### Option 2: Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/Predicting-the-Spark.git
cd Predicting-the-Spark
pip install -r requirements.txt
jupyter notebook notebooks/Predict_the_spark.ipynb
```

**Note:** The FBFM40 raster data files need to be downloaded separately from [LANDFIRE](https://landfire.gov/fuel/fbfm40) and placed in the appropriate directory. See the notebook for detailed instructions.

---

## Data Sources

- [US Wildfire Dataset — Kaggle](https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset)
- [GRIDMET — Gridded Surface Meteorological Dataset](https://www.drought.gov/data-maps-tools/gridded-surface-meteorological-gridmet-dataset)
- [IRWIN — Integrated Reporting of Wildland-Fire Information](https://www.wildfire.gov/application/irwin-integrated-reporting-wildfire-information)
- [LANDFIRE FBFM40 — Vegetation Fuel Model](https://landfire.gov/fuel/fbfm40)
