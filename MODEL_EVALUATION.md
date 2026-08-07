# Model Evaluation Report

## Objective
Predict residential house sale prices using regression models, trained 
on the Kaggle "House Prices: Advanced Regression Techniques" dataset.

## Dataset Summary
- **Source:** Kaggle House Prices dataset (`train.csv`)
- **Initial size:** 1,460 houses, 81 columns
- **After cleaning:** 1,458 houses, 76 columns
  - Dropped 5 columns with >50% missing values (`Alley`, `PoolQC`, 
    `Fence`, `MiscFeature`, `FireplaceQu`)
  - Filled remaining missing values (median for numeric, "None" for 
    categorical where absence is meaningful, e.g. no basement)
  - Removed 2 outlier houses (very large living area, unusually low price)
- **After feature engineering:** 241 columns (categorical variables 
  one-hot encoded)

## Key Findings from EDA
- `SalePrice` is right-skewed — most houses cluster between 
  $100K–$250K, with a long tail of higher-priced homes
- Top price-correlated features:
  1. `OverallQual` (0.79) — overall material/finish quality
  2. `GrLivArea` (0.71) — above-ground living area
  3. `GarageCars` (0.64) — garage capacity
  4. `TotalBsmtSF` (0.61) — basement square footage
  5. `FullBath` (0.56) — number of full bathrooms
- Overall quality rating was a stronger price driver than raw square footage

## Models Trained

### 1. Linear Regression (baseline)
| Metric | Value |
|---|---|
| MAE | $17,931 |
| R² | 0.881 |

### 2. Random Forest Regressor (n_estimators=100)
| Metric | Value |
|---|---|
| MAE | $17,005 |
| R² | 0.893 |

## Model Comparison

| Model | MAE | R² | 
|---|---|---|
| Linear Regression | $17,931 | 0.881 |
| **Random Forest** | **$17,005** | **0.893** |

**Random Forest outperformed Linear Regression** on both metrics, with 
~$926 lower average error and higher explained variance. This is 
expected given the high number of features (241) relative to training 
rows (1,166), which tends to make Linear Regression less stable, while 
Random Forest handles high-dimensional, non-linear data more robustly.

## Sample Prediction
| | Value |
|---|---|
| Actual Price | $190,000 |
| Predicted Price | $213,571 |
| Error | $23,571 (~12%) |

This is a single example within normal range of the model's average 
error (~$17K MAE across all test houses).

## Interpretation
- An R² of 0.893 means the model explains about 89% of the variation 
  in house prices — a strong result for this dataset.
- An average error (MAE) of ~$17,000 is reasonable given house prices 
  in this dataset range roughly from $35,000 to $750,000.
- Errors tend to be larger for very expensive or unusually featured 
  houses, which are underrepresented in the training data.

## Final Model
**Random Forest Regressor** was selected as the final model based on 
superior performance on both MAE and R².

## Possible Future Improvements
- Try Gradient Boosting (XGBoost/LightGBM) for potentially better accuracy
- Apply log transformation to `SalePrice` to address skewness
- Use ordinal encoding for quality-based features instead of one-hot 
  encoding, to preserve their natural ranking
- Perform hyperparameter tuning (e.g. GridSearchCV) on Random Forest
- Address multicollinearity between related features 
  (e.g. `GarageCars` and `GarageArea`)