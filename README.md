# House Price Prediction

A machine learning project that predicts residential property prices 
based on features like living area, quality, garage size, and more.

## Dataset
Kaggle "House Prices - Advanced Regression Techniques" 
(1460 houses, 79 features)

## Approach
1. Data Cleaning — dropped columns with >50% missing values, 
   filled remaining gaps (median for numeric, "None" for categorical)
2. EDA — analyzed price distribution and correlations; found 
   OverallQual, GrLivArea, and GarageCars as top price drivers
3. Outlier Removal — removed 2 houses with abnormally large area 
   but low price
4. Feature Engineering — one-hot encoded all categorical variables 
   (76 → 241 columns)
5. Model Training — compared Linear Regression vs Random Forest

## Results
| Model | MAE | R² |
|---|---|---|
| Linear Regression | $17,931 | 0.881 |
| Random Forest | $17,005 | 0.893 |

**Best model: Random Forest Regressor**

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Open `notebooks/house_price_prediction.ipynb`
3. Run all cells

## Files
- `notebooks/` — full analysis and model training
- `models/house_price_model.pkl` — trained model
- `data/` — dataset used