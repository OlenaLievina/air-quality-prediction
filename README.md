# Air Quality Prediction (UCI Dataset)
Machine learning project for air quality prediction.
This project is a learning experiment with **time series regression**.
## What the script does
1. **Downloads** the dataset directly from GitHub (UCI CSV format).
2. **Cleans** the data:
   - replaces `-200` with `NaN` (sensor failure indicator);
   - drops columns that are completely empty;
   - removes `NMHC(GT)` due to excessive missing values.
3. **Generates time-based features** (sin/cos transforms for month, day of week, hour, etc.).
4. **Scales** numeric features and imputes missing values.
5. **Trains 3 models** using `RandomizedSearchCV` + `TimeSeriesSplit`:
   - `RandomForestRegressor`
   - `XGBRegressor`
   - `HistGradientBoostingRegressor`
6. **Saves**:
   - trained pipelines (`joblib`);
   - a leaderboard with **cross-validation RMSE** and best parameters (`leaderboard.csv`).
**Note:**  
`leaderboard.csv` contains the **cross-validation RMSE** scores and best hyperparameters found during training,  
but **does NOT include the test set evaluation metrics**.

## Final test metrics
| Model                   | RMSE   | R² Score |
|-------------------------|--------|----------|
| XGBoost                 | 0.1847 |  0.9992  |
| HistGradientBoosting    | 0.2841 |  0.9981  |
| RandomForestRegressor   | 0.6090 |  0.9913  |
