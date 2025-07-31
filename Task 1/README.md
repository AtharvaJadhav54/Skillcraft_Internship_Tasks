# Task 1: House Price Prediction 🏠

**Objective** 
Build and evaluate a linear regression model to predict house prices based on square footage, number of bedrooms, and total bathrooms.

**Tools & Libraries**  
- Python, Pandas, NumPy  
- Scikit-Learn (LinearRegression, train_test_split, metrics)  
- Plotly Express & Graph Objects  

---

## 🔍 Overview of Steps

1. **Data Upload & Loading**  
   Interactive CSV upload in Google Colab, then read into a DataFrame.
2. **Data Preparation**  
   - Select and rename features (`GrLivArea`, `BedroomAbvGr`, `FullBath`, `HalfBath`, `SalePrice`)  
   - Engineer a `TotalBathrooms` feature (counting half-baths as 0.5)  
   - Handle missing values by filling with medians
3. **Model Training**  
   - Split data 80/20 (train/test)  
   - Fit a `LinearRegression` model
4. **Evaluation**  
   - Compute R², RMSE, MAE  
   - Interpret results (explained variance and average error)
5. **Visualization**  
   - Actual vs. Predicted scatter plot with identity line  
   - Residuals vs. Predicted scatter plot  
   - 3D scatter of features vs. price, colored by bedrooms
6. **Interactive Prediction**  
   - Prompt for square footage, bedrooms, bathrooms  
   - Output a predicted house price

---

## 📈 Results

- **R²**: 0.6286  
- **RMSE**: \$53,371.56  
- **MAE**: \$36,569.64  

> *The model explains 62.86% of the variance in house prices, and on average its predictions are off by \$36.6k.*

---

## 🔧 Usage

1. Clone the repo (or Task 1 folder) to Google Colab.  
2. Upload your `train.csv` when prompted.  
3. Run:
   python main.py
