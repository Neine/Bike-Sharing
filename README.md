# 🚴 Bike Sharing Demand Prediction — Linear Regression Assignment

**Submitted by:** Neine Arora  
**Course:** Machine Learning — UpGrad & IIIT-B  

---

## 📋 Problem Statement

A bike-sharing system allows users to rent bikes for short durations. BoomBikes, a US-based provider, wants to understand the factors affecting the demand for shared bikes — especially post-pandemic — to better plan their business strategy.

The goal is to build a **Multiple Linear Regression model** to predict the daily count of bike rentals (`cnt`) using weather, seasonal, and calendar variables.

**Business Questions:**
1. Which variables are significant in predicting bike-sharing demand?
2. How well do those variables describe the demand?

---

## 📁 Repository Structure

```
├── Bike Sharing Assignment.ipynb   # Main Jupyter Notebook with full analysis
├── day.csv                         # Dataset (daily bike rental records)
├── Linear_Regression_Subjective_Questions.pdf  # Subjective answers
└── README.md                       # Project documentation
```

---

## 📊 Dataset Description

**File:** `day.csv`  
**Source:** UCI Machine Learning Repository — Bike Sharing Dataset  
**Records:** 730 daily observations (2018–2019)

| Variable | Description |
|---|---|
| `instant` | Record index |
| `dteday` | Date |
| `season` | Season (1: Spring, 2: Summer, 3: Fall, 4: Winter) |
| `yr` | Year (0: 2018, 1: 2019) |
| `mnth` | Month (1–12) |
| `holiday` | Whether the day is a holiday |
| `weekday` | Day of the week |
| `workingday` | Whether the day is a working day |
| `weathersit` | Weather situation (1: Clear, 2: Mist, 3: Light Snow, 4: Snow+Fog) |
| `temp` | Normalised temperature in Celsius |
| `atemp` | Normalised feeling temperature in Celsius |
| `hum` | Normalised humidity |
| `windspeed` | Normalised wind speed |
| `casual` | Count of casual (non-registered) users |
| `registered` | Count of registered users |
| `cnt` | Total rental count (target variable) |

---

## 🔍 Approach & Methodology

### Step 1: Data Understanding
- Loaded and explored the dataset (shape, dtypes, descriptive stats)
- Identified no missing values
- Renamed columns for clarity (`yr` → `Year`, `mnth` → `month`, `hum` → `humidity`, `cnt` → `count`)
- Converted numeric codes to readable categories (seasons, months, weather, weekdays)

### Step 2: Exploratory Data Analysis (EDA)
- **Pairplot** to visualise correlations among numerical variables
- **Heatmap** (Pearson correlation) to quantify relationships — `count` highly correlated with `registered` (r = 0.95), `temp`, and `atemp`
- **Boxplots** for all categorical variables vs `count`
- **Barplots** for season, weather, year, and month vs `count`

**Key EDA Insights:**
- Bike demand increased from 2018 to 2019
- Highest demand in Fall, followed by Summer, Winter, Spring
- Demand peaks in mid-year (July) and dips at year start/end
- Clear weather drives higher rentals than misty or snowy conditions
- Holidays show slightly lower demand than working days

### Step 3: Data Preparation
- Dropped irrelevant/redundant columns: `instant`, `dteday`, `atemp`, `casual`, `registered`
- Applied **One-Hot Encoding** (with `drop_first=True`) to all categorical variables: `season`, `month`, `weekday`, `weathersit`

### Step 4: Train-Test Split
- Split ratio: **70% train / 30% test** (`random_state=100`)
- Train set: ~511 records | Test set: ~219 records

### Step 5: Feature Scaling
- Applied **MinMax Scaling** on numerical variables: `temp`, `humidity`, `windspeed`, `count`
- Used `scaler.fit_transform()` on training data and `scaler.transform()` on test data

### Step 6: Model Building
- Used **RFE (Recursive Feature Elimination)** with `LinearRegression` to shortlist features
- Iteratively built OLS models using **statsmodels**, removing features with:
  - p-value > 0.05
  - VIF > 5 (to handle multicollinearity)
- Variables removed across iterations: `humidity` (VIF=29.73), `workingday` (VIF=13.48), `Sat` (p=0.175)

### Step 7: Residual Analysis
- Plotted distribution of residuals (`y_train − y_train_pred`)
- Confirmed residuals follow **normal distribution** with mean ≈ 0 — validating linear regression assumptions

### Step 8: Predictions & Model Evaluation
- Predicted on test set using final model
- Evaluated using **R² score** on both train and test sets
- Plotted `y_test` vs `y_pred` scatter plot to visually assess model fit

---

## 📈 Final Model & Regression Equation

```
Count = 0.273 + 0.235×Year + 0.430×temp - 0.149×windspeed
        - 0.104×spring + 0.039×winter - 0.044×Jan - 0.066×July
        + 0.053×Sep - 0.044×Sun - 0.287×Light Snow - 0.078×Mist+Cloudy
```

### Top 3 Features by Importance

| Rank | Feature | Coefficient | Impact |
|---|---|---|---|
| 1 | Temperature (`temp`) | +0.430 | Higher temp → more rentals |
| 2 | Light Snow/Rain (`weathersit`) | -0.287 | Bad weather → fewer rentals |
| 3 | Year (`yr`) | +0.235 | Demand growing year on year |

---

## 🛠️ Libraries Used

```python
numpy
pandas
matplotlib
seaborn
sklearn (train_test_split, MinMaxScaler, RFE, LinearRegression, r2_score)
statsmodels (OLS, variance_inflation_factor)
```

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook "Bike Sharing Assignment.ipynb"
   ```

4. Run all cells: **Kernel → Restart & Run All**

> ⚠️ Make sure `day.csv` is in the **same directory** as the notebook before running.

---

## 📌 Key Conclusions

- Bike demand is strongly driven by **temperature**, **weather conditions**, and **year-over-year growth**
- Adverse weather (light snow, mist) significantly **reduces** demand
- Seasonal patterns show peak demand in **Fall** and lowest in **Spring**
- The model achieves comparable R² on both train and test sets, indicating a **well-generalised model** with no significant overfitting

---

## 👤 Author

**Neine Arora**  
Linear Regression Assignment — UpGrad & IIIT-B AI/ML Programme
