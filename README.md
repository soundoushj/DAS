# Cars Price Analysis Dashboard

## Project Overview

The Cars Price Analysis Dashboard is an interactive data science and machine learning application developed using Streamlit.  
The project analyzes car datasets, performs exploratory data analysis (EDA), and predicts car prices using regression models.

The dashboard includes:

- Dataset exploration
- Statistical analysis
- Data visualization
- Machine learning modeling
- Price prediction system

---

# Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# Project Structure

```bash
project/
│
├── app.py
├── requirements.txt
├── README.md
└── Data_Set/
    └── Cars_dataset.csv
```

---

# Features

## 1. Overview Dashboard

The Overview page provides:

- Total number of cars
- Average car price
- Maximum car price
- Minimum car price
- Number of brands
- Dataset preview
- Statistical summary

---

## 2. Exploratory Data Analysis (EDA)

The EDA section includes:

### Visualizations

- Price distribution histogram
- Cars by brand bar chart
- Correlation heatmap
- Mileage vs Price scatter plot

### Insights

- Price trends
- Feature relationships
- Correlation analysis
- Mileage impact on car prices

---

## 3. Machine Learning Models

The application supports multiple regression models:

### Models Included

- Linear Regression
- Ridge Regression
- Lasso Regression

### Evaluation Metrics

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Cross Validation Score

### Model Visualizations

- Actual vs Predicted Prices
- Residual Distribution
- Feature Importance Analysis

---

# Data Preprocessing

The following preprocessing steps are applied:

## Cleaning

- Missing value handling
- Duplicate removal

## Encoding

Categorical variables are encoded using:

```python
pd.get_dummies()
```

## Feature Scaling

Standardization is performed using:

```python
StandardScaler()
```

---

# Machine Learning Workflow

## Step 1 — Data Preparation

The dataset is divided into:

- Features (X)
- Target variable (y)

Target variable:

```python
Price
```

---

## Step 2 — Train/Test Split

Dataset split:

```python
80% Training
20% Testing
```

Using:

```python
train_test_split()
```

---

## Step 3 — Model Training

The selected regression model is trained using:

```python
model.fit(X_train, y_train)
```

---

## Step 4 — Prediction

Predictions are generated using:

```python
model.predict(X_test)
```

---

## Step 5 — Evaluation

Performance metrics are calculated:

```python
r2_score()
mean_absolute_error()
mean_squared_error()
```

---

# Price Prediction System

The dashboard contains an interactive prediction system where users can enter:

- Year
- Mileage
- Horsepower
- Engine Size
- Brand
- Fuel Type

The trained model then estimates the car price.

---

# Key Insights

## Mileage Impact

Mileage has a negative correlation with car price.

Higher mileage generally leads to lower resale value.

---

## Engine Power

Cars with higher horsepower and larger engine sizes tend to have higher prices.

---

## Brand Influence

Premium brands generally have higher average prices.

---

# Dashboard Design

The dashboard uses:

- Dark theme UI
- Responsive layout
- Custom CSS styling
- Interactive charts
- Sidebar navigation

---

# How to Run the Project

## Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Run Streamlit App

```bash
streamlit run app.py
```

---

# Requirements

Example `requirements.txt`:

```txt
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

# Future Improvements

Possible future enhancements:

- Add more machine learning algorithms
- Deploy online using Streamlit Cloud
- Add advanced filtering options
- Improve prediction accuracy
- Add real-time database support
- Add car image visualization

---

# Conclusion

This project demonstrates a complete machine learning workflow:

1. Data collection
2. Data preprocessing
3. Exploratory data analysis
4. Model training
5. Model evaluation
6. Interactive deployment

The Cars Price Analysis Dashboard provides an easy-to-use interface for analyzing car datasets and predicting prices using machine learning techniques.

---

# Author

Soundous Hadj

ENSTA — National Higher School of Advanced Technologies

Algeria