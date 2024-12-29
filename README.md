# **Audience Rating Prediction Project**

## **Project Overview**
This project aims to build a machine learning model to predict the audience rating of movies based on various features. The pipeline demonstrates the end-to-end process, from data preprocessing to model evaluation and hyperparameter tuning, ensuring accurate predictions and robust performance.

---
### **List of Files in the Project**

1. **Predicting_Audience_Rating.ipynb**  
   - The main Jupyter notebook containing the entire pipeline for predicting audience ratings. It includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, hyperparameter tuning, and final pipeline implementation.

2. **Rotten_Tomatoes_Movies3.xls**  
   - The raw dataset in `.xls` format containing movie information such as runtime, genre, cast, critic ratings, and audience ratings.

3. **Rotten_tomatoes_Movies3.csv**  
   - The converted version of the dataset in `.csv` format for easier handling and compatibility with data processing libraries. This file is generated during the initial steps of the notebook.

These files are essential for running the project and reproducing the results.
---

## **Features**
- **Data Preprocessing**:
  - Conversion of `.xls` dataset to `.csv` for better compatibility.
  - Handling missing values and encoding categorical variables.
  - Splitting data into training and testing sets.
- **Exploratory Data Analysis (EDA)**:
  - Visualization of target variable distribution.
  - Statistical summaries to understand data trends and outliers.
- **Model Building**:
  - Implementation of multiple regression models: Linear Regression, Random Forest, Decision Tree, and Support Vector Regression (SVR).
  - Evaluation using metrics like Mean Squared Error (MSE) and R² score.
- **Hyperparameter Tuning**:
  - RandomizedSearchCV for optimizing model parameters.
  - Early stopping for Random Forest to prevent overfitting.
- **Final Pipeline**:
  - A robust pipeline with Random Forest Regressor as the final model.
  - StandardScaler for feature normalization.

---

## **Dataset**
The dataset contains information about movies, including:
- **Numerical Features**: `runtime_in_minutes`, `tomatometer_rating`, `tomatometer_count`, `audience_rating` (target variable).
- **Categorical Features**: `rating`, `genre`, `directors`, `writers`, `cast`, `studio_name`, etc.
- **Text-based Features**: `movie_info`, `critics_consensus`.
- **Date Features**: `in_theaters_date`, `on_streaming_date`.

### Dataset Summary:
- Total rows: 16,638
- Target variable: `audience_rating` (float64)
- Missing values in several columns handled during preprocessing.

---

## **Steps to Run the Project**

### 1. **Setup Environment**

Here is the complete list of import statements used in the notebook:

```python
# Import libraries for data handling and manipulation
import pandas as pd  # For handling tabular data (DataFrames)
import numpy as np  # For performing numerical operations

# Import libraries for data visualization
import matplotlib.pyplot as plt  # For creating basic plots and graphs
import seaborn as sns  # For creating more advanced and visually appealing plots

# Import libraries for data preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For scaling data and encoding labels

# Import libraries for model building and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score  # For splitting data and hyperparameter tuning
from sklearn.linear_model import LinearRegression  # For Linear Regression model
from sklearn.tree import DecisionTreeRegressor  # For Decision Tree Regression model
from sklearn.ensemble import RandomForestRegressor  # For Random Forest Regression model
from sklearn.svm import SVR  # For Support Vector Regression model

# Import libraries for performance metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # For evaluating model performance

# Import library for model persistence
import joblib  # For saving and loading models

# Import library for formatted output in tables
from tabulate import tabulate  # For displaying results in a formatted table

# Import library for in-memory file handling
import io  # For working with files in memory

# Import library for safely evaluating strings into Python objects
import ast  # For converting strings to Python objects (e.g., dict, list) safely
```

These imports cover all the necessary modules and packages used throughout the notebook for data handling, visualization, preprocessing, model building, evaluation, and utility functions.

### 2. **Run the Notebook**
Follow these steps in the Jupyter Notebook:
1. Load the dataset (`Rotten_Tomatoes_Movies3.xls`) and convert it to `.csv`.
2. Perform exploratory data analysis to understand the dataset.
3. Preprocess the data by handling missing values and encoding categorical features.
4. Train multiple models and evaluate their performance.
5. Tune hyperparameters using RandomizedSearchCV.

### 3. **Final Pipeline**
The final pipeline uses a Random Forest Regressor with optimized hyperparameters:
```python
RandomForestRegressor(
    n_estimators=150,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42
)
```
The pipeline includes a StandardScaler for normalization.

---

## **Results**

### Model Performance Comparison:
| Model                  | Best MSE     | Best R²   | Notes                          |
|------------------------|--------------|-----------|--------------------------------|
| Linear Regression      | Moderate     | ~0.561    | Simple model; underfits data. |
| Random Forest          | Best (low)   | ~0.601    | Best balance; slight overfit. |
| Decision Tree          | Moderate     | ~0.507    | Overfits; lacks generalization. |
| Support Vector Regressor (SVR) | Poor        | ~0.049    | Severe underfitting observed. |

### Final Model Performance:
- R² Score: ~0.601
- MSE: Low error compared to other models.

---

## **Visualization**
1. Histogram with KDE for audience rating distribution.
2. Bar chart comparing training vs testing scores for each model.

---

## **Key Takeaways**
1. The Random Forest model performed the best after hyperparameter tuning, balancing complexity and accuracy.
2. Early stopping helped prevent overfitting in Random Forest by limiting tree depth and size.
3. Proper preprocessing (e.g., handling missing values, encoding) significantly improved model performance.

---

## **Future Improvements**
1. Explore advanced models like Gradient Boosting or XGBoost for better performance.
2. Use feature engineering to create new predictors from existing data (e.g., text analysis on `critics_consensus`).
3. Address potential outliers in runtime (`min=1`, `max=2000`) with domain-specific knowledge.

---

## **Contact**
For questions or suggestions regarding this project, feel free to reach out!

