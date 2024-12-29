# Zoho Project: Predicting Audience Rating

This project aims to build a model to predict the 'audience_rating' for movies using the Rotten Tomatoes dataset. The notebook demonstrates the complete pipeline from data preprocessing to model evaluation.

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Exploratory Data Analysis](#exploratory-data-analysis)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Building](#model-building)
5. [Model Evaluation](#model-evaluation)
6. [Conclusion](#conclusion)
7. [Conclusion](#conclusion)
8. [Project Requirements](#project-requirements)

## Data Preparation

- The original dataset is converted from .xls to .csv format for easier processing.
- Libraries used: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, tabulate, io, ast

## Exploratory Data Analysis

- Basic statistical analysis of the dataset
- Visualization of key features and their relationships with the target variable

## Data Preprocessing

- Handling missing values
- Feature engineering
- Encoding categorical variables
- Scaling numerical features

## Model Building

- Multiple regression models are built and compared:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Support Vector Regressor
- Hyperparameter tuning using GridSearchCV and RandomizedSearchCV

## Model Evaluation

- Models are evaluated using various metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R2) Score
- Cross-validation is performed to ensure robustness

## Conclusion

The Random Forest Regressor is selected as the best-performing model. The final model is saved using joblib for future use.

To use this notebook:

1. Ensure all required libraries are installed
2. Run the cells in order
3. Adjust hyperparameters or try different models as needed

For any questions or improvements, please open an issue or submit a pull request.

## Project Requirements

- **Python Version**: 3.10.0
- **Libraries and Versions**:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - joblib
  - tabulate

## Key Components

1. **Data Handling and Manipulation**:
   - pandas
   - numpy

2. **Data Visualization**:
   - matplotlib
   - seaborn

3. **Machine Learning and Data Preprocessing**:
   - scikit-learn

4. **Model Persistence**:
   - joblib

5. **Formatted Output**:
   - tabulate

6. **Additional Standard Libraries**:
   - io
   - ast

## Development Environment

The project was developed using a Jupyter Notebook environment, as evidenced by the cell structure and markdown formatting in the provided code.

To ensure compatibility and reproducibility, it's recommended to list the exact versions of these libraries in a requirements.txt file or to use a virtual environment management tool like conda or venv.

