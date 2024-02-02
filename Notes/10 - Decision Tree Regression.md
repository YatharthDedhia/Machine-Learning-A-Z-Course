# Decision Tree Regression

## Introduction
- Decision trees: Classification and Regression Trees (CART)
- Two types: Classification Trees and Regression Trees
- Focus on Regression Trees in this section

## Regression Trees Basics
- Scatterplot representation with independent variables (x1, x2) and dependent variable (y)
- Algorithm splits scatterplot into segments (leaves) based on conditions
- Splits determined by information entropy, optimizing data grouping
- Stopping criteria for algorithm to decide when to stop splitting (e.g., minimum information gain)

## Decision Tree Structure
- Nodes represent splits or decisions
- Leaves are terminal nodes with predicted values
- Splits determined by conditions (e.g., x1 < 20, x2 < 170)

## Prediction in Decision Tree
- New observation falls into a terminal leaf
- Predicted value is the average of dependent variable (y) in that leaf

## Implementation Steps in Python
### Step 1: Import Libraries and Dataset
```python
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('your_dataset.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values
```

### Step 2: Training the Decision Tree Regression Model
```python
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, Y)
```

### Step 3: Predicting a New Result
```python
# Predicting a new result
y_pred = regressor.predict([[6.5]])
```

### Step 4: Visualizing Decision Tree Regression Results
```python
# Visualizing the Decision Tree Regression results
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```

### Step 5: High-Resolution Visualization
```python
# Visualizing the Decision Tree Regression results (High Resolution)
X_grid = np.arange(min(X), max(X), 0.01)  # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Decision Tree Regression (High Resolution)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```

## Conclusion
- Decision Tree Regression provides a non-linear model
- No need for feature scaling
- Visualizations show step-wise prediction structure
- Useful for understanding correlations in data with multiple features


Note: Replace 'your_dataset.csv' with the actual name of your dataset file.



## Decision Tree Regression

### Building and Training the Decision Tree Regression Model

- The class used for Decision Tree Regression in Scikit-learn is `DecisionTreeRegressor` from the `tree` module.
- To create an instance of the class, use `DecisionTreeRegressor(random_state=0)`, where `random_state` ensures reproducibility.
- Training the model is done by calling the `fit` method on the created regressor, passing the matrix of features (`X`) and the dependent variable vector (`y`).

```python
from sklearn.tree import DecisionTreeRegressor

# Creating Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=0)

# Training the model
regressor.fit(X, y)
```

### Making Predictions with Decision Tree Regression

- Predictions are made using the `predict` method of the trained Decision Tree Regressor.
- Since no feature scaling is required for Decision Tree Regression, the process is straightforward.
- To predict a new result (e.g., salary for position level 6.5), use `regressor.predict([[6.5]])`.

```python
# Making predictions
predicted_salary = regressor.predict([[6.5]])
print(f"Predicted Salary for Position Level 6.5: ${predicted_salary[0]:,.2f}")
```

### Visualization of Decision Tree Regression Results

- Visualizing Decision Tree Regression results involves plotting the regression curve.
- In 2D, the curve may appear stair-like, with constant predictions within certain ranges of the feature.
- Visualization code is simple and does not require feature scaling transformations.

```python
import matplotlib.pyplot as plt
import numpy as np

# Plotting the Decision Tree Regression results
plt.scatter(X, y, color='red', label='Actual Salaries')
plt.plot(X_grid, regressor.predict(X_grid.reshape(-1, 1)), color='blue', label='Decision Tree Regression')
plt.title('Decision Tree Regression Results')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()
```

**Note:** Decision Tree Regression may not be visually appealing in 2D, but its strength lies in handling higher-dimensional datasets effectively.
