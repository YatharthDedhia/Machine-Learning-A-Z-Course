# Machine Learning A to Z Course - Support Vector Regression

## Section 1: Support Vector Regression Intuition

- Support Vector Regression (SVR) invented by Vladimir Vapnik in the '90s.
- Discussed in Vladimir Vapnik's book "The Nature of Statistical Learning, 1992."
- Covers Support Vector Machine for classification and Support Vector Regression.
- Kernel Support Vector Machine, Kernel Support Vector Regression, and the Kernel Trick will also be discussed.

### Linear Support Vector Regression

- Ordinary least squares method used in simple linear regression to minimize errors.
- SVR uses a tube (epsilon-insensitive tube) around the regression line.
- Epsilon is the width of the tube, and errors within the tube are disregarded.
- Errors outside the tube are measured as distances between the point and the tube.
- Support vectors are points outside the tube, influencing the tube's formation.
- SVR minimizes the sum of distances between points and the tube.

### Additional Reading
- "Efficient Learning Machines, Theories, Concepts and Applications for Engineers and System Design" by Mariette Awad and Rahul Khanna.
- Chapter 4: Support Vector Regression.

## Section 2: Non-linear Support Vector Regression Heads-Up

- Overview of non-linear SVR.
- Linear SVR intuition covered, non-linear SVR models mentioned.
- Non-linear SVR models and concepts will be covered later in the course.
- Two options for learners: proceed to non-linear SVR Python tutorials or go through additional tutorials for intuition first.

## Section 3: Practical Activity - Support Vector Regression Implementation

- Building a Support Vector Regression model in Python.
- Focus on feature scaling due to the implicit equation in SVR.
- Dataset: Predicting salary based on position level.
- Comparison with Polynomial Regression model.
- Visualization of SVR results in low and high resolution.

### Key Implementation Steps:

1. Importing libraries.
2. Loading the dataset.
3. Applying feature scaling.
4. Training the SVR model on the entire dataset.
5. Predicting salary for a given position level (e.g., 6.5).
6. Visualizing SVR results in low and high resolution.

**Note**: SVR model performance is compared with Polynomial Regression on the same dataset.



# Support Vector Regression (SVR) - Part 1

## Overview
- SVR is a regression technique introduced by Vladimir Vapnik in the '90s.
- It's part of the Support Vector Machine (SVM) family.
- Covered in "The Nature of Statistical Learning, 1992" by Vladimir Vapnik.
- In this course: Support Vector Machine (classification), Support Vector Regression, Kernel Support Vector Machine, Kernel Support Vector Regression, Kernel Trick, etc.

## Linear Support Vector Regression
- SVR uses a tube around the regression line called the epsilon-insensitive tube.
- The tube has a width of epsilon measured vertically along the axis.
- Points inside the tube are disregarded in terms of error.
- Points outside the tube contribute to the error calculation, measured as the distance between the point and the tube.
- Distances are represented as xC* if below the tube, xC if above, known as slack variables.
- Objective: Minimize the sum of these distances.

### Formulas
- OLS (Ordinary Least Squares) in simple linear regression:
  \[ \min_{a, b} \sum_{i=1}^{n} (y_i - (a + b \cdot x_i))^2 \]
- SVR Objective:
  \[ \min_{\textbf{w}, b, \zeta, \zeta^*} \frac{1}{2} \|\textbf{w}\|^2 + C \sum_{i=1}^{n} (\zeta_i + \zeta_i^*) \]
  subject to: 
  \[ \begin{align*} y_i - (\textbf{w} \cdot \textbf{x}_i + b) &\leq \varepsilon + \zeta_i^* \\ (\textbf{w} \cdot \textbf{x}_i + b) - y_i &\leq \varepsilon + \zeta_i \\ \zeta_i, \zeta_i^* &\geq 0 \end{align*} \]

## Support Vectors
- Points outside the tube are support vectors.
- They dictate the tube's structure.
- These vectors support the formation of the tube.
- Key feature: Flexibility by allowing some error inside the tube.

## Additional Reading
- Chapter 4, "Support Vector Regression" in "Efficient Learning Machines" by Mariette Awad and Rahul Khanna.



# Support Vector Regression (SVR) - Part 2

## Non-linear Support Vector Regression
- Extension beyond Linear SVR.
- Various models covered, including RBF kernel SVR and the kernel trick.
- Python tutorials will demonstrate non-linear SVR with the RBF kernel.

## Heads-up for Non-linear SVR
- Atlan's Python tutorials will feature non-linear SVR right after this section.
- Non-linear SVR topics include SVM Intuition, Kernel SVM Intuition, Mapping to Higher Dimension, Kernel Trick, Types of Kernel Function, and Non-linear Kernel SVR.

### Choice for Learners
1. Follow Atlan's Python tutorials directly for practical implementation.
2. Gain intuition first by going through the theoretical tutorials mentioned above and then dive into the practical side.

## Important Reminder
- Non-linear SVR content comes at the end of a series of tutorials.
- Choices: Direct implementation or understanding theoretical intuition first.

---

# Support Vector Regression (SVR) - Part 3

## Practical Implementation with Atlan
- Advanced SVR model with feature scaling.
- Expect to master feature scaling during this implementation.

## Feature Scaling Considerations
- Apply feature scaling to both the feature matrix \(X\) and the dependent variable vector \(Y\).
- Unlike previous situations, here, \(Y\) takes high values compared to feature values.
- Reshape \(Y\) into a 2D array for compatibility with standard scaling.

### Feature Scaling Formulas
- Standardization Formula:
  \[ x_{\text{standardized}} = \frac{x - \text{mean}(X)}{\text{std}(X)} \]
- Reshape \(Y\) into a 2D array: \(Y = Y.\text{reshape}(-1, 1)\).

## Next Steps
- Feature scaling is a critical step for SVR.
- Future steps include model training, predictions, and inverse scaling for result interpretation.

```

```markdown
# Support Vector Regression (SVR) - Part 4

## Feature Scaling Implementation

### Reshaping Dependent Variable \(Y\)
- \(Y\) needs to be reshaped into a 2D array for compatibility with standard scaling.
- Apply the reshape function: \(Y = Y.\text{reshape}(-1, 1)\).

```python
# Reshape Y into a 2D array
Y = Y.reshape(-1, 1)
print("Reshaped Y:")
print(Y)
```

### Standard Scaling
- Apply standard scaling to both the feature matrix \(X\) and the dependent variable \(Y\).
- Ensure both \(X\) and \(Y\) are in the same range for effective model training.

```python
from sklearn.preprocessing import StandardScaler

# Standardize feature matrix X
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Standardize dependent variable Y
sc_Y = StandardScaler()
Y = sc_Y.fit_transform(Y)

print("Standardized X:")
print(X)
print("Standardized Y:")
print(Y)
```

## Feature Scaling Output
- After standard scaling, both \(X\) and \(Y\) are in a standardized form.
- This ensures that the SVR model can effectively learn the correlations between position levels and salaries.

### Important Note
- Feature scaling is crucial for SVR, especially when the dependent variable (\(Y\)) has significantly higher values than the features.
- Next steps involve SVR model training, making predictions, and inverse scaling for result interpretation.

# Support Vector Regression (SVR) - Part 5

## Conclusion of Feature Scaling
- Feature scaling implemented successfully.
- Both feature matrix \(X\) and dependent variable \(Y\) are standardized using standard scaling.
- Essential for effective training and accurate predictions in SVR.
- Next steps: Proceed to SVR model training and predictions.


# Support Vector Regression (SVR) - Part 6

## Model Training and Predictions

### SVR Model Creation
- Create an SVR model using the radial basis function (RBF) kernel.
- Fit the model with the standardized feature matrix \(X\) and dependent variable \(Y\).

```python
from sklearn.svm import SVR

# Create SVR model with RBF kernel
regressor = SVR(kernel='rbf')

# Fit the model with standardized X and Y
regressor.fit(X, Y)
```

### Making Predictions
- Use the trained SVR model to make predictions.
- Apply inverse scaling to get predictions in the original scale.

```python
# Predict the result
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))

# Inverse scaling to get predictions in the original scale
y_pred = sc_Y.inverse_transform(y_pred)

print("Predicted Salary for Position Level 6.5:")
print(y_pred)
```

## Result Interpretation
- The SVR model is trained and used to predict the salary for a specific position level (6.5).
- Inverse scaling applied to obtain predictions in the original scale.

### Next Steps
- Evaluate the model, visualize results, and understand the SVR predictions for different position levels.

# Support Vector Regression (SVR) - Part 7

## Model Evaluation and Visualization

### Evaluating SVR Model
- Assess the performance of the SVR model using appropriate metrics.
- Metrics may include Mean Squared Error (MSE), R-squared, etc.

```python
# Evaluate the model (use appropriate evaluation metrics)
# Example: mse = mean_squared_error(true_values, predicted_values)
```

### Visualizing SVR Results
- Plot the SVR results, including the original data points and the SVR regression line.

```python
# Visualize the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y), color='red', label='Original Data')
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor.predict(X)), color='blue', label='SVR Regression Line')
plt.title('SVR Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()
```

## Conclusion
- SVR model evaluated and results visualized.
- Understanding the SVR predictions for different position levels.
- Concludes the implementation of SVR for predicting salaries based on position levels.

# Support Vector Regression (SVR) - Part 8

## Inverse Scaling for Result Interpretation

### Inverse Scaling of Predictions
- After obtaining predictions using SVR, it's crucial to inverse scale them for result interpretation.
- This step brings predictions back to the original scale.

```python
# Inverse scaling of predictions
y_pred_original_scale = sc_Y.inverse_transform(y_pred)

print("Predicted Salary for Position Level 6.5 (Original Scale):")
print(y_pred_original_scale)
```

## Result Interpretation
- Predictions from the SVR model are now in the original scale.
- This step ensures that the final results are easily interpretable and applicable in real-world scenarios.

### Conclusion of SVR Implementation
- SVR model successfully implemented, trained, and evaluated.
- Predictions obtained and interpreted in the original scale for practical use.

# Support Vector Regression (SVR) - Part 9

## Final Thoughts

### Recap
- Explored Linear SVR, Non-linear SVR, and practical implementation with feature scaling.
- Covered essential steps: reshaping, standard scaling, model training, predictions, evaluation, and inverse scaling.

### Next Steps
- Apply SVR in real-world scenarios, customize for specific datasets, and explore advanced topics like hyperparameter tuning.

### Conclusion
- Completes the SVR module in the machine learning course.
- Next tutorials may cover additional regression techniques, ensemble methods, or other advanced topics.

```

# Machine Learning Course - Additional Notes

## Feature Scaling Recap

### Standardization Formula
\[ x_{\text{standardized}} = \frac{x - \text{mean}(X)}{\text{std}(X)} \]

### Reshape Dependent Variable \(Y\) into a 2D Array
```python
# Reshape Y into a 2D array
Y = Y.reshape(-1, 1)
```

### Standard Scaling Implementation
```python
from sklearn.preprocessing import StandardScaler

# Standardize feature matrix X
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Standardize dependent variable Y
sc_Y = StandardScaler()
Y = sc_Y.fit_transform(Y)
```

### Inverse Scaling of Predictions
```python
# Inverse scaling of predictions
y_pred_original_scale = sc_Y.inverse_transform(y_pred)
```

## SVR Implementation

### SVR Model Creation
```python
from sklearn.svm import SVR

# Create SVR model with RBF kernel
regressor = SVR(kernel='rbf')

# Fit the model with standardized X and Y
regressor.fit(X, Y)
```

### Making Predictions
```python
# Predict the result
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))

# Inverse scaling to get predictions in the original scale
y_pred = sc_Y.inverse_transform(y_pred)
```

### Evaluating SVR Model
```python
# Evaluate the model (use appropriate evaluation metrics)
# Example: mse = mean_squared_error(true_values, predicted_values)
```

### Visualizing SVR Results
```python
# Visualize the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y), color='red', label='Original Data')
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor.predict(X)), color='blue', label='SVR Regression Line')
plt.title('SVR Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()
```

### Inverse Scaling for Result Interpretation
```python
# Inverse scaling of predictions
y_pred_original_scale = sc_Y.inverse_transform(y_pred)
```

## Conclusion
- Comprehensive notes covering Linear SVR, Non-linear SVR, feature scaling, model training, predictions, evaluation, and result interpretation.
- Ready to apply SVR in real-world scenarios and explore advanced machine learning topics.

```

These comprehensive notes cover various aspects of Support Vector Regression (SVR) implementation, including linear and non-linear SVR, feature scaling, model training, predictions, evaluation, and result interpretation. The provided markdown code snippets can be easily integrated into your course materials.



## SVR Model Training and Prediction

### SVR Model Training
```python
# Import SVR class from sklearn.svm module
from sklearn.svm import SVR

# Create SVR regressor with RBF kernel
regressor = SVR(kernel='rbf')

# Fit the SVR model on the entire dataset
regressor.fit(X, Y)
```

### Making Predictions with SVR Model
```python
# Predict the result for the position level 6.5
scaled_prediction = regressor.predict(sc_X.transform(np.array([[6.5]])))

# Inverse transform to get the prediction in the original scale
original_scale_prediction = sc_Y.inverse_transform(scaled_prediction.reshape(-1, 1))
```

### SVR Result Interpretation
- The predicted salary for position level 6.5 is approximately $170,370.

## Next Steps - Visualizing SVR Results
- In the next tutorial, we will visualize the SVR results and further interpret the model's performance. This involves using `SC_X.inverse_transform` and `SC_Y.inverse_transform` for visualization purposes.



## Visualizing SVR Results

### Adaptation of Polynomial Regression Visualization Code

#### Code Cell 1 - Visualization Basics
```python
# Replace "polynomial regression" by "SVR"
# Reverse scaling for X and Y to get original scale
X = sc_X.inverse_transform(X)
Y = sc_Y.inverse_transform(Y)
```

#### Code Cell 2 - Visualization with Original Input and Predictions
```python
# Replace polynomial regression prediction with SVR prediction
# Adapt input to X, as it is already scaled
scaled_prediction = regressor.predict(X)
# Inverse transform to get predictions in original scale
original_scale_prediction = sc_Y.inverse_transform(scaled_prediction.reshape(-1, 1))
```

#### Code Cell 3 - High-Resolution Visualization
```python
# Reverse scaling for x_grid to get original scale
x_grid = np.array(x_grid).reshape(-1, 1)
x_grid = sc_X.inverse_transform(x_grid)

# SVR prediction for high resolution x_grid
scaled_predictions_high_res = regressor.predict(sc_X.transform(x_grid))
# Inverse transform to get predictions in original scale
predictions_high_res = sc_Y.inverse_transform(scaled_predictions_high_res.reshape(-1, 1))
```