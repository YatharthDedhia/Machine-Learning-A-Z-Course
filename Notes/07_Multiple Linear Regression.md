# Machine Learning Course Notes

## Multiple Linear Regression

### Introduction to Building Models Step-by-Step
Welcome to Dodson's ultimate course! Today, we explore the crucial topic of building models step by step. We will cover a framework with various methods, providing a comprehensive understanding of systematic model construction.

#### Evolution from Simple to Complex
Recall the simplicity of the past, leading to straightforward linear regression models. Today's complex datasets demand careful selection of predictors due to data abundance and model complexity.

#### Why Variable Selection Matters
1. **Garbage In, Garbage Out:** Including unnecessary variables leads to unreliable models.
2. **Interpretability:** Explaining numerous variables becomes impractical.

### Methods of Model Construction
Let's delve into five methods:

#### 1. All-In Method
- **Scenario:** Utilize when prior knowledge or external frameworks dictate using all variables.
- **Approach:** Include all variables in the model.

#### 2. Backward Elimination
**Step-by-Step Process:**
1. **Significance Level:** Choose a significance level (default: 5%).
2. **Full Model:** Fit the model with all predictors.
3. **P-Value Examination:** Identify the predictor with the highest P-value.
4. **Elimination:** If P-value > Significance Level, remove the predictor and refit the model.
5. **Iteration:** Repeat steps 3-4 until all remaining variables have P-values below the significance level.

#### 3. Forward Selection
**Step-by-Step Process:**
1. **Significance Level:** Choose a significance level (default: 5%).
2. **Simple Regression Models:** Fit all possible simple regression models.
3. **Select Predictor:** Choose the model with the lowest P-value for one variable.
4. **Iterative Growth:** Add one more variable at a time and select based on P-value.
5. **Termination:** Stop when adding a new variable yields a P-value above the significance level.

#### 4. Bidirectional Elimination (Stepwise Regression)
- **Combination:** Combines elements of backward elimination and forward selection.
- **Dual Significance Levels:** Selects variables based on significance levels for both entry and stay.

#### 5. Score Comparison
- **Exhaustive Approach:** Evaluate all possible models.
- **Criterion Selection:** Choose a goodness-of-fit criterion (e.g., R-squared).
- **Selection:** Pick the model with the best criterion.

#### Choosing Variable Selection Methods
- **Stepwise Regression Terminology:** Sometimes refers to bidirectional elimination, forward selection, and backward elimination collectively.
- **Common Usage:** "Stepwise regression" often implies bidirectional elimination, acknowledging its more comprehensive nature.

### Practical Implementation
For practical exercises, focus on backward elimination. It is efficient, provides a step-by-step understanding, and sets the stage for exploring additional tricks to enhance model robustness.

### Conclusion
With five diverse methods explored, you now have a roadmap for building models step by step. As we delve into practical examples, stay engaged and get ready to apply these methodologies. Building robust models is both a science and an art, and we're here to master it together. Let's dive into the world of multiple linear regression and explore its applications. Get ready for exciting tutorials ahead!

## Multiple Linear Regression: Exploring the Potatoes Dataset

### Introduction
Embark on a practical journey into multiple linear regression using a fascinating dataset. While not predicting potato yields, the principles are universal and applicable to various scenarios.

#### Dataset Overview
- **Objective:** Assist a venture capitalist fund in optimizing investment decisions.
- **Variables:** Profit (Dependent Variable), R&D Spend, Administration Spend, Marketing Spend, State of Operation.

#### Business Challenge
- **Venture Capitalist's Goal:** Maximize profit through strategic investment decisions.
- **Analytical Approach:** Build a model to understand factors influencing profit across diverse companies.

### Multiple Linear Regression Equation
The multiple linear regression equation introduces multiple independent variables:

\[ Y = b_0 + b_1*X_1 + b_2*X_2 + b_3*X_3 + b_4*X_4 + ... + b_n*X_n \]

Here:
- \( Y \) is the dependent variable (Profit).
- \( b_0 \) is the y-intercept or constant.
- \( b_1, b_2, ..., b_n \) are slope coefficients for each independent variable.
- \( X_1, X_2, ..., X_n \) are the corresponding independent variables.

#### Practical Example
Explore real-world applications of multiple linear regression in agriculture through a paper titled "The Application of Multiple Linear Regression and Artificial Neural Network Models for Yield Prediction of Very Early Potato Cultivators before Harvest."

#### Next Steps
Enjoy hands-on tutorials as we apply multiple linear regression to our venture capitalist dataset. Stay curious, and see you in the practical sessions!

## Tutorial Excerpts: Linear Regression Assumptions and Dummy Variables

In these tutorial excerpts, the instructor discusses the assumptions of linear regression and introduces the concept of dummy variables for categorical variables.

### Assumptions of Linear Regression:
1. **Linearity:** Ensure a linear relationship between the dependent variable and each independent variable.

   - Anscombe's quartet illustrates the potential pitfalls of blindly applying linear regression.

2. **Homoscedasticity:** The variance should be constant; avoid cone-shaped patterns in charts.

3. **Multivariate Normality (Normality of Error Distribution):** Along the linear regression line, aim for a normal distribution of data points.

4. **Independence of Observations (No Autocorrelation):** Avoid patterns in the data; rows should be independent.

5. **Lack of Multicollinearity:** Independent variables should not be highly correlated to ensure reliable coefficient estimates.

6. **Outlier Check (Extra Check):** Evaluate whether outliers significantly affect the linear regression line.

### Dummy Variables:
- **Purpose:** Used for categorical variables, such as the "state" in this example.
- **Creation:** Create a new column for each category (e.g., New York, California) and populate with 1s and 0s.
- **Usage:** Use one less dummy variable than the number of categories to avoid the "dummy variable trap."
- **Interpretation:** Dummy variables act like switches, turning "on" or "off" for a specific category, with coefficients being implicit.

### Dummy Variable Trap:
- **Definition:** Including both dummy variables can lead to multicollinearity.
- **Solution:** Always omit one dummy variable. If there are N categories, include N-1 dummy variables. Avoid duplicating variables in the model.

## Multiple Linear Regression

### Introduction
- **Objective**: Predict profit of startups based on R&D spend, administration spend, marketing spend, and state.
- **Dataset**: 50 startups with information on R&D spend, administration spend, marketing spend, state, and profit.

### Assumptions of Linear Regression
1. **Linearity:** Ensure a linear relationship between the dependent variable and each independent variable. Visual inspection of scatter plots is essential.

2. **Homoscedasticity (Equal Variance):** The variance of the errors should be constant across all levels of the independent variable. Check for cone-shaped patterns in residual plots.

3. **Multivariate Normality (Normality of Error Distribution):** Residuals should follow a normal distribution. Observe if residuals form a normal

 distribution along the linear regression line.

4. **Independence of Observations (No Autocorrelation):** Ensure no pattern in the data, and observations are independent. Detect autocorrelation by examining residual plots.

5. **Lack of Multicollinearity:** Independent variables should not be highly correlated with each other. Use variance inflation factor (VIF) to assess multicollinearity.

6. **Outlier Check (Extra Check):** Identify and handle outliers that significantly affect the linear regression model.

### Dummy Variables
- **Purpose:** Encode categorical variables (e.g., state) into numerical format for regression analysis.
- **Procedure**: Create dummy variables, one for each category, and include only (n-1) dummy variables to avoid the dummy variable trap.

### Dummy Variable Trap
- **Issue:** Including all dummy variables can lead to multicollinearity.
- **Solution:** Exclude one dummy variable from each set of dummy variables. \(D_2 = 1 - D_1\) to avoid duplicating variables in the model.

# P Values and Statistical Significance

## Statistical Significance
- **Key Question:** Is the result statistically significant?
- **P Value:** Probability of obtaining the observed results under the assumption that the null hypothesis is true.
- **Significance Level (Alpha):** Commonly set at 5%.

## Hypothesis Testing
- **Null Hypothesis (H-0):** Assumes no effect or no difference.
- **Alternative Hypothesis (H-1):** Assumes an effect or difference.
- **Decision Rule:** If p-value < alpha, reject the null hypothesis.

## P Value Intuition
- Lower P value indicates stronger evidence against the null hypothesis.
- P value < alpha: Reject the null hypothesis.
- P value > alpha: Fail to reject the null hypothesis.

## Confidence Level
- Set the confidence level based on the application's requirements. Commonly used levels: 95%, 99%.

## Conclusion
- P values help determine statistical significance.
- Significance level (alpha) sets the threshold for rejecting the null hypothesis.
- Confidence level reflects the confidence in rejecting the null hypothesis.

# ML Course Video 3: Data Pre-processing and Multiple Linear Regression

## Data Pre-processing

### Importing Libraries and Data
- **Instructor:** Import libraries and dataset.
- **Code**:

```python
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Importing dataset
dataset = pd.read_csv('your_dataset.csv')
```

### Matrix of Features and Dependent Variable
- Obtain matrix of features \(X\) and dependent variable vector \(y\).
- **Code**:

```python
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

### One-Hot Encoding for Categorical Data
- Explain one-hot encoding for categorical variable "State."
- **Code**:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```

### Dummy Variable Trap and Feature Scaling
- Discuss dummy variable trap and why feature scaling is unnecessary for multiple linear regression.
- **Explanation**:
  - Coefficients will handle variable scaling.
  - No need to scale features.

## Multiple Linear Regression

### Model Building
- Build Multiple Linear Regression model using scikit-learn.
- **Code**:

```python
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

### Model Training
- Train the Multiple Linear Regression model on the training set.
- **Code**:

```python
regressor.fit(X_train, y_train)
```

### Model Evaluation
- Discuss the plan to evaluate the model on the test set.
- **Explanation**:
  - Compare real profits with predicted profits for each startup in the test set.

### Conclusion and Next Steps
- Emphasize the efficiency of scikit-learn's linear regression class.
- Preview the evaluation metrics in the next tutorial.
- **Instructor:** Encourage learners to proceed to the next tutorial for further evaluation techniques.

# ML Course Video 4: Predicting Test Set Results

### Introduction
- Briefly introduce the focus on predicting test set results with Multiple Linear Regression.

### Real and Predicted Profits
- Explain the plan to display vectors of real and predicted profits.
- **Code**:

```python
# Get the vector of predicted profits
y_pred = regressor.predict(X_test)
```

### Model Evaluation Visualization
- Discuss the method of visually comparing real and predicted profits.
- **Explanation**:
  - Display vectors side by side for comparison.
  - Evaluate model performance based on closeness of predictions to real results.

### Next Tutorial
- Encourage learners to proceed to the next tutorial for more evaluation techniques.
- **Instructor:** Conclude the video and prompt learners to enjoy machine learning.

## Conclusion

### Backward Elimination for Model Optimization
- **Overview:** Optimize the model by selecting statistically significant variables.
- **Steps:** Significance level, full model fit, find highest P-value, remove variable, fit new model.
- **Implementation:**
  - Select significance level.
  - Fit full model with all predictors.
  - Find predictor with highest P-value above significance level.
  - Remove predictor with highest P-value.
  - Fit new model without removed predictor.
  - Repeat steps until no predictor exceeds significance level.

#### Key Decision Points
- Use significance level to decide whether to remove predictors.
- Evaluate P-values to assess statistical significance.

```python
# Example Code (Step 2):
regressors = X[:, [0, 1, 2, 3, 4]]  # Assuming X contains all features
regressor_ols = sm.OLS(endog=Y, exog=regressors).fit()
summary = regressor_ols.summary()
print(summary)
```

### Homework Assignment
- Complete backward elimination for the provided model.
- Evaluate and decide which predictors to retain.
- Compare results with provided solution in the next tutorial.

## Conclusion
- **ML Toolkit Expansion:** Multiple Linear Regression.
- **Model Selection:** Experiment with different models for efficiency.
- **Next:** Explore polynomial regression for nonlinear datasets.