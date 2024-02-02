# Machine Learning Course: Polynomial Regression

## Introduction
- Welcome back to the machine learning course.
- Today's topic: Polynomial Regression.

## Types of Regressions
1. Simple Linear Regression
   - Represented as \(y = B_0 + B_1x_1\).
2. Multiple Linear Regression
   - Represented as \(y = B_0 + B_1x_1 + B_2x_2 + \ldots + B_nx_n\).
3. Polynomial Linear Regression
   - Represents a variation of multiple linear regression.
   - Uses the same variable \(x_1\) but with different powers.

## Why Polynomial Regression?
- Situations where simple linear regression fails:
  - Data doesn't fit well (e.g., parabolic data).
- Example: Describing disease spread, pandemics, and epidemics.
- Polynomial regression formula: \(y = B_0 + B_1x_1 + B_2x_1^2 + B_3x_1^3 + \ldots\).
- The power of \(x_1\) introduces a parabolic effect, improving data fit.

## Linear vs. Non-linear
- Polynomial regression is a form of linear regression.
- Linear refers to coefficients, not the X variables.
- Example of non-linear regression formula: \(y = \frac{0 + B_1x_1}{B_2 + x_2}\).

## Polynomial Linear Regression as a Special Case
- Polynomial linear regression is a version of multiple linear regression.
- It's not an entirely new type but a specialized case.

## Practical Activity: Polynomial Regression Implementation
### Scenario
- HR department predicting previous salary based on position.
- Dataset: Position_Salaries.

### Implementation Steps
1. Importing Libraries
   - Standard Python data science libraries.
2. Importing Dataset
   - Load Position_Salaries dataset.
3. Skip Data Splitting
   - Retain maximum data for predicting salary.
4. Train Linear Regression Model
   - On the entire dataset.
5. Train Polynomial Regression Model
   - On the entire dataset.
6. Visualize Results
   - Compare linear and polynomial regression.
7. Make Predictions
   - Predict salary using both models for a position level.

## Conclusion
- Polynomial regression adapts to non-linear relationships.
- Valuable tool in various scenarios.
- Polynomial linear regression is a specialized form of multiple linear regression.

## Next Steps
- Further exploration in upcoming tutorials.
- Enhance skills in real-world and complex datasets.

## Instructor's Note
- Polynomial regression enhances model flexibility.
- Understand the dataset and choose regression types accordingly.


### Polynomial Regression Model Implementation

The instructor continues by guiding the audience through the implementation of the polynomial regression model. The key steps involve:

1. **Importing PolynomialFeatures Class:**
   - Importing the `PolynomialFeatures` class from the `pre-processing` module of the scikit-learn library. This class is crucial for creating a matrix of powered features.

    ```python
    from sklearn.preprocessing import PolynomialFeatures
    ```

2. **Creating Polynomial Regressor Object:**
   - Creating an instance of the `PolynomialFeatures` class. This object, named `poly_reg`, is responsible for generating the powered features matrix.
   - Specifying the degree (`N`) for the polynomial regression model; initially, it is set to 2.

    ```python
    poly_reg = PolynomialFeatures(degree=2)
    ```

3. **Transforming Features Matrix:**
   - Applying the `fit_transform` method of `poly_reg` to transform the original matrix of features (`X`) into a new matrix (`X_poly`) containing features at different powers.

    ```python
    X_poly = poly_reg.fit_transform(X)
    ```

4. **Creating Polynomial Regressor Object for Training:**
   - Creating a new linear regressor object (`lin_reg2`) to integrate the powered features into the model. This is distinct from the linear regressor used earlier (`lin_reg`).

    ```python
    lin_reg2 = LinearRegression()
    ```

5. **Training the Polynomial Regression Model:**
   - Using the `fit` method of `lin_reg2` to train the polynomial regression model on the new matrix of powered features (`X_poly`) and the dependent variable vector (`y`).

    ```python
    lin_reg2.fit(X_poly, y)
    ```

6. **Visualization of Polynomial Regression Results:**
   - Transitioning to visualizing the polynomial regression results using Matplotlib. The code for plotting real results (red points) and predictions (blue curve) is similar to the linear regression visualization.

    ```python
    plt.scatter(X, y, color='red', label='Real Results')
    plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue', label='Polynomial Regression')
    ```

7. **Enhancing the Polynomial Regression Graph:**
   - Adding a title, X label, and Y label to improve the overall appearance of the graph.

    ```python
    plt.title("Truth or Bluff (Polynomial Regression)")
    plt.xlabel("Position Level")
    plt.ylabel("Salary")
    plt.show()
    ```

These steps collectively demonstrate the construction and training of a polynomial regression model, offering more flexibility in capturing non-linear relationships within the dataset.


## Notes for Linear Regression and Polynomial Regression Implementation

### Linear Regression Implementation:

1. **Introduction:**
   - Linear regression model implemented using scikit-learn library.
   - Focus on predicting salaries based on position levels.

2. **Building the Linear Regression Model:**
   - Used `LinearRegression` class from the `linear_model` module of scikit-learn.
   - Created a `lin_reg` object as an instance of the `LinearRegression` class.
   - Utilized the `fit` method to train the model on the entire dataset.

3. **Visualization of Linear Regression Results:**
   - Plotted real salaries (red points) against position levels.
   - Plotted the linear regression line (blue) using the `plot` function.
   - Highlighted the limitations of the linear regression model for the given dataset.
   
### Polynomial Regression Implementation:

1. **Introduction:**
   - Extended the regression model to polynomial regression to capture non-linear relationships.
   - Used the `PolynomialFeatures` class from the `preprocessing` module of scikit-learn.

2. **Building the Polynomial Regression Model:**
   - Created a `poly_reg` object as an instance of the `PolynomialFeatures` class.
   - Applied the `fit_transform` method to transform the single-feature matrix into a matrix with powers of the feature.
   - Created a new linear regressor (`lin_reg_2`) to train on the transformed matrix of features.

3. **Visualization of Polynomial Regression Results:**
   - Displayed the polynomial regression curve, which fits the data better.
   - Showed the need for adjusting the power of the polynomial based on the dataset.

4. **Improving the Curve Visualization:**
   - Implemented a smoother curve by densifying the points for visualization.
   - Provided additional code for enhanced graph aesthetics.

### Prediction using Regression Models:

1. **Prediction using Linear Regression:**
   - Showed the process of predicting the salary for a position level using the linear regression model.
   - Highlighted the limitation of the linear model in this context.

2. **Prediction using Polynomial Regression:**
   - Explained the necessary adjustments for predicting using the polynomial regression model.
   - Emphasized the improved accuracy of predictions using polynomial regression.

3. **Comparing Predictions:**
   - Compared the predicted salary using linear regression (330K) with the actual salary (160K).
   - Demonstrated the more accurate prediction using polynomial regression (159K).

### Conclusion:

- Successfully implemented linear and polynomial regression models for predicting salaries.
- Polynomial regression provided a more accurate fit for the given dataset.
- Prepared for the next section, where Support Vector Regression (SVR) would be introduced and implemented.
- Encouraged the learner to continue exploring non-linear regression models.



