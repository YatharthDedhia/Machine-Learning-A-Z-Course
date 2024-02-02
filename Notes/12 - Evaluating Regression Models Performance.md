# R Squared in Model Evaluation

## Introduction
- R Squared is a crucial concept for evaluating the goodness of fit in regression models.
- It measures the proportion of the variance in the dependent variable that is predictable from the independent variables.
- R Squared ranges from 0 to 1, with 1 indicating a perfect fit.

## Calculation of R Squared
- **Residual Sum of Squares (RSS):** Represents the sum of squared differences between actual and predicted values.
- **Total Sum of Squares (TSS):** Represents the sum of squared differences between actual values and the mean.
- **Formula:** \( R^2 = 1 - \frac{RSS}{TSS} \)
- R Squared is a ratio of how well the regression line fits the data compared to a simple average line.

## Interpretation of R Squared
- R Squared values range between 0 and 1.
- Closer to 1 implies a better fit, where the model explains more variance.
- Values around 0.9 are considered very good, while below 0.7 may indicate a less effective model.

## Adjusted R Squared
- **Problem:** Adding more independent variables tends to artificially increase R Squared, even if they do not contribute meaningfully.
- **Adjusted R Squared:** Introduces a penalty term for adding new variables, preventing overfitting.
- **Formula:** \( Adjusted R^2 = 1 - \frac{(1 - R^2)(n - 1)}{(n - k - 1)} \)
  - \( n \): Sample size
  - \( k \): Number of independent variables

## Practical Considerations
- Adjusted R Squared helps avoid the pitfall of adding irrelevant variables for the sake of increasing R Squared.
- The decision to add a new variable should be justified by a substantial increase in R Squared compensating for the penalty.

## Conclusion
- R Squared and Adjusted R Squared are valuable metrics for evaluating regression models.
- They provide insights into the goodness of fit and help prevent overfitting by considering the impact of additional variables.

# Model Evaluation: Adjusted R Squared

## Introduction
- Adjusted R Squared builds upon the concept of R Squared but addresses the issue of adding unnecessary variables.
- Adding new variables tends to artificially inflate R Squared, making it a less reliable measure of model improvement.

## Calculation of Adjusted R Squared
- **Formula:** \( Adjusted R^2 = 1 - \frac{(1 - R^2)(n - 1)}{(n - k - 1)} \)
- The penalty term adjusts for the number of independent variables, discouraging the addition of variables that do not significantly improve the model.

## Practical Implications
- Adjusted R Squared guides the decision-making process when considering the inclusion of new variables.
- A higher Adjusted R Squared indicates that the added variables contribute meaningfully to the model.

## Conclusion
- Adjusted R Squared is a refined metric for model evaluation, offering a more nuanced perspective by accounting for the impact of additional variables.
- It promotes a more thoughtful approach to model building, preventing the inclusion of irrelevant variables.
```

These notes cover the key concepts, formulas, and practical considerations for both R Squared and Adjusted R Squared in markdown format.