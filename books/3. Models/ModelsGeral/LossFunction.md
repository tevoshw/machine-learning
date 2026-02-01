# LOSS FUNCTION
**DEFINITION**
- An internal model calculation that measures prediction error, we can choose the better for the model.
- It is the criterion the model uses to evaluate how wrong or right its predictions are.
- Based on this value, the optimization algorithm updates the parameters (coefficients and bias).
- It measures how far the prediction (ŷ) is from the true value (y) on the training data.

**REGULARIZATIONN**
- Regularization is an additional (extra) term added to the loss function. (LOSS = Error + Regularization)
- It penalizes model complexity, usually large coefficients.
- The goal is to reduce overfitting and improve generalization.
- Common Regularization term: L1 (Lasso), L2 (Ridge), ElasticNet (L1 + L2)

## IMPORTANT NOTE !
- Loss functions are used during training to guide parameter updates.
- Evaluation metrics (RMSE, R², Accuracy) are computed after training on validation or test data.

# REGRESSION

## MSE (Mean Squared Error)
- **LOSS FUNCTION** Loss = Error (MSE)
- **DEFINITION**: Measures the average squared error between predicted and true values
- **CHARACTERISTICS**:
    1. Big mistakes are heavily penalized.
    2. Very sensitive to outliers.
    3. Differentiable → great for Gradient Descent.
- **WHEN TO USE**: Your data is clean, and you want to punish big mistakes. It's the "standard" choice.
- **HANDS-OK-**:
    1.  SKLearn Model: LinearRegression()

## MAE (Mean Absolute Error)
- **LOSS FUNCTION** Loss = Error (MAE)
- **DEFINITION**: Measures the average absolute error between predicted values and true values.
- **CHARACTERISTICS**:
    0. Regression Models.
    1. All errors are penalized equally.
    2. Less sensitive to outliers than MSE.
    3. Not fully differentiable at zero (gradient is constant).
- **WHEN TO USE**: You have "crazy" outliers (dirty data) and you don't want them to ruin your model's logic.
- **HANDS-OK-**:
    1.  SKLearn Model: QuantileRegression()

## HUBER LOSS
- **LOSS FUNCTION** Loss = Error (HUBER)
- **DEFINITION**: A loss function that combines MSE and MAE, behaving like MSE for small errors and MAE for large errors.
- **CHARACTERISTICS**:
    1. Quadratic for small errors → stable optimization.
    2. Linear for large errors → robust to outliers.
    3. Fully differentiable, suitable for Gradient Descent.
- **WHEN TO USE**: You have some outliers, but you don't want to go as extreme as MAE. It's the "balanced" middle ground.
- **HANDS-OK-**:
    1.  SKLearn Model: HuberRegression()

## LASSO REGRESSION (L1 Regularization)
- **LOSS FUNCTION** Loss = Error (MSE) + Regularization (L1)
- **DEFINITION**: A linear model that adds a penalty term equal to the absolute value of the magnitude of coefficients to the loss function ($Loss = MSE + \alpha \sum |w_j|$).
- **CHARACTERISTICS**:
    1. **Sparsity**: It can force the coefficients of unimportant features to be exactly zero
    2. **Feature Selection**: Acts as an automated feature selection tool by effectively removing irrelevant variables.
    3. **Geometric Nature**: Due to the diamond-shaped constraint, it tends to hit the axes, resulting in zero weights.
- **WHEN TO USE**: You have a "messy" dataset with too many columns (50+) and you want the model to discard the useless ones for you.
- **HANDS-OK-**:
    1.  SKLearn Model: Lasso()
- **REGULARIZATION**:
1. Reset the coeficients

## RIDGE REGRESSION (L2 Regularization)
- **LOSS FUNCTION** Loss = Error (MSE) + Regularization  (L2)
- **DEFINITION**: A linear model that adds a penalty term equal to the square of the magnitude of coefficients to the loss function ($Loss = MSE + \alpha \sum w_j^2$).
- **CHARACTERISTICS**:
    1. **Weight Shrinkage**: It shrinks coefficients towards zero, but they never reach exactly zero.
    2. **Multicollinearity Handling**: Excellent for datasets where input variables are highly correlated; it distributes the weight among them.
    3. **Non-Sparse Solution**: Keeps all features in the model, making it less ideal for feature selection but great for retaining information.
- **WHEN TO USE**: You have too many features (50+) and they are all "fighting" for importance (correlation). Use this to stabilize the model.
- **HANDS-OK-**:
    1.  SKLearn Model: Ridge()
- **REGULARIZATION**:
1. Shrinks the coeficients (but dont reset)

## ELASTIC NET REGRESSION
- **LOSS FUNCTION** Loss = Error (MSE) + Regularization (L1 + L2)
- **DEFINITION**: A regularization technique that combines both L1 (Lasso) and L2 (Ridge) penalties into a single loss function.
- **CHARACTERISTICS**:
    1. **Balanced Penalty**: Uses a convex combination of L1 and L2 (controlled by a ratio parameter).
    2. **Group Effect**: Unlike Lasso, which might pick one variable at random from a group of correlated features, Elastic Net tends to include the whole group.
    3. **Flexibility**: It overcomes the limitations of Lasso when the number of predictors ($n$) is much larger than the number of observations ($m$).
- **WHEN TO USE**: You are in doubt or have groups of correlated variables. It's the "safe bet" when you don't know if you need L1 or L2.
- **HANDS-OK-**:
    1.  SKLearn Model: ElasticNet()
- **REGULARIZATION**:
1. Reset some coeficients, and shrinks others

## TOBIT MODEL (CENSORED REGRESSION)

- **LOSS FUNCTION** LOss = Negative Log-Likelihood (Censored Gaussian)
- **DEFINITION**: A regression model designed for situations where the dependent variable is censored, meaning that values below or above a certain threshold are not fully observed.
- **CHARACTERISTICS**:
    1. **Latent Variable**: Assumes an unobserved continuous variable \( y^* = X\beta + \varepsilon \), with \( \varepsilon \sim \mathcal{N}(0, \sigma^2) \).
    2. **Censoring Mechanism**: The observed variable \( y \) is censored at a known limit (e.g., \( y = 0 \) if \( y^* \le 0 \)).
    3. **Likelihood-Based Estimation**: Combines probability density (PDF) for uncensored observations and cumulative distribution (CDF) for censored ones.
- **WHEN TO USE**: When the dependent variable has a natural censoring point (e.g., income ≥ 0, expenditures with many zeros, detection limits).
- **HANDS-ON**:
    1. Statsmodels: Tobit (via custom MLE)
    2. Other libraries: lifelines, PyTorch (custom likelihood)
- **REGULARIZATION**:
    1. Not inherent to the model
    2. Can be extended with L1 (Lasso Tobit) or L2 (Ridge Tobit) penalties


# ERROR CLASSIFICATION