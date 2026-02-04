First thing we need to understanding it's, how a model learn? Through math.
- 1. When we define a model (like LinearRegression) a math function are defined inside the model
- 2. The model get the splited daya X_TRAIN (with the features) and y_train (with the label)
- 3. The model analysis the features numbers (target) and do predictions (y')
HOW A MODEL DO THESE PREDICTIONS?
- Every math functions have parameters (learn more in https://mathinsight.org/definition/parameter), and the model inicialize these parameters with random numbers, we can say that the model do predictions the number for the parameters
- The number of parameters it's proportional to the n_features, so 15 features == 15 parameters, and the model tries to predict all theses parameters
- So with these random numbers for parameters and the features targets, a complete math function are created, and consequently a result the (y')
- But the model doesn't know if the predictions for the parameters are correct, then a model need other math function to verify that accuracy of parameters numbers, and we call that function **LOSS FUNCTION**, a function that calcule the error of model predictions
- Error in machine learning it's how the model will compare the y' with the y, **Error = L(y', y)**, so doesn't have a fixed math to calculate that, always depends the objective of the model


# LOSS FUNCTION
**DEFINITION**
- A math function inside the model, that measure how far the y' (parameters prediction) is from the y (parameter number)
- There are a many types of loss functions, in other words, a many math functions to calculate the loss, but why? we'll see soon in this page (before you read following content, try to think a little and find the answer)

**HOW THIS WORK?**
- The model tries to predict all the parameters numbers, with others math functions that we'll see later, so the loss function determine if the model are going in a good way or not with the predictions, with a number that we call error
- Every situation have a different type of error that matters more:

```
Example: Delivering pizza late

Error: 2 minutes → no problem
Error: 30 minutes → angry customer

- So in this case a big error (30) will important more than a small (2), and we need to 'alert' this for the model, so we'll use a loss function that penalizes big mistakes, and we'll this math function
- Error = L(y', y) ** 2, so will become [4, 900], and the model will know that error like 30 it's a big mistake

```

- But let's think about, the data have y extremes values (outliers), so the model will try to adjust the paraneters for these outliers, and if will use the same thinking in the last example, the real data will be penalized trying to get a low error for the outliers, and the parameters goes to be a total disaster. So in this case we need to use other loss function, that doesn't penalized.


```
Example: Delivering pizza late

Error: 2 minutes → no problem
Error: 30 minutes → angry customer

- So in this case a the errors nedd to become the lowest (for the model doesn't adjust for outliers), so we don't change them
- Error = L(y', y) , so will become [2, 30], and the model doesn't will penalized so far big mistakes, and it won't adjust as well to outliers

```


**REGULARIZATIONN**
- Regularization is an additional (extra) term added to the loss function. (LOSS = Error + Regularization)
- It penalizes model complexity, usually large coefficients.
- The goal is to reduce overfitting and improve generalization.
- Common Regularization term: L1 (Lasso), L2 (Ridge), ElasticNet (L1 + L2)
- HYPERPARAMETER:

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
    1.  SKLearn Model: HuberRegressor()

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


