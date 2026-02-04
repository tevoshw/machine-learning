First thing we need to understanding it's, how a model learn? Through math.
- When we define a model (like LinearRegression) a math function are defined inside the model
- The model get the splited daya X_TRAIN (with the features) and y_train (with the label)
- The model analysis the features numbers (target) and do predictions (y')
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

Error: 2 minutes → Normal dayta
Error: 30 minutes → OUTLIERS (INVALID DATA)

- So in this case a the errors nedd to become the lowest (for the model doesn't adjust for outliers), so we don't change them
- Error = L(y', y) , so will become [2, 30], and the model doesn't will penalized so far big mistakes, and it won't adjust as well to outliers

```

## IMPORTANT NOTE !
- Loss functions are used during training to guide parameter updates.
- Evaluation metrics (RMSE, R², Accuracy) are computed after training on validation or test data.

# TYPES OF LOSS FUNCTIONS

## MSE (Mean Squared Error)
- **MATH FUNCTION:** $$\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
- **TYPE:** Regression problems
- **LOSS FUNCTION** Loss = Error (MSE)
- **DEFINITION**: Measures the average squared error between predicted and true values
- **WHEN TO USE**: Your data is clean, and you want to punish big mistakes. It's the "standard" choice.
- **HANDS-OK-**:
    1.  SKLearn Model: `LinearRegression()`

## MAE (Mean Absolute Error)
**MATH FUNCTION:** $$\frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
- **TYPE:** Regression problems
- **LOSS FUNCTION** Loss = Error (MAE)
- **DEFINITION**: Measures the average absolute error between predicted values and true values.
- **WHEN TO USE**: You have "crazy" outliers (dirty data) and you don't want them to ruin your model's logic.
- **HANDS-OK-**:
    1.  SKLearn Model: `QuantileRegression()`

## HUBER LOSS
**MATH FUNCTION:**
$$ {\delta}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\\delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{else}\end{cases} $$
- **TYPE:** Regression problems
- **LOSS FUNCTION** Loss = Error (HUBER)
- **DEFINITION**: A loss function that combines MSE and MAE, behaving like MSE for small errors and MAE for large errors.
- **WHEN TO USE**: You have some outliers, but you don't want to go as extreme as MAE. It's the "balanced" middle ground.
- **HANDS-OK-**:
    1.  SKLearn Model: `HuberRegressor()`


Now that we understanding how models kmow if they're doing good or bad predictions for parameter, and know the good of bad way, other problemns appears.
1. The model tries to arrive in a 0 from error, in other words, the model tries to minimize as much as possible the loss function, from 10 to 5, from 5 to 1, from 1 to 0
2. To do that, the model starts to capture the all the data and noise values, adjusting the parameters for the outliers and invalid values too
3. To minimize the loss function, the model can do anything, like starts to jump for have a big numbers for the parameters like, from 1.5 from 1500 (cuz of outliers and more)
4. If you train too much, the model will start 'memoryzing' the data, and not learning, like u just need a simples linear regression, and you you use a complex model (such as a 10th-degree polynomial), the model will create curves in a line model
5. If the model has a little dataset, the model will learning these values, not learning, and others problems, so we need to do something.
How we can treat that?
- We can add more terms in the loss function, that we these 'extras' terms of **REGULARIZATION**, like $$LF = Error + Regularization$$ 
- The regression part of loss function are to avoid overfitting, lot of features and others problems mentioned.

# REGULARIZATION
**DEFINITION**
- "EXTRA TERMS" for the loss functions, that helps the model to avoid overfitting and high parameters numbers.
- Like error math, there are a lot of differents types of regularization terms
- The models that have these extra terms, are renamed, but them still do the samme work (so you can identify if the model has regularization terms, just looking at the models name)


WITHOUT REGULARIZATION:

```
L = 100

- So in this case the error it's 10, but how we're using the MSE turn to 100
```

WITH REGULARIZATION
```
L = 100 + (2 . (100) )
L = 300

- So in this case the error i'ts the same, and we're using a regularization, so the loss function increase, and the model will alert for that choose of parameters i'ts bad, avoid overfitting
```

**ALPHA ($\alpha $)**
- Like we said, there are many differents types of terms in the regularization part, but one it's fixed, and we call that alpha
- It's a hyperparameter that multiply the penalty part and then sum with the error, like:   $$ alpha . (Penalty) $$
- How it's a hyperparameter, we can define that before the train, but for standard the value are 1

# TYPES OF PENALTY

> [!IMPORTANT]
> All of these models, are a know models (liner regression and more), THE ONLY THING THAT CHANGED it's tHE LOSS FUNCTIONS, that add EXTRA TERMS (REGULARIZATION), and CONSEQUENTELY CHANGED THE NAME ( idk why, just accept too :/ ), BUT INSIDE IT'S A KNOW MODEL, THE SAME OBJECTIVE AND MORE.

## LASSO REGULARIZATION (L1)
**PENALTY FUNCTION:**
$$P = \sum_{j=1}^{n} |w_j|$$
- **TYPE:** Regression (Sparsity / Feature Selection)
- **LOSS FUNCTION:** $Loss = Error + Regularization$
- **DEFINITION**: Adds the "absolute magnitude" of coefficients as a penalty term. Unlike L2, the pressure to decrease the weight remains constant regardless of the weight's size.
- **BEHAVIOR**: It can force the coefficients of irrelevant features to be **exactly zero**. This results in a "sparse" model where only the most important features remain.
- **WHEN TO USE**: When you have a high number of features and you suspect that many of them are irrelevant or redundant (noise).
- **HANDS-ON**:
    1. SKLearn Model: `Lasso(alpha=1.0)`

## RIDGE REGULARIZATION (L2)
**PENALTY FUNCTION:**
$$P = \sum_{j=1}^{n} w_j^2$$
- **TYPE:** Regression (Weight Decay)
- **LOSS FUNCTION:** $Loss = Error + Regularization$
- **DEFINITION**: Adds the "squared magnitude" of coefficients as a penalty term to the loss function. Because it squares the weights, large coefficients are penalized much more heavily than small ones.
- **BEHAVIOR**: It shrinks the parameters asymptotically toward zero. The weights become very small (e.g., 0.00001) but **never reach absolute zero**. All features are kept in the model.
- **WHEN TO USE**: When you have many features with small/medium effects and you want to prevent any single one from exploding (handles Multicollinearity well).
- **HANDS-ON**:
    1. SKLearn Model: `Ridge(alpha=1.0)`

## ELASTIC NET (L1 + L2)
**PENALTY FUNCTION:**
$$P = \rho \sum |w| + \frac{\alpha(1-\rho)}{2} \sum w^2$$
- **TYPE:** Hybrid Regression
- **LOSS FUNCTION:** $Loss = Error + Regularization$
- **DEFINITION**: A combination of both L1 and L2 regularization. It uses a second hyperparameter ($\rho$ or `l1_ratio`) to control the mix between the two.
- **WHEN TO USE**: It is the "best of both worlds." Use it when there are multiple features that are correlated with each other; Lasso might pick one at random, while Elastic Net will likely keep both (Ridge effect) while still removing dead weight (Lasso effect).
- **HANDS-ON**:
    1. SKLearn Model: `ElasticNet(alpha=1.0, l1_ratio=0.5)`