# Preparing Data for Machine Learning Algorithms

## 1. Why Do We Need to Split the Data?

### The Fundamental Problem: Overfitting vs Underfitting

When we train an ML model, we want it to **generalize well** on new data it has never seen. There are two critical problems:

#### 1. Overfitting (Over-fitting)
```
The model "memorizes" the training data instead of learning patterns
- Learns noise and specific peculiarities of the data
- Excellent performance on training, terrible on new data
- Example: model memorizes that customer with ID 42 always buys
  → on new customers, it fails completely
```

**When it occurs:**
- Model too complex for the amount of data
- Training for too long
- Features too specific or noisy

#### 2. Underfitting (Under-fitting)
```
The model is too simple and doesn't capture important patterns
- Doesn't learn even on training data
- Poor performance on training AND on new data
- Example: linear model trying to predict non-linear relationship
```

**When it occurs:**
- Model too simple
- Insufficient data
- Non-representative features

### The Solution: Validation with Separate Data

To ensure the model generalizes well, we need **data the model never saw during training**. If we evaluate only on data it saw, we have an illusion of quality.

```
WRONG:
  1. Train on full dataset
  2. Test on same dataset
  3. Result: 99% accuracy (but it's memorization!)
  4. In production: fails

CORRECT:
  1. Split data into train and test
  2. Train on train set
  3. Test on test set (data model never saw)
  4. Realistic result
  5. In production: works as expected
```

### Why Not Just Train and Test?

If we have train and test, why do we need validation too?

```
During training, we adjust:
- MODEL PARAMETERS (weights, tree splits)
  → learned automatically by the algorithm
  
- HYPERPARAMETERS (learning rate, tree depth, K in KNN)
  → adjusted MANUALLY by us

If we adjust hyperparameters looking at test results:
- We're "using" test as indirect train
- Overfitting also happens in hyperparameters
- Test is no longer "unseen data"

Solution: VALIDATION
- Separate data to adjust hyperparameters
- Test remains completely untouched until the end
```

### X and Y: Separating Features and Target

Before splitting into train/test, we separate **features (X)** from **target (Y)**:

```python
# X: independent features (what model uses to predict)
X = df.drop("price")  # all columns EXCEPT the target

# Y: dependent target (what we want to predict)
Y = df["price"]  # only the column we want to predict

# Quick verification
print(X.shape)  # (1000, 15) → 1000 samples, 15 features
print(Y.shape)  # (1000,) → 1000 samples, 1 target
```

### Random Seed: Reproducibility

When we split data randomly, different runs generate different splits. To **reproduce results exactly**, we use a seed:

```python
# Without seed: different executions = different splits
train_test_split(X, y)  # run 1: [idx 2, 5, 1, 8...]
train_test_split(X, y)  # run 2: [idx 7, 3, 9, 1...]

# With seed: always same division
train_test_split(X, y, random_state=42)  # run 1: [idx 2, 5, 1, 8...]
train_test_split(X, y, random_state=42)  # run 2: [idx 2, 5, 1, 8...]
                                          # IDENTICAL!
```

**Why use a seed?**
- Your colleagues can reproduce your work
- You can reproduce your results later
- Comparisons between models are fair (same data)

---

## 2. Types of Sets and Their Characteristics

### 2.1 Training Set

#### What is it?
The largest portion of the data, used to **teach the model** to find patterns and relationships in the data.

#### What is it for?
- The algorithm adjusts all **model parameters** (weights in neural network, splits in tree, centroids in clustering)
- The model "learns" the relationships between X and Y

#### Main characteristics:
```
Size: 60-80% of total data
  - Example: 1000 samples → 600-800 in training
  - Must be large enough for model to learn
  - Too small → underfitting

Objective: Reduce training error
  - Model optimizes to be good on this data
  - High performance here is expected

Problem: Doesn't measure generalization
  - High performance on training DOESN'T mean generalization
  - Model might be overfitting
```

#### Practical example:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# X_train has 80% of samples
# X_test has 20% of samples

# Train
model.fit(X_train, y_train)

# Check performance on training
train_accuracy = model.score(X_train, y_train)
print(f"Train Accuracy: {train_accuracy}")  # Usually high: 0.95
```

---

### 2.2 Validation Set (Development Set)

#### What is it?
Data separated to **adjust hyperparameters** and monitor if the model is overfitting during training.

#### What is it for?
1. **Hyperparameter Tuning**: choose learning rate, tree depth, regularization
2. **Early Stopping**: stop training when validation performance worsens
3. **Overfitting Detection**: compare train vs validation performance

#### Main characteristics:
```
Size: 10-20% of data
  - Created from training set
  - Less data than train, but enough to evaluate
  - Example: 1000 samples → 800 train, 100 val, 100 test

Creation: HOLDOUT from training set
  - Data that did NOT participate in training
  - Guarantee it wasn't "memorized"

Critical: We DON'T change model parameters here
  - We only adjust HYPERPARAMETERS (external settings)
  - We don't refit the model based on validation
  - (except in methods like early stopping which is automatic)

Objective: Smart tuning
  - Avoids adjusting hyperparameters looking at test
  - Validates patterns before evaluating on test
```

#### When validation is NOT needed:
- Very large datasets (model won't overfit easily)
- Hyperparameters already well-known
- Simple problems with simple models

#### When validation IS needed:
- Small/medium dataset (< 100k samples)
- Many hyperparameters to adjust
- Complex model (neural networks, ensemble)
- Performance on train very different from test

#### Practical example:
```python
from sklearn.model_selection import train_test_split

# 1st Split: train + val vs test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2nd Split: train vs val (from temporary set)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
# Train: 640 | Val: 160 | Test: 200

# Train and use validation for monitoring
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate on each set
train_score = model.score(X_train, y_train)
val_score = model.score(X_val, y_val)
test_score = model.score(X_test, y_test)

print(f"Train: {train_score:.3f} | Val: {val_score:.3f} | Test: {test_score:.3f}")
# Expected: Train > Val > Test (slightly)
# If Train >> Val: overfitting!
```

---

### 2.3 Test Set

#### What is it?
Data completely **separated and untouched** until the end, used to evaluate the final performance of the model on "never seen" data.

#### What is it for?
- Measure **realistic performance** of the model
- Simulate how the model will behave in production
- Final validation before deploying to production

#### Main characteristics:
```
Size: 10-20% of data
  - Smaller: fewer data to test, noisier results
  - Larger: fewer data to train, underfitting
  - Golden ratio: 80/20 (train/test) in general

Untouchable until the end
  - Never use to adjust hyperparameters
  - Never observe during development
  - First "observation": final evaluation
  - Any contact = contamination

Objective: Measure real generalization
  - If test performance ≈ train performance: good generalist
  - If test performance << train performance: overfitting

Metrics for test:
  - Classification: Accuracy, Precision, Recall, F1, ROC-AUC
  - Regression: MAE, RMSE, R², MAPE
  - Always use metric appropriate to problem
```

#### What NOT to do with test:
```
WRONG: Observe test during development
  - Adjust hyperparameters based on test
  - Do feature engineering looking at test error
  - Remove features because test performed poorly
  → This is overfitting on hyperparameters!

WRONG: Use test for cross-validation
  - GridSearchCV on test data
  - Adjust K in KNN observing test
  - Any adjustment based on test is invalid

CORRECT: Evaluate only once
  - After all optimization is complete
  - Use test as FINAL and UNIQUE evaluation
```

#### Practical example:
```python
# Train model with final hyperparameters
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# FIRST AND ONLY observation of test
y_pred = model.predict(X_test)

# Evaluate performance
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# This is the FINAL report you report
```

---

## 3. Splitting Strategies

### Hold-Out (Simple Split)

#### What is it?
Split the dataset **only once** with fixed sets and defined roles.

```
Original Dataset
       ↓
    ┌──────┴──────┐
    ↓             ↓
  Train (80%)   Test (20%)
```

#### Characteristics:
```
Advantages:
- Simple to implement
- Fast computationally
- Good for large datasets (100k+)
- Easy to understand and explain

Disadvantages:
- On small datasets, might not be representative
- One "bad" split (by chance) hurts entire evaluation
- Doesn't use all data for train and validation
- Performance can vary greatly depending on split
```

#### When to use:
- Large datasets (100k+ samples)
- Computational time is critical
- Data already well-distributed

#### When NOT to use:
- Small datasets (< 1000 samples)
- Problems where representativeness is crucial (imbalanced data)

#### Implementation:
```python
from sklearn.model_selection import train_test_split

# Simple Hold-Out: 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Hold-Out with validation: 60/20/20
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)
# 0.25 of 80% = 20%
```

---

### K-Fold Cross-Validation

#### What is it?
Split data into **K parts** and train K times, each time using a different part as validation.

```
Original Dataset (divided into 5 parts)
┌─────┬─────┬─────┬─────┬─────┐
│ 1   │ 2   │ 3   │ 4   │ 5   │
└─────┴─────┴─────┴─────┴─────┘

Fold 1: Val=1,    Train=2,3,4,5
Fold 2: Val=2,    Train=1,3,4,5
Fold 3: Val=3,    Train=1,2,4,5
Fold 4: Val=4,    Train=1,2,3,5
Fold 5: Val=5,    Train=1,2,3,4

Final result = Average of all metrics
```

#### Characteristics:
```
Advantages:
- Uses more data for both training AND validation
- More representative on small datasets
- Reduces variance of evaluation (multiple measurements)
- Excellent for hyperparameter tuning
- Less sensitive to "bad" split by chance

Disadvantages:
- MUCH slower (trains K times)
- More complex to implement
- Not recommended for VERY large datasets
- Higher memory consumption
```

#### When to use:
- Small/medium datasets (< 100k)
- When you need careful hyperparameter tuning
- When you want more confidence in results
- ML competitions (Kaggle, etc)

#### When NOT to use:
- Very large datasets (computational time)
- When training time is critical
- Very heavy models (large neural networks)

#### Implementation:
```python
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

# KFold with 5 folds
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate model using cross-validation
scores = cross_val_score(
    model,
    X, y,
    cv=kfold,
    scoring='accuracy'
)

print(f"Scores per fold: {scores}")
print(f"Mean: {scores.mean():.3f}")
print(f"Standard deviation: {scores.std():.3f}")

# Example output:
# Scores per fold: [0.91, 0.93, 0.89, 0.92, 0.90]
# Mean: 0.910
# Standard deviation: 0.015
```

#### Tuning with KFold:
```python
from sklearn.model_selection import GridSearchCV

# GridSearchCV uses K-Fold internally
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 1]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,  # 5-fold cross-validation
    n_jobs=-1  # use all cores
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_}")

# Evaluate on test (data never seen before)
test_score = grid_search.score(X_test, y_test)
print(f"Test score: {test_score}")
```

---

### Stratified K-Fold

#### What is it?
K-Fold that **maintains class proportions** in each fold. Essential for **imbalanced** problems.

```
Problem: K-Fold normal on imbalanced data
Original data: 95% class 0, 5% class 1
Fold 1: Might have 100% class 0 → biased validation!

Solution: Stratified K-Fold
Fold 1: 95% class 0, 5% class 1 ✓
Fold 2: 95% class 0, 5% class 1 ✓
Fold 3: 95% class 0, 5% class 1 ✓
All folds have same distribution
```

#### Characteristics:
```
Advantages:
- Each fold is representative of the whole
- Fair evaluation on imbalanced data
- Reduces variance between folds

Disadvantages:
- Slightly slower (sorts data)
- Requires Y to be categorical
```

#### When to use:
- **ALWAYS when doing classification with imbalanced classes**
- Standard in competitions (Kaggle, etc)
- When you want reliable results

#### When NOT to use:
- Regression (no "classes")
- Already well-balanced data (normal K-Fold is ok)

#### Implementation:
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Stratified 5-Fold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate
scores = cross_val_score(
    model,
    X, y,
    cv=skfold,
    scoring='f1'  # Use F1 for imbalanced, not accuracy!
)

print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.3f}")
```

---

## 4. Common Problems and How to Avoid Them

### Data Leakage (Information Leakage)

#### What is it?
When data that **should be invisible** to the model during training **leaks** and contaminates the training.

#### Consequences:
```
High performance on tests
  - 99% Accuracy but fails in production
  
Model memorized instead of learned
  - Gained "forbidden" information during training
  - Non-generalizable patterns
  
Illusory metrics
  - Report shows 95% accuracy
  - But data had information from future/test
```

#### Common types of leakage:

**1. Test → Train Leakage**
```python
# WRONG: Preprocess BEFORE splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses ALL data!
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
# Training learned the scale of test data

# CORRECT: Split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
scaler.fit(X_train)  # Only on training
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**2. Future Information Leakage**
```python
# WRONG: Use future information to predict future
# Sales data + sale result on same line
X = [price, quantity, PROFIT_FROM_SALE]  # PROFIT is the future!
y = [sale_success]  # Sale success

# CORRECT: Use only available information
X = [price, quantity, seller_experience]
y = [sale_success]
```

**3. OverSampling Leakage**
```python
# WRONG: Do oversampling BEFORE splitting
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)
# Synthetic training data appears in test!

# CORRECT: Do oversampling AFTER splitting
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)
# Training has synthetic data, test is 100% real
```

#### How to detect leakage:

```python
# Warning signs:
# 1. Test performance >> train performance
train_score = 0.75
test_score = 0.94  # Very different! Suspicious!

# 2. Unrealistic metrics (100% accuracy)
# Rarely happens without leakage

# 3. Feature highly correlated with target
# Especially if that feature wouldn't make sense
feature_importance = [0.01, 0.02, 0.97]
# One feature with 97% importance? Might be leakage!

# 4. Model works in train but fails in production
# Classic: leakage not reproduced in production
```


---

## Checklist

- [ ] Separated X (features) from Y (target)
- [ ] Defined random_state for reproducibility
- [ ] Split data into train and test (80/20 or 70/30)
- [ ] If needed, split train into train + validation
- [ ] Fit scalers/encoders **only on training**
- [ ] Transform scalers/encoders on all sets
- [ ] Did NOT observe test data during development
- [ ] Used StratifiedKFold for imbalanced problems
- [ ] Did oversampling **after** splitting train/test
- [ ] Verified no data leakage exists
- [ ] Evaluated final model **only on test**