# PREPROCESSING

## STRUCTURAL CLEANING

**CONSTANT FEATURE**
- TREAT: REMOVE
- VISUALIZE: .nunique()
- REMOVE: .drop(columns = [])

**REDUNDANT FEATURE**
- TREAT: Analyses and choose the feature who it's more advantageous, and remove the other, or just maintain  both
- REMOVE: .drop(columns = [])

**DUPLICATES**
- TREAT: Check if are duplicated samples (rows) or duplicated values in the features, so check if this duplicated are relevant data or not
- DUPLICATES: .duplicated() + .sum()
- REMOVE: .drop(columns = [])
- OTHER WAY TO REMOVE: .drop_duplicated(subset = [])


## MISSING & VALID DATA TREAT

**NULL VALUES**
- TREAT: Check if we have null values, so them remove the samples with this missing data, if are a lot of samples, we remove the feature with that high value with null data
- NULL: .isnull() // .isnull() +.sum()
- REMOVE: .drop(columns = [])
- IMPUTATION:  `SimpleInputer(strategy = 'median')` from sklearn
- WHEN DO? IF REMOVE BEFORE SPLIT, IF IMPUTATION AFTER SPLIT

**NEGATIVE DATA**
- TREAT: Check if data makes sense, if makes ok, doesnt change, but not we need to remove or transform
- VISUALIZE: (dataset < 0).sum().sum()
- REMOVE: .drop(columns = [])

**CENSORING TREATMENT**
- TREAT: Remove or fililtering, so you can remove the "capped" values if they represent a small portion of the data to avoid biasing the trend.
- VISUALIZE: Histogram about (y)
- FLAGGING: Create a new binary feature `is_censored` (1 for limit values, 0 for others) to help the model distinguish the "wall".
- TOBIT MODEL: Use specific regression models designed to handle censored variables.


## TYPE OF THE DATA TREAT

### CATEGORIAL
**CATEGORIAL FEATURES**
- TREAT: Verify the cardinality and null values, if are to high we remove, altough if we don't have that, we have to put it into numeric values.
- REMOVE: df.drop(columns = []) // + .value_counts()
- TRANSFORM: ENCODING ALGORITHMS (ONE-HOT-ENCODING or  ORDINAL-ENCODING -> based on the type of data ordinal or nominal)
- WHEN DO? AFTER THE SPLIT

**CARDINALITY** 
- TREAT: Check the difference between the data in the features, if it's to high, we need to analyze and if necessary remove it
- CARDINALITY: .nunique() // dataset['column'].nunique() // dataset['column'].value_counts()
- REMOVE: .drop(columns = [])

### NUMERIC

**TIME FEATURES**
- TREAT: Convert into datetime data and then, verify cardinality, null values, sort them and see if it's relevant feature to remove or maintain, *CAUTION WITH DATA LEAKAGE*
- CONVERT: .to_datetime()

## RELATION BETWEEN THE FEATURES

**HIGH CORRELATION**
- TREAT: Check the correlation, and we can check if we can remove or do something (only numeric numbers works, so we need to transform if we have to numeric numbers all the features)
- CORRELATION: .corr() 
- REMOVE: .drop(columns = [])


## DISTRIBUTION OF DATA AND FEATURES TREAT

**SKEWED DATA**
- TREAT: Check the asymmetric, if have, right or left, and then remove these data (extreme values) or drop rows with unrealistic outliers
- SKEWED: .skew() (if skew > 1 (right and treat) or if skew < -1 (left and treat))
- TREAT: np.lop1p (to transform) // np.expm1 (to back original values)
- REMOVE: .drop(columns = [])

**OUTLIERS**
- TREAT: Check if have, and them remove if's necessary
- CHECK OUTLIERS: BOXPLOT, IQR methods
- REMOVE: .drop(columns = [])
- IQR methods
- WHEN DO? AFTER THE SPLIT (to avoid data leakage)

**SCALE AWARENESS**
- TREAT: Check the different between the values (range, magnitude, units) and apply STARDANTIZATON OR NORMALIZATION
- DIFFERENT SCALES: .describe()
- STANDARDIZATION: StandardScaler() -> few outlliers, (sklearn)
- NORMALIZATION: MinMaxScaler() (sklearn)
- WHEN DO? AFTER THE SPLIT (avoid data leakage)

**TARGET BALANCE**
- TREAT: Verify if the balance are OK
- CHECK BALANCE: .values_counts (the target feature)
- TREAT SKLEARN: `train_test_split(stratify = y)`

# FEATURE ENGINEERING

**FEATURE ENGINEERING**
- TREAT: Create and transform exist features in new features, or just delete it
- CREATE: dataset['new_feature'] = dataset['feature_one'] + dataset['feature_two']



# DATA LEAKAGE
## Data Leakage Prevention Guide

**Data Leakage** occurs when information from outside the training dataset (the "future" or the test set) is used to create the model. This leads to overly optimistic performance metrics that will fail in real-world production.

---

## The Gold Rule: X vs Y
* **X (After Split):** Any transformation that requires calculations based on data distribution (mean, standard deviation, quartiles, frequencies).
* **Y (Before Split):** Structural cleaning and logic that do not depend on group statistics (row-by-row operations).

---

##  What to do AFTER the Split (X) and Why?

### 1. Scale Awareness (Scaling & Normalization)
* **Techniques:** `StandardScaler`, `MinMaxScaler`.
* **Why X?** Scaling uses the **mean** and **standard deviation**. If you calculate these on the whole dataset, the training set "knows" the global average.
* **The Risk:** The model adjusts its weights knowing the range of data it shouldn't have seen yet.

### 2. Missing Data Treatment (Imputation)
* **Technique:** Filling nulls with **Mean, Median, or Mode**.
* **Why X?** The value used to "plug the hole" must represent only what the model learned from the training data.
* **The Risk:** If the test set has outliers, the global mean changes. The training set will receive "spoilers" of these future trends.

### 3. Categorical Encoding
* **Techniques:** `OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder`.
* **Why X?** To ensure the mapping of categories comes strictly from the training set.
* **The Risk:** If a new category appears only in the test set, encoding before the split gives the training set a "ghost" column it shouldn't know exists.

### 4. Outlier Treatment
* **Techniques:** `IQR (Interquartile Range)`, `Z-Score`.
* **Why X?** What is considered "abnormal" depends on the distribution. You must define "normal" based on training evidence only.
* **The Risk:** Extreme values in the test set will shift your outlier boundaries before training begins.

### 5. High Correlation & Feature Selection
* **Techniques:** `.corr()`, `SelectKBest`.
* **Why X?** The decision on which features are redundant should be made by looking only at the training samples.
* **The Risk:** You might discard a feature that seems redundant globally but is actually unique to the training distribution.

## The Professional Solution: Pipelines

The best way to handle **X** tasks is using Scikit-Learn **Pipelines**. This ensures that transformations are re-fitted only on the training folds during cross-validation.



```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Correct workflow:
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])