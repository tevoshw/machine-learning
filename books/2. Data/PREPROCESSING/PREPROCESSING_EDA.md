# PREPROCESSING

## STRUCTURAL CLEANING

**CONSTANT FEATURE**
- TREAT: REMOVE
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
- VISUALIZE: BARPLOT
- NULL: .isnull() // .isnull() +.sum()
- REMOVE: .drop(columns = [])

**NEGATIVE DATA**
- TREAT: Check if data makes sense, if makes ok, doesnt change, but not we need to remove or transform
- VISUALZIE: HISTOGRAM, BOXPLOT, KDEPLOT, SCATTERPLOT
- REMOVE: .drop(columns = [])

## TYPE OF THE DATA TREAT

### CATEGORIAL


**CATEGORIAL FEATURES**
- TREAT: Verify the cardinality and null values, if are to high we remove, altough if we don't have that, we have to put it into numeric values.
- VISUALIZE: BARPLOT, COUNTPLOT
- REMOVE: df.drop(columns = []) // + .value_counts()
- TRANSFORM: ENCODING ALGORITHMS (ONE-HOT-ENCODING or  ORDINAL-ENCODING -> based on the type of data ordinal or nominal) 

**CARDINALITY** 
- TREAT: Check the difference between the data in the features, if it's to high, we need to analyze and if necessary remove it
- CARDINALITY: .nunique() // dataset['column'].nunique() // dataset['column'].value_counts()
- REMOVE: .drop(columns = [])

### NUMERIC

**TIME FEATURES**
- TREAT: Convert into datetime data and then, verify cardinality, null values, sort them and see if it's relevant feature to remove or maintain
- VISUALIZE: LINE PLOT, ROLLING MEAN PLOT
- CONVERT: .to_datetime()
- CARDINALITY: .nunique()
- NULL VALUES: isnull() + .sum()
- SORT: .sort()
- REMOVE: .drop(columns = [])

## RELATION BETWEEN THE FEATURES

**HIGH CORRELATION**
- TREAT: Check the correlation, and we can check if we can remove or do something (only numeric numbers works, so we need to transform if we have to numeric numbers all the features)
- VISUALIZE: HEATMAP: sns.heatmap(), PAIRPLOT, SCATTER PLOT + HUE
- CORRELATION: .corr() 
- REMOVE: .drop(columns = [])


## DISTRIBUTION OF DATA AND FEATURES TREAT

**SKEWED DATA**
- TREAT: Check the asymmetric, if have, right or left, and then remove these data (extreme values) or drop rows with unrealistic outliers
- VISUALIZE: HISTOGRAM, BOXPLOT, VIOLINPLOT, SCATTERPLOT
- SKEWED: .skew()
- REMOVE: .drop(columns = [])

**OUTLIERS**
- TREAT: Check if have, and them remove if's necessary
- VISUALIZE: SCATTER AND BOXPLOT
- CHECK OUTLIERS: BOXPLOT
- REMOVE: .drop(columns = [])

**SCALE AWARENESS**
- TREAT: Check the different between the values (range, magnitude, units) and apply STARDANTIZATON OR NORMALIZATION
- VISUALIZE: HISTOGRAMS, BOXPLOT
- DIFFERENT SCALES: .describe()
- STANDARDIZATION: StandardScaler() -> few outlliers, (sklearn)
- NORMALIZATION: MinMaxScaler() (sklearn)

**TARGET BALANCE**
- TREAT: Verify if the balance are OK
- VISUALIZE: BARPLOT, COUNTPLOT,
- CHECK BALANCE: .values_counts (the target feature)

# FEATURE ENGINEERING

**FEATURE ENGINEERING**
- TREAT: Create and transform exist features in new features, or just delete it
- VISUALIZE: HEATMAP, HISTOGRAM, SCATTER
- CREATE: dataset['new_feature'] = dataset['feature_one'] + dataset['feature_two']
