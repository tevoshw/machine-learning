
# EXPLORATORY DATA ANALYSIS (EDA) 

## GERAL ANALYZE OF THE DATA
**OBJECTIVE**
- The real objective of EDA, it's to understanding better the data, how works and help to choose the better model

**TYPE OF DATA**
- HEAD, TAIL, DESCRIBE

**SHAPE**
- .SHAPE

## STRUCTURE AND QUALITY OF THE FEATURES
**CONSTANT FEATURE**
- DEFINITON: Are features that have all the same value for all the samples, they don't have a variability, so don't have a value to predict
- CHARACTERISTICS: Can be numeric, object or something else
- EXAMPLE: Country: 1. BR 2. BR 3. BR 4. BR
- HOW FIND:
1. `.nunique()`
2. `.value_counts()`
- HOW TREAT:
1. REMOVE: `.drop()`

**REDUNDANT FEATURE**
- DEFINITION: Are more than one feature which contains the same data in a different way.
- CHARACTERISTICS: Can be in differents units (cm or meter)
- EXAMPLE: Weight: 100cm // 1 meter
- HOW FIND:
1. Geral analyze of the data
- HOW TREAT:
1. Analyses and choose the feature who it's more advantageous, and remove the other, or just maintain  both

**DUPLICATES**
- DEFINITION: Are repeated samples, here's the problem are rows and not the features
- CHARACTERISTICS: Can be all the fatures repeated, or some only
- EXAMPLE: SAMPLE 1: 1, 2, 3, 4 // SAMPLE 2: 1, 2, 3, 4 (all the features) // SAMPLE 3: 1,2,2,4 (some only)
- HOW FIND:
1. `.duplicated()`
- HOW TREAT:
1. `.drop_duplicated()`

**CARDINALITY** 
- DEFINITION: A feature with high quantity of differents labels or targets (it's much more common in categorial features)
- CHARACTERISTICS: Some numeric features have a high cardinality, and this don't harm the model, so it's more for categorial.
- EXAMPLE: FEATURE X: LION, CAT, DOG, WHALE, ANT, BUG (AND MORE 100 EXAMPLES), all presents in one single feature 
- HOW FIND:
1. `.nunique()`
- HOW TREAT:
1. REMOVE: `.drop()`

## MISSING & INVALID DATA
**NULL VALUES**
- DEFINITION: Are missing values in the dataset.
- CHARACTERISTICS: The null valeus isn't a 0, or '' string, just doens't exist like a ghost value
- EXAMPLE: 5 features: 1, 2, 3, ,5 (one feature is missing)
- HOW FIND:
1. .isnull()
- HOW TREAT:
1. IMPUTATION (FILL): `SimpleInputer(strategy = '')`
2. REMOVE: `.dropna`

**NEGATIVE DATA**
- DEFINITION: Features with negatives values.
- CHARACTERISTICS: 
1. Not necessarily an error, in some case are (it makese sense un real life?)
2. Can be VALID (weather, profit, variaton and more) or INVALID (error)
- EXAMPLE: Weather = -48 (valid) // Age = -39 (invalid)
- HOW  FIND:
1. `(data < 0).sum().sum()`
- HOW TREAT:
1. REMOVE: `.query('column >= 0')`

**CENSORING**
- DEFINITION: When the value of a measurement or observation is restricted at a certain threshold, masking the true underlying value.
- CHARACTERISTICS:
1. The "Ceiling" Effect: Data points accumulate in a straight line at the maximum or minimum limit (as seen in your Regplot at 500k).
- TYPES:
1. Right-Censoring: The limit is at the top. (Ex: Surveys where the highest option is "Income > $100k").
2. Left-Censoring: The limit is at the bottom. (Ex: A chemical test that cannot detect substances below 0.01mg).
- EXAMPLES:
1. Housing: Prices capped at $500,000 for privacy or system limits (VALID but restricted).
2. Visual: A horizontal "wall" of dots at the edge of a Scatter Plot.
- HOW FIND: 
1. HISTOGRAMS: `.hist()`
- HOW TREAT:
1.  Flagging
2. Tobit model

## TYPE OF THE DATA
**CATEGORIAL FEATURES**
- DEFINITION: Features that represents classes or objects and not numerics continuous, the model can't learning without numeric values, so it's necessary analyses that
- CHARACTERISTICS: 
1. Can be string, object or category;
2. They can be very important or useless;
3. Nominal (no order of importance), Ordinal (with order of importance) or Label (create a numeric order that doesnt exist, and we use for the Y feature)
- EXAMPLE: Country: BR, US, GER (nominal feature) // Position: Boss, employeer, trainee (ordinal feature)
- HOW FIND:
1. `.dtype`
- HOW TREAT:
1. ENCODERS: `OneHotEncoder(), OrdinalEncoder()` from sklearn
2. `.get_dummies()`
> ONLY **TREAT AFTER THE SPLIT**

**TIME FEATURES**
- DEFINITION: Features that represents time or dates 
- CHARACTERISTICS:
1. Can be object, datetime or int datatype
2. Frequency (daily, monthly or annual)
- EXAMPLE: (01/01/2026 2024-08-15 14:32:10 timestamp) 
1. If we put these datas like int, can be 20200101 and 20201231, and a model can think that 20200101 < 20201231
- HOW FIND:
1. `.dtype`
2. Analyze the data
- HOW TREAT:
1. CONVERT: `.to_datetime()`

## DISTRIBUTION OF THE DATA
**SKEWED DATA**
- DEFINITION: Occurs when data distribution it's not symmetrical, pull to one side (right-skewed or left-skewed)
- CHARACTERISTICS: It's not an error, but can harm the model
- EXAMPLE: 90 persons has 10k wage, 10 persons has a 100k wage, the media it's a unreal number, because the 10 persons, change the values.
- VISUALIZE: 
1. Histogram
- HOW FIND:
1. `.skew()`
- HOW TREAT
1. MATH FUNCTIONS: `.lop1p`


**OUTLIERS**
- DEFINITION: Values presents in the data, that are completely different from the others, extreme or minimal, and this can harm the model
- CHARACTERISTICS:
1. Normally are few values
2. Different from the skewed that are a most, outliers are a few.
- EXAMPLE: Height: 1.90, 1.91, 1.85. 5.00 -> these are a outlier
- HOW FIND:
1. Histogram
2. Box plot
3. Scatter plot
- HOW TREAT:
1. IQR METHODS
> ONLY **TREAT AFTER THE SPLIT**

**TARGET Y**
- DEFINITION: Check the target understanding better the data and try to improve more.
- CHARACTERISTICS:
1. Type of the target
2. Distribution of the target
3. Check outliers in target
- EXAMPLE: 90% of the targets are 'spam' and 10% are 'not spam'
- VISUALIZE:
1. Histograms
2. Box plots
3. `.value_counts`


## RELATION BETWEEN FEATURES
**HIGH CORRELATION**
- DEFINITION: Features with a correlation in the data between them (if one increases, the other also increases.) 
- CHARACTERISTICS: 
1. +1 and -1 are the maximum values that features can have, positives it's when they increase together, negative it's when one increase and the other decrease
2. When features are too correlation, when we can do a feature engineering, trying to improve the model.
3. Low correlation does not necessarily mean the feature is useless, especially for non-linear models.
4. To see the correlation it's better to do a HEATMAP
5. The correlation just can be acessed with NUMERIC VALUES, so objects, string and more goes be an error
- EXAMPLE: TARGET: house_price // FEATURE: total_roomns // CORRELATION: 0.6
- HOW FIND:
1. `.corr()`
2. Plots

**FEATURE X TARGET**
- DEFINITION: Check the features X features
- CHARACTERISTICS:
1. Helps identify predictive features
2. Supports feature selection
3. Can reveal non-linear relationships
- HOW FIND:
1. Scatter plots
2. Lam plots

## MODEL SENSITIVITY
**SCALE AWARENESS**
- DEFINITION: The difference between the valeus of the features, e the impact about this data in the model.
- When Use?: Max/Min >= 10
-  CHARACTERISTICS: 
1. Without a scaling, a model can think that big numbers are more imporants them small numbers
2. Harm models based in median or gradient (linear regression, nn and more)
- EXAMPLE: Age: 50 // Wage : 100.000, in this case the model can give more importance to the wage because the big numbers, and harm the model 
- HOW FIND:
1. `.describe()
2. Analyze the data`
- HOW TREAT:
1. Scaler: `StandardScaler(), MinMaxScaler()` from sklearn
> ONLY **TREAT AFTER THE SPLIT**

## FEATURE ENGINEERING
**FEATURE ENGINEERING**
- DEFINITION: The process of select, transform and create new feature (unreal), based in real features, trying always to improve the model.
- CHARACTERISTICS: Involves thinking and know about the data
- EXAMPLE: Instead have 2 features: weigth and height // we can do a IMC feature = weight / height²

## PROBLEMS
**TIME SERIES**
- DEFINITION: We cannot always rely solely on mathematical equations or deterministic models. For unstable and dynamic phenomena—such as the financial market, sports, human behavior, weather, traffic, and the economy—the relationships do not address a fixed or exact mathematical question.
- CHARACTERISTICS:
1. The past influences the future.
2. The order of the data matters -> we cannot shuffle it.
3. Whenever using the chart, always use the 'DATE' feature with DD/MM/YY format.
4. Lag: It's a value fron past, a feature (LAG7: 7 days of data)
5. Forecasting: The term means predicting the future based on the past. It's similary with time series (Time series is the analysis of data points collected sequentially over time to understand past trends, patterns, and behaviors. Forecasting is the process of using those historical patterns to predict future values)

