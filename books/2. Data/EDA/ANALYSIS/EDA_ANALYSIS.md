
# EXPLORATORY DATA ANALYSIS (EDA) 

## GERAL ANALYZE OF THE DATA

**TYPE OF DATA**
- HEAD, TAIL, DESCRIBE

**SHAPE**
- .SHAPE

## STRUCTURE AND QUALITY OF THE FEATURES
**CONSTANT FEATURE**
- DEFINITON: Are features that have all the same value for all the samples, they don't have a variability, so don't have a value to predict
- CHARACTERISTICS: Can be numeric, object or something else
- EXAMPLE: Country: 1. BR 2. BR 3. BR 4. BR

**REDUNDANT FEATURE**
- DEFINITION: Are more than one feature which contains the same data in a different way.
- CHARACTERISTICS: Can be in differents units (cm or meter)
- EXAMPLE: Weight: 100cm // 1 meter

**DUPLICATES**
- DEFINITION: Are repeated samples, here's the problem are rows and not the features
- CHARACTERISTICS: Can be all the fatures repeated, or some only
- EXAMPLE: SAMPLE 1: 1, 2, 3, 4 // SAMPLE 2: 1, 2, 3, 4 (all the features) // SAMPLE 3: 1,2,2,4 (some only)

**CARDINALITY** 
- DEFINITION: A feature with high quantity of differents labels or targets (it's much more common in categorial features)
- CHARACTERISTICS: Some numeric features have a high cardinality, and this don't harm the model, so it's more for categorial.
- EXAMPLE: FEATURE X: LION, CAT, DOG, WHALE, ANT, BUG (AND MORE 100 EXAMPLES), all presents in one single feature 

## TYPE OF THE DATA
**CATEGORIAL FEATURES**
- DEFINITION: Features that represents classes or objects and not numerics continuous, the model can't learning without numeric values, so it's necessary analyses that
- CHARACTERISTICS: 
1. Can be string, object or category;
2. They can be very important or useless;
3. Nominal (no order of importance), Ordinal (with order of importance) or Label (create a numeric order that doesnt exist, and we use for the Y feature)
- EXAMPLE: Country: BR, US, GER (nominal feature) // Position: Boss, employeer, trainee (ordinal feature)


**TIME FEATURES**
- DEFINITION: Features that represents time or dates 
- CHARACTERISTICS:
1. Can be object, datetime or int datatype
2. Frequency (daily, monthly or annual)
- EXAMPLE: (01/01/2026 2024-08-15 14:32:10 timestamp) 
1. If we put these datas like int, can be 20200101 and 20201231, and a model can think that 20200101 < 20201231

## MISSING & INVALID DATA


**NULL VALUES**
- DEFINITION: Are missing values in the dataset.
- CHARACTERISTICS: The null valeus isn't a 0, or '' string, just doens't exist like a ghost value
- EXAMPLE: 5 features: 1, 2, 3, ,5 (one feature is missing)

**NEGATIVE DATA**
- DEFINITION: Features with negatives values.
- CHARACTERISTICS: 
1. Not necessarily an error, in some case are (it makese sense un real life?)
2. Can be VALID (weather, profit, variaton and more) or INVALID (error)
- EXAMPLE: Weather = -48 (valid) // Age = -39 (invalid)

## DISTRIBUTION OF THE DATA

**SKEWED DATA**
- DEFINITION: Occurs when data distribution it's not symmetrical, pull to one side (right-skewed or left-skewed)
- CHARACTERISTICS: It's not an error, but can harm the model
- EXAMPLE: 90 persons has 10k wage, 10 persons has a 100k wage, the media it's a unreal number, because the 10 persons, change the values.


**OUTLIERS**
- DEFINITION: Values presents in the data, that are completely different from the others, extreme or minimal, and this can harm the model
- CHARACTERISTICS:
1. Normally are few values
2. Different from the skewed that are a most, outliers are a few.
- EXAMPLE: Height: 1.90, 1.91, 1.85. 5.00 -> these are a outlier


## RELATION BETWEEN FEATURES

**HIGH CORRELATION**
- DEFINITION: Features with a correlation in the data between them (if one increases, the other also increases.) 
- CHARACTERISTICS: 
1. +1 and -1 are the maximum values that features can have, positives it's when they increase together, negative it's when one increase and the other decrease
2. When features are too correlation, when we can do a feature engineering, trying to improve the model.
3. If the correlation between a feature and the target it's bad, we can delete it, focus to improve the model ()
4. To see the correlation it's better to do a HEATMAP
5. The correlation just can be acessed with NUMERIC VALUES, so objects, string and more goes be an error
- EXAMPLE: TARGET: house_price // FEATURE: total_roomns // CORRELATION: 0.6

## MODEL SENSITY

**SCALE AWARENESS**
- DEFINITION: The difference between the valeus of the features, e the impact about this data in the model.
-  CHARACTERISTICS: 
1. Without a scaling, a model can think that big numbers are more imporants them small numbers
2. Harm models based in median or gradient (linear regression, nn and more)
- EXAMPLE: Age: 50 // Wage : 100.000, in this case the model can give more importance to the wage because the big numbers, and harm the model 

**TARGET BALANCE**
- DEFINITION: Verify and ensure the balance of target, to see if the targets (y) are well distributed or not
- CHARACTERISTICS: 
1. It's more common in classifier problems
2. Can contains rares targets
- EXAMPLE: 90% of the targets are 'spam' and 10% are 'not spam' 

## FEATURE ENGINEERING

**FEATURE ENGINEERING**
- DEFINITION: The process of select, transform and create new feature (unreal), based in real features, trying always to improve the model.
- CHARACTERISTICS: Involves thinking and know about the data
- EXAMPLE: Instead have 2 features: weigth and height // we can do a IMC feature = weight + height