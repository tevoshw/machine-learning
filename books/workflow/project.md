_HOW TO THINK LIKE A PROFESSIONAL, TO DO MACHINE LEARNING PROJECTS?_

# 1. ANALYZE THE SITUATION 
### 1.1 QUESTIONS?
**What's the real problem?**

**What's the final objective?**
- Predict: numbers
- Classifier: classes/groups
- Recommend
- Found patterns

**How the model will be used?**
- A people or a system will use?
- Need to be fast

**Wich type of data will be used?**
- Real time or historic
- Supervised or non-supervised
- How the dataset it's defined (big or small) (have null values, skewed data or outliers)

**The model have restrictions?**
- Memory
- Time
- Money

**The model in the future**
- Problem will be changed?
- New data will come?

**Define the systems of the machine learning project**
- A (supervised or other)
- B (batch or offline)
- C (instances or modules)

# 2. GET THE DATA (Import the data in the envoriment) 
**Where's the data comes**
- Kaggle's suport
- A database
- A file

# 3.  IDENTIFY AND VISUALIZE THE DATA TO GET IMPORTANT INFORMATIONS 
### 3.1 Identify the priorites in the dataset and select the features in the dataset (Fix the data to analyze) (EDA PART)
**CONSTANT FEATURE**
- The value are the sames, doesn't change, are irrelevant features to the model

**REDUNDANT FEATURE**
- Two features have the same information in diferrent data
- Weight (meters (m) ), (centimeters (cm) ) and etc

**CATEGORIAL FEATURES**
- Some features are objects, classes or strings, we need to analyze the important of the feature and then choose what we gonna do
- Features like name, id, address, are irrelevant in 99% of the cases
- Features like male or female, can be relevant in some contexts, always depending
- The model can't learning directly with these type of data (object, string or classes), they need to be encoded into numbers for the model learning
- Imporant categorial features, we need to transform into numbers, so we gonna use the one-hot or label encoding method, so that features will become a good feature (we call that ENCONDIG, in sklearn we have some encoder functions, like LABEL, ONE-HOT and ORDINAL )

**TIME FEATURES**
- These type of data are dates, like 01/01/2026
- We need to treat this data to the model doesn't learning wrong or something else
- Can be a day, year, hour, day of week or anything related

**CARDINALITY** 
- Some features have a high cardinality, so depending on the model it's better to delete theses data
- See all the values per features, and then see which feature have the high cardinality and then delete the feature (depeding) with a pandas option (.nunique)

**NULL VALUES**
- High null features values (some features have a high null values, so we need to delete the SAMPLES who doesn't have the value)
- See all the values per features, and then see which feature doesn't agree with the other and then DELETE THE SAMPLES WITH THE NULL VALUES
- See only the null values with pandas options and then delete the SAMPLES WITH THE NULL VALUES with pandas (.dropna)

**HIGH CORRELATION**
- !Sometimes a feature doesn't have a hight correlation with the label, but have with others features, so we delethe that feature (!DEPENDING THE MODEL AND THE OBJECTIVE!) !
- HEATMAP 2 DIMENSIONAL TABLE, TO SEE THE CORRELATION, the HEATMAP IT'S MORE EFFECTIVE TO NUMERICAL FEATURES, so we need to do a pre-select features, or 'delete' some str and object features
- It's more common we use create a new df to heatmap 'df_heatmap' just with number features, and then use seaborn

**SCALE AWARENESS**
- Identify features with very different numeric ranges
- Age: 50 // Wage : 100.000, in this case the model can give more importance to the wage because the big numbers, and harm the model 

**FEATURE ENGINEERING**
- In this case we (dev's) create new features or transform (like we do in categorial features) to improve the model
- We can to combine features like weight and height to IMC, or price and quantity to value
- Create bins 0-10, 10-20, 20-30
- We can treat the data to get a more right, simple and real data

### 3.2 Analyze the daya, and understanding how works (EDA YET)
**DISTRIBUTION OF THE DATA (FIND 'ERRORS' IN THE DATA -> SKEWED, OUTLIERS, SCALE AWARANESS OR NEGATIVE DATA)**
- Scatter plot -> see the correlation between 2 features (how 2 features works together), just do with some features with the most correlation
- Histogram -> see if the data are balanced and skewed (in this case very different values ​​can alter the average) (how a feature work)
- Box Plot -> resume the distribution of the feature, and find outliers 
- AND OTHERS PLOT
- The meaning of visualyze the data it's to decide who models will fit better, find outlies, skewed data and others thing invisible to human eye, that can be a problem to the model learning


**NEGATIVE DATA**
1. Geral (EDA)
- Numbers numbers, in some case these type can harm the model, but in other cases can be useful
- We can see with the .info() or .describe(), to verify if we have theses type of data, and then analyze if this data harm or  help the model
- Have high quantity of this data or some a little?
2.  How to treat the negatives numbers if harm the model? (PREPROCESSING)
- IF THE NEGATIVE HAVE A MEANING, WE KEEP HOW LOOKS LIKE
- IF THE NEGARTIVES DOENS'T HAVE A MEANING, WE TREAT IN 3 WAY (REMOVE, TRANSFORM IN NAN NUMBER OR CLIPPING)

**SKEWED DATA**
1. Geral (EDA)
- If you found the skewed data and outliers, the first thing u need to fix it's the skewed data, and then the outliers
- SKEWED DATA IT'S THE ASYMMETRICAL DATA LIKE IN THE EXAMPLE
- We are see 100 data about followers in the instagram, 90 of then have 100 followers and the other 10 have 100.000 followers, so the average are 10.009 followers, what doesn't matches with the real life.
- Pulls more to one side, LIKE A WAVE YESSSS DAMN
2. How to treat the skewed data? (PREPROCESSING)
- Check if it's really skewed data,  so we use the .skew()
- Then we need to understanding the type of the skewed data, if it's right or left skewed
- Choose the right transformation: log, square root, box-cox, yeo-johnson (all a numpy, scipy and sklearn functions)
- Then we need to compare the old data with the new to choose who gonna be the better

**OUTLIERS**
1. Geral (EDA)
- Different to the skewed, outliers are some (little) quantity of extremitys data, skewed are more data, outliers are points, a lower quantity
- Doesn't pulls more to one side, it's more like a nemo point, in nowhere but exits and compromises the model learning
2. How to treat the outlier data? (PREPROCESSING)
- _FIRST WE NEED TO TREAT THE SKEWED DATA AND THEN THE OUTLIERS DATA_
- Choose a way: remove, capping (limit the outliers) or keep them.
- The capping strategy we define a limit for the data, looks like (10, 10, 15, 20, 20, 25, *500) -> (10, 10, 15, 20, 20, 25, *25)

**SCALE AWARENESS**
1. Geral (EDA)
- A data with numbers on different scales, and can harm the model
- Max value in A feature: 10 // Max value in B feature: 100000 // so can harm the model
2. How to treat (PREPROCESSING)
- To treat the data, we use SKLEARNK SCALER functions (STANDARD, ROBUST, MINMAX)

**TARGET BALANCE**
- See if the classes are good distribution, (not like 99% spam and 1% not spam)

# 4. PREPARE THE DATA FOR THE ML ALGORITHMNS (Separate the data, in test, train, validation and more)
**THERE ARE DIFFERENTS PARTS FOR THE DATA IN A MACHINE LEARNING MODE, AND WE'LL GO EXPLORE ALL OF THESE STEPS RIGHT NOW**


### 4.1 WHY AND HOW STARING?
**WHY WE NEED TO SEPARE THE DATASET IN LITTLE DATASETS?**
- A model can learning, and can also memorize the data, or do something else that harm the model to be efective (over and underfitting)
- To avoid these thing of situation and improve the model learning, we split in split datasets, to test, train, verify and more things, that garanted greater power to the model and data

**HOW TO STARTING**
- We need to separe in X  and Y variables
- The X variable contains the independent features, and the Y the dependent features
- We can use the .drop() from pandas to remove the dependent features (to get the X data)
- And also we use x['y'] to get the Y feature

**RANDOM SEED**
- A way in functions that generate a random choose of the data
- DATASET ->(1,2,3,4,5)
- RANDOM 2 -> (2,4,3,5,1)
- RANDOM 33 -> (1,3,5,4,2)

### 4.2 TYPES OF SET
**TRAINING SET**
- Part of the dataset that will be used to train the model
- Adjust the bias and wight
- Forward, loss, backpropagation
- Gradient descent
- 60-80% of the data
- train_test_split() from sklearn

**VALIDATION SET**
- Part of the dataset where we'll avaliation the train step, to see if are memorizing or learning
- In this steep it's to avaliable the hyperparameters
- We can check also, if the model have troublesl like under and overfitting
- 10-20% of the data

**TESTING SET**
- Part that will avaliable the model with new instancies (samples)
- Calculate the generelazation
- 10-20% of the data


### 4.3 STRATEGIES TO SPLIT THE SET'S
**HOLDOUT**
- Divide it only once the dataset, each part have a fixed role
- Used in medium/big datasets
- Simple, fast and easy
- We can use with SKLEARN train_test_split
- *But in small datasets, may not represent well*
![alt text](image-1.png)

**K-FOLD CROSS VALIDATION**
- Divide in k trains, in each train the dataset it's splited in a different way
- And after verify the accurancy of each train
- the test set, stay separate
- Used in small/medium datasets
- Compare models with more efficiency, but more slow
- We can use with SKLEARN KFold
![alt text](image-2.png)

**STRATIFIED SPLIT**
- Only used in classes problems
- A way to split the data that garanted proportion
- Train: 72 gatos / 8 cachorros
- Validation: 9 gatos / 1 cachorro
- Test: 9 gatos / 1 cachorro
- We can use with SKLEARN StratiedKFold



#### 4.4 PROBLEMS
**DATA LEAKAGE**
- It's when a data that should be unavailable at training test, leaks at the model
- Test data to train data, and more
- The model goes with a high accurancy, a overfitting case (high accurancy with data leakage means nothing)



# 5. SELECT AND TRAIN THE MODEL (Analyze the models, identify who is better and train the modelsupervised or not, regression or classification, batch (offline) or online, per instancies (similar) or per model (maths) )


### 5.0 GERAL


### 5.1 SYSTEM A




#### 5.2 SYSTEM B



### 5.3 SYSTEM C




# 6. IMPROVE THE MODEL See the erros, and try to improve the error accurancy, here goes the news predicr (with scikit model.predict) and after visualyze the performance measure (MRSE, MAE and more)
And after, select news hyperparameters or something to improve the error measure


# 7. SHOW THE SOLUTION Share to the other people and the manager
# 8. PUBLIC THE SYSTEM, ANALYZE AND ADJUST Publish to the internet, and see how goes work with new instancies




# FLOWCHART
![alt text](image.png)