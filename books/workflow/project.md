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
### 3.1 Identify the priorites in the dataset and select the features in the dataset (Fix the data to analyze)
**CONSTANT FEATURE**
- The value are the sames, doesn't change, are irrelevant features to the model

**REDUNDANT FEATURE**
- Two features have the same information in diferrent data
- Weight (meters (m) ), (centimeters (cm) ) and etc

**High correlation**
- !Sometimes a feature doesn't have a hight correlation with the label, but have with others features, so we delethe that feature (!DEPENDING THE MODEL AND THE OBJECTIVE!) !
- HEATMAP 2 DIMENSIONAL TABLE, TO SEE THE CORRELATION, the HEATMAP IT'S MORE EFFECTIVE TO NUMERICAL FEATURES, so we need to do a pre-select features, or 'delete' some str and object features
- It's more common we use create a new df to heatmap 'df_heatmap' just with number features, and then use seaborn

**Cardinality** 
- Some features have a high cardinality, so depending on the model it's better to delete theses data
- See all the values per features, and then see which feature have the high cardinality and then delete the feature (depeding) with a pandas option (.nunique)

**NULL VALUES**
- High null features values (some features have a high null values, so we need to delete the SAMPLES who doesn't have the value)
- See all the values per features, and then see which feature doesn't agree with the other and then DELETE THE SAMPLES WITH THE NULL VALUES
- See only the null values with pandas options and then delete the SAMPLES WITH THE NULL VALUES with pandas (.dropna)

**CATEGORIAL FEATURES**
- Some features are objects, classes or strings, we need to analyze the important of the feature and then choose what we gonna do
- Features like name, id, address, are irrelevant in 99% of the cases
- Features like male or female, can be relevant in some contexts, always depending
- The model can't learning directly with these type of data (object, string or classes), they need to be encoded into numbers for the model learning
- Imporant categorial features, we need to transform into numbers, so we gonna use the one-hot or label encoding method, so that features will become a good feature

**TIME FEATURES**
- These type of data are dates, like 01/01/2026
- We need to treat this data to the model doesn't learning wrong or something else
- Can be a day, year, hour, day of week or anything related

**FEATURE ENGINEERING**
- In this case we (dev's) create new features or transform (like we do in categorial features) to improve the model
- We can to combine features like weight and height to IMC, or price and quantity to value
- Create bins 0-10, 10-20, 20-30
- We can treat the data to get a more right, simple and real data

### 3.2 Analyze the daya, and understanding how works
**DISTRIBUTION OF THE DATA (FIND 'ERRORS' IN THE DATA -> SKEWED DATA AND OUTLIERS)**
- Scatter plot -> see the correlation between 2 features (how 2 features works together), just do with some features with the most correlation
- Histogram -> see if the data are balanced and skewed (in this case very different values ​​can alter the average) (how a feature work)
- Box Plot -> resume the distribution of the feature, and find outliers 
- AND OTHERS PLOT
- The meaning of visualyze the data it's to decide who models will fit better, find outlies, skewed data and others thing invisible to human eye, that can be a problem to the model learning

**SKEWED DATA**
1. Geral 
- If you found the skewed data and outliers, the first thing u need to fix it's the skewed data, and then the outliers
- SKEWED DATA IT'S THE ASYMMETRICAL DATA LIKE IN THE EXAMPLE
- We are see 100 data about followers in the instagram, 90 of then have 100 followers and the other 10 have 100.000 followers, so the average are 10.009 followers, what doesn't matches with the real life.
- Pulls more to one side, LIKE A WAVE YESSSS DAMN
2. How to treat the skewed data?
- Check if it's really skewed data,  so we use the .skew()
- Then we need to understanding the type of the skewed data, if it's right or left skewed
- Choose the right transformation: log, square root, box-cox, yeo-johnson (all a numpy, scipy and sklearn functions)
- Then we need to compare the old data with the new to choose who gonna be the better

**OUTLIERS**
1. Geral
- Different to the skewed, outliers are some (little) quantity of extremitys data, skewed are more data, outliers are points, a lower quantity
- Doesn't pulls more to one side, it's more like a nemo point, in nowhere but exits and compromises the model learning
2. How to treat the outlier data?
- _FIRST WE NEED TO TREAT THE SKEWED DATA AND THEN THE OUTLIERS DATA_
- Choose a way: remove, capping (limit the outliers) or keep them.
- The capping strategy we define a limit for the data, looks like (10, 10, 15, 20, 20, 25, *500) -> (10, 10, 15, 20, 20, 25, *25)

**TARGET BALANCE**
- See if the classes are good distribution, (not like 99% spam and 1% not spam)

# 4. PREPARE THE DATA FOR THE ML ALGORITHMNS (Separate the data, in test, train, validation and more)

# 5. SELECT AND TRAIN THE MODEL (Analyze the models, identify who is better and train the modelsupervised or not, regression or classification, batch (offline) or online, per instancies (similar) or per model (maths) )




# 6. IMPROVE THE MODEL See the erros, and try to improve the error accurancy, here goes the news predicr (with scikit model.predict) and after visualyze the performance measure (MRSE, MAE and more)
And after, select news hyperparameters or something to improve the error measure


# 7. SHOW THE SOLUTION Share to the other people and the manager
# 8. PUBLIC THE SYSTEM, ANALYZE AND ADJUST Publish to the internet, and see how goes work with new instancies




# FLOWCHART
![alt text](image.png)