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
- D (dataset split: holdout, crossvalidation, kfold, stratified and etc)
- E (model strategy (not the final model): linear regression, logistic and etc)
- F (perfomance measure: RMSE, MAE and etc)


# 2. GET THE DATA (Import the data in the envoriment) 
**Where's the data comes**
- Kaggle's suport
- A database
- A file

# 3.  IDENTIFY, VISUALIZE AND PREPROCESSING THE DATA TO MODEL.

### 3.1 EXPLORATORY DATA ANALYSIS (EDA) 

**GERAL**
- Where we goona analyze the data, don't change, only for analysis and see the data for things that can harm or improve the model


### 3.2 CHANGE AND TRANSFORM THE DATA (PREPROCESSING) 

**GERAL**
- After find in the data things that can harm or improve, we need to change that, here in the preprocessing that happened, here the data are transform for use in the model

--------------------PLOTS NECESSARYS TO EXPLORE THE DATA (BONUS)---------------------
- JOINTPLOT
- STRIPLOT/SWARMPLOT
- RIDGELINEPLOT
- HEXBIN PLOT
- BOXPLOT GROUPED
- CDF/ ECDF PLOT
- HEATMAP DE MISSING VALUES
- AUTOCORRELATION PLOT
- RADAR, PARALLEL, PCA SCATTER, CLUSTERMAP

--------------------------------- PART 2 ----------------------------
- Correlation → heatmap, pairplot, jointplot, hexbin
- Distribution → hist, kde, ecdf, ridgeline
- Categorical → countplot, stripplot, box grouped
- Outliers → box, violin, scatter, hexbin
- Time → line, rolling mean, autocorrelation
- Missing → missing heatmap

----------------------------------------.---------------------------------------------------


# 4. PREPARE THE DATA FOR THE ML ALGORITHMNS (Separate the data, in test, train, validation and more)
**THERE ARE DIFFERENTS PARTS FOR THE DATA IN A MACHINE LEARNING MODE, AND WE'LL GO EXPLORE ALL OF THESE STEPS RIGHT NOW**


### 4.1 WHY AND HOW STARING?
**WHY WE NEED TO SEPARE THE DATASET IN LITTLE DATASETS?**
- A model can learning, and can also memorize the data, or do something else that harm the model to be efective (over and underfitting)
- To avoid these thing of situation and improve the model learning, we split in split datasets, to test, train, verify and more things, that garanted greater power to the model and data

**X and Y**
- DEFINITION: Split the X (independent features) and Y (dependent features), and garanted that all it's ok
- EXAMPLE:
``` 
X = df.drop("target", axis=1)
Y = df["target"]
```

**RANDOM SEED**
- DEFINITION: A way in functions that generate a random choose of the data
- EXAMPLE: DATASET ->(1,2,3,4,5) RANDOM 2 -> (2,4,3,5,1) RANDOM 33 -> (1,3,5,4,2)
- HANDS-ON: SKLEARN PARAMETERS in train_test_split(x,y random_seed = 42)

### 4.2 TYPES OF SET
**TRAINING SET**
- DEFINITION: Part of the dataset that will be used to train the model, where's train will find the patterns and relationships in the data.
- CHARACTERISTICS: 
1. Largest portion of the dataset, often 60-80% of the data.
2. Used to fit model parameters (weight and bias, trees split, clusters, and more parameters)
- HANDS-ON: We can use the SKLEARN function train_test_split(), or a function with numpy to create that. 


**VALIDATION SET or DEV TRAIN SET**
- DEFINITION: Primarily used for hyperparameter tunning, helps to select the best model configuration without using the test data
- CHARACTERISTICS: 
1. Created from training data set
2. Check if the model is learning or memorize and avoid under/overfitting
3. Help to fit the hyperparameters (learning rate, epochs, number of trees, layes and etc) and !DOESNT CHANGE THE MODEL PARAMETERS!
4. Get a 10-20% of the data
5. In some cases are not necessary, just when the accuracy goes bad, (or) and we need do tunning the hyperparameters
6. HOLDOUT: When we do the split from the train set, we remove these data (a few), and add in the validation set, in other words, the !validation data will not be data that has been trained!
7. KFOLD: A part of the split data will become the validation set, so it's more easy working with KFOLD systems when we need to tunning the hyperparameters
- HANDS-ON: 
1. We can do a second split from train set, with the SKLEARN train_test_split() -> this to get the data split for validation test and then try manually 
2. Using SEARCH FUNCTIONS from SKLEARN: GridSearchCV, RandomizerSearchCV and more.

**TESTING SET**
- DEFITION: Check the model's perfomance on unseen data, crucial for evaluation how well the model generalization real-world examples
- CHARACTERISTICS: 
1. 10-20% of the data
2. Compare the accuracy of the model with MEASURE PERFORMANCE FUNCTIONS (MSE, RMSE, MAE, R2)
- HANDS-OM: train_test_split get the data for split test set (normally 80/20 for train test)


### 4.3 STRATEGIES TO SPLIT THE SET'S
**HOLDOUT**
- DEFINITION: Divide it only once the dataset, each part have a fixed role
- CHARACTERISTICS:
1. Used in medium/big datasets
2. Simple, fast and easy
3. But in small datasets, may not represent well (overfitting problem)
- HANDS-ON: We can use with SKLEARN train_test_split(), or some function to do that

**K-FOLD CROSS VALIDATION**
- DEFINITION: Divide in k trains and splits data, in each train the dataset it's splited in a different way, and verify the accuracy of each train with the new split data
- CHARACTERISTICS:
1. Train and validation set are using, test set don't enter here
2. Used in small/medium datasets
3. Compare models with more efficiency, but it's slower
4. It's so powerfull to tunning hyperparameters, like we said in VALIDATION PART
- HANDS-ON: We can use with SKLEARN KFold

**STRATIFIED SPLIT**
- DEFINITION: A variant of K-Fold for classification problems
- CHARACTERISTICS:
1. Only used in classes problems
2. A way to split the data that garanted proportion
- EXAMPLE:
- Train: 72 gatos / 8 cachorros
- Validation: 9 gatos / 1 cachorro
- Test: 9 gatos / 1 cachorro
- HANDS-ON: Use from SKLEARN StratiedKFold()




#### 4.4 PROBLEMS
**DATA LEAKAGE**
- DEFINITION: When a data that should be unavailable at training test, leaks at the model
- CHARACTERISITCS: 
1. The train data goes to test data
2. High accuracy, overfitting problem
3. High accuracy with data leake means nothing, the model memorize it
g


# 5. SELECT THE SYSTEMS FOR THE MODEL


### 5.1 SYSTEM A    
**GERAL QUESTIONS AND THINGS**
- How the data? labeled or not
- Do we need to transform the problem into another formulation? (classification → regression, multi-class → binary)

**SUPERVISED**
- DEFINITION: type of Machine Learning where the model learns from abeled data, each input sample has a corresponding target (label), and the model’s goal is to learn a function that maps inputs to outputs.
- CHARACTERISTICS:
1. Used labeled datasets
2. Learn a direct relantionship between features (X) ands labels (Y)
3. Can used for: REGRESSION (continuous values) and CLASSIFICATION (classes)
4. Perfomance is evaluated using metrics

**UNSUPERVISED**
- DEFINITION: type of Machine Learning where the model learns from unlabeled data, without a target variable, aiming to discover patterns, structures, or relationships in the data.
- CHARACTERISTICS:
1. Uses unlabeled datasets
2. Does not have labels (Y)
3. Learns hidden patterns or data structures
4. Commonly used for: CLUSTERING and DIMENSIONALITY REDUCTION

**SEMI-SUPERVISED**
- DEFINITION: type of Machine Learning that uses a small amount of labeled data combined with a large amount of unlabeled data to improve learning performance.
- CHARACTERISTICS:
1. Uses both labeled and unlabeled datasets
2. Reduces labeling cost
3. Improves model performance when labeled data is scarce
4. Commonly used for: CLASSIFICATION and REGRESSION


**REINFORCEMENT LEARNING**
- DEFINITION: type of Machine Learning where an agent learns by interacting with an environment, taking actions and receiving rewards or penalties.
- CHARACTERISTICS:
1. No labeled data
2. Learns through trial and error
3. Uses reward signals instead of labels
4. Goal is to maximize cumulative reward over time



#### 5.2 SYSTEM B

**GERAL QUESTIONS**


**BATCH or OFFLINE**
- DEFINITION: type of Machine Learning where the model is trained using the entire dataset at once, and the model is not updated after deployment unless it is retrained from scratch.
- CHARACTERISTICS:
1. Trained on a fixed dataset
2. Does not learn incrementally
3. Requires retraining to incorporate new data
4. Commonly used when data does not change frequently

**ONLINE**
- DEFINITION: type of Machine Learning where the model learns incrementally, updating itself continuously as new data arrives.
- CHARACTERISTICS:
1. Learns one sample or a small batch at a time
2. Adapts to new data in real time
3. Suitable for data streams
4. Handles concept drift better than batch learning

### 5.3 SYSTEM C

**INSTANCE-BASED LEARNING**
- DEFINITION: type of Machine Learning where the model stores training instances and makes predictions by comparing new data points to similar stored examples.
- CHARACTERISTICS:
1. Does not build an explicit global model
2. Predictions are based on similarity measures
3. High memory usage
4. Fast training, slower inference

**MODEL-BASED LEARNING**
- DEFINITION: type of Machine Learning where the model learns a generalized function from the training data and uses this function to make predictions.
- CHARACTERISTICS:
1. Builds an explicit model during training
2. Learns patterns and relationships in the data
3. Lower memory usage at inference
4. Requires a training phase


### 5.4 SYSTEM D
**THIS PART WAS DEFINIED IN #4**

### 5.5 SYSTEM E and F
**THIS PART WILL BE DEFINED IN OTHER FOLDER, WE GONNA ANALYZE PART PER PART TO SEE THE BEST OF MACHINE LEARNING**
- WORKFLOW/SYSTEM_E_f



# 6. TRAIN AND IMPROVE THE MODEL

# 7. SHOW THE SOLUTION Share to the other people and the manager
# 8. PUBLIC THE SYSTEM, ANALYZE AND ADJUST Publish to the internet, and see how goes work with new instancies




# FLOWCHART
![alt text](image.png)