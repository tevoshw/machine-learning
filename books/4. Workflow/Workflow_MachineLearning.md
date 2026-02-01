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
- Problems Addressed in /2. Data/SPLIT

**WHY AND HOW STARING?**
- Prepare the ambient and the data
**TYPES OF SET**
- Choose about train, validation and test sets
**STRATEGIES TO SPLIT THE SET'S**
- Holdout, KFold or Stratified
**PROBLEMS**
- Data Leakage



# 5. SELECT THE SYSTEMS FOR THE MODEL


### 5.1 SYSTEM A  (SUPERVISED, UNSUPERVISED..)
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



#### 5.2 SYSTEM B (ONLINE vs OFFLINE...)

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

### 5.3 SYSTEM C (INSTANCIES vs MODULES)

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


### 5.4 SYSTEM D (DATA SPLIT)
**SPLIT**
- Problems Addressed in /2. Data/SPLIT


### 5.5 SYSTEM E (MODEL)
**GERAL**
- More details addressed in /3. Models/Models Geral
- We need to define 5 characteristics in this system.
1. Parameters: The things that the model learn and stores
2. Objective Function/Loss Function + Regularization (optional): A function that measure the quality of the model for the data and parameters (MSE, CROSS-ENTROPY and more)
3. Optimization method: How the model update the parameters (Gradient Descent, Adam, SGD)
4. Hyperparameters: Control how the model learning (Complexity, regularization and more)5
5. Metrics: A function the calcule the measure to test set



# 6. TRAIN AND IMPROVE THE MODEL
**GERAL**
- Improve the model, with some reviews and adjusts
1. Transform and update the hyperparameters
2. Try others models, loss functions and optimization methods
3. EDA part 2.
- More details addressed in:
1. /3. Models/Models Geral/HyperParameters
2. Problems Addressed in /2. Data/EDA


# 7. SHOW THE SOLUTION Share to the other people and the manager
# 8. PUBLIC THE SYSTEM, ANALYZE AND ADJUST Publish to the internet, and see how goes work with new instancies




# FLOWCHART
![alt text](image.png)