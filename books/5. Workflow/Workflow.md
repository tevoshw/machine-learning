_HOW TO THINK LIKE A PROFESSIONAL, TO DO MACHINE LEARNING PROJECTS?_

# 1. Analyze the situation


### 1.1 Questions?
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
- A.1 (Classification or regression if A = Supervized)
- B (batch or offline)
- C (instances or modules)
- D (dataset split: holdout, crossvalidation, kfold, stratified and etc)
- E (model strategy (not the final model): linear regression, logistic and etc)
- F (perfomance measure: RMSE, MAE and etc)


# 2. Get the data (Import the data in the envoriment) 
**Where's the data comes**
- Kaggle's suport
- A database
- A file

# 3. Identify, visualize and preprocessing the data.   

### 3.1 Exploratory Data Analysis (EDA) 

**Geral**
- Where we gonna analyze the data, don't change, only for analysis and see the data for things that can harm or improve the model


### 3.2 Preprocessing 

**Geral**
- After find in the data things that can harm or improve, we need to change that, here in the preprocessing that happened, here the data are transform for use in the model

> Some times you need to split before the preprocessing e.g. (Scaler, Outliers, OverSampling and more)

> All loaded in the preprocessing part


# 4. Split the data 
**Types of set**
- Choose about train, validation and test sets
**Strategies to split**
- Holdout, KFold or Stratified


# 5. Select the systems (Model PARTS)


### 5.1 System A (The MODELS)

- Define the Model's that we gonna test (searching for the best) 

### 5.2 System B (HyperParameters)

- Define the hyperparameters for each model 

### 5.3 System C (EXTRAS)

1. The batch size
2. The METRICS that we gonna use


# 6. Train the model and Tunning Model

- See the metrics
- Tunning model strategies
- ReLOOP

# 7. Evaluate the model for the World

- Creates a website or something to others people use