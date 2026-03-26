# PREPARE THE DATA FOR THE ML ALGORITHMNS (Separate the data, in test, train, validation and more)
**THERE ARE DIFFERENTS PARTS FOR THE DATA IN A MACHINE LEARNING MODE, AND WE'LL GO EXPLORE ALL OF THESE STEPS RIGHT NOW**


## 4.1 WHY AND HOW STARING?
**WHY WE NEED TO SEPARE THE DATASET IN LITTLE DATASETS?**
- A model can learning, and can also memorize the data, or do something else that harm the model to be efective (over and underfitting)
- To avoid these thing of situation and improve the model learning, we split in split datasets, to test, train, verify and more things, that garanted greater power to the model and data

**X and Y**
- DEFINITION: Split the X (independent features) and Y (dependent features), and garanted that all it's ok
- EXAMPLE:
``` 
X = df.drop("target")
Y = df["target"]
```

**RANDOM SEED**
- DEFINITION: A way in functions that generate a random choose of the data
- EXAMPLE: DATASET ->(1,2,3,4,5) RANDOM 2 -> (2,4,3,5,1) RANDOM 33 -> (1,3,5,4,2)
- HANDS-ON: SKLEARN PARAMETERS in train_test_split(x,y random_seed = 42)

**DATAFRAME TO ?**
- The EDA and PREPROCESSING are doing with the data in DATAFRAME, but the model doesn't learning with these type of data, so we need to transform the data (dataframe -> ?)
- We need to transform the dataframe in a numeric matrix , more common in NUMPY/TENSORS matrix, so `X.to_numpy() for NUMPY` and `torch.tensor(X) for TENSORS`
> IN SKLEARN the lib does that for us, in the `model.fit()` so it doesn't necessary,

## 4.2 TYPES OF SET
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


## 4.3 STRATEGIES TO SPLIT THE SET'S
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




## 4.4 PROBLEMS
**DATA LEAKAGE**
- DEFINITION: When a data that should be unavailable at training test, leaks at the model
- CHARACTERISITCS: 
1. The train data goes to test data
2. High accuracy, overfitting problem
3. High accuracy with data leake means nothing, the model memorize it
g