# HOW TO THINK LIKE A PROFESSIONAL, TO DO MACHINE LEARNING PROJECTS?

- 1. **ANALYZE THE SITUATION:** For what we're searching, what is the objective? How we are gonna use that?


- 2. **GET THE DATA:** Import the data in the envoriment 


# 3.  IDENTIFY AND VISUALIZE THE DATA TO GET IMPORTANT INFORMATIONS 
**Identify the priorites in the dataset and select the features in the dataset**
- High correlation (sometimes a feature doesn't have a hight correlation with the label, but have with others features, so we delethe that feature (!DEPENDING THE MODEL AND THE OBJECTIVE!) )
- Cardinality (some features have a high cardinality, so depending on the model it's better to delete theses data)
- High null features values (some features have a high null values, so we need to delete the SAMPLES who doens't have the value)

- 4. **PREPARE THE DATA FOR THE ML ALGORITHMNS:** Separate the data, in test, train, validation and more

- 5. **SELECT AND TRAIN THE MODEL:** Analyze the models, identify who is better and train the model
supervised or not, regression or classification, batch (offline) or online, per instancies (similar) or per model (maths)




- 6. **IMPROVE THE MODEL:** See the erros, and try to improve the error accurancy, here goes the news predicr (with scikit model.predict) and after visualyze the performance measure (MRSE, MAE and more)
And after, select news hyperparameters or something to improve the error measure


- 7. **SHOW THE SOLUTION:** Share to the other people and the manager
- 8. **PUBLIC THE SYSTEM, ANALYZE AND ADJUST:** Publish to the internet, and see how goes work with new instancies




# FLOWCHART
![alt text](image.png)