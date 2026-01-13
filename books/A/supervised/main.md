# Supervised Learning

Supervised Learning is a type of Machine Learning where the model learns from **labeled data**.

That means:  
for each input (features), there is a **known correct output**.

---

## Features, Labels and Targets

### Features (X)
- **Independent variables**
- Represent the input data
- Examples: age, height, number of rooms, words in a text, pixels in an image

### Labels / Targets (y)
- **Dependent variables**
- Represent the **correct answer**
- The model learns a function that maps `X → y`

> In practice, **labels and targets mean the same thing**  
> “Label” is more common in classification  
> “Target” is the general term

---

## Goal of Supervised Learning

Learn a function: ' f(X) ≈ y '

So the model can make accurate predictions on **unseen data**.

---

## Problem Types

### 📈 Regression
- Target is **continuous / numeric**
- Examples: house prices, temperature, student grades

### 🏷️ Classification
- Target is **discrete / categorical**
- Examples: spam vs not spam, disease detection, image classes

---

## Common Algorithms

### 🔹 Linear Regression
- Simple and fundamental model
- Learns a linear relationship between features and target
- Core for understanding loss, gradients, weights, and bias

---

### 🔹 Logistic Regression
- Used for **classification**
- Outputs probabilities (via sigmoid function)
- Common in binary classification problems

---

### 🔹 K-Nearest Neighbors (KNN)
- Distance-based algorithm
- Predicts using the **nearest neighbors**
- No explicit training phase (lazy learning)

---

### 🔹 Support Vector Machines (SVM)
- Finds a hyperplane that separates classes
- Powerful in high-dimensional spaces

---

### 🔹 Decision Trees
- Highly interpretable
- Learns rule-based decisions like:

