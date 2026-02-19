# Error Dictionary: Case #01 - Overfitting

## 1. What is it?
**Overfitting** occurs when a model learns the "noise" and specific details in the training data to the extent that it negatively impacts the performance of the model on new data. The model essentially memorizes the training set instead of learning the underlying patterns.

> **Slogan:** "Perfect on training, a disaster in the real world."

---

## 2. Most Frequent Models
While it can happen in any architecture, it is most common in:
* **Deep Neural Networks:** Due to the massive number of trainable parameters ($W$).
* **Decision Trees:** When they grow without "pruning" or depth limits.
* **CNNs (Convolutional Neural Networks):** Especially when trained on small datasets without augmentation.
* **High-Degree Polynomial and Linear Regression:** When the curve tries to touch every single data point.

---

## 3. Why does it happen? (Root Causes)

1. **Excessive Model Complexity:** Having more "memory capacity" (neurons/layers) than the actual problem requires.
2. **Small Dataset:** With few examples, the model can easily create a "unique rule" for every single data point.
3. **Training for too many Epochs:** Running the optimizer for too long until it starts fitting the random fluctuations of the data.
4. **Lack of Regularization:** Absence of constraints like *Dropout* or *Weight Decay* (L2) that prevent weights from becoming too large and unstable.
5. **Data Leakage:** When information from the test/validation set "leaks" into the training process, making the model look artificially perfect.

---

## 4. Identification via Logs


## GAP A 
- **How to treat GAP A:**
1. Simplify the model
2. Using Weight decay (regularization)

### (Classification Example)
Context: You are training a model to detect fraudulent bank transactions. The dataset has 100,000 rows (samples), but fraud is a rare event. You decided to use a Deep Neural Network with 10 layers and did not apply any regularization techniques.

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.850 | 55% | 0.860 | 54% | Underfitting (Learning) |
| 10 | 0.420 | 85% | 0.440 | 83% | Optimal (Generalizing) |
| 20 | 0.210 | 94% | 0.250 | 90% | Warning (Divergence starts) |
| 30 | 0.080 | 98% | 0.550 | 85% | Overfitting (Memorizing) |
| 40 | 0.010 | 100% | 1.100 | 82% | Hard Overfit |


### (Regression Example)
Context: You have 1,000 samples and 50 features. You are using a very deep network with Mean Squared Error (MSE) as the loss.

| Epoch | Train MSE | Val MSE | Status |
| :--- | :--- | :--- | :--- |
| 1 | 500.0 | 510.0 | Initializing |
| 20 | 120.0 | 135.0 | Convergence |
| 50 | 45.0 | 110.0 | **Gap Opening** |
| 100 | 12.0 | 350.0 | **Critical Failure** |

## GAP B
- **How to treat GAP B:**
1. Verify if have DATA LEAKAGE


### Regression Example
The Loan Approval Predictor
You are predicting if a customer will be granted a **Loan** (Yes/No).
* **Dataset:** 5,000 samples / 15 features.
* **The Trap:** The feature `Loan_Account_Number` was accidentally included in the training features.

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.550 | 0.540 | 65% | 66% | Initializing |
| 2 | 0.001 | 0.001 | 99.9% | 99.9% | **Suspicious** |
| 3 | 0.000 | 0.000 | 100% | 100% | **Total Leakage** |