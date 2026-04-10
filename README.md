# MACHINE LEARNING BOOKS

This repository is organized as a **complete learning and reference structure for Machine Learning**, combining theory, data workflows, models, and hands-on projects.

---

## Repository Structure

The repository is divided into two main parts: **BOOKS** (theoretical and conceptual content) and **SRC** (practical implementation).

### BOOKS

## 📂 Repository Structure

```text
📦 books ........................... Theoretical foundations and study guides
 ┣ 📂 1. MachineLearningGeral ...... General ML concepts and definitions
 ┃ ┣ 📜 1. Geral.md ................ Overview of ML types (Supervised, Unsupervised, etc.)
 ┃ ┣ 📜 2. Librarys.md ............. Guide to essential libraries (Pandas, NumPy, Matplotlib)
 ┃ ┣ 📜 3. Dictionary.md ........... Glossary of technical ML terms
 ┃ ┗ 📜 4. Books_to_learn.md ....... Curated reading list and resources
 ┣ 📂 2. Data ...................... Data lifecycle and manipulation
 ┃ ┣ 📂 1. Data .................... Fundamentals of data types and sources
 ┃ ┃ ┗ 📜 1. Data.md ............... Theory on raw data and structured information
 ┃ ┣ 📂 2. Eda and Preprocessing ... Exploratory Data Analysis and Cleaning
 ┃ ┃ ┣ 📜 0. PreRequisites.md ...... Baseline knowledge required for EDA
 ┃ ┃ ┣ 📜 1. Eda.md ................ Theory on statistical data exploration
 ┃ ┃ ┣ 📜 1.1 EdaPlots.ipynb ....... Practical visualizations for data insights
 ┃ ┃ ┣ 📜 1.2 EdaSumary.md ......... Executive summary of EDA techniques
 ┃ ┃ ┗ 📜 2. Preprocessing.md ...... Encoding, scaling, and handling missing values
 ┃ ┗ 📂 3. Splits .................. Data partitioning strategies
 ┃ ┃ ┗ 📜 Split.md ................. Theory on Train/Test/Validation and Cross-Validation
 ┣ 📂 3. Models .................... Machine Learning algorithm architectures
 ┃ ┣ 📂 1. Non-Gradient Models ..... Logic-based and distance-based algorithms
 ┃ ┃ ┣ 📜 0. PreRequisites.md ...... Mathematical requirements for non-gradient models
 ┃ ┃ ┣ 📜 1. TreeBasedModels.md .... Decision Trees and their structural logic
 ┃ ┃ ┣ 📜 2. InstancesBasedModels .. Distance-based models like KNN
 ┃ ┃ ┣ 📜 3. EnsembleBasedModels ... Bagging, Boosting, and Random Forests
 ┃ ┃ ┣ 📜 4. SVMsBasedModels ....... Support Vector Machines and Kernel theory
 ┃ ┃ ┗ 📜 5. ProbabilisticModels ... Bayesian models and Naive Bayes
 ┃ ┗ 📂 Gradients Models ........... Optimization-based algorithms
 ┃ ┃ ┗ 📂 ModelsGeral .............. General principles of optimization models
 ┃ ┃ ┃ ┣ 📜 0. PreRequisites.md .... Calculus and Linear Algebra foundations
 ┃ ┃ ┃ ┣ 📜 1. ForwardPass.md ...... How data flows through the model
 ┃ ┃ ┃ ┣ 📜 1.1 ActivationFuncs .... Non-linearity: ReLU, Sigmoid, Softmax
 ┃ ┃ ┃ ┣ 📜 2. LossFunction.md ..... Error measurement: MSE, Cross-Entropy
 ┃ ┃ ┃ ┣ 📜 3. Optimization ........ Gradient Descent and Backpropagation
 ┃ ┃ ┃ ┣ 📜 3.1 TunningModels.md ... GridSearch, RandomSearch, and Hyperparameters
 ┃ ┃ ┃ ┗ 📜 4. Metrics.md .......... Performance evaluation: Accuracy, F1-Score, AUC-ROC
 ┣ 📂 4. Workflow .................. Standardized ML project steps
 ┃ ┗ 📜 Workflow_ML.md ............. End-to-end pipeline from problem to deploy
 ┣ 📂 5. Math ...................... The mathematical engine behind ML
 ┃ ┣ 📂 1. Linear Algebra .......... Vectors, Matrices, and Tensors
 ┃ ┃ ┣ 📂 books .................... Theoretical Linear Algebra notes
 ┃ ┃ ┗ 📂 src ...................... Numpy implementations of algebraic operations
 ┃ ┣ 📂 Calculus .................. Derivatives and Gradients
 ┃ ┃ ┣ 📂 books .................... Calculus theory for optimization
 ┃ ┃ ┗ 📂 src ...................... Numerical calculus implementations
 ┃ ┗ 📂 Prob & Stats .............. Distributions and Uncertainty
 ┃ ┃ ┣ 📂 books .................... Statistical theory for Data Science
 ┃ ┃ ┗ 📂 src ...................... Practical statistical analysis with Python
 ┣ 📂 6. SKLearn ................... Practical framework implementation
 ┃ ┣ 📂 1. Preprocessing ........... Automating data prep with Scikit-Learn
 ┃ ┣ 📂 2. Pipeline ................ Creating robust and reproducible ML pipelines
 ┃ ┗ 📂 3. Models .................. Applying algorithms using the SKLearn API
 ┗ 📂 exercises_hanson_ml .......... Solutions for "Hands-On Machine Learning" book
📦 src ............................. Practical implementations and projects
 ┣ 📂 datasets ..................... Local storage for training data
 ┗ 📂 projects ..................... End-to-end applied Machine Learning portfolio
 ┃ ┣ 📂 Classification ............. Predicting discrete categories/labels
 ┃ ┗ 📂 Regression ................. Predicting continuous numerical values
```

---
# How to Navigate this Repository

This repository is structured to serve as both a theoretical textbook and a practical laboratory. Choose the path that best fits your current goal:

### 🎓 I want to LEARN (Beginner)
If you are just starting your journey, follow the numerical order of the folders in `BOOKS/`:
1. Start with **General Theory** (folders 1 and 2) to understand the landscape.
2. Master the **Mathematical Foundations** (folder 5) before moving to algorithms.
3. Finish with the **Models and Workflow** (folders 3 and 4) to see how all the pieces fit together in a real-world project.

### 🔄 I want to REVIEW (Intermediate / Experienced)
For those who already know the concepts but need a quick technical refresh:
* Jump to **Models/Gradients Models** to review cost functions and optimization logic.
* Consult the **SKLearn** folder for the correct syntax of professional framework implementations.

### 💻 I want CODE & PROJECTS (Practitioner)
If your goal is hands-on implementation:
* **`src/projects`**: End-to-End projects categorized by problem type (Regression/Classification).
* **`Math/src`**: Algorithm logic implemented from scratch using pure **NumPy** (no high-level libraries).
* **`SKLearn/Pipeline`**: Best practices for creating clean, production-ready ML code.

---

# Important: Scope Guide

This repository is focused exclusively on **Classical Machine Learning** (Linear Models, Trees, Ensembles, SVMs, etc.).

> [!IMPORTANT]
> **Deep Learning** (Complex Neural Networks, Computer Vision, Advanced NLP) is hosted in a **separate repository** to maintain focus and organizational clarity. If you are looking for Deep Learning implementations, please visit the link below:
> [Link to your Deep Learning Repo here]

---
---

## Purpose of This Repository

This repository aims to be:

- A **guide for my self study**
- A **learning guide** for beginners  
- A **theoretical reference** for Machine Learning concepts  
- A **hands-on resource** with real implementations  
- A **reusable workflow template** for Machine Learning projects  
