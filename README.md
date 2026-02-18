# MACHINE LEARNING BOOKS

This repository is organized as a **complete learning and reference structure for Machine Learning**, combining theory, data workflows, models, and hands-on projects.

---

## Repository Structure

The repository is divided into two main parts: **BOOKS** (theoretical and conceptual content) and **SRC** (practical implementation).

### BOOKS

```text
# 📂 Machine Learning Repository Structure

├── 📁 books ........................... (Theoretical foundations and ML study notes)
│   │
│   ├── 📁 1. MachineLearningGeral ..... (Core concepts and terminology)
│   │   ├── 📝 Books_to_learn.md ....... (Curated reading list)
│   │   ├── 📝 Dictionary.md ........... (Key terms and definitions)
│   │   ├── 📝 Geral.md ................ (High-level overview of ML)
│   │   └── 📝 Librarys.md ............. (Notes on essential libraries like Pandas/NumPy)
│   │
│   ├── 📁 2. Data ..................... (Data lifecycle and manipulation)
│   │   ├── 📁 EDA, PREPROCESSING ....... (Data cleaning and visualization)
│   │   │   ├── 📝 PREPROCESSING.md .... (Feature engineering notes)
│   │   │   └── 📄 EDA_PLOTS.ipynb ..... (Exploratory graphs and charts)
│   │   └── 📁 SPLIT ................... (Data partitioning strategies)
│   │       └── 📝 Split.md ............ (Train/Test/Validation splitting logic)
│   │
│   ├── 📁 3. Models ................... (Deep dive into model architecture)
│   │   ├── 📁 Gradients Models ........ (Math behind gradient-based learning)
│   │   │   └── 📁 ModelsGeral ......... (Step-by-step model logic)
│   │   │       ├── 📝 0. PreReq.md .... (Required math and logic)
│   │   │       ├── 📝 1. Forward.md ... (Data flow through the model)
│   │   │       ├── 📝 2. Loss.md ...... (Measuring prediction error)
│   │   │       ├── 📝 3. OptIntern.md . (Internal parameter updates)
│   │   │       └── 📝 4. Metrics.md ... (Model evaluation methods)
│   │   └── 📁 Non-Gradient Models ..... (Models not using gradient descent)
│   │
│   ├── 📁 4. Workflow ................. (Standard operating procedures for ML)
│   │   └── 📝 Workflow_ML.md .......... (The end-to-end pipeline)
│   │
│   ├── 📁 5. SKLearn .................. (Practical Scikit-Learn implementation)
│   │   ├── 📁 Models .................. (Built-in model applications)
│   │   ├── 📁 pipeline ................ (Automating the ML workflow)
│   │   └── 📁 preprocessing ........... (Data scaling and encoding)
│   │
│   └── 📁 exercises_hanson ............ (Practical tasks from Hands-On ML book)
│
├── 📁 src (Practical Implementation & Experiments)
│   ├── 📁 datasets: Local data storage and Scikit-learn dataset loaders.
│   └── 📁 projects: End-to-end ML projects (e.g., Housing Price Prediction).
````

---
## How to Read This Repository

### ! For Beginners

If you are starting in Machine Learning, follow this recommended order:

1. **BOOKS / MachineLearningGeral**  
   → Learn the fundamentals, terminology, and core concepts

2. **BOOKS / Data**  
   → Understand EDA, preprocessing, and data splitting

3. **BOOKS / Models**  
   → Study different modeling systems and algorithms

> **Important:**  
> Always read and explore the corresponding code in **`SRC/PROJECTS`** while studying the theory.

---

### For Practitioners / Experienced Users

If you already have experience and are looking for something specific, feel free to jump directly to the relevant section:

- **BOOKS/MachineLearningGeral**  
  → General Machine Learning concepts and theory

- **BOOKS/Data**  
  → EDA, preprocessing, and data preparation techniques

- **BOOKS/Models**  
  → Modeling systems and algorithm implementations

---

## Purpose of This Repository

This repository aims to be:

- A **learning guide** for beginners  
- A **theoretical reference** for Machine Learning concepts  
- A **hands-on resource** with real implementations  
- A **reusable workflow template** for Machine Learning projects  
