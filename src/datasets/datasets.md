# Scikit-learn Datasets

This repository/document explains how **datasets work in scikit-learn**, how to **load them**, and where to **find more official datasets** for Machine Learning practice.

Scikit-learn provides built-in datasets mainly for **learning, experimentation, and benchmarking** machine learning algorithms.

---

## Types of Datasets in Scikit-learn

Scikit-learn datasets are generally divided into:

### 1. Toy Datasets
Small, clean datasets mainly used for **learning concepts** and **testing algorithms**.

Examples:
- Iris
- Digits
- Wine
- Breast Cancer

These datasets usually fit entirely in memory and are easy to work with.

---

### 2. Real-world Datasets
Larger and more realistic datasets, often requiring downloading.

Examples:
- California Housing
- Olivetti Faces
- 20 Newsgroups

These datasets are closer to real Machine Learning problems.

---

## How to Import Datasets

All datasets are loaded from the `sklearn.datasets` module.

### Basic Import

```python
from sklearn import datasets
from sklearn.datasets import <namedataset>
