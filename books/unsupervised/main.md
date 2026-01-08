# Unsupervised Learning

Unsupervised Learning is a type of Machine Learning where the model learns from **unlabeled data**.

There are **no labels, no targets, and no ground truth answers**.

---

## Features Only (No Labels)

In unsupervised learning, the dataset contains **only features (X)**.

Example:

| Height (m) | Weight (kg) |
|-----------:|------------:|
| 1.80 | 75 |
| 1.65 | 90 |
| 1.90 | 82 |
| 1.60 | 70 |
| 1.85 | 78 |
| 1.62 | 88 |

Each sample:
[height, weight]

---

## Goal of Unsupervised Learning

To **discover hidden structure** in data, such as:
- Natural groupings
- Similarity patterns
- Outliers and anomalies
- Lower-dimensional representations

---

## Main Categories

---

## 🔹 Clustering

Clustering algorithms group similar samples together **without knowing the correct labels**.

The output is a **cluster ID**, also called a **pseudo-label**.

### Common Clustering Algorithms

#### • K-Means
- Partitions data into **K clusters**
- Each sample is assigned to the nearest centroid
- Fast and widely used

#### • Hierarchical Clustering
- Builds a **tree of clusters (dendrogram)**
- Does not require a predefined number of clusters
- Can be agglomerative or divisive

#### • DBSCAN
- Density-based clustering
- Finds clusters of arbitrary shape
- Automatically detects **outliers**
- No need to specify number of clusters

#### • Mean Shift
- Centroid-based algorithm
- Automatically finds the number of clusters
- Works well for dense regions

#### • Gaussian Mixture Models (GMM)
- Probabilistic clustering
- Each point belongs to a cluster with a probability
- Soft clustering (not hard assignments)

---

## 🔹 Anomaly Detection

Anomaly detection focuses on identifying **rare, abnormal, or unexpected data points**.

### Common Anomaly Detection Algorithms

#### • Isolation Forest
- Isolates anomalies instead of profiling normal data
- Efficient for high-dimensional datasets

#### • One-Class SVM
- Learns a boundary around normal data
- Anything outside is considered an anomaly

#### • Local Outlier Factor (LOF)
- Detects anomalies based on local density deviation
- Points with much lower density than neighbors are anomalies

#### • Elliptic Envelope
- Assumes data follows a Gaussian distribution
- Detects outliers based on covariance

---

## 🔹 Dimensionality Reduction

Reduces the number of features while preserving important structure.

Used for:
- Visualization
- Noise reduction
- Feature extraction
- Speeding up training

### Common Dimensionality Reduction Algorithms

#### • Principal Component Analysis (PCA)
- Linear method
- Maximizes variance along new axes
- Widely used and fast

#### • t-SNE
- Non-linear method
- Preserves local structure
- Excellent for visualization (2D / 3D)

#### • UMAP
- Non-linear and scalable
- Preserves both local and global structure
- Faster than t-SNE

#### • Autoencoders
- Neural-network-based dimensionality reduction
- Learns compressed latent representations

---

## Relationship with Supervised Learning

Unsupervised learning is often used to:
- Explore data before labeling
- Create features for supervised models
- Generate pseudo-labels for semi-supervised learning

Example:
Unsupervised → Clustering → Pseudo-labels → Supervised Model

---

## Summary

- ✔ No labels or targets
- ✔ Discovers patterns and structure
- ✔ Includes clustering, anomaly detection, and dimensionality reduction
- ✔ Essential for exploratory data analysis (EDA)

---

> Unsupervised Learning helps you understand your data  
> before trying to predict anything.


## 70% of the time is spent transforming unsupervised data into supervised data, using cluster and anomaly algorithms.