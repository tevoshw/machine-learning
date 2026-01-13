# Semi-Supervised Learning

## What is it?
Semi-Supervised Learning is a Machine Learning paradigm that uses **a small amount of labeled data** together with **a large amount of unlabeled data** to train models.

It lies between:
- **Supervised Learning**
- **Unsupervised Learning**

---

## More idea
Unlabeled data contains **structure**.
The model learns this structure and uses a few labeled examples to guide decision boundaries.

---

## Why use Semi-Supervised Learning?
- Labeling data is expensive and time-consuming
- Unlabeled data is abundant
- Improves performance with fewer labeled samples

---

## Main approaches

### Self-Training
The model is trained on labeled data, predicts labels for unlabeled data, and reuses the most confident predictions as new labels.

**Related algorithms:**
- Pseudo-Labeling
- Self-Training Classifier

---

### Graph-Based Methods
Build a similarity graph and propagate labels across nearby samples.

**Algorithms:**
- Label Propagation
- Label Spreading

---

### Consistency Regularization
The model is encouraged to produce consistent predictions under data perturbations.

**Algorithms:**
- Π-Model
- Mean Teacher
- FixMatch
- MixMatch

---

### Co-Training
Two models (or two different feature views) label data for each other.

**Algorithms:**
- Co-Training
- Tri-Training

---

## When to use?
- Few labeled samples
- Large unlabeled datasets
- Mostly classification problems

---

## Common Libraries
- scikit-learn
- PyTorch
- TensorFlow

---

## Summary
Semi-Supervised Learning leverages the structure of unlabeled data to reduce labeling costs while maintaining strong performance.
