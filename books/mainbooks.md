# 🧠 Machine Learning — System Organization

This repository is organized to explain **how Machine Learning systems are classified**.  
The structure is divided into **three main folders (A, B, and C)**, each representing a **different classification criterion**.

The goal is that anyone who opens this repository can immediately understand:
- what each folder represents
- what type of content can be found inside
- how the concepts are connected

---

## 📁 A — Type of Data

This folder groups Machine Learning systems according to **the type of data available during training**.

Inside this folder you will find:
- **Supervised Learning**  
  Models trained with labeled data (features + labels).
- **Unsupervised Learning**  
  Models that work with unlabeled data to discover patterns and structures.
- **Semi-supervised Learning**  
  A mix of a small amount of labeled data with a large amount of unlabeled data.
- **Reinforcement Learning**  
  Systems that learn by interacting with an environment through rewards and penalties.

📌 In short:  
This folder answers the question **“What kind of information does the algorithm receive?”**

---

## 📁 B — Data Update Strategy

This folder organizes ML systems according to **how the model learns over time**.

Inside this folder you will find:
- **Batch Learning (Offline)**  
  Models trained on the entire dataset at once and not updated automatically.
- **Online Learning**  
  Models that learn continuously as new data arrives.

📌 In short:  
This folder answers the question **“Does the model learn all at once or continuously?”**

---

## 📁 C — Learning Strategy

This folder organizes ML systems according to **how they learn and generalize from data**.

Inside this folder you will find:
- **Instance-based Learning**  
  The algorithm memorizes training examples and makes predictions based on similarity.
- **Model-based Learning**  
  The algorithm learns a mathematical model that represents patterns in the data.

📌 In short:  
This folder answers the question **“Does the algorithm memorize examples or learn a model?”**

---

## How to read this folder

Any Machine Learning system can be described as a **combination of the three folders**:

> **A (type of data) + B (data update) + C (learning strategy)**

Example:
> Supervised Learning + Batch Learning + Model-based Learning

---

## 🎯 Repository Purpose

- Serve as a study and reference material
- Organize fundamental Machine Learning concepts
- Provide a clear mental model of how ML systems are structured
