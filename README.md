# Decision Tree & Random Forest - Python Implementation (No Libraries)

---

## Overview

This repository contains simple Python implementations of **Decision Tree** and **Random Forest** classifiers built from scratch without using any machine learning libraries.

These implementations demonstrate the core concepts and algorithms behind these popular models using basic Python code and standard libraries like `math`, `random`, and `collections`.

---

## 1) Decision Tree

### What is a Decision Tree?

A decision tree is a flowchart-like structure used for making decisions based on input features. It splits the data into branches based on feature values and conditions until it reaches a final decision (class label).

### Why Use a Decision Tree?

| Reason                    | Description                                            |
|---------------------------|--------------------------------------------------------|
| 🔍 Easy to Understand     | Works like human decision making: "If this, then that" |
| 🧠 No Complex Math Needed | No calculus or linear algebra required                 |
| 🔀 Handles Mixed Data     | Supports both numerical and categorical features       |
| 💡 Built-in Feature Selection | Automatically chooses the most informative features   |
| 📊 Versatile             | Can be used for classification and regression tasks    |

### What Problems Can It Solve?

- Spam detection (Yes/No)
- Loan approval (Approve/Reject)
- Disease diagnosis (Positive/Negative)
- Weather prediction
- Price prediction (regression)

### How Decision Tree Works - Step by Step

1. Calculate **Entropy** to measure impurity in the data.
2. Calculate **Information Gain** to find the best feature to split the data.
3. Split the data based on that feature.
4. Repeat recursively for each subset until all data is pure or no features left.
5. Build the tree structure in nested dictionaries.

### Code Overview

- `entropy(data)`: Calculates the entropy of class labels in the dataset.
- `split_data(data, column, value)`: Splits dataset based on feature column and value.
- `best_split(data)`: Finds the best feature to split using information gain.
- `build_tree(data, features)`: Recursively builds the decision tree.

---

## 2) Random Forest

### What is Random Forest?

Random Forest is an ensemble learning method that builds many decision trees on random subsets of data and features. It combines their predictions by voting to improve accuracy and reduce overfitting.

### Why Use Random Forest?

| Reason                | Description                                          |
|-----------------------|------------------------------------------------------|
| 🌲 Multiple Trees     | Combines many decision trees for better results      |
| 🎲 Randomness         | Uses random subsets of data and features for diversity|
| 🤝 More Accurate      | Usually performs better than a single decision tree  |
| 💪 Less Overfitting   | Robust to noisy data and less likely to overfit      |
| 👩‍💻 Easy to Use      | Minimal tuning needed and handles both classification and regression |

### Problems Random Forest Can Solve

- Spam detection
- Credit risk evaluation
- Medical diagnosis
- Stock price prediction
- Weather forecasting

### How Random Forest Works - Step by Step

1. **Bootstrap Sampling:** Create multiple random samples with replacement from the training data.
2. **Build Trees:** Build a decision tree on each sample using a random subset of features.
3. **Predict:** Each tree predicts the class of an input.
4. **Majority Vote:** Aggregate predictions by voting; the majority class wins.
5. **Final Prediction:** The combined output is more stable and accurate.

### Code Overview

- `bootstrap_sample(data)`: Generates a random sample of data with replacement.
- `build_tree(data, features)`: Builds a decision tree (with randomness in feature selection).
- `predict(tree, features, sample)`: Predicts a label for a sample using one decision tree.
- `random_forest_predict(trees, features, sample)`: Aggregates predictions from multiple trees via majority voting.
- `train_random_forest(data, features, n_trees)`: Trains multiple trees on bootstrap samples to create a forest.

---

## How to Use

1. Clone the repository.
2. Run the Python script to train decision tree or random forest models on your dataset.
3. Modify the `data` and `features` variables as needed.
4. Use `train_random_forest` and `random_forest_predict` functions for ensemble predictions.
