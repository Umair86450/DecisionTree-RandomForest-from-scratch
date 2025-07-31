from decision_tree import DecisionTree
from random_forest import RandomForest

# Example Data
data = [
    ['Sunny', 'Hot', 'No'],
    ['Sunny', 'Hot', 'No'],
    ['Sunny', 'Mild', 'Yes'],
    ['Sunny', 'Cool', 'Yes'],
    ['Overcast', 'Hot', 'Yes'],
    ['Overcast', 'Mild', 'Yes'],
    ['Overcast', 'Cool', 'Yes'],
    ['Rainy', 'Hot', 'No'],
    ['Rainy', 'Mild', 'Yes'],
    ['Rainy', 'Cool', 'Yes'],
]

features = ['Outlook', 'Temp']

# Train Decision Tree
print("### Training Decision Tree ###")
decision_tree = DecisionTree(data, features)
decision_tree.fit(data)

# Predict with Decision Tree
sample = ['Sunny', 'Cool']
print(f"Prediction for {sample} with Decision Tree: {decision_tree.predict(sample)}")

# Train Random Forest
print("\n### Training Random Forest ###")
random_forest = RandomForest(data, features, n_trees=5)
random_forest.train()

# Predict with Random Forest
prediction = random_forest.predict(sample)
print(f"Prediction for {sample} with Random Forest: {prediction}")
