import random
import math
from collections import Counter
from base_model import BaseModel

class DecisionTree(BaseModel):
    def __init__(self, data, features, learning_rate=0.01, max_iterations=1000):
        super().__init__(learning_rate, max_iterations)
        self.data = data
        self.features = features
        self.tree = None

    def split_data(self, column, value):
        split = []
        for row in self.data:
            if row[column] == value:
                reduced_row = row[:column] + row[column + 1:]
                split.append(reduced_row)
        return split

    def best_split(self):
        labels = [row[-1] for row in self.data]
        base_entropy = self.calculate_entropy(labels)
        best_gain = 0
        best_column = -1
        n_features = len(self.data[0]) - 1

        for col in range(n_features):
            values = set(row[col] for row in self.data)
            new_entropy = 0
            for val in values:
                subset = self.split_data(col, val)
                subset_entropy = self.calculate_entropy([row[-1] for row in subset])
                p = len(subset) / len(self.data)
                new_entropy += p * subset_entropy
            info_gain = base_entropy - new_entropy
            if info_gain > best_gain:
                best_gain = info_gain
                best_column = col

        return best_column

    def calculate_entropy(self, labels):
        total = len(labels)
        label_counts = Counter(labels)
        ent = 0.0
        for label in label_counts:
            p = label_counts[label] / total
            ent_component = -p * math.log2(p)
            ent += ent_component
        return ent

    def build_tree(self):
        labels = [row[-1] for row in self.data]

        if labels.count(labels[0]) == len(labels):
            return labels[0]

        if len(self.data[0]) == 1:
            return Counter(labels).most_common(1)[0][0]

        best_feat = self.best_split()
        best_feat_name = self.features[best_feat]
        tree = {best_feat_name: {}}
        values = set(row[best_feat] for row in self.data)

        for val in values:
            subset = self.split_data(best_feat, val)
            sub_features = self.features[:best_feat] + self.features[best_feat + 1:]
            subtree = DecisionTree(subset, sub_features)
            tree[best_feat_name][val] = subtree.build_tree()

        return tree

    def fit(self, data):
        self.tree = self.build_tree()
        print(f"Decision Tree Built:\n{self.tree}")
        return self

    def predict(self, sample):
        tree = self.tree
        while isinstance(tree, dict):
            root = next(iter(tree))
            feature_value = sample[self.features.index(root)]
            tree = tree[root].get(feature_value)
            if tree is None:
                return None
        return tree
