from collections import Counter
from decision_tree import DecisionTree
import random

class RandomForest(DecisionTree):
    def __init__(self, data, features, n_trees=5, learning_rate=0.01, max_iterations=1000):
        super().__init__(data, features, learning_rate, max_iterations)
        self.n_trees = n_trees
        self.trees = []

    def bootstrap_sample(self):
        n = len(self.data)
        return [random.choice(self.data) for _ in range(n)]

    def train(self):
        for _ in range(self.n_trees):
            sample = self.bootstrap_sample()
            tree = DecisionTree(sample, self.features)
            self.trees.append(tree.fit(sample))

    def predict(self, sample):
        predictions = [self._predict_one_tree(tree, sample) for tree in self.trees]
        return Counter(predictions).most_common(1)[0][0]

    def _predict_one_tree(self, tree, sample):
        return tree.predict(sample)
