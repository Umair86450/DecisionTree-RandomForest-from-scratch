from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self._learning_rate = learning_rate
        self._max_iterations = max_iterations

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def get_learning_rate(self):
        return self._learning_rate

    def set_learning_rate(self, value):
        if value > 0:
            self._learning_rate = value
        else:
            print("Invalid learning rate")
