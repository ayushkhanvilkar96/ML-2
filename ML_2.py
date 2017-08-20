import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

# indexes of data we want to test
test_idx = [0, 50, 100]
    
# Training Data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

# Testing Data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print (test_target)

# Prediction
print (clf.predict(test_data))
