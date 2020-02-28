from sklearn import linear_model
import numpy as np
X = [[20, 3],
     [23, 7],
     [31, 10],
     [42, 13],
     [50, 7],
     [60, 5]]
Y = [0, 1, 1, 1, 0, 0]

lr = linear_model.LogisticRegression()
lr.fit(X, Y)
test_x = [[20, 10]]
y = lr.predict(test_x)
print(y)
prob = lr.predict_proba(test_x)
print("预计概率", prob)
x = np.random.random([3,3])
print(dir(x))
print("version1")
