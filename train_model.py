import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np


X = np.array([
    [25, 40],
    [45, 60],
    [30, 35],
    [50, 50],
    [23, 20],
])
y = [0, 1, 0, 1, 0]  

model = LogisticRegression()
model.fit(X, y)


with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
