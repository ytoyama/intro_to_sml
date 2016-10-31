import numpy as np



def softmax(X):
  N = X.shape[0]
  exp = np.exp(X - X.max(1).reshape((N, 1)))
  return exp / exp.sum(1).reshape(N, 1)

a = np.array([[1, 2, 3], [4, 5, 6]])
print(softmax(a))
a = np.exp(a)
print(softmax(a))
a = np.exp(a)
print(softmax(a))
