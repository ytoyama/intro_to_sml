import numpy as np

N = 100
C = 10
d = 31

X = np.ones((N, d))
Y = np.zeros((N, C))
Y[0, :] = 1
W = np.random.randn(C, d)
b = np.zeros(C)
model = { 'weight': W, 'bias': b }


def calcGrad(X, Y, model):
  W = model['weight']
  b = model['bias']

  second = 1 / (X.dot(W.transpose()) + b).sum(1)
  dW = (np.repeat(Y.reshape((N, C, 1)), d, axis=2) - second.reshape((N, 1, 1))) * X.reshape((N, 1, d))
  db = Y - second.reshape((N, 1))
  return dW.mean(0), db.mean(0)

dW, db = calcGrad(X, Y, model)
assert dW.shape == W.shape, dW.shape
assert db.shape == b.shape, db.shape
