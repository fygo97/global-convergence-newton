import numpy as np
from scipy.special import expit

class CELoss:
    def loss(self, weights, x, y):
        probs = expit(x @ weights)
        loss = -np.mean(y * np.log(probs + 1e-12) + (1 - y) * np.log(1 - probs + 1e-12))
        return loss

    def grad(self, weights, x, y):
        probs = expit(x @ weights.T)
        grad = (x.T @ (probs - y)) / len(y)
        return grad

    def hessian(self, weights, x, y):
        probs = expit(x @ weights)
        D_diag = probs * (1 - probs)
        D = np.diag(D_diag)
        hess = (x.T @ D @ x) / len(y)
        return hess

class NCCELoss:
    def __init__(self, lambda_, alpha):
        self.lambda_ = lambda_
        self.alpha = alpha

    def loss(self, weights, x, y):
        z = x @ weights
        f = np.mean(np.log(1 + np.exp(-y * z)))
        r = self.lambda_ * np.sum(self.alpha * weights**2 / (1 + self.alpha * weights**2))
        return f + r

    def grad(self, weights, x, y):
        z = x @ weights
        sigma = expit(-y * z)
        f = - (x.T @ (y * sigma)) / len(y)
        r = 2 * self.lambda_ * self.alpha * weights / (1 + self.alpha * weights**2)**2
        return f + r

    def hessian(self, weights, x, y):
        z = x @ weights
        sigma = expit(-y * z)
        W_diag = sigma * (1 - sigma)
        W = np.diag(W_diag)
        f = (x.T @ W @ x) / len(y)
        r_diag = 2 * self.lambda_ * self.alpha * (1 - 3 * self.alpha * weights**2) / (1 + self.alpha * weights**2)**3
        r = np.diag(r_diag)
        return f + r
