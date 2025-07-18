import numpy as np

class CE:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def sigmoid(self, x):
        return np.where(x >= 0,1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def value(self, weights):
        sigmoid = self.sigmoid(self.X @ weights)
        loss = -np.mean(self.y * np.log(sigmoid) + (1 - self.y) * np.log(1 - sigmoid))
        return loss

    def grad(self, weights):
        sigmoid = self.sigmoid(self.X @ weights)
        grad = (self.X.T @ (sigmoid - self.y)) / len(self.y)
        return grad

    def hessian(self, weights):
        sigmoid = self.sigmoid(self.X @ weights)
        D_diag = sigmoid * (1 - sigmoid)
        D = np.diag(D_diag)
        hess = (self.X.T @ D @ self.X) / len(self.y)
        return hess


class NCCE:
    def __init__(self, X, y, lambda_, alpha):
        self.X = X
        self.y = y
        self.lambda_ = lambda_
        self.alpha = alpha

    def sigmoid(self, x):
        return np.where(x >= 0,1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def value(self, weights):
        f = np.mean(np.log(1 + np.exp(-self.y * (self.X @ weights))))
        R = self.lambda_ * np.sum(self.alpha * weights**2 / (1 + self.alpha * weights**2))
        return f + R

    def grad(self, weights):
        sigmoid = self.sigmoid(-self.y * (self.X @ weights))
        f = - (self.X.T @ (self.y * sigmoid)) / len(self.y)
        R = 2 * self.lambda_ * self.alpha * weights / (1 + self.alpha * weights**2)**2
        return f + R

    def hessian(self, weights):
        sigmoid = self.sigmoid(-self.y * (self.X @ weights))
        W_diag = sigmoid * (1 - sigmoid)
        W = np.diag(W_diag)
        f = (self.X.T @ W @ self.X) / len(self.y)
        R_diag = 2 * self.lambda_ * self.alpha * (1 - 3 * self.alpha * weights**2) / (1 + self.alpha * weights**2)**3
        R = np.diag(R_diag)
        return f + R
