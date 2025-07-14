import numpy as np

class CrossEntropyLoss:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def sigmoid(self, x):
        if np.isscalar(x):
            if x >= 0:
                z = np.exp(-x)
                return 1 / (1 + z)
            else:
                z = np.exp(x)
                return z / (1 + z)
        else:
            return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def value(self, omega):
        probs = self.sigmoid(self.X @ omega)
        loss = -np.mean(self.y * np.log(probs + 1e-12) + (1 - self.y) * np.log(1 - probs + 1e-12))
        return loss

    def grad(self, omega):
        probs = self.sigmoid(self.X @ omega)
        grad = (self.X.T @ (probs - self.y)) / len(self.y)
        return grad

    def hessian(self, omega):
        probs = self.sigmoid(self.X @ omega)
        D_diag = probs * (1 - probs)
        D = np.diag(D_diag)
        hess = (self.X.T @ D @ self.X) / len(self.y)
        return hess

class LogisticRegularizedLoss:
    def __init__(self, X, y, lambda_, alpha):
        self.X = X
        self.y = y
        self.lambda_ = lambda_
        self.alpha = alpha

    def sigmoid(self, x):
        if np.isscalar(x):
            if x >= 0:
                z = np.exp(-x)
                return 1 / (1 + z)
            else:
                z = np.exp(x)
                return z / (1 + z)
        else:
            return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def value(self, omega):
        z = self.X @ omega
        f = np.mean(np.log(1 + np.exp(-self.y * z)))
        r = self.lambda_ * np.sum(self.alpha * omega**2 / (1 + self.alpha * omega**2))
        return f + r

    def grad(self, omega):
        z = self.X @ omega
        sigma = self.sigmoid(-self.y * z)
        f = - (self.X.T @ (self.y * sigma)) / len(self.y)
        r = 2 * self.lambda_ * self.alpha * omega / (1 + self.alpha * omega**2)**2
        return f + r

    def hessian(self, omega):
        z = self.X @ omega
        sigma = self.sigmoid(-self.y * z)
        W_diag = sigma * (1 - sigma)
        W = np.diag(W_diag)
        f = (self.X.T @ W @ self.X) / len(self.y)
        r_diag = 2 * self.lambda_ * self.alpha * (1 - 3 * self.alpha * omega**2) / (1 + self.alpha * omega**2)**3
        r = np.diag(r_diag)
        return f + r
