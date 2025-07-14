import copy
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from methods import Method, LossFunction
from scipy.special import expit
from tqdm import trange
from loss_functions import CELoss, NCCELoss


class CustomLogReg():

    def __init__(self, method, loss_type = LossFunction.CE):
        self.losses = []
        self.train_accuracies = []
        self.method = method
        self.loss_type = loss_type
        self.loss_function = None


    def fit(self, x, y, epochs, lr = 0.1, batch_size = 2048, lbd = 1e-7, alpha = 0.5, mu=0.1):
        ones = np.ones(x.shape[0]).reshape((-1, 1))
        x = np.hstack([ones, x])  # Add bias column

        self.weights = np.random.rand(x.shape[1]) * 0.1

        n_samples = x.shape[0]

        if self.loss_type == LossFunction.CE:
            self.loss_function = CELoss()
        else:
            self.loss_function = NCCELoss(lambda_=mu,alpha=alpha)

        for epoch in trange(epochs, desc="Training Epochs"):
            # Shuffle indices
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            x_shuffled = x[indices]
            y_shuffled = y[indices]

            done = False

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                x_dot_weights = np.dot(x_batch, self.weights)
                pred = self._sigmoid(x_dot_weights)

                if self.method == Method.GD:
                    done = self.perform_GD_update_step(x_batch, y_batch, pred, lr / (epoch + 1))  # decreasing step size
                elif self.method == Method.NEWTON:
                    done = self.perform_Newton_update_step(x_batch, y_batch, pred, lr, lbd=lbd)
                if done:
                    break

            # Evaluate full-batch performance after epoch
            pred_full = self._sigmoid(np.dot(x, self.weights))
            pred_class = [1 if p > 0.5 else 0 for p in pred_full]
            self.train_accuracies.append(accuracy_score(y, pred_class))
            self.losses.append(self.compute_loss(y, pred_full))

            if done:
                break


    def perform_GD_update_step(self, x, y, pred, lr):
        error_w = self.compute_gradients(x, y, pred)
        self.weights = self.weights - lr * error_w
        return False
        # print(np.linalg.norm(error_w))

    def perform_Newton_update_step(self, x, y, pred, lr, lbd):
        error_w = self.compute_gradients(x, y, pred) + lbd * self.weights
        hessian = self.compute_hessian(x, pred) + np.identity(x.shape[1]) * lbd
        H_inv = np.linalg.inv(hessian)
        self.weights = self.weights - lr * np.dot(H_inv, error_w)
        if np.linalg.norm(error_w, ord=np.inf) <= 1e-9:
            print("Abort criteria reached")
            return True
        else:
            return False


    def compute_loss(self, y_true, y_pred, alpha = 0.5, mu = 0.01):
        # Clamp predicted values to avoid log(0) and values outside (0,1)
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)

        y_zero_loss = y_true * np.log(y_pred)
        y_one_loss = (1 - y_true) * np.log(1 - y_pred)

        if self.loss_function == LossFunction.CE:
            loss = -np.sum(y_zero_loss + y_one_loss) / y_true.shape[0]
        else:
            reg = mu * np.sum((alpha * self.weights**2) / (1 + alpha * self.weights**2))
            loss = (-np.sum(y_zero_loss + y_one_loss) / y_true.shape[0]) + reg

        return loss

    def compute_gradients(self, x, y_true, y_pred, alpha = 0.5, mu = 0.01):
        # derivative of binary cross entropy
        difference =  y_pred - y_true

        gradients_w = np.dot(x.transpose(), difference)
        gradients_w = gradients_w / x.shape[0]

        if self.loss_function == LossFunction.NCCE:
            reg = mu * np.sum( (2 * alpha * self.weights) / (1 + alpha * self.weights**2)**2)
            gradients_w = gradients_w + reg

        return gradients_w

    def compute_hessian(self, x, y_pred, alpha = 0.5, mu = 0.01):
        D = np.diag(y_pred * (1 - y_pred))
        H = np.dot(np.dot(x.T, D), x) * (1.0 / x.shape[0])

        if self.loss_function == LossFunction.NCCE:
            reg = mu * np.sum((2 * alpha * (1 - alpha * self.weights**2) ) / (1 + alpha * self.weights**2)**3)
            H = H + np.diag(reg)

        return H

    def predict(self, x):
        ones = np.ones(x.shape[0]).reshape((-1, 1))
        x = np.hstack([ones, x])
        x_dot_weights = np.dot(x, self.weights.transpose())
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]

    def _sigmoid(self, x):
        return expit(x)
