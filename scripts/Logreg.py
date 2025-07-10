import copy
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from methods import Method

class CustomLogReg():
    def __init__(self, method):
        self.losses = []
        self.train_accuracies = []
        self.method = method

    def fit(self, x, y, epochs, lr):
        ones = np.ones(x.shape[0]).reshape((-1, 1))
        x = np.hstack([ones, self._transform_x(x)])
        y = self._transform_y(y)

        self.weights = np.ones(x.shape[1])

        for i in range(epochs):
            print(i)
            x_dot_weights = np.dot(x, self.weights)
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            if self.method == Method.GD:
                self.perform_GD_update_step(x, y, pred, lr)
            elif self.method == Method.NEWTON:
                self.perform_Newton_update_step(x, y, pred, lr)
            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)

    def perform_GD_update_step(self, x, y, pred, lr):
            error_w = self.compute_gradients(x, y, pred)
            self.update_model_parameters(error_w, lr)

    def perform_Newton_update_step(self, x, y, pred, lr):
            error_w = self.compute_gradients(x, y, pred)
            hessian = self.compute_hessian(x, pred)
            H_inv = np.linalg.pinv(hessian)
            self.weights = self.weights - lr * np.dot(H_inv, error_w)

    def compute_loss(self, y_true, y_pred):
        # Clamp predicted values to avoid log(0) and values outside (0,1)
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)

        y_zero_loss = y_true * np.log(y_pred)
        y_one_loss = (1 - y_true) * np.log(1 - y_pred)

        return -np.mean(y_zero_loss + y_one_loss)

    def compute_gradients(self, x, y_true, y_pred):
        # derivative of binary cross entropy
        difference =  y_pred - y_true
        # gradient_b = np.mean(difference)
        gradients_w = np.dot(x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w

    def compute_hessian(self, x, y_pred):
        D = np.diag(y_pred * (1 - y_pred))
        H = np.dot(np.dot(x.T, D), x)
        return H

    def update_model_parameters(self, error_w, lr):
        self.weights = self.weights - lr * error_w

    def predict(self, x):
        ones = np.ones(x.shape[0]).reshape((-1, 1))
        x = np.hstack([ones, x])
        x_dot_weights = np.dot(x, self.weights.transpose())
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])

    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def _transform_x(self, x):
        # x = copy.deepcopy(x)
        return x

    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y
