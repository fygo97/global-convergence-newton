import copy
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from methods import Method, LossFunction
from scipy.special import expit
from tqdm import trange
from loss import CELoss, NCCELoss


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
                pred = expit(x_dot_weights)

                if self.method == Method.GD:
                    done = self.perform_GD_update_step(x_batch, y_batch, lr / (epoch + 1))  # decreasing step size
                elif self.method == Method.NEWTON:
                    done = self.perform_Newton_update_step(x_batch, y_batch, lr, lbd=lbd)
                if done:
                    break

            # Evaluate full-batch performance after epoch
            pred_full = expit(np.dot(x, self.weights))
            pred_class = [1 if p > 0.5 else 0 for p in pred_full]
            self.train_accuracies.append(accuracy_score(y, pred_class))
            self.losses.append(self.loss_function.loss(self.weights, x, y))

            if done:
                break


    def perform_GD_update_step(self, x, y, lr):
        assert(self.loss_function != None)
        error_w = self.loss_function.grad(self.weights, x, y)
        self.weights = self.weights - lr * error_w
        return False


    def perform_Newton_update_step(self, x, y, lr, lbd):
        assert(self.loss_function != None)
        error_w = self.loss_function.grad(self.weights, x, y) + lbd * self.weights
        hessian = self.loss_function.hessian(self.weights, x, y) + np.identity(x.shape[1]) * lbd
        H_inv = np.linalg.inv(hessian)
        self.weights = self.weights - lr * np.dot(H_inv, error_w)
        if np.linalg.norm(error_w, ord=np.inf) <= 1e-9:
            print("Abort criteria reached")
            return True
        else:
            return False


    def predict(self, x):
        ones = np.ones(x.shape[0]).reshape((-1, 1))
        x = np.hstack([ones, x])
        x_dot_weights = np.dot(x, self.weights.transpose())
        probabilities = expit(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]



class MultivarLogReg():

    def __init__(self, method, loss_type = LossFunction.CE, num_classes = 10):
        self.losses = []
        self.train_accuracies = []
        self.method = method
        self.loss_type = loss_type
        self.loss_function = None
        self.num_classes = num_classes


    def fit(self, x, y, epochs, lr = 0.1, batch_size = 2048, lbd = 1e-7, alpha = 0.5, mu=0.1):
        ones = np.ones(x.shape[0]).reshape((-1, 1))
        x = np.hstack([ones, x])  # Add bias column

        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        self.weights = np.random.rand(num_classes, x.shape[1]) * 0.1

        n_samples = x.shape[0]

        if self.loss_type == LossFunction.CE:
            self.loss_function = CELoss()
        else:
            self.loss_function = NCCELoss(lambda_=mu,alpha=alpha)


        for class_idx, cls in enumerate(self.classes_):
            if num_classes > 1:
                binary_y = (y == cls).astype(np.float32)
            else:
                binary_y = y
            weights = np.random.rand(x.shape[1]) * 0.1

            per_class_loss = []
            per_class_train_acc = []

            for epoch in trange(epochs, desc="Training Epochs"):
                # Shuffle indices
                indices = np.arange(n_samples)
                np.random.shuffle(indices)

                x_shuffled = x[indices]
                y_shuffled = binary_y[indices]

                done = False

                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    x_batch = x_shuffled[start:end]
                    y_batch = y_shuffled[start:end]

                    if self.method == Method.GD:
                        weights = self.perform_GD_update_step(weights, x_batch, y_batch, lr / (epoch + 1))  # decreasing step size
                    elif self.method == Method.NEWTON:
                        weights = self.perform_Newton_update_step(weights, x_batch, y_batch, lr, lbd=lbd)
                    elif self.method == Method.M22:
                        weights = self.perform_Mishchenko22_update_step(weights, x_batch, y_batch, lbd=lbd)
                    elif self.method == Method.CUBIC:
                        weights = self.perform_Cubic_update_step(weights, x_batch, y_batch, lbd=lbd, L_est=10)

                    if done:
                        break

                # Evaluate full-batch performance after epoch
                pred_class = self.predict(x)
                per_class_train_acc.append(accuracy_score(y, pred_class))
                per_class_loss.append(self.loss_function.loss(weights, x, y))

                if done:
                    break

            self.weights[class_idx] = weights
            self.train_accuracies.append(per_class_train_acc)
            self.losses.append(per_class_loss)


    def perform_GD_update_step(self, weights, x, y, lr):
        assert(self.loss_function != None)
        error_w = self.loss_function.grad(weights, x, y)
        weights = weights - lr * error_w
        return weights


    def perform_Newton_update_step(self, weights, x, y, lr, lbd):
        assert(self.loss_function != None)
        error_w = self.loss_function.grad(weights, x, y) + lbd * weights
        hessian = self.loss_function.hessian(weights, x, y) + np.identity(x.shape[1]) * lbd
        H_inv = np.linalg.inv(hessian)
        weights = weights - lr * np.matmul(H_inv, error_w)
        return weights

    def perform_Mishchenko22_update_step(self, weights, x, y, lbd, H_param = 0.1 ):
        assert(self.loss_function != None)
        g = self.loss_function.grad(weights, x, y) + lbd * weights
        grad_norm = np.linalg.norm(g)
        lambda_k = np.sqrt(H_param * grad_norm)
        Hk = self.loss_function.hessian(weights, x, y) + np.identity(x.shape[1]) * lbd
        reg_Hk = Hk + lambda_k * np.eye(len(Hk))

        try:
            p = np.linalg.solve(reg_Hk, g)
        except np.linalg.LinAlgError:
            p = np.linalg.lstsq(reg_Hk, g, rcond=None)[0]

        weights = weights - p
        return weights

    def perform_Cubic_update_step(self, weights, x, y, L_est, lbd):
        assert(self.loss_function != None)
        g = self.loss_function.grad(weights, x, y)
        H = self.loss_function.hessian(weights, x, y)
        g_norm = np.sqrt(g @ np.linalg.solve(H, g))
        alpha = (-1 + np.sqrt(1 + 2 * L_est * g_norm)) / (L_est * g_norm + 1e-12)
        s = np.linalg.solve(H, g)
        weights = weights - alpha * s
        return weights


    def predict(self, x):
        logits = np.matmul(x, self.weights.T)        # shape: (n_samples, num_classes)
        probs = expit(logits)
        return self.classes_[np.argmax(probs, axis=1)]

