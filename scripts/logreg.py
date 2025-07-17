import copy
import logging
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from methods import Method, LossFunction
from scipy.special import expit
from tqdm import trange
from loss import CELoss, NCCELoss
import time

logger = logging.getLogger(__name__)


class MultivarLogReg():

    def __init__(self, method, loss_type = LossFunction.CE, num_classes = 10):
        self.losses = []
        self.train_accuracies = []
        self.method = method
        self.loss_type = loss_type
        self.loss_function = None
        self.num_classes = num_classes
        self.grad_norm = []
        self.criterion_reached = -1
        self.time_to_convergence = 0


    def fit(self, x, y, epochs, lr = 1, batch_size = None, lbd = 0, alpha = 1.0, mu=0.001, H_adan_0 = 0.1, epsilon = 1e-8):

        # ones = np.ones(x.shape[0]).reshape((-1, 1))
        # x = np.hstack([ones, x])  # Add bias column

        start_time = time.time()

        H_adan = 4 * H_adan_0

        batch_size = x.shape[0] if batch_size == None else 2048

        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        self.weights = np.ones(x.shape[1]) * 0.1
        self.train_accuracies.append(accuracy_score(y_true=y, y_pred=self.predict(x)))
        logger.info(f"full weights shape = {self.weights.shape}")

        n_samples = x.shape[0]

        if self.loss_type == LossFunction.CE:
            self.loss_function = CELoss()
        else:
            self.loss_function = NCCELoss(lambda_=mu,alpha=alpha)
            y = y * 2 - 1

        weights = np.random.randn(x.shape[1]) * 0.1

        logger.info(f"single weights shape = {weights.shape}")

        for epoch in trange(epochs, desc="Training Epochs"):

            # Batch loop: Allows epochs to be split into batches. we do not split @fynn) Only performs one iteration when batch_size is chosen as None 
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_batch = x[start:end]
                y_batch = y[start:end]

                if self.method == Method.GD:
                    weights = self.perform_GD_update_step(weights, x_batch, y_batch, lr / (epoch + 1))  # decreasing step size
                elif self.method == Method.NEWTON:
                    weights = self.perform_Newton_update_step(weights, x_batch, y_batch, lr, lbd=lbd)
                elif self.method == Method.GRN:
                    weights = self.perform_GRN_update_step(weights, x_batch, y_batch, lbd=lbd)
                elif self.method == Method.AICN:
                    weights = self.perform_AICN_update_step(weights, x_batch, y_batch, L_est=10)
                elif self.method == Method.ADAN:
                    weights, H_adan = self.perform_ADAN_update_step(weights, x_batch, y_batch, _H = H_adan)

            # Evaluate full-batch performance after epoch
            self.losses.append(self.loss_function.loss(weights, x, y))
            self.grad_norm.append(np.linalg.norm(self.loss_function.grad(weights, x, y)))

            if (np.linalg.norm(self.loss_function.grad(weights, x, y), ord=np.inf) < epsilon and self.criterion_reached == -1):
                self.criterion_reached = epoch
                stop_time = time.time()
                self.time_to_convergence = stop_time - start_time

            self.weights = weights
            self.train_accuracies.append(accuracy_score(y_true=y, y_pred=self.predict(x)))

            print(f"y = {y}, prediction = {self.predict(x)}")



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
    
    def perform_ADAN_update_step(self, weights_0, x, y, _H):
        assert(self.loss_function != None)
        weights_k = weights_0.copy()
        d = len(weights_k)
        _H = _H / 4

        grad_norm_k = np.linalg.norm(self.loss_function.grad(weights_k,x,y))
        hessian_k = self.loss_function.hessian(weights_k,x,y)
        g = self.loss_function.grad(weights_k, x, y)

        while True:

            _H = 2 *_H #line 5
            lambda_k = np.sqrt(_H*grad_norm_k) #line 7 (line 6 is pointless)

            #define weights_plus <=> line 8--------
            reg_hessian_k = hessian_k + lambda_k * np.eye(d)
            p = np.linalg.solve(reg_hessian_k, g)
            weights_plus = weights_k - p
            #--------------------------------------

            r_plus = np.linalg.norm(weights_plus - weights_k)# <=> line 9

            # line 10 
            grad_norm_plus = np.linalg.norm(self.loss_function.grad(weights_plus,x,y))
            loss_weights_plus = self.loss_function.loss(weights_plus,x,y)
            loss_weights_k = self.loss_function.loss(weights_k,x,y)
            if ((grad_norm_plus <= 2 * lambda_k * r_plus) and loss_weights_plus <= (loss_weights_k - (2.0 / 3.0) * lambda_k * r_plus**2) ):
                weights_k = weights_plus
                break

        return weights_k, _H

    def perform_GRN_update_step(self, weights, x, y, lbd, H_param = 0.1 ):
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

    def perform_AICN_update_step(self, weights, x, y, L_est):
        assert(self.loss_function != None)
        g = self.loss_function.grad(weights, x, y)
        H = self.loss_function.hessian(weights, x, y)
        p = np.linalg.solve(H, g)
        _G = L_est * (g @ p)
        alpha = (-1 + np.sqrt(1 + 2 * _G)) / _G
        print(alpha)
        weights = weights - alpha * p
        return weights


    def predict(self, x):
        '''
            Returns a class label prediction.
            Applies expit to X*w elmentwise and derives a label prediction for accuracy plot.
        '''
        logits = x @ self.weights        # shape: (n_samples, num_classes)
        probs = expit(logits)
        prediction = np.array([1 if p > 0.5 else 0 for p in probs])
        return prediction
