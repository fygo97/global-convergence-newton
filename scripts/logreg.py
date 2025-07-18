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

        weights_new = None
        weights_old = None
        _H_old = None

        if self.method == Method.ADANP:
            self.weights_new = weights + np.random.randn(*weights.shape) * 2
            self.weights_old = weights + np.random.randn(*weights.shape) * 2
            g_new = self.loss_function.grad(self.weights_new, x, y)
            g_old = self.loss_function.grad(self.weights_old, x, y)
            _Hessian_old = self.loss_function.hessian(self.weights_old, x, y)
            diff = self.weights_new - self.weights_old
            p = _Hessian_old @ diff
            _H_old = np.linalg.norm(g_new - g_old - p) / (np.linalg.norm(diff)**2)

        sigma_k = 0.1


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
                elif self.method == Method.ADANP:
                    self.weights_new, self.weights_old, _H_old = self.perform_ADANP_update_step(self.weights_old, self.weights_new, x, y, _H_old)
                    weights = self.weights_new
                elif self.method == Method.CREG:
                    weights, sigma_k = self.perform_CREG_update_step(weights, x, y, sigma_k=sigma_k)

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

    def perform_ADANP_update_step(self, weights_old, weights_new, x, y, _H_old):
        assert(self.loss_function != None)
        g_new = self.loss_function.grad(weights_new, x, y)
        g_old = self.loss_function.grad(weights_old, x, y)
        _Hessian_new = self.loss_function.hessian(weights_new, x, y)
        _Hessian_old = self.loss_function.hessian(weights_old, x, y)
        d = len(weights_new)
        diff = weights_new - weights_old
        p = _Hessian_old @ diff
        _M_k = np.linalg.norm(g_new - g_old - p) / (np.linalg.norm(diff)**2)
        print(f"M_k = {_M_k}")
        _H_new = np.maximum(_M_k, _H_old / 2)
        _lambda_new = np.sqrt(_H_new * np.linalg.norm(g_new))
        s = np.linalg.solve(_Hessian_new + _lambda_new * np.eye(d), g_new)
        weights_new_new = weights_new - s
        return weights_new_new, weights_new, _H_new

    def perform_CREG_update_step(self, weights, x, y, 
                                sigma_k=1.0, 
                                eta_1=0.1, eta_2=0.9,
                                gamma_1=2.0, gamma_2=4.0,
                                max_cauchy_search=50):
        """
        Performs one Adaptive Regularization with Cubics (ARC) step.

        Args:
            x_k (np.ndarray): Current point (weights).
            x (np.ndarray): Training data.
            y (np.ndarray): Labels.
            sigma_k (float): Regularization parameter.
            eta_1, eta_2 (float): Acceptance thresholds.
            gamma_1, gamma_2 (float): Factors for sigma update.
            max_cauchy_search (int): Max iterations to estimate α^C.

        Returns:
            x_{k+1} (np.ndarray): Updated weights.
            sigma_{k+1} (float): Updated regularization parameter.
        """
        assert self.loss_function is not None
        g_k = self.loss_function.grad(weights, x, y)
        H_k = self.loss_function.hessian(weights, x, y)
        
        d = g_k.shape[0]

        # --- Step 1: Compute Cauchy point s_k^C = -alpha^C * g_k ---
        def m_k(s):
            return g_k @ s + 0.5 * s @ H_k @ s + (sigma_k / 6) * np.linalg.norm(s)**3

        # Approximate argmin over α of m_k(-α g_k)
        alpha_vals = np.logspace(-4, 2, max_cauchy_search)
        min_val = float("inf")
        alpha_c = 0
        for alpha in alpha_vals:
            s_cand = -alpha * g_k
            val = m_k(s_cand)
            if val < min_val:
                min_val = val
                alpha_c = alpha

        s_c_k = -alpha_c * g_k
        m_c_k = m_k(s_c_k)

        # --- Step 1 cont: Find any step s_k such that m_k(s_k) ≤ m_k(s_c_k) ---
        # For simplicity, we use s_k = s_c_k in this implementation
        s_k = s_c_k
        m_s_k = m_k(s_k)

        # --- Step 2: Compute actual improvement and ρ_k ---
        f_k = self.loss_function.loss(weights, x, y)
        f_new = self.loss_function.loss(weights + s_k, x, y)
        actual_reduction = f_k - f_new
        predicted_reduction = f_k - m_s_k
        rho_k = actual_reduction / predicted_reduction if predicted_reduction != 0 else 0

        # --- Step 3: Accept or reject step ---
        if rho_k >= eta_1:
            x_k_plus1 = weights + s_k
        else:
            x_k_plus1 = weights

        # --- Step 4: Update σ_k ---
        if rho_k > eta_2:
            sigma_k_plus1 = np.random.uniform(0, sigma_k)
        elif eta_1 <= rho_k <= eta_2:
            sigma_k_plus1 = np.random.uniform(sigma_k, gamma_1 * sigma_k)
        else:
            sigma_k_plus1 = np.random.uniform(gamma_1 * sigma_k, gamma_2 * sigma_k)

        return x_k_plus1, sigma_k_plus1

    def predict(self, x):
        '''
            Returns a class label prediction.
            Applies expit to X*w elmentwise and derives a label prediction for accuracy plot.
        '''
        logits = x @ self.weights        # shape: (n_samples, num_classes)
        probs = expit(logits)
        prediction = np.array([1 if p > 0.5 else 0 for p in probs])
        return prediction
