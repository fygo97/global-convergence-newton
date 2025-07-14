import numpy as np

class Solvers:

    def __init__(self, func, grad, hess):
        self.func = func
        self.grad = grad
        self.hess = hess

    def mishchenko(self, x0, H_param=None, max_iter=100, tol=1e-6):
        """
        Implements:
          λ_k = sqrt(H) * ||∇f(x_k)||
          x_{k+1} = x_k - (Hess + λ_k * I)^{-1} ∇f(x_k)
        Exactly matching the algorithm in the given screenshot.

        Parameters:
        - x0 : initial point (numpy array)
        - H_param : positive constant H
        - max_iter : maximum number of iterations
        - tol : stopping criterion on gradient norm

        Convergence:
        - local: quadratic
        - global:1/k^2
        """
        if H_param is None:
            H_param = 0.1  # Default value
        x = x0.copy()
        for _ in range(max_iter):
            g = self.grad(x)
            grad_norm = np.linalg.norm(g)
            if grad_norm < tol:
                break

            lambda_k = np.sqrt(H_param * grad_norm)
            Hk = self.hess(x)
            reg_Hk = Hk + lambda_k * np.eye(len(x))

            try:
                p = np.linalg.solve(reg_Hk, g)
            except np.linalg.LinAlgError:
                p = np.linalg.lstsq(reg_Hk, g, rcond=None)[0]

            x = x - p

        return x
