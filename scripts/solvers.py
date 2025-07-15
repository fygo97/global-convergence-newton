import numpy as np

class Solvers:

    def __init__(self, func, grad, hess):
        self.func = func
        self.grad = grad
        self.hess = hess
    # Requirement: hess (i.e. H) L-Lipschitz
    # Regularization factor: lambda_k = sqrt(H * ||∇f(x_k)||)
    def mishchenko(self, x0, H_param=None, max_iter=100, tol=1e-6):
        """
        Mishchenko Newton-type method with affine-invariant scaling and cubic regularization.
        Reference: K. Mishchenko, "A Newton-type method with cubic regularization and affine-invariant scaling," 2022.
        https://arxiv.org/abs/2206.01575

        @param x0: Initial point (numpy array)
        @param H_param: Positive constant H for scaling (float, default 0.1)
        @param max_iter: Maximum number of iterations
        @param tol: Tolerance for stopping criterion on gradient norm

        @return x: Final iterate (numpy array)

        Convergence:
          - Local: Quadratic
          - Global: O(1/k^2)
        """
        if H_param is None:
            H_param = 0.1
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
    
    # Requirement: hess (i.e. H) self-concordant
    # Regularization factor: -
    def affine_invariant_cubic_newton(self, x0, L_est, n_iters=10):
        """
        Affine-Invariant Cubic Newton (AICN) method.
        Reference: S. Hanzely, D. Kamzolov, D. Pasechnyuk, A. Gasnikov, 
                   P. Richtárik, M. Takáč. 
                   "A Damped Newton Method Achieves Global O(1/k^2) and Local Quadratic Convergence Rate."
                   NeurIPS 2022. (Algorithm 1)
                   https://arxiv.org/abs/2206.01894

        @param x0: Initial point (numpy array)
        @param L_est: Cubic regularization parameter (must be >= Hessian-Lipschitz constant)
        @param n_iters: Number of iterations

        @return xs: List of iterates (each is a numpy array)
        """
        x = x0.copy()
        xs = [x.copy()]
        for k in range(n_iters):
            g = self.grad(x)
            H = self.hess(x)
            g_norm = np.sqrt(g @ np.linalg.solve(H, g))
            alpha = (-1 + np.sqrt(1 + 2 * L_est * g_norm)) / (L_est * g_norm + 1e-12)
            s = np.linalg.solve(H, g)
            x_new = x - alpha * s
            xs.append(x_new.copy())
            x = x_new
        return xs
