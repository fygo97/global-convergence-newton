import numpy as np

class Solvers:

    def __init__(self, func, grad, hess):
        self.func = func
        self.grad = grad
        self.hess = hess
    # Requirement: hess (i.e. H) L-Lipschitz
    # Regularization factor: lambda_k = sqrt(H * ||∇f(x_k)||)

    import numpy as np

    """
    Implements Algorithm 2.1(Adaptive Newton / AdaN).
    Inputs:
        x0  - initial point (numpy array)
        func - function f(x)
        grad - gradient ∇f(x)
        hess - Hessian ∇²f(x)
        H0   - initial regularization parameter
        max_outer_iters - number of outer Newton iterations
        line_search_factor - multiplier for increasing H
        tol  - stopping tolerance for gradient norm
    Returns:
        history of iterates [x0, x1, ..., xk]
    """
    def AdaN(self, weights_0, H0,X,y,batch_size):
        global AdaN_k, AdaN_Hk
        weights_k = weights_0.copy()
        k=0
        d = len(weights_k)
        while True: #"for k = 0,1,... do" <=> line 2
            if k == 0:
                Hk = H0
            else:
                Hk = Hk/4

            grad_norm_k = np.linalg.norm(self.loss_function.grad(weights_k,X,y))
            hessian_k = grad_norm_k = np.linalg.norm(self.loss_function.hessian(weights_k,X,y))
            while True: #"repeat" <=> line 4
                Hk = 2*Hk #line 5
                lambda_k = np.sqrt(Hk*grad_norm_k) #line 7 (line 6 is pointless)

                #define weights_plus <=> line 8--------
                reg_hessian_k = hessian_k + lambda_k*np.eye(d)
                p = np.linalg.solve(reg_hessian_k,grad_norm_k)
                weights_plus = weights_k - p
                #--------------------------------------

                r_plus = np.linalg.norm(weights_plus - weights_k)# <=> line 9


                # line 10 
                grad_norm_plus = np.linalg.norm(self.loss_function.grad(weights_plus,X,y))
                loss_weights_plus = self.loss_function.loss(weights_plus,X,y)
                loss_weights_k = self.loss_function.loss(weights_k,X,y)
                if ((grad_norm_plus <= 2* lambda_k * r_plus) and loss_weights_plus <= loss_weights_k -2/3 *lambda_k*r_plus**2 ):
                    weights_k = weights_plus
                    k += 1
                    break
        return weights_k




    def mishchenko(self, x0, H_param=None, max_iter=100, tol=1e-6):
        """
        Mishchenko2023
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
            grad = self.grad(x)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < tol:
                break

            lambda_k = np.sqrt(H_param * grad_norm)
            Hk = self.hess(x)
            reg_Hk = Hk + lambda_k * np.eye(len(x))

            try:
                p = np.linalg.solve(reg_Hk, grad)
            except np.linalg.LinAlgError:
                p = np.linalg.lstsq(reg_Hk, grad, rcond=None)[0]

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
            alpha = (-1 + np.sqrt(1 + 2 * L_est * g_norm)) / (L_est * g_norm)
            s = np.linalg.solve(H, g)
            x_new = x - alpha * s
            xs.append(x_new.copy())
            x = x_new
        return xs
