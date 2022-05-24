import numpy as np
from sklearn.base import BaseEstimator
from .utils.utils import fnorm, l2norm, nuclear_prox, l1shrink

class rpca(BaseEstimator):
    """
        Parameters
        ----------
        lambda_:
            penalty for the sparsity error
        mu_:
            initial lagrangian penalty
        max_iter:
            maximum number of iterations
        rho_:
            learning rate
        tau_:
            mu update criterion parameter
        REL_TOL:
            relative tolerence
        ABS_TOL:
            absolute tolerance
    """
    def __init__(self, lambda_=None, mu=None, rho=2, tau=10, rel_tol=1e-3, abs_tol=1e-4, max_iter=100, verbose=False):
        self.STATS = None
        self.lambda_ = lambda_
        self.mu_ = mu
        self.tau_ = tau
        self.rho_ = rho
        self.REL_TOL = rel_tol
        self.ABS_TOL = abs_tol
        self.max_iter = max_iter
        self.verbose = verbose

    def __cost(self, L, S):
        nuclear_norm = np.linalg.svd(L, full_matrices=False, compute_uv=False).sum()
        l1_norm = np.linalg.norm(S, 1)

        return nuclear_norm + self.lambda_ * l1_norm, nuclear_norm, l1_norm


    def _calculate_residuals(self, x, S, L, S_old):
        primal_residual = fnorm(x - S - L)
        dual_residual = self.mu_ * fnorm(S - S_old)

        return primal_residual, dual_residual

    def _update_tols(self, x, S, L, Y):
        primal_tol = self.REL_TOL * max(l2norm(x), l2norm(S), l2norm(L))
        dual_tol = self.REL_TOL * l2norm(Y)
        return primal_tol, dual_tol

    def _J(self, x, mu):
        return max(np.linalg.norm(x), np.max(np.abs(x)) / mu)

    def fit(self, x):
        assert x.ndim == 2

        x = x.astype(np.float)
        xmin = np.min(x)
        rescale = max(1e-8, np.max(x - xmin))

        xt = (x - xmin) / rescale

        if self.lambda_ is None:
            self.lambda_ = 1 / np.sqrt(max(xt.shape))
        if self.mu_ is None:
            self.mu_ = xt.size / (4.0 * np.linalg.norm(xt, 1))

        Y = xt / self._J(xt, self.mu_)
        S = np.zeros_like(xt)
        L = np.zeros_like(xt)

        self.STATS = {
            'err_primal': [],
            'err_dual': [],
            'eps_primal': [],
            'eps_dual': [],
            'mu': []
        }

        if self.verbose:
            print(f"Lambda: {self.lambda_}; mu: {self.mu_}")

        for i in range(self.max_iter):
            if self.verbose:
                print(f"Iteration: {i}; Current mu: {self.mu_}")
            # argmin_L ||X - (L + S) + Y/mu||_F^2 + (lmb/mu)*||L||_*
            L = nuclear_prox(xt - S + Y / self.mu_, 1 / self.mu_)

            # argmin_S ||X - (L + S) + Y/mu||_F^2 + (lmb/mu)*||S||_1
            S_old = S.copy()
            S = l1shrink(x=xt - L + Y / self.mu_, eps=self.lambda_ / self.mu_)

            # Update Y
            Y += (xt - S - L)*self.mu_

            primal, dual = self._calculate_residuals(xt, S, L, S_old)
            primal_tol, dual_tol = self._update_tols(xt, S, L, Y)

            eps_primal = np.sqrt(xt.size) * self.ABS_TOL + primal_tol
            eps_dual = np.sqrt(xt.size) * self.ABS_TOL + dual_tol

            if self.verbose:
                print(f"Primal: {primal}, Dual: {dual}")
                print(f"Primal tol: {primal_tol}, Dual tol: {dual_tol}")
                print(f"Eps Primal: {eps_primal}, Eps Dual: {eps_dual}")

            # add the stats
            self.STATS['err_primal'].append(primal_tol)
            self.STATS['err_dual'].append(dual_tol)
            self.STATS['eps_primal'].append(eps_primal)
            self.STATS['eps_dual'].append(eps_dual)
            self.STATS['mu'].append(self.mu_)

            # check for stopping criterion

            if primal < eps_primal and dual < eps_dual:
                break

            #if primal <= primal_tol and dual <= dual_tol:
            #    break

            # update mu_
            if primal > self.tau_ * dual:
                self.mu_ *= self.rho_
            elif dual > self.tau_ * primal:
                self.mu_ /= self.rho_

        # Scale back up to the original data scale
        if self.verbose:
            if i < self.max_iter - 1:
                print('Converged in %d steps' % i)
            else:
                print('Reached maximum iterations')

        S = (S + xmin) * rescale
        L = x - S
        self.L = L
        self.S = S

        self.STATS['cost'] = self.__cost(L, S)

        return L, S