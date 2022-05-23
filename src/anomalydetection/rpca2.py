import numpy as np
from sklearn.base import BaseEstimator
from utils.utils import fnorm, l2norm, nuclear_prox, l1shrink

class rpca2(BaseEstimator):
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
    def __cost(self, L, S):
        nuclear_norm = np.linalg.svd(L, full_matrices=False, compute_uv=False).sum()

        l1_norm = np.abs(S).sum()

        return nuclear_norm + self.lambda_ * l1_norm, nuclear_norm, l1_norm

    def __init__(self, lambda_=None, mu=None, abs_tol=1e-4, rel_tol=1e-3, rho=2,tau=10, max_iter=100, verbose=False):
        self.lambda_ = lambda_
        self.mu_ = mu
        self.tau_ = tau
        self.rho_ = rho
        self.REL_TOL = rel_tol
        self.ABS_TOL = abs_tol
        self.max_iter = max_iter
        self.verbose = verbose



    def _calculate_residuals(self, x, S, L, S_old):
        primal_residual = fnorm(x - S - L)
        dual_residual = self.mu_ * fnorm(S - S_old)

        return primal_residual, dual_residual

    def _update_tols(self, x, S, L, Y):
        tol_primal = self.REL_TOL * max(l2norm(x), l2norm(S), l2norm(L))
        tol_dual = self.REL_TOL * l2norm(Y)
        return tol_primal, tol_dual

    def fit(self, x):
        self.fit_transform(x)
        return self

    def fit_transform(self, x):
        MIN_MU = 1e0
        MAX_MU = 1e5

        # Scale the data to a workable range
        x = x.astype(np.float)
        xmin = np.min(x)
        rescale = max(1e-8, np.max(x - xmin))

        xt = (x - xmin) / rescale

        L = xt.copy()
        S = np.zeros_like(xt)
        Y = np.zeros_like(xt)

        if self.lambda_ is None:
            self.lambda_ = max(xt.shape) ** (-0.5)
        if self.mu_ is None:
            self.mu_ = xt.size / (4.0 * np.linalg.norm(xt, 1))
        else:
            if self.mu_ > MAX_MU or self.mu_ < MAX_MU:
                print(f"{self.mu_} is not in the accepted range ({MIN_MU}-{MAX_MU})")
                self.mu_ = MIN_MU

        self.STATS = {
            'err_primal': [],
            'err_dual': [],
            'eps_primal': [],
            'eps_dual': [],
            'mu': []
        }

        for i in range(self.max_iter):
            if self.verbose:
                print(f"Iteration: {i}")
                print(f"Lambda: {self.lambda_}; mu: {self.mu_}")

            L = nuclear_prox(xt - S - Y, 1.0 / self.mu_)
            S_old = S.copy()
            S = l1shrink(x=xt - L - Y, eps=self.lambda_ / self.mu_)

            primal, dual = self._calculate_residuals(xt, S, L, S_old)
            primal_tol, dual_tol = self._update_tols(xt, S, L, Y)

            #  Y+ <- fnorm(x - L - S)
            Y = Y + primal

            eps_primal = np.sqrt(xt.size) * self.ABS_TOL + primal_tol
            eps_dual = np.sqrt(xt.size) * self.ABS_TOL + dual_tol

            if self.verbose:
                print(f"Primal: {primal_tol}, Dual: {dual_tol}")
                print(f"Eps Primal: {eps_primal}, Eps Dual: {eps_dual}")

            self.STATS['eps_primal'].append(eps_primal)
            self.STATS['eps_dual'].append(eps_dual)
            self.STATS['err_primal'].append(primal_tol)
            self.STATS['err_dual'].append(dual_tol)
            self.STATS['mu'].append(self.mu_)

            if primal <= eps_primal and dual <= eps_dual:
                break

            # update mu and Y
            if primal_tol > self.tau_ * dual_tol:
                self.mu_ = self.mu_ * self.rho_
                Y = Y / self.rho_

            elif dual_tol > self.tau_ * primal_tol:
                self.mu_ = self.mu_ / self.rho_
                Y = Y * self.rho_

        if self.verbose:
            if i < self.max_iter - 1:
                print('Converged in %d steps' % i)
            else:
                print('Reached maximum iterations')

        # Scale back up to the original data scale
        S = (S + xmin) * rescale
        self.L = x - S
        self.S = S

        self.STATS['cost'] = self.__cost(L, S)

        return L, S