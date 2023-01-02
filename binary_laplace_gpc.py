from __future__ import annotations

from abc import ABCMeta, abstractmethod
import math
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn


class Likelihood(metaclass=ABCMeta):
    registry: Dict[str, Likelihood] = {}

    def __init__(self):
        pass

    def __init_subclass__(cls):
        assert cls.__name__ not in cls.registry, f"Likelihood name {cls.__name__} already exists."
        cls.registry[cls.__name__] = cls

    @abstractmethod
    def get_log_likelihood(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Get log-likelihood for given target 'y' and latent function 'f'.

        Args:
            y (torch.Tensor, (N,1)): classification targets on the points.
            f (torch.Tensor, (N,1)): the value of sampled latent function on the points.

        Returns:
            torch.Tensor: likelihood (scalar value)
        """
        pass

    @abstractmethod
    def get_jacobian_of_log_likelihood(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Get Jacobian of log-likelihood for given target 'y' and latent function 'f'.

        Args:
            y (torch.Tensor, (N,1)): classification targets on the points.
            f (torch.Tensor, (N,1)): the value of sampled latent function on the points.

        Returns:
            torch.Tensor, (N,1): Jacobian
        """
        pass

    @abstractmethod
    def get_hessian_of_log_likelihood(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Get Hessian of log-likelihood for given target 'y' and latent function 'f'.

        Args:
            y (torch.Tensor, (N,1)): classification targets on the points.
            f (torch.Tensor, (N,1)): the value of sampled latent function on the points.

        Returns:
            torch.Tensor, (N,N): Hessian
        """
        pass


class Logistic(Likelihood):
    def get_log_likelihood(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return (-torch.log(1 + torch.exp(-y * f))).sum()

    def get_jacobian_of_log_likelihood(self, y, f):
        t = (y + 1) / 2
        pi = 1 / (1 + torch.exp(-f))
        return t - pi

    def get_hessian_of_log_likelihood(self, y, f):
        pi = 1 / (1 + torch.exp(-f))
        return torch.diag((-pi * (1 - pi)).squeeze(-1))


class CumulativeGaussian(Likelihood):
    def get_log_likelihood(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return torch.log(self.get_cdf(y * f))

    def get_jacobian_of_log_likelihood(self, y, f):
        return y * self.get_prob(f) / self.get_cdf(y * f)

    def get_hessian_of_log_likelihood(self, y, f):
        prob = self.get_prob(f)
        cdf = self.get_cdf(y * f)
        return torch.diag((-(prob**2) / (cdf**2) - (y * f * prob) / cdf).squeeze(-1))

    @staticmethod
    def get_prob(z):
        return torch.exp(-((z**2) / 2 - math.log(math.sqrt(2 * math.pi))))

    @staticmethod
    def get_cdf(z):
        return 0.5 * (
            1 + torch.erf(z / math.sqrt(2))
        )  # See https://github.com/pytorch/pytorch/blob/master/torch/distributions/normal.py (cdf)


class BinaryLaplaceGPC(nn.Module):
    def __init__(
        self,
        length_scale: float = 1.0,
        amplitude_scale: float = 1.0,
        likelihood_func: str = "Logistic",
        eps: float = 0.001,
        n_samples: int = 10,
    ):
        super().__init__()
        self.length_scale_ = nn.Parameter(torch.tensor(np.log(length_scale)))
        self.amplitude_scale_ = nn.Parameter(torch.tensor(np.log(amplitude_scale)))

        self.likelihood_func = Likelihood.registry[likelihood_func]()
        self.eps = eps
        self.n_samples = n_samples

    @property
    def length_scale(self):
        return torch.exp(self.length_scale_)

    @property
    def amplitude_scale(self):
        return torch.exp(self.amplitude_scale_)

    def forward(self, x):
        """compute prediction. fit() must have been called.
        x: test input data point. N x D tensor for the data dimensionality D."""
        L = self.L
        sqrt_W = self.sqrt_W
        k = self.kernel_mat(self.X, x)
        v = torch.linalg.solve(L, sqrt_W.mm(k))
        mu = k.T.mm(self.likelihood_func.get_jacobian_of_log_likelihood(self.y, self.f))  # (N',1)
        var = self.amplitude_scale - torch.diag(v.T.mm(v))  # (N')

        z = mu.repeat(1, self.n_samples) + torch.sqrt(var).unsqueeze(-1) * torch.randn_like(
            mu.repeat(1, self.n_samples)
        )  # (N',{self.n_samples})

        pi = torch.sigmoid(z).mean(-1)
        return mu, var, pi

    def fit(self, X, y):
        """should be called before forward() call.
        X: training input data point. N x D tensor for the data dimensionality D.
        y: training target data point. N x 1 tensor."""
        f = torch.zeros_like(y).float()
        N = X.shape[0]
        K = self.kernel_mat(X, X)
        while True:
            f = f.detach()
            W = -self.likelihood_func.get_hessian_of_log_likelihood(y, f)
            sqrt_W = W.sqrt()
            L = torch.linalg.cholesky(torch.eye(N, device=y.device) + sqrt_W.mm(K.mm(sqrt_W)))
            b = W.mm(f) + self.likelihood_func.get_jacobian_of_log_likelihood(y, f)
            a = b - sqrt_W.mm(torch.linalg.solve(L.T, torch.linalg.solve(L, sqrt_W.mm(K.mm(b)))))
            diff = (torch.abs(K.mm(a) - f)).max()
            f = K.mm(a)
            if diff < self.eps:
                break

        approx_marginal_likelihood = (
            -0.5 * a.T.mm(f)
            - torch.log(torch.diag(L)).sum()
            + self.likelihood_func.get_log_likelihood(y, f)
        )
        self.X = X
        self.y = y
        self.sqrt_W = sqrt_W
        self.L = L
        self.K = K
        self.f = f
        return approx_marginal_likelihood

    def kernel_mat(self, X, Z):
        Xsq = (X**2).sum(dim=1, keepdim=True)
        Zsq = (Z**2).sum(dim=1, keepdim=True)
        sqdist = Xsq + Zsq.T - 2 * X.mm(Z.T)
        return self.amplitude_scale * torch.exp(-0.5 * sqdist / self.length_scale)

    def train_step(self, X, y, opt):
        """gradient-based optimization of hyperparameters
        opt: torch.optim.Optimizer object."""
        opt.zero_grad()
        nll = -self.fit(X, y).sum()
        nll.backward()
        opt.step()
        return {
            "loss": nll.item(),
            "length": self.length_scale.detach().cpu(),
            "amplitude": self.amplitude_scale.detach().cpu(),
        }
