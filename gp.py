import numpy as np
import torch
import torch.nn as nn


class GP(nn.Module):
    def __init__(self, length_scale=1.0, noise_scale=1.0, amplitude_scale=1.0):
        super().__init__()
        self.length_scale_ = nn.Parameter(torch.tensor(np.log(length_scale)))
        self.noise_scale_ = nn.Parameter(torch.tensor(np.log(noise_scale)))
        self.amplitude_scale_ = nn.Parameter(torch.tensor(np.log(amplitude_scale)))

    @property
    def length_scale(self):
        return torch.exp(self.length_scale_)

    @property
    def noise_scale(self):
        return torch.exp(self.noise_scale_)

    @property
    def amplitude_scale(self):
        return torch.exp(self.amplitude_scale_)

    def forward(self, x):
        """compute prediction. fit() must have been called.
        x: test input data point. N x D tensor for the data dimensionality D."""
        y = self.y
        L = self.L
        alpha = self.alpha
        k = self.kernel_mat(self.X, x)
        v = torch.linalg.solve(L, k)
        mu = k.T.mm(alpha)
        var = self.amplitude_scale + self.noise_scale - torch.diag(v.T.mm(v))
        return mu, var

    def fit(self, X, y):
        """should be called before forward() call.
        X: training input data point. N x D tensor for the data dimensionality D.
        y: training target data point. N x 1 tensor."""
        D = X.shape[1]
        K = self.kernel_mat_self(X)
        L = torch.linalg.cholesky(K)
        alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, y))
        marginal_likelihood = (
            -0.5 * y.T.mm(alpha) - torch.log(torch.diag(L)).sum() - D * 0.5 * np.log(2 * np.pi)
        )
        self.X = X
        self.y = y
        self.L = L
        self.alpha = alpha
        self.K = K
        return marginal_likelihood

    def kernel_mat_self(self, X):
        sq = (X**2).sum(dim=1, keepdim=True)
        sqdist = sq + sq.T - 2 * X.mm(X.T)
        return self.amplitude_scale * torch.exp(
            -0.5 * sqdist / self.length_scale
        ) + self.noise_scale * torch.eye(len(X))

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
            "noise": self.noise_scale.detach().cpu(),
            "amplitude": self.amplitude_scale.detach().cpu(),
        }
