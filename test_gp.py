import numpy as np
import torch
from torch.optim import SGD
from gp import GP


def test_gp():
    X = torch.randn(10, 1)
    f = torch.sin(X * 2 * np.pi /4).flatten()
    y = f + torch.randn_like(f) * 0.1
    y = y[:,None]
    grid = torch.linspace(-5, 5, 20)[:,None]

    gp = GP()
    gp.fit(X, y)
    mu, var = gp.forward(grid)
    mu = mu.detach().numpy().flatten()
    std = torch.sqrt(var).detach().numpy().flatten()


def test_gp_opt():
    X = torch.randn(10, 1)
    f = torch.sin(X * 2 * np.pi /4).flatten()
    y = f + torch.randn_like(f) * 0.1
    y = y[:,None]
    grid = torch.linspace(-5, 5, 20)[:,None]

    gp = GP()
    opt = SGD(gp.parameters(), lr=0.01)
    for i in range(2):
        d_train = gp.train_step(X, y, opt)
