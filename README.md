# pytorch-minimal-gaussian-process

In search of truth, simplicity is needed. There exist heavy-weighted libraries, but as you know, we need to go bare bone sometimes.
Here is a minimal implementation of Gaussian process regression in PyTorch.

The implementation generally follows Algorithm 2.1 in [Gaussian Process for Machine Learning (Rassmussen and Williams, 2006)](http://www.gaussianprocess.org/gpml/).


* Author: [Sangwoong Yoon](https://swyoon.github.io/), [Hyeokjun Kwon](https://www.linkedin.com/in/hyeokjun-kwon-24b992210)

## Features

* Gaussian process regression with squared exponential kernel.
* Hyperparameter optimization via marginal likelihood maximization using Pytorch built-in autograd functionality. (See `demo.ipynb`)
* Unittesting using Pytest.

## Updates

* 2022-01-01: Bugfix in predictive variance computation
* 2023-01-03: Implement binary Laplace Gaussian process regression

## Dependency

* Numpy
* PyTorch
* PyTest
* Matplotlib (for demo)

## How to Use
### Gaussian process regression
```python
from gp import GP

# generate data
X = torch.randn(100,1)
y = torch.sin(X * 2 * np.pi /4). + torch.randn(100, 1) * 0.1
grid = torch.linspace(-5, 5, 200)[:,None]

# run GP
gp = GP()  # you may specify initial hyperparameters using keyword arguments
gp.fit(X, y)
mu, var = gp.forward(grid)
```

### Gaussian process classification
```python
from gp import GP

# generate data
X = torch.randn(100,1)
f = torch.sin(X * 3 * np.pi / 4)
y = (f > 0.).int() * 2 - 1
grid = torch.linspace(-5, 5, 200)[:,None]

# run GP
gp = BinaryLaplaceGPC()  # you may specify initial hyperparameters using keyword arguments
gp.fit(X, y)
mu, var, pi = gp.forward(grid)
```
## Unittesting

```
$ pytest
```

## See also

* [GPyTorch](https://gpytorch.ai/): A full-featured Gaussian process package based on PyTorch.
