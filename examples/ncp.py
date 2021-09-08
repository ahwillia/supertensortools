"""
Nonnegative CP tensor decomposition.
"""
import torch
import matplotlib.pyplot as plt

# Create tensor from nonnegative factors.
true_factors = [
    torch.rand(3, 50),
    torch.rand(3, 51),
    torch.rand(3, 52),
]
tensor = torch.einsum("ri,rj,rk->ijk", *true_factors)
tensor_sqnorm = torch.vdot(tensor.view(-1), tensor.view(-1))

# Create list of parameters.
init_params = [
    torch.rand(3, 50),
    torch.rand(3, 51),
    torch.rand(3, 52)
]
_s = (
    tensor_sqnorm / torch.einsum("ri,rj,rk,ri,rj,rk->", *(init_params * 2))
) ** (1 / 3)
params = [torch.nn.Parameter(p * _s) for p in init_params]
proxs = [lambda x: x.relu_()] # enforce all parameters to be nonnegative.

# Create function that computes the loss.
def closure():
    # Enforce nonnegativity
    for p in params:
        if (p < 0).any():
            return torch.tensor(float("Inf"))
    # Compute loss
    est = torch.einsum("ri,rj,rk->ijk", *params)
    resids = est - tensor
    return torch.vdot(resids.view(-1), resids.view(-1)) / tensor_sqnorm

# Define proximal operators.
optimizer = AcceleratedProxGrad(
    params, proxs,
    lr=1000.0, init_momentum=1.0,
    momentum_backtrack_factor=0.9,
    lr_backtrack_factor=0.5,
)

loss_hist = []
lr_hist = []
beta_hist = []
for i in range(100):
    lr_hist.append(optimizer.prxgrad.param_groups[0]["lr"])
    beta_hist.append(optimizer.param_groups[0]["beta"])
    loss_hist.append(optimizer.step(closure).item())
print(max(lr_hist))
print(loss_hist[-1])

fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(loss_hist[1:], color="k", label="objective")
axes[0].set_yscale("log")
axes[1].plot(beta_hist[1:], color="k", label="momentum")
axes[2].plot(lr_hist[1:], color="k", label="step size")
fig.subplots_adjust(hspace=0)

# Define proximal operators.
params = [torch.nn.Parameter(p * _s) for p in init_params]
optimizer = AcceleratedProxGrad(
    params, proxs,
    lr=1000.0, init_momentum=0.0,
    momentum_backtrack_factor=1.0,
    lr_backtrack_factor=0.5,
)

loss_hist = []
lr_hist = []
beta_hist = []
for i in range(100):
    lr_hist.append(optimizer.prxgrad.param_groups[0]["lr"])
    beta_hist.append(optimizer.param_groups[0]["beta"])
    loss_hist.append(optimizer.step(closure).item())
print(max(lr_hist))
print(loss_hist[-1])

axes[0].plot(loss_hist[1:], color="r", label="objective")
axes[0].set_yscale("log")
axes[1].plot(beta_hist[1:], color="r", label="momentum")
axes[2].plot(lr_hist[1:], color="r", label="step size")

plt.show()
