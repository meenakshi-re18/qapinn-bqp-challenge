"""
pinn_losses.py — physics-informed loss terms for the Burgers' equation:

    PDE residual:  f(x,t) = u_t + u*u_x - nu*u_xx      (should be ~0)
    IC loss:       u(x,0) - (-sin(pi*x))                (should be ~0)
    BC loss:       u(-1,t), u(1,t)                       (should be ~0)

Derivatives are obtained via automatic differentiation (torch.autograd),
NOT finite differences — this is what makes the network's *residual*
trainable everywhere in the domain, not just at grid points.
"""
import torch


def pde_residual(model, x: torch.Tensor, t: torch.Tensor, nu: float) -> torch.Tensor:
    """Computes f = u_t + u*u_x - nu*u_xx at the given collocation points."""
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)

    u = model(x, t)

    grad_outputs = torch.ones_like(u)
    u_x = torch.autograd.grad(u, x, grad_outputs=grad_outputs,
                               create_graph=True, retain_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=grad_outputs,
                               create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True, retain_graph=True)[0]

    f = u_t + u * u_x - nu * u_xx
    return f


def compute_losses(model, batch, nu, weights):
    """
    batch: dict with keys x_f, t_f, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc
    Returns dict of individual losses + total.
    """
    # --- PDE residual loss ---
    f = pde_residual(model, batch["x_f"], batch["t_f"], nu)
    loss_pde = torch.mean(f ** 2)

    # --- Initial condition loss ---
    u_ic_pred = model(batch["x_ic"], batch["t_ic"])
    loss_ic = torch.mean((u_ic_pred - batch["u_ic"]) ** 2)

    # --- Boundary condition loss ---
    u_bc_pred = model(batch["x_bc"], batch["t_bc"])
    loss_bc = torch.mean((u_bc_pred - batch["u_bc"]) ** 2)

    total = (weights["pde"] * loss_pde
             + weights["ic"] * loss_ic
             + weights["bc"] * loss_bc)

    return {
        "total": total,
        "pde": loss_pde.detach(),
        "ic": loss_ic.detach(),
        "bc": loss_bc.detach(),
    }
