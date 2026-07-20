"""
reference_solution.py — high-resolution ground-truth solver for the
viscous Burgers' equation, used ONLY for evaluation (never for training
the PINN). This keeps the comparison honest: the PINN never sees this
data.

Method: method-of-lines. The PDE is discretized in space with 2nd-order
central differences, and the resulting stiff ODE system is integrated in
time with SciPy's implicit BDF integrator (stable for the diffusion
term even though nu is small, without requiring a restrictively tiny
explicit timestep).

    u_t = -u * u_x + nu * u_xx
    u(-1, t) = u(1, t) = 0          (Dirichlet BC, enforced by only
                                      integrating interior nodes)
    u(x, 0)  = -sin(pi * x)          (IC)

The solution is cached to disk (.npz) so it is computed once and reused
by every future run / by Model B for a fair, identical comparison.
"""
import os
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags


def _burgers_rhs(t, u_int, nu, dx, nx):
    """Right-hand side of du/dt for the interior nodes."""
    u = np.empty(nx)
    u[0] = 0.0
    u[-1] = 0.0
    u[1:-1] = u_int

    ux = (u[2:] - u[:-2]) / (2.0 * dx)
    uxx = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx ** 2)

    return -u[1:-1] * ux + nu * uxx


def _jac_sparsity(nx):
    """Tridiagonal sparsity pattern -> lets BDF/Radau use sparse linear
    algebra internally instead of a dense (nx-2)^2 Jacobian."""
    n = nx - 2
    return diags([1.0, 1.0, 1.0], [-1, 0, 1], shape=(n, n))


def solve_burgers_reference(
    nu: float,
    x_min: float, x_max: float,
    t_min: float, t_max: float,
    nx: int, nt: int,
    method: str = "BDF",
):
    """
    Returns
    -------
    x : (nx,) ndarray
    t : (nt,) ndarray
    U : (nt, nx) ndarray   U[i, j] = u(x[j], t[i])
    """
    x = np.linspace(x_min, x_max, nx)
    dx = x[1] - x[0]
    t_eval = np.linspace(t_min, t_max, nt)

    u0 = -np.sin(math.pi * x)
    u0_int = u0[1:-1]

    sol = solve_ivp(
        fun=_burgers_rhs,
        t_span=(t_min, t_max),
        y0=u0_int,
        method=method,
        t_eval=t_eval,
        args=(nu, dx, nx),
        jac_sparsity=_jac_sparsity(nx),
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"Reference solver failed: {sol.message}")

    U = np.zeros((nt, nx))
    U[:, 0] = 0.0
    U[:, -1] = 0.0
    U[:, 1:-1] = sol.y.T  # sol.y shape (n_interior, nt) -> (nt, n_interior)

    return x, t_eval, U


def get_reference_solution(cfg, logger=None):
    """Load cached reference solution if available, otherwise compute and
    cache it. `cfg` is the Config class/object."""
    cache_path = cfg.REFERENCE_CACHE_PATH
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        if logger:
            logger.info(f"Loaded cached reference solution from {cache_path}")
        return data["x"], data["t"], data["U"]

    if logger:
        logger.info("No cached reference solution found — solving Burgers' "
                     f"equation numerically (nx={cfg.REF_NX}, nt={cfg.REF_NT})...")

    x, t, U = solve_burgers_reference(
        nu=cfg.NU,
        x_min=cfg.X_MIN, x_max=cfg.X_MAX,
        t_min=cfg.T_MIN, t_max=cfg.T_MAX,
        nx=cfg.REF_NX, nt=cfg.REF_NT,
        method=cfg.REF_SOLVER_METHOD,
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, x=x, t=t, U=U)
    if logger:
        logger.info(f"Reference solution cached at {cache_path}")

    return x, t, U
