"""
This file implements some helper functions, including:
    (1) get_properties
        Get the components' properties.

    (2) cardano_cubic_solver
        Solve the cubic equation of state based on the Cardano formula

    (3) halley_cubic_solver
        Solve the cubic equation of state based on the Halley's method

    (4) rr_solver
        Solve the Rachford-Rice equation using the method proposed in the 
        article "A new look at the Rachford-Rice equation"

    (5) hebden_solver
        Solve the linear system: (H + alpha * I) * delta = -grads using the
        method proposed in the article "AN ALGORITHM FOR MINIMIZATION USING
        EXACT SECOND DERIVATIVES"
"""

import torch
from torch import Tensor
from itertools import combinations

from ptflash.bank import ELEMENTS, INTERACTIONS


def get_properties(components, dtype, device):
    """
    Get the components' properties defined in [bank](bank.py), including:
        critical pressure (pc), critical temperature (pt), acentric factor (omega),
        binary interaction parameters (kij, kijt, kijt2).

    Parameters
    ----------
    components : list
        CAS number of components

    dtype: torch.dtype

    device: torch.device
    """

    n = len(components)
    pcs = torch.as_tensor(
        [ELEMENTS[i]["CriticalPressure"] for i in components], dtype=dtype, device=device
    )
    tcs = torch.as_tensor(
        [ELEMENTS[i]["CriticalTemperature"] for i in components],
        dtype=dtype,
        device=device,
    )
    omegas = torch.as_tensor(
        [ELEMENTS[i]["AcentricFactor"] for i in components], dtype=dtype, device=device
    )
    pairs = list(combinations(components, 2))
    kij = torch.zeros(n, n, dtype=dtype, device=device)
    kijt = torch.zeros_like(kij)
    kijt2 = torch.zeros_like(kij)

    associated_pairs = list(INTERACTIONS.keys())
    bips = list(INTERACTIONS.values())
    for pair in pairs:
        if pair in associated_pairs:
            idx = associated_pairs.index(pair)
            kijs = bips[idx]
            row = components.index(pair[0])
            col = components.index(pair[1])
            kij[[row, col], [col, row]] += kijs["Kij"]
            kijt[[row, col], [col, row]] += kijs["Kijt"]
            kijt2[[row, col], [col, row]] += kijs["Kijt2"]

    return pcs, tcs, omegas, kij, kijt, kijt2


def cardano_cubic_solver(coefs: Tensor):
    """
    Solve the cubic equation of state based on the Cardano formula.
    For more details about the Cardano formula, refer to:
        https://en.wikipedia.org/wiki/Cubic_equation

    Parameters
    ----------
    coefs : Tensor
        Coefficients of the cubic equation.
    """

    c1 = coefs[:, 0]
    c2 = coefs[:, 1]
    c3 = coefs[:, 2]
    c4 = coefs[:, 3]
    delta0 = c2**2 - 3 * c1 * c3
    delta1 = 2 * c2**3 - 9 * c1 * c2 * c3 + 27 * c1**2 * c4
    delta = 4 * delta0**3 - delta1**2
    # delta < 0, one real root. Otherwise, three real roots.
    if_1_real = delta < 0
    # Handle the cases of one real root
    k1 = (delta1[if_1_real] - torch.sqrt(-delta[if_1_real])) / 2
    k2 = (delta1[if_1_real] + torch.sqrt(-delta[if_1_real])) / 2
    k3 = k1.sign() * k1.abs() ** (1 / 3) + k2.sign() * k2.abs() ** (1 / 3)
    one_real_root = -(k3 + c2[if_1_real]) / (3 * c1[if_1_real])
    # Handle the cases of three real roots
    # Set all variables complex so that torch.sqrt also works for negative values
    c1 = c1[~if_1_real] + 0j
    c2 = c2[~if_1_real] + 0j
    c3 = c3[~if_1_real] + 0j
    c4 = c4[~if_1_real] + 0j
    delta0 = delta0[~if_1_real] + 0j
    delta1 = delta1[~if_1_real] + 0j
    delta = delta[~if_1_real] + 0j
    # C1 and C2 use "-" and "+" respectively. If C1 is zero, C2 will be chosen so
    # that C is non-zero always.
    C1 = (delta1 - torch.sqrt(-delta)) / 2
    C2 = (delta1 + torch.sqrt(-delta)) / 2
    C = torch.where(C1 != 0, C1, C2) ** (1 / 3)
    # Define two constants 1 and 3 with the dtype and device as coefs
    const_1 = torch.tensor(1.0, dtype=coefs.dtype, device=coefs.device)
    const_3 = torch.tensor(3.0, dtype=coefs.dtype, device=coefs.device)
    xi = torch.complex(-const_1, const_3.sqrt()) / 2
    three_real_roots = torch.stack(
        [-1 / (3 * c1) * (c2 + xi**k * C + delta0 / (xi**k * C)) for k in range(3)],
        dim=1,
    )

    return if_1_real, one_real_root.unsqueeze(-1), three_real_roots.real


def update_z(z: Tensor, x0: Tensor, x1: Tensor, x2: Tensor):
    """Subfunction of `halley_cubic_solver` to calculate the updates on z."""

    grad2 = 3.0 * z - x2
    grad = z * (grad2 - x2) + x1
    f = z**3 - x2 * z**2 + x1 * z - x0
    dz = f / grad
    dz.mul_(1.0 + dz * grad2 / grad)
    return dz


def halley_cubic_solver(
    ps: Tensor,
    ts: Tensor,
    b: Tensor,
    a_alpha: Tensor,
    c1: float,
    c2: float,
    max_nit: int = 100,
    tol: float = 1.0e-10,
):
    """
    Cubic EoS solver based on Halley's method.
        (1) Start with a single liquid-like guess which is solved precisely
        (2) Deflate the cubic analytically
        (3) Solve the quadratic equation for the next two volumes,
        (4) Perform one halley step on each of them to obtain the final solutions.
    Note that this method does not calculate imaginary roots, which are set to zero on detection.

    The cubic EoS takes on the form:
        P = RT / (V - b) - a_alpha / ((V + c1 * b) * (V + c2 * b))
          = RT / (V - b) - a_alpha / (V ** 2 + delta * V + epsilon)
        where delta = (c1 + c2) * b and epsilon = c1 * c2 * b ** 2

    The cubic EoS can also be described as a function of the compressibility coefficient:
        Z ** 3 - x2 * Z ** 2 + x1 * z - x0 = 0
        where x2 = 1 + k1 - k2
              x1 = k3 + k4 - k2 * (k1 + 1)
              x0 = k4 * (k1 + 1) + k1 * k3

              k1 = b * P / (R * T)
              k2 = delta * P / (R * T)
              k3 = a_alpha * P / (R * T) ** 2
              k4 = epsilon * (P / (R * T)) ** 2

    Parameters
    ----------
    ps : array_like
        Pressure

    ts : array_like
        Temperature

    b : array_like
        Co-volume parameter.

    a_alpha : array_like
        Energy parameter.

    c1 : float
        EOS-specific coefficient, e.g., c1_SRK = 1 and c1_PR = 1 + sqrt(2)

    c2 : float
        EOS-specific coefficient, e.g., c2_SRK = 0 and c2_PR = 1 - sqrt(2)

    max_nit: int, default=100
            The maximum number of iterations.

    tol: float, default=1.0e-10
        Absolute tolerance for termination.
    """

    R = 8.314472
    delta = (c1 + c2) * b
    epsilon = c1 * c2 * b**2
    pt_ratio = ps / ts
    k1 = b / R * pt_ratio
    k2 = delta / R * pt_ratio
    k3 = a_alpha * ps / (R * ts) ** 2
    k4 = epsilon * (pt_ratio / R) ** 2

    x0 = k4 * (k1 + 1) + k1 * k3
    x1 = k3 + k4 - k2 * (k1 + 1)
    x2 = 1 + k1 - k2

    # The inflection point where the second derivative is zero
    inflect = x2 / 3.0
    f = inflect**3 - x2 * inflect**2 + x1 * inflect - x0
    # z0 = k1 or z0 = k1+1 depending on the value of f at the inflection point
    cond1 = (f < 0.0) | (k1 > inflect)
    z = k1.clone()
    z[cond1] = k1[cond1] + 1

    x0_cp = x0.clone()
    x1_cp = x1.clone()
    x2_cp = x2.clone()
    k1_cp = k1.clone()

    unconverged = torch.arange(ps.shape[0], device=ps.device)
    Z0 = torch.zeros_like(ps)
    for nit in torch.arange(max_nit, device=ps.device):
        dz = update_z(z, x0_cp, x1_cp, x2_cp)
        new_z = z - dz
        cond2 = new_z < k1_cp
        new_z[cond2] = (k1_cp[cond2] + z[cond2]) / 2.0
        errors = (new_z - z).abs().squeeze()
        converged = errors <= tol
        z = new_z

        if converged.all() | (nit == max_nit - 1):
            Z0[unconverged] = z
            break
        else:
            if converged.any():
                converged_indices = unconverged[converged]
                Z0[converged_indices] = z[converged]
                z = z[~converged]
                x0_cp = x0_cp[~converged]
                x1_cp = x1_cp[~converged]
                x2_cp = x2_cp[~converged]
                k1_cp = k1_cp[~converged]
                unconverged = unconverged[~converged]

    # Deflation of the cubic EoS to find the other two roots
    # (Z - Z0) * (Z ** 2 + e1 * Z + e0) = 0
    e1 = Z0 - x2
    e0 = Z0 * e1 + x1
    d = e1**2 - 4 * e0  # discriminant
    # If discriminant < 0, only one real root
    if_1_real = (d < 0.0).reshape(-1)
    if_3_real = ~if_1_real
    one_real_root = Z0[if_1_real]
    # Other two real roots
    Z1 = (-e1[if_3_real] + d[if_3_real].sqrt()) / 2.0
    Z2 = (-e1[if_3_real] - d[if_3_real].sqrt()) / 2.0
    # Perform one halley's update on Z1 and Z2 to obtain the final solution
    x0 = x0[if_3_real]
    x1 = x1[if_3_real]
    x2 = x2[if_3_real]
    dz1 = update_z(Z1, x0, x1, x2)
    dz2 = update_z(Z1, x0, x1, x2)
    Z1.sub_(dz1)
    Z2.sub_(dz2)

    three_real_roots = torch.concat([Z0[if_3_real], Z1, Z2], dim=1)
    return if_1_real, one_real_root, three_real_roots


def rachford_rice(theta_v: Tensor, zi: Tensor, ki: Tensor):
    """Subfunction of `rr_solver` to evaluate the Rachford-Rice equation."""

    v = (ki - 1) * zi / (1 + (ki - 1) * theta_v)
    return v.sum(dim=1, keepdim=True)


def rr_grad(theta_v: Tensor, zi: Tensor, ki: Tensor):
    """
    Subfunction of rr_solver to calculate the gradient of the
    Rachford-Rice equation.
    """
    v = -zi * ((ki - 1) / (1 + (ki - 1) * theta_v)) ** 2
    return v.sum(dim=1, keepdim=True)


def rr_solver(zi: Tensor, ki: Tensor, max_nit: int = 100, tol: float = 1.0e-8):
    """Solve the Rachford-Rice equation based on the reference:
            "A new look at the Rachford-Rice equation"

    Parameters
    ----------
    zi: array_like
        The composition.

    ki: array_like
        The distribution coefficients.

    max_nit: int, default=100
        The maximum number of iterations.

    tol: float, default=1.0e-8
        Tolerance for termination.
    """
    alpha_l = 1 / (1 - ki.max(dim=1, keepdim=True)[0])
    alpha_r = 1 / (1 - ki.min(dim=1, keepdim=True)[0])

    theta_l = (ki * zi - 1) / (ki - 1)
    theta_l[ki <= 1.0] = -float("inf")
    theta_l = theta_l.max(dim=1, keepdim=True)[0]
    theta_l = torch.maximum(alpha_l, theta_l)

    theta_r = (1 - zi) / (1 - ki)
    theta_r[ki >= 1.0] = float("inf")
    theta_r = theta_r.min(dim=1, keepdim=True)[0]
    theta_r = torch.minimum(alpha_r, theta_r)

    rr0 = rachford_rice(torch.zeros_like(theta_r), zi, ki)  # RR equation at x=0
    rr1 = rachford_rice(torch.ones_like(theta_r), zi, ki)  # RR equation at x=1
    theta_l[(rr0 > 0.0) & (theta_l < 0.0)] = 0.0
    theta_l[rr1 > 0.0] = 1.0
    theta_r[(rr1 < 0.0) & (theta_r > 1.0)] = 1.0
    theta_r[rr0 < 0.0] = 0.0

    theta = (theta_l + theta_r) / 2
    unconverged = torch.arange(zi.shape[0], device=zi.device)
    THETA = torch.zeros_like(theta)
    NITS = torch.zeros_like(theta, dtype=torch.int64)
    ERRORS = torch.zeros_like(theta)

    for nit in torch.arange(max_nit, device=zi.device):
        rr = rachford_rice(theta, zi, ki)  # rr is the value of the RR equation
        dummy = (theta - alpha_l) * (alpha_r - theta)
        func = dummy * rr  # func is the value of the helper function in the article
        grad = dummy * rr_grad(theta, zi, ki) - (2 * theta - alpha_l - alpha_r) * rr
        test = ((theta - theta_r) * grad - func) * ((theta - theta_l) * grad - func)
        # further restrict the bounds of theta
        theta_l[func > 0] = theta[func > 0]
        theta_r[func < 0] = theta[func < 0]
        # If test >=0, use bisection algorithm. Otherwise, use newton method.
        # update theta
        new_theta = torch.where(test < 0, theta - func / grad, (theta_l + theta_r) / 2.0)
        dx = new_theta - theta
        theta = new_theta
        # stopping criteria
        converged = ((dx.abs() < tol) & (func.abs() < tol)).view(-1)

        if converged.all() | (nit == max_nit - 1):
            THETA[unconverged] = theta
            NITS[unconverged] = nit + 1
            ERRORS[unconverged] = func.abs()
            unconverged = unconverged[~converged]
            break
        else:
            if converged.any():
                converged_indices = unconverged[converged]
                THETA[converged_indices] = theta[converged]
                NITS[converged_indices] = nit + 1
                ERRORS[converged_indices] = func[converged].abs()

                alpha_l = alpha_l[~converged]
                alpha_r = alpha_r[~converged]
                theta_r = theta_r[~converged]
                theta_l = theta_l[~converged]
                theta = theta[~converged]
                zi = zi[~converged]
                ki = ki[~converged]
                unconverged = unconverged[~converged]

    return THETA, NITS, ERRORS, unconverged


def hebden_solver(
    hess: Tensor, grads: Tensor, d: Tensor, max_nit: int = 20, tol: float = 0.1
):
    """
    Solve the linear system: (H + alpha * I) * delta = -grads
        This function adjusts the value of alpha to:
            (1) handle the cases where H is not positive-definite
            (2) make the 2-norm of delta equal to d to ensure convergence

        Reference:
            AN ALGORITHM FOR MINIMIZATION USING EXACT SECOND DERIVATIVES.

    Parameters
    ----------

    hess: array_like
        The hessian matrix of the Gibbs energy w.r.t. the mole numbers of the
        vapour phase. hess has the shape of (N, n, n) where N is the batch size
        and n is the number of components.

    grads: array_like
        The gradients of the Gibbs energy w.r.t. the mole numbers of the vapour
        phase. grads has the shape (N, n) where N is the batch size and n is
        the number of components.

    d: array_like
        The restricted step to avoid divergence. Note that its shape is N instead of
        (N, 1), that is, d is a row rather than a column.

    max_nit: int, default=100
        The maximum number of iterations.

    tol: float, default=0.1
        The tolerance to determine the convergence between the norm of delta and d.
    """

    alpha = torch.zeros(hess.shape[0], dtype=hess.dtype, device=hess.device)
    alpha_min = torch.zeros_like(alpha)
    alpha_max = torch.zeros_like(alpha)
    set_min = torch.zeros_like(alpha, dtype=torch.bool)
    set_max = torch.zeros_like(set_min)

    unconverged = torch.arange(hess.shape[0], device=hess.device)
    DELTA = torch.zeros_like(grads)
    ALPHA = torch.zeros_like(alpha)
    NITS = torch.zeros_like(alpha, dtype=unconverged.dtype)

    for nit in torch.arange(max_nit, device=hess.device):
        eye = torch.diag_embed(torch.ones_like(grads) * alpha.unsqueeze(-1))
        corrected_hess = hess + eye

        # -------------------------------------------------------------------------
        # lu decomposition is deprecated in favor of cholesky which works for cpu and cuda
        # I retain the following code for comparison

        # lu decomposition without pivoting
        # hess_lu = torch.lu(corrected_hess, pivot=False)
        # determine if hess is positive-definite
        # diag = torch.diagonal(hess_lu[0], dim1=1, dim2=2)
        # negative = (diag <= 0).any(dim=1)
        # positive = ~negative
        # torch.lu_solve without pivoting only available for cuda
        # note that lu_solve receives the RHS of size (*, m, k)
        # so add a dimension for grads
        # delta = torch.lu_solve(-grads.unsqueeze(-1), *hess_lu)
        # gamma = torch.lu_solve(delta, *hess_lu)
        # delta and gamma gotten above have 3 dimensions of which the last one is 1
        # reshape them into 2 dimensions
        # delta = delta.reshape(delta.shape[0], -1)
        # gamma = gamma.reshape(delta.shape[0], -1)
        # --------------------------------------------------------------------------

        # Check if hess is positive-definite

        # If hess is positive, cholesky_ex returns 0, otherwise, a positive integer
        negative = torch.linalg.cholesky_ex(corrected_hess)[1].to(torch.bool)
        positive = ~negative
        # delta = torch.linalg.solve(corrected_hess, -grads)
        # gamma = torch.linalg.solve(corrected_hess, delta)
        hess_inv = torch.linalg.inv(corrected_hess)
        delta = -hess_inv @ grads.unsqueeze(-1)
        gamma = hess_inv @ delta
        delta = delta.reshape(delta.shape[0], -1)
        gamma = gamma.reshape(gamma.shape[0], -1)

        d1 = (delta**2).sum(dim=1)
        d2 = (delta * gamma).sum(dim=1)
        size = d1.sqrt()

        # If hess is negative, increase alpha to guarantee its positiveness
        # based on three conditions
        alpha_min[negative] = alpha[negative]  # first set min=alpha
        set_min[negative] = True
        cond1 = negative & (alpha == 0.0)
        cond2 = negative & (alpha != 0.0) & set_max
        cond3 = negative & (alpha != 0.0) & (~set_max)
        # d1 / d2 is the estimate of the dominant eigenvalue
        alpha[cond1] = 1.2 * (d1[cond1] / d2[cond1]).abs()
        alpha[cond2] = (alpha[cond2] + alpha_max[cond2]) / 2
        alpha[cond3] = 3.0 * alpha[cond3]

        # If hess is positive but size != d
        # cond4: if size < d, set max=alpha and try to decrease it
        cond4 = positive & (size < d)
        alpha_max[cond4] = alpha[cond4]
        set_max[cond4] = True
        # cond5: if size > d, set min=alpha and try to increase it
        cond5 = positive & (size > d)
        alpha_min[cond5] = alpha[cond5]
        set_min[cond5] = True

        # determine convergence
        errors = ((d1 / d**2.0 - 1.0) / 2.0).abs()
        # cond6: If size < d and alpha == 0, cannot further proceed,
        # considered convergent.
        cond6 = (alpha == 0.0) & (size < d)
        converged = (errors <= tol) | cond6

        if converged.all() | (nit == max_nit - 1):
            DELTA[unconverged] = delta
            ALPHA[unconverged] = alpha
            NITS[unconverged] = nit + 1
            break
        else:
            if converged.any():
                converged_indices = unconverged[converged]
                DELTA[converged_indices] = delta[converged]
                ALPHA[converged_indices] = alpha[converged]
                NITS[converged_indices] = nit + 1

                grads = grads[~converged]
                hess = hess[~converged]
                alpha = alpha[~converged]
                alpha_min = alpha_min[~converged]
                alpha_max = alpha_max[~converged]
                set_min = set_min[~converged]
                set_max = set_max[~converged]
                d = d[~converged]
                size = size[~converged]
                d1 = d1[~converged]
                d2 = d2[~converged]
                positive = positive[~converged]
                unconverged = unconverged[~converged]

            # only correct alpha for which hess is positive-definite
            correction = (size / d - 1.0) * d1 / d2
            alpha[positive] += correction[positive]
            # if the corrected alpha is out of bounds, set it (min+max)/2
            set_bounds = set_min & set_max
            out_bounds = (alpha < alpha_min) | (alpha > alpha_max)
            cond7 = set_bounds & out_bounds
            alpha[cond7] = (alpha_min[cond7] + alpha_max[cond7]) / 2.0
            alpha.clamp_(min=0.0)

    return DELTA, ALPHA, NITS
