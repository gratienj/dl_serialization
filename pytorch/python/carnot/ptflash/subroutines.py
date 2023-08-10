import torch
from torch import Tensor
from typing import Optional

from ptflash.pteos import PTSRK
from ptflash.utils import rr_solver, hebden_solver


class PTFlash:
    """Flash calculation based on PyTorch, including the methods:
        (1) Phase split calculations: `ss_splitter` and `newton_splitter`
        (2) Stability analysis: `ss_analyser` and `newton_analyser`

    Parameters
    ----------

    pcs: array-like, critical pressure

    tcs: array-like, critical temperature

    omegas: array-like, acentric parameters

    kij: square matrix of shape (n_components, n_components)
        Temperature-independent binary interaction parameters

    kijt: square matrix of shape (n_components, n_components)
        Temperature-dependent binary interaction parameters

    kijt2: square matrix of shape (n_components, n_components)
        Squared-temperature-dependent binary interaction parameters

    cubic_solver: str, halley or cardano, default="halley"

    dtype: torch.dtype, default=torch.float64

    device: torch.device, default=None
    """

    def __init__(
        self,
        pcs: Tensor,
        tcs: Tensor,
        omegas: Tensor,
        kij: Tensor,
        kijt: Tensor,
        kijt2: Tensor,
        cubic_solver: str = "halley",
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        self.pcs = pcs.to(dtype=dtype, device=device)
        self.tcs = tcs.to(dtype=dtype, device=device)
        self.omegas = omegas.to(dtype=dtype, device=device)
        self.eos = PTSRK(pcs, tcs, omegas, kij, kijt, kijt2, cubic_solver, dtype, device)
        self.dtype = dtype
        self.device = device

    def init_ki(self, inputs: Tensor, log: bool = False):
        """Initialize the distribution coefficients (ki) using Wilson approximation"""

        part1 = torch.log(self.pcs / inputs[:, :1])
        part2 = 5.373 * (1 + self.omegas) * (1 - self.tcs / inputs[:, 1:2])
        lnki = part1 + part2
        if log:
            return lnki
        else:
            return lnki.exp()

    def update_ki(self, inputs: Tensor, lnki: Tensor, lnphis: Tensor):
        """
        Subfunction of `ss_splitter` to update the distribution coefficients ki.

        Parameters
        ----------
        inputs: array_like
            The first column is the pressure, the second is the temperature, and the others
            are the composition.

        lnki: array_like
            The log distribution coefficients

        lnphis: array_like, default=None
            The log fugacity coefficients of inputs used to calculate Tangent Plan Distance
        """

        ps = inputs[:, :1]
        ts = inputs[:, 1:2]
        zi = inputs[:, 2:]
        # solve Rachford-Rice equation
        ki = lnki.exp()
        theta_v = rr_solver(zi, ki)[0]
        theta_v.clamp_(min=0, max=1)
        xi = zi / (1 + (ki - 1) * theta_v)
        yi = ki * xi
        xi = xi / xi.sum(dim=1, keepdim=True)
        yi = yi / yi.sum(dim=1, keepdim=True)
        # update ki
        lnphil = self.eos(torch.cat([ps, ts, xi], dim=1))[0]
        lnphiv = self.eos(torch.cat([ps, ts, yi], dim=1))[0]
        lnki = lnphil - lnphiv
        # calculate tpd
        zi_plus_lnphis = zi.log() + lnphis
        tpdx = (xi * (xi.log() + lnphil - zi_plus_lnphis)).sum(dim=1, keepdim=True)
        tpdy = (yi * (yi.log() + lnphiv - zi_plus_lnphis)).sum(dim=1, keepdim=True)
        tpd = (1 - theta_v) * tpdx + theta_v * tpdy

        return theta_v, xi, yi, lnki, tpdx, tpdy, tpd

    def ss_splitter(
        self,
        inputs: Tensor,
        ki: Optional[Tensor] = None,
        lnphis: Optional[Tensor] = None,
        max_nit: int = 100,
        tol: float = 1.0e-8,
    ):
        """
        Phase split calculations with successive substitution using Dominant Eigenvalue Method

        Parameters
        ----------

        inputs: array_like
            The first column is the pressure, the second is the temperature, and the others
            are the composition.

        ki: array_like, default=None
            Initial distribution coefficients. If not passed, call `init_ki` to initialize it.

        lnphis: array_like, default=None
            The log fugacity coefficients of inputs used to calculate Tangent Plan Distance (tpd)

        max_nit: int, default=100
            The maximum number of iterations.

        tol: float, default=1.0e-8
            Relative tolerance for termination.
        """

        if ki is None:
            lnki = self.init_ki(inputs, log=True)
        else:
            lnki = ki.log()

        n_samples = inputs.shape[0]
        n_components = inputs.shape[1] - 2
        unconverged = torch.arange(n_samples, device=self.device)
        # The following variables are used to save converged results after each iteration
        THETA_V = torch.zeros(n_samples, 1, dtype=self.dtype, device=self.device)
        Xi = torch.zeros(n_samples, n_components, dtype=self.dtype, device=self.device)
        Yi = torch.zeros_like(Xi)
        LNKi = torch.zeros_like(Yi)
        NITS = torch.zeros(n_samples, 1, dtype=torch.int64, device=self.device)
        ERRORS = torch.zeros_like(THETA_V)

        if lnphis is None:
            lnphis = self.eos(inputs)[0]  # used to calculate tpd
        lnki_0 = torch.zeros_like(LNKi)
        lnki_1 = torch.zeros_like(LNKi)

        for nit in torch.arange(max_nit, device=self.device):
            theta_v, xi, yi, new_lnki, _, _, tpd = self.update_ki(
                inputs, lnki=lnki, lnphis=lnphis
            )
            if nit % 3 == 0:
                lnki_0 = new_lnki
            elif nit % 3 == 1:
                lnki_1 = new_lnki
            elif nit % 3 == 2:
                d0 = lnki_1 - lnki_0
                d1 = new_lnki - lnki_1
                sum0 = (inputs[:, 2:] * d0**2).sum(dim=1, keepdim=True)
                sum1 = (inputs[:, 2:] * d1**2).sum(dim=1, keepdim=True)
                lambdas = (sum1 / sum0).sqrt()
                lambdas.clamp_(max=0.95)
                extrapolated_factors = lambdas / (1 - lambdas)
                new_lnki2 = new_lnki + extrapolated_factors * d1
                theta_v2, xi2, yi2, new_lnki2, _, _, tpd2 = self.update_ki(
                    inputs, lnki=new_lnki2, lnphis=lnphis
                )
                # accept the extrapolation iif tpd decreases and theta != 0 and 1
                cond1 = tpd2 < tpd
                cond2 = (theta_v2 != 0.0) & (theta_v2 != 1.0)
                accepted = (cond1 & cond2).view(-1)
                theta_v[accepted] = theta_v2[accepted]
                xi[accepted] = xi2[accepted]
                yi[accepted] = yi2[accepted]
                new_lnki[accepted] = new_lnki2[accepted]

            errors = ((new_lnki - lnki) / lnki).abs().max(dim=1)[0]
            converged = errors < tol

            if converged.all() | (nit == max_nit - 1):
                THETA_V[unconverged] = theta_v
                Xi[unconverged] = xi
                Yi[unconverged] = yi
                LNKi[unconverged] = new_lnki
                NITS[unconverged] = nit + 1
                ERRORS[unconverged] = errors.reshape(-1, 1)
                unconverged = unconverged[~converged]
                break
            else:
                if torch.all(~converged):
                    lnki = new_lnki
                else:
                    converged_indices = unconverged[converged]
                    THETA_V[converged_indices] = theta_v[converged]
                    Xi[converged_indices] = xi[converged]
                    Yi[converged_indices] = yi[converged]
                    LNKi[converged_indices] = new_lnki[converged]
                    NITS[converged_indices] = nit + 1
                    ERRORS[converged_indices] = errors[converged].reshape(-1, 1)

                    inputs = inputs[~converged]
                    lnki = new_lnki[~converged]
                    unconverged = unconverged[~converged]
                    lnphis = lnphis[~converged]
                    lnki_0 = lnki_0[~converged]
                    lnki_1 = lnki_1[~converged]

        return THETA_V, Xi, Yi, LNKi.exp(), NITS, ERRORS, unconverged

    def calculate_gibbs(self, inputs: Tensor, vi: Tensor, return_grads: bool = True):
        """Subfunction of `newton_splitter` to calculate the Gibbs energy."""

        ps = inputs[:, :1]
        ts = inputs[:, 1:2]
        zi = inputs[:, 2:]
        theta_v = vi.sum(dim=1, keepdim=True)
        li = zi - vi
        xi = li / (1 - theta_v)
        yi = vi / theta_v

        res_l = self.eos(torch.cat([ps, ts, xi], dim=1), return_grads)
        res_v = self.eos(torch.cat([ps, ts, yi], dim=1), return_grads)
        lnphil, lnphiv = res_l[0], res_v[0]
        dlnphil_dni = res_l[-1]
        dlnphiv_dni = res_v[-1]
        # evaluate the Gibbs energy
        term_l = xi.log() + lnphil
        term_v = yi.log() + lnphiv
        gibbs_l = (li * term_l).sum(dim=1, keepdim=True)
        gibbs_v = (vi * term_v).sum(dim=1, keepdim=True)
        gibbs = gibbs_l + gibbs_v
        return gibbs, theta_v, xi, yi, term_l, term_v, dlnphil_dni, dlnphiv_dni

    def newton_splitter(
        self,
        inputs: Tensor,
        ki: Optional[Tensor] = None,
        max_nit: int = 100,
        tol: float = 1.0e-8,
    ):
        """
        Phase split calculations using second-order method

        Reference:
                AN ALGORITHM FOR MINIMIZATION USING EXACT SECOND DERIVATIVES.

        Parameters
        ----------

        inputs: array_like
            The first column is the pressure, the second is the temperature, and the others
            are the composition.

        ki: array_like, default=None
            Initial distribution coefficients. If not passed, call `init_ki` to initialize it.

        max_nit: int, default=100
            The maximum number of iterations.

        tol: float, default=1.0e-8
            If the maximum of abs(gradients) is less than tol, then stop.
        """

        if ki is None:
            ki = self.init_ki(inputs, log=False)
        # calculate theta_v, xi, yi, and vi based on ki
        theta_v = rr_solver(inputs[:, 2:], ki)[0]
        theta_v.clamp_(min=0, max=1)
        xi = inputs[:, 2:] / (1 + (ki - 1) * theta_v)
        yi = ki * xi
        xi = xi / xi.sum(dim=1, keepdim=True)
        yi = yi / yi.sum(dim=1, keepdim=True)
        vi = theta_v * yi
        inputs2 = inputs.clone()  # copy inputs to calculate other properties in the end

        n_samples = inputs.shape[0]
        n_components = inputs.shape[1] - 2
        unconverged = torch.arange(n_samples, device=self.device)
        # The following variables are used to save converged results after each iteration
        Vi = torch.zeros(n_samples, n_components, dtype=self.dtype, device=self.device)
        NITS = torch.zeros(n_samples, 1, dtype=unconverged.dtype, device=self.device)
        ERRORS = torch.zeros(n_samples, 1, dtype=self.dtype, device=self.device)

        # define the parameters of Hebden optimization
        opt_tol = 0.1
        opt_d = torch.where(theta_v <= 0.5, theta_v, 1.0 - theta_v) / 2.0

        for nit in torch.arange(max_nit, device=self.device):
            res = self.calculate_gibbs(inputs, vi=vi, return_grads=True)
            gibbs, theta_v, xi, yi, term_l, term_v, dlnphil_dni, dlnphiv_dni = res
            # calculate the gradients and hessian matrix
            frac = torch.sqrt(inputs[:, 2:] / (xi * yi))
            grads = term_v - term_l
            theta_v_l = theta_v * (1.0 - theta_v)
            pseudo_grads = grads * theta_v_l / frac
            # reshape theta_v into (N, 1, 1) to calculate hess
            theta_v = theta_v.reshape(-1, 1, 1)
            hess = theta_v * dlnphil_dni + (1 - theta_v) * dlnphiv_dni - 1.0
            hess.div_(torch.einsum("ij,ik->ijk", frac, frac))
            hess.add_(torch.diag_embed(torch.ones_like(pseudo_grads)))

            # solve (hess + alpha * I) * delta = -grads
            delta, alpha, _ = hebden_solver(
                hess, pseudo_grads, opt_d.squeeze(), tol=opt_tol
            )

            delta_vi = delta / frac
            # update the smaller between vi and li to keep both of them in [0, 1]
            li = inputs[:, 2:] - vi
            smaller = (vi + delta_vi) < (li - delta_vi)
            # If vi + delta < vi / 2, then delta = - vi / 2
            cond1 = smaller & (vi < -2 * delta_vi)
            delta_vi[cond1] = -vi[cond1] / 2
            # If li - delta < li / 2, then delta = li / 2
            cond2 = (~smaller) & (li < 2 * delta_vi)
            delta_vi[cond2] = li[cond2] / 2
            # update vi
            new_vi = vi + delta_vi
            # reset delta
            delta = delta_vi * frac
            # calculate new gibbs energy
            new_gibbs = self.calculate_gibbs(inputs, vi=new_vi, return_grads=False)[0]
            # compare the real and predicted reductions to adjust d
            diff = new_gibbs - gibbs
            pred1 = (delta * pseudo_grads).sum(dim=1, keepdim=True)
            pred2 = (delta.unsqueeze(1) @ hess @ delta.unsqueeze(-1)).reshape(-1, 1)
            pred = (pred1 + pred2 / 2.0) / theta_v_l
            ratio = (diff / pred).abs().view(-1)
            size = delta.norm(p=2, dim=-1, keepdim=True)
            # ratio <= 0.25, decrease d
            cond3 = ratio <= 0.25
            opt_d[cond3] = size[cond3] / 2.0
            # if alpha > 0.0 and the prediction is good, increase d
            cond4 = (ratio >= 0.75) & (alpha > 0.0)
            opt_d[cond4] = size[cond4] * 2.0
            cond5 = ~(cond3 | cond4)
            opt_d[cond5] = size[cond5]
            # if the gibbs energy increases, discard the updates and reduce d
            rejected = (diff > 1.0e-10).view(-1)
            accepted = ~rejected
            vi[accepted] = new_vi[accepted]
            opt_d[rejected] = size[rejected] / 5.0

            errors = grads.abs().max(dim=1)[0]
            converged = errors < tol

            if converged.all() | (nit == max_nit - 1):
                Vi[unconverged] = vi
                NITS[unconverged] = nit + 1
                ERRORS[unconverged] = errors.reshape(-1, 1)
                unconverged = unconverged[~converged]
                break
            else:
                if converged.any():
                    converged_indices = unconverged[converged]
                    Vi[converged_indices] = vi[converged]
                    NITS[converged_indices] = nit + 1
                    ERRORS[converged_indices] = errors[converged].reshape(-1, 1)

                    inputs = inputs[~converged]
                    vi = vi[~converged]
                    unconverged = unconverged[~converged]
                    opt_d = opt_d[~converged]

        THETA_V = Vi.sum(dim=1, keepdim=True)
        Xi = (inputs2[:, 2:] - Vi) / (1 - THETA_V)
        Yi = Vi / THETA_V
        LNPHIS_L = self.eos(torch.cat([inputs2[:, :2], Xi], dim=1))[0]
        LNPHIS_V = self.eos(torch.cat([inputs2[:, :2], Yi], dim=1))[0]
        Ki = (LNPHIS_L - LNPHIS_V).exp()

        return THETA_V, Xi, Yi, Ki, NITS, ERRORS, unconverged

    def update_wi(self, inputs: Tensor, lnwi: Tensor, lnphis: Tensor):
        """
        Subfunction of `ss_analyser` to update the lnwi

        Parameters
        ----------
        inputs: array_like
            The first column is the pressure, the second is the temperature, and the others
            are the composition.

        lnwi: array_like

        lnphis: array_like
            The logarithmic fugacity coefficients of inputs
        """

        ps = inputs[:, :1]
        ts = inputs[:, 1:2]
        zi = inputs[:, 2:]
        wi = lnwi.exp()
        normalized_wi = wi / wi.sum(dim=1, keepdim=True)
        lnphiw = self.eos(torch.cat([ps, ts, normalized_wi], dim=1))[0]
        new_lnwi = lnphis + zi.log() - lnphiw
        grads = lnwi - new_lnwi  # the gradients of tm w.r.t. wi
        diff = (wi * grads**2).sum(dim=1, keepdim=True)  # used for dem extrapolation
        # modified tangent plane distance
        tm = (wi * (grads - 1)).sum(dim=1) + 1.0

        return new_lnwi, grads, diff, tm

    def ss_analyser(
        self,
        inputs: Tensor,
        wi: Optional[Tensor] = None,
        ki: Optional[Tensor] = None,
        lnphis: Optional[Tensor] = None,
        vapour_like: bool = True,
        max_nit: int = 100,
        tol: float = 1.0e-8,
    ):
        """
        Stability analysis with successive substitution using Dominant Eigenvalue Method. Determine
        if the feed is stable by locating the global minima of the tangent plane distance.

        Parameters
        ----------

        inputs: array_like
            The first column is the pressure, the second is the temperature, and the others
            are the composition.

        wi: array_like, default=None
            The initial estimate to initiate the minimization search, which is the vapour-like
            or liquid-like. If not provided, use a vapour-like estimate ki * zi or a liquid-like
            estimate zi / ki based on `vapour_like`.

        ki: array_like, default=None
            Initial distribution coefficients. If not passed, call `init_ki` to initialize ki.

        lnphis: array_like, default=None
            The log fugacity coefficients of inputs used to calculate Tangent Plan Distance (tpd)

        vapour_like: boolean, default=True
            Take effect when `wi` is None. If True, use a vapour-like estimate ki * zi, otherwise,
            use a liquid-like estimate zi / ki.

        max_nit: int, default=100
            The maximum number of iterations.

        tol: float, default=1.0e-8
            Relative tolerance for termination.
        """

        if wi is None:
            if ki is None:
                ki = self.init_ki(inputs, log=False)
            wi = ki * inputs[:, 2:] if vapour_like else inputs[:, 2:] / ki
        lnwi = wi.log()

        n_samples = inputs.shape[0]
        stable = torch.zeros(n_samples, 1, dtype=torch.bool, device=self.device)
        # code indicates the condition of convergence.
        # code=0, tm < -tol, unstable, converged
        # code=1, max(grad) < tol, converged
        # code=2, trivial solution, stable, converged
        code = torch.zeros_like(stable, dtype=torch.int)
        unconverged = torch.arange(n_samples, device=self.device)
        # The following variables are used to save converged results after each iteration
        STABLE = torch.zeros_like(stable)
        CODE = torch.zeros_like(code)
        LNWi = torch.zeros_like(lnwi)
        TM = torch.zeros(n_samples, dtype=self.dtype, device=self.device)
        NITS = torch.zeros_like(stable, dtype=unconverged.dtype)

        d0 = torch.zeros_like(LNWi)
        d1 = torch.zeros_like(LNWi)
        # update wi once before the loop
        if lnphis is None:
            lnphis = self.eos(inputs)[0]
        lnwi = self.update_wi(inputs, lnwi=lnwi, lnphis=lnphis)[0]
        # main loop
        for nit in torch.arange(max_nit, device=self.device):
            lnwi, grads, diff, tm = self.update_wi(inputs, lnwi=lnwi, lnphis=lnphis)
            if nit % 2 == 0:
                d0 = diff
            elif nit % 2 == 1:
                d1 = diff
                lambdas = (d1 / d0).sqrt()
                lambdas.clamp_(max=0.95)
                extrapolated_factors = lambdas / (1 - lambdas)
                # -grads is also the difference between new_lnwi and lnwi
                lnwi2 = lnwi - extrapolated_factors * grads
                lnwi2, grads2, _, tm2 = self.update_wi(inputs, lnwi=lnwi2, lnphis=lnphis)
                # accept the extrapolation iif tm decreases
                accepted = tm2 < tm
                lnwi[accepted] = lnwi2[accepted]
                grads[accepted] = grads2[accepted]
                tm[accepted] = tm2[accepted]

            # convergence criteria
            # If tm < 0, we conclude that the feed is unstable.
            cond1 = tm < -1.0e-10
            stable[cond1] = False
            code[cond1] = 0
            # If the max of the gradients of tm w.r.t. wi is small enough,
            # we arrive at the stationary point.
            cond2 = grads.abs().max(dim=1)[0] < tol
            stable[cond2 & (~cond1)] = True
            code[cond2] = 1
            # If wi close to zi, we get the trivial solution
            # sqrt_wi = lnwi.exp().sqrt()
            # sqrt_zi = inputs[:, 2:].sqrt()
            # distance = torch.norm(lnwi.exp() - inputs[:, 2:], dim=1)
            # prd = ((lnwi.exp() - sqrt_wi * sqrt_zi) * grads).sum(dim=1)
            # ratio = tm / prd
            # cond3 = (distance < 0.01) & (ratio > 0.8)
            # stable[cond3] = True
            # code[cond3] = 2
            # All conditions converge.
            # converged = cond1 | cond2 | cond3
            converged = cond2  # only use cond2 temporarily

            if converged.all() | (nit == max_nit - 1):
                STABLE[unconverged] = stable
                CODE[unconverged] = code
                LNWi[unconverged] = lnwi
                TM[unconverged] = tm
                NITS[unconverged] = nit + 1
                unconverged = unconverged[~converged]
                break
            else:
                if converged.any():
                    converged_indices = unconverged[converged]
                    STABLE[converged_indices] = stable[converged]
                    CODE[converged_indices] = code[converged]
                    LNWi[converged_indices] = lnwi[converged]
                    TM[converged_indices] = tm[converged]
                    NITS[converged_indices] = nit + 1

                    lnwi = lnwi[~converged]
                    inputs = inputs[~converged]
                    lnphis = lnphis[~converged]
                    stable = stable[~converged]
                    code = code[~converged]
                    unconverged = unconverged[~converged]
                    d0 = d0[~converged]
                    d1 = d1[~converged]

        return STABLE, CODE, LNWi.exp(), TM, NITS, unconverged

    def newton_analyser(
        self,
        inputs: Tensor,
        wi: Optional[Tensor] = None,
        ki: Optional[Tensor] = None,
        vapour_like: bool = True,
        max_nit: int = 100,
        tol: float = 1.0e-8,
    ):
        """
        Stability analysis using the second-order method

        Parameters
        ----------

        inputs: array_like
            The first column is the pressure, the second is the temperature, and the others
            are the composition.

        wi: array_like, default=None
            The initial estimate to initiate the minimization search, which is the vapour-like
            or liquid-like. If not provided, use a vapour-like estimate ki * zi or a liquid-like
            estimate zi / ki based on `vapour_like`.

        ki: array_like, default=None
            Initial distribution coefficients. If not passed, call `init_ki` to initialize ki.

        vapour_like: boolean, default=True
            Take effect when `wi` is None. If True, use a vapour-like estimate ki * zi, otherwise,
            use a liquid-like estimate zi / ki.

        max_nit: int, default=100
            The maximum number of iterations.

        tol: float, default=1.0e-8
            Relative tolerance for termination.
        """

        if wi is None:
            if ki is None:
                ki = self.init_ki(inputs, log=False)
            wi = ki * inputs[:, 2:] if vapour_like else inputs[:, 2:] / ki

        n_samples = inputs.shape[0]
        lnphis = self.eos(inputs)[0]
        stable = torch.zeros(n_samples, 1, dtype=torch.bool, device=self.device)
        # code indicates the condition of convergence
        code = torch.zeros_like(stable, dtype=torch.int)
        unconverged = torch.arange(n_samples, device=self.device)
        # The following variables are used to save converged results after each iteration
        STABLE = torch.zeros_like(stable)
        CODE = torch.zeros_like(code)
        Wi = torch.zeros_like(wi)
        TM = torch.zeros(n_samples, dtype=self.dtype, device=self.device)
        NITS = torch.zeros_like(stable, dtype=unconverged.dtype)

        opt_tol = 0.1
        opt_d = torch.ones_like(stable, dtype=self.dtype) * 0.25

        for nit in torch.arange(max_nit, device=self.device):
            # calculate the gradients
            sum_wi = wi.sum(dim=1, keepdim=True)
            normalized_wi = wi / sum_wi
            lnphiw, _, _, dlnphiw_dni = self.eos(
                torch.cat([inputs[:, :2], normalized_wi], dim=1), return_grads=True
            )
            swi = wi.sqrt()
            sgd = wi.log() + lnphiw - inputs[:, 2:].log() - lnphis
            grads = swi * sgd
            # modified tangent plane distance
            tm = 1.0 + (wi * (sgd - 1.0)).sum(dim=1)
            # calculate the hessian matrices
            hess = torch.einsum("ij,ik->ijk", swi, swi) * dlnphiw_dni
            hess.div_(sum_wi.unsqueeze(-1))
            hess.add_(torch.diag_embed(1.0 + sgd / 2.0))
            # solve (hess + alpha * I) * delta = -grads
            delta, alpha, _ = hebden_solver(hess, grads, opt_d.squeeze(), tol=opt_tol)
            gamma = 2.0 * swi
            cond1 = gamma < -delta
            delta[cond1] = -0.9 * gamma[cond1]
            gamma.add_(delta)
            new_wi = gamma**2 / 4.0
            # evaluate the new modified tangent plane distance
            normalized_wi = new_wi / new_wi.sum(dim=1, keepdim=True)
            lnphiw = self.eos(torch.cat([inputs[:, :2], normalized_wi], dim=1))[0]
            sgd = new_wi.log() + lnphiw - inputs[:, 2:].log() - lnphis
            new_grads = new_wi.sqrt() * sgd
            new_tm = 1.0 + (new_wi * (sgd - 1.0)).sum(dim=1)
            # compare the real and predicted reductions to adjust d
            diff = new_tm - tm
            pred1 = (delta * grads).sum(dim=1)
            pred2 = (delta.unsqueeze(1) @ hess @ delta.unsqueeze(-1)).reshape(-1)
            pred = pred1 + pred2 / 2.0
            ratio = (diff / pred).view(-1)
            size = delta.norm(p=2, dim=-1, keepdim=True)
            # ratio <= 0.25, decrease d
            cond2 = ratio <= 0.25
            opt_d[cond2] = size[cond2] / 2.0
            # if alpha > 0.0 and the prediction is good, increase d
            cond3 = (ratio >= 0.75) & (alpha > 0.0)
            opt_d[cond3] = size[cond3] * 2.0
            cond4 = ~(cond2 | cond3)
            opt_d[cond4] = size[cond4]
            # if tm increases, discard the updates and reduce d
            rejected = (diff > 1.0e-10).view(-1)
            accepted = ~rejected
            wi[accepted] = new_wi[accepted]
            grads[accepted] = new_grads[accepted]
            tm[accepted] = new_tm[accepted]
            opt_d[rejected] = size[rejected] / 3.0

            errors = grads.abs().max(dim=1)[0]
            converged = errors < tol
            stable[converged & (tm > -1.0e-10)] = True
            code[converged] = 1

            if converged.all() | (nit == max_nit - 1):
                STABLE[unconverged] = stable
                CODE[unconverged] = code
                Wi[unconverged] = wi
                TM[unconverged] = tm
                NITS[unconverged] = nit + 1
                unconverged = unconverged[~converged]
                break
            else:
                if converged.any():
                    converged_indices = unconverged[converged]
                    STABLE[converged_indices] = stable[converged]
                    CODE[converged_indices] = code[converged]
                    Wi[converged_indices] = wi[converged]
                    TM[converged_indices] = tm[converged]
                    NITS[converged_indices] = nit + 1

                    wi = wi[~converged]
                    inputs = inputs[~converged]
                    lnphis = lnphis[~converged]
                    stable = stable[~converged]
                    code = code[~converged]
                    unconverged = unconverged[~converged]
                    opt_d = opt_d[~converged]

        return STABLE, CODE, Wi, TM, NITS, unconverged
