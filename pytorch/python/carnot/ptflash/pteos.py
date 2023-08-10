import torch
from torch import Tensor
from typing import Optional

from ptflash.utils import cardano_cubic_solver, halley_cubic_solver


class PTSRK:
    """SRK Equation of State based on PyTorch.

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
        self.dtype = dtype
        self.device = device
        self.cubic_solver = cubic_solver

        self.pcs = pcs.to(dtype=dtype, device=device)
        self.tcs = tcs.to(dtype=dtype, device=device)
        self.omegas = omegas.to(dtype=dtype, device=device)

        R = 8.314472
        tc_pc_ratio = self.tcs / self.pcs
        const1 = 0.42748023354034154
        const2 = 0.08664034996495773
        self.ais = const1 * (R**2) * tc_pc_ratio * self.tcs
        self.bis = const2 * R * tc_pc_ratio
        self.mis = self.omegas * (1.574 - 0.176 * self.omegas) + 0.480
        # binary interaction parameters
        self.kij = kij.to(dtype=dtype, device=device)
        self.kijt = kijt.to(dtype=dtype, device=device)
        self.kijt2 = kijt2.to(dtype=dtype, device=device)

    def calculate_lnphis(self, a_alpha_ij, a_alpha, A, b, B, zi, comp_coefs):
        """Evaluate logarithmic fugacity coefficients (lnphis)"""

        part1 = self.bis / b * (comp_coefs - 1) - torch.log(comp_coefs - B)
        part2 = torch.einsum("ijk,ik->ij", a_alpha_ij, zi)
        part3 = A / B * (self.bis / b - 2 / a_alpha * part2)
        part4 = torch.log(1 + B / comp_coefs)
        return part1 + part3 * part4

    def calculate_grads(self, ps, ts, zi, a_alpha_ij, a_alpha, b, comp_coefs):
        """Calculate the partial derivatives of lnphis w.r.t. ni"""

        R = 8.314472
        vs = R * comp_coefs * ts / ps
        f = ((vs + b) / vs).log() / (R * b)
        # The partial derivatives of B
        Bi = self.bis
        # The partial derivative of the sqrt of a_alphas w.r.t. t
        # saat = -self.ais.sqrt() * self.mis / (2 * (ts * self.tcs).sqrt())
        # The partial derivative of a_alpha_ij w.r.t. t
        # aat1 = (1 - kij) * torch.einsum("ij,ik->ijk", saat, a_alphas.sqrt())
        # aat2 = torch.einsum("ij,ik->ijk", a_alphas.sqrt(), saat)
        # The partial derivatives of D
        Di = 2 * torch.einsum("ik,ijk->ij", zi, a_alpha_ij)
        Dij = 2 * a_alpha_ij
        # The first partial derivatives of F, g, f
        gv = b / (vs * (vs - b))
        gb = -1 / (vs - b)
        fv = -1 / (R * vs * (vs + b))
        fb = -(f + vs * fv) / b
        Fd = -f / ts
        # The second order partial derivatives of F, g, and f
        gvv = -1 / (vs - b) ** 2 + 1 / vs**2
        gbv = 1 / (vs - b) ** 2
        gbb = -gbv
        fvv = (1 / vs**2 - 1 / (vs + b) ** 2) / (R * b)
        fbv = -(2 * fv + vs * fvv) / b
        fbb = -(2 * fb + vs * fbv) / b
        Fnv = -gv
        Fnb = -gb
        Fbv = -gbv - a_alpha * fbv / ts
        Fbb = -gbb - a_alpha * fbb / ts
        Fdv = -fv / ts
        Fbd = -fb / ts
        Fvv = -gvv - a_alpha * fvv / ts
        # The second total derivatives of F
        Bi_plus_Bj = Bi + Bi.unsqueeze(1)
        BiBj = Bi * Bi.unsqueeze(1)
        BiDj = torch.einsum("ik, j->ijk", Di, Bi)
        BjDi = torch.einsum("ij, k->ijk", Di, Bi)
        part1 = torch.einsum("ij, jk->ijk", Fnb, Bi_plus_Bj)
        part2 = torch.einsum("ij,ijk->ijk", Fbd, BiDj + BjDi)
        part3 = torch.einsum("ij,jk->ijk", Fbb, BiBj)
        part4 = torch.einsum("ij,ijk->ijk", Fd, Dij)
        total_Fij = part1 + part2 + part3 + part4
        total_Fiv = Fnv + Fbv * Bi + Fdv * Di
        total_Fvv = Fvv
        # Partial derivatives of p w.r.t. ni and v
        pi = R * ts * (-total_Fiv + 1 / vs)
        pv = -R * ts * (total_Fvv + 1 / vs**2)
        # Partial derivatives of lnphis w.r.t. ni
        dlnphis_dni = (
            total_Fij
            + torch.einsum("ij, ik->ijk", pi, pi) / (R * ts * pv).unsqueeze(-1)
            + 1
        )
        return dlnphis_dni

    def __call__(self, inputs: Tensor, return_grads: bool = False):
        """For the relevant equations, refer to the following page:
            https://thermo.readthedocs.io/thermo.eos_mix.html#srk-family-eoss

        inputs: array_like
            The first column is the pressure, the second is the temperature, and the others
            are the composition.

        return_grads: boolean, default=False
            If True, return the partial derivatives of logarithmic fugacity coefficients
            with respect to mole numbers, i.e., dlnphis_dni. If False, return a zero matrix.
        """

        ps = inputs[:, :1]
        ts = inputs[:, 1:2]
        zi = inputs[:, 2:]
        R = 8.314472
        alphas = (1 + self.mis * (1 - torch.sqrt(ts / self.tcs))) ** 2
        b = zi @ self.bis.view(-1, 1)
        a_alphas = self.ais * alphas
        kij = (
            self.kij
            + torch.einsum("ij, jk->ijk", ts, self.kijt)
            + torch.einsum("ij, jk->ijk", ts**2, self.kijt2)
        )
        a_alpha_ij = (1 - kij) * torch.sqrt(
            torch.einsum("ij,ik->ijk", a_alphas, a_alphas)
        )
        z_ij = torch.einsum("ij,ik->ijk", zi, zi)
        a_alpha = (z_ij * a_alpha_ij).sum(dim=(1, 2)).reshape(-1, 1)
        A = a_alpha * ps / (R * ts) ** 2
        B = b * ps / (R * ts)

        # solve EoS to get the compressibility coefficients
        if self.cubic_solver == "halley":
            if_1_real, one_real_root, three_real_roots = halley_cubic_solver(
                ps, ts, b, a_alpha, c1=1.0, c2=0.0, max_nit=100, tol=1.0e-10
            )
        elif self.cubic_solver == "cardano":
            ones = torch.ones_like(A)
            cubic_coefs = torch.cat([ones, -ones, A - B * (1 + B), -A * B], dim=1)
            if_1_real, one_real_root, three_real_roots = cardano_cubic_solver(cubic_coefs)
        else:
            raise ValueError(
                f"The cubic solver is halley or cardano, not {self.cubic_solver}."
            )

        if if_1_real.all():
            """Only one real root, directly return results."""
            comp_coefs = one_real_root
            lnphis = self.calculate_lnphis(a_alpha_ij, a_alpha, A, b, B, zi, comp_coefs)
            gibbs = (zi * lnphis).sum(dim=1, keepdim=True)
            if return_grads:
                dlnphis_dni = self.calculate_grads(
                    ps, ts, zi, a_alpha_ij, a_alpha, b, comp_coefs
                )
            else:
                dlnphis_dni = torch.zeros_like(zi)

            return lnphis, comp_coefs, gibbs, dlnphis_dni
        else:
            """
            If three real roots, assign the smallest root to the liquid phase and the biggest root
            to the vapour phase. Return the root corresponding to the smaller Gibbs energy.
            """
            a_alpha_ij1 = a_alpha_ij[if_1_real]
            a_alpha1 = a_alpha[if_1_real]
            A1 = A[if_1_real]
            b1 = b[if_1_real]
            B1 = B[if_1_real]
            zi1 = zi[if_1_real]
            comp_coefs1 = one_real_root

            # create placeholder variables to save results
            comp_coefs = torch.zeros_like(ps)
            comp_coefs[if_1_real] = comp_coefs1
            lnphis = torch.zeros_like(zi)
            lnphis1 = self.calculate_lnphis(
                a_alpha_ij1, a_alpha1, A1, b1, B1, zi1, comp_coefs1
            )
            lnphis[if_1_real] = lnphis1
            gibbs = torch.zeros_like(ps)
            gibbs[if_1_real] = (zi1 * lnphis1).sum(dim=1, keepdim=True)

            # calculate and compare gibbs energy
            liq_coefs = (three_real_roots).max(dim=1, keepdim=True).values
            vap_coefs = (three_real_roots).min(dim=1, keepdim=True).values

            a_alpha_ij3 = a_alpha_ij[~if_1_real]
            a_alpha3 = a_alpha[~if_1_real]
            A3 = A[~if_1_real]
            b3 = b[~if_1_real]
            B3 = B[~if_1_real]
            zi3 = zi[~if_1_real]
            liq_lnphis = self.calculate_lnphis(
                a_alpha_ij3, a_alpha3, A3, b3, B3, zi3, liq_coefs
            )
            vap_lnphis = self.calculate_lnphis(
                a_alpha_ij3, a_alpha3, A3, b3, B3, zi3, vap_coefs
            )
            liq_gibbs = (zi3 * liq_lnphis).sum(dim=1, keepdim=True)
            vap_gibbs = (zi3 * vap_lnphis).sum(dim=1, keepdim=True)

            comparison = liq_gibbs < vap_gibbs
            comp_coefs[~if_1_real] = torch.where(comparison, liq_coefs, vap_coefs)
            lnphis[~if_1_real] = torch.where(comparison, liq_lnphis, vap_lnphis)
            gibbs[~if_1_real] = torch.where(comparison, liq_gibbs, vap_gibbs)

            if return_grads:
                dlnphis_dni = self.calculate_grads(
                    ps, ts, zi, a_alpha_ij, a_alpha, b, comp_coefs
                )
            else:
                dlnphis_dni = torch.zeros_like(zi)

            return lnphis, comp_coefs, gibbs, dlnphis_dni
