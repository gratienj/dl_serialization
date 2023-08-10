import torch
from torch import Tensor
from typing import Optional
from timeit import default_timer

from ptflash.subroutines import PTFlash


def negative_tpd_detector(
    flasher: PTFlash, inputs: Tensor, ki: Tensor, lnphis: Tensor, max_nit: int = 3
):
    """
    Run successive substitution of the split calculations `max_nit` times to detect if tpd
    could be negative. If tpd < 0, the feed is unstable and no need for stability analysis.
    """
    n_samples = inputs.shape[0]
    indices = torch.arange(n_samples, device=inputs.device)
    stable = torch.ones(n_samples, dtype=torch.bool, device=inputs.device)
    lnki = ki.log()
    zi = inputs[:, 2:]  # used to re-estimate ki
    it = 0
    theta_v, xi, yi, lnki, tpdx, tpdy, tpd = flasher.update_ki(
        inputs, lnki=lnki, lnphis=lnphis
    )
    in_0_1 = ((theta_v > 0.0) & (theta_v < 1.0)).view(-1)
    inputs = inputs[in_0_1]
    lnki = lnki[in_0_1]
    lnphis = lnphis[in_0_1]
    indices = indices[in_0_1]
    for _ in torch.arange(max_nit, device=inputs.device):
        it = it + 1
        theta_v, xi, yi, lnki, tpdx, tpdy, tpd = flasher.update_ki(
            inputs, lnki=lnki, lnphis=lnphis
        )
        in_0_1 = ((theta_v > 0.0) & (theta_v < 1.0)).view(-1)
        inputs = inputs[in_0_1]
        lnki = lnki[in_0_1]
        lnphis = lnphis[in_0_1]
        indices = indices[in_0_1]

    xi = xi[in_0_1]
    yi = yi[in_0_1]
    tpdx = tpdx[in_0_1]
    tpdy = tpdy[in_0_1]
    tpd = tpd[in_0_1]
    negative_tpdx = (tpdx < -1.0e-10).view(-1)
    negative_tpdy = (tpdy < -1.0e-10).view(-1)
    negative_tpd = (tpd < -1.0e-10).view(-1)
    negative = negative_tpdx | negative_tpdy | negative_tpd
    negative_indices = indices[negative]
    stable[negative_indices] = False

    # tpdx < 0 but tpd > 0, ki = zi / xi
    case1 = negative_tpdx & (~negative_tpd)
    ki[indices[case1]] = zi[indices[case1], :] / xi[case1]
    # tpdy < 0 but tpd > 0, ki = yi / zi
    case2 = negative_tpdy & (~negative_tpd)
    ki[indices[case2]] = yi[case2] / zi[indices[case2], :]
    # tpd < 0, ki = lnki.exp()
    ki[indices[negative_tpd]] = lnki[negative_tpd].exp()
    return stable, ki


def analyser(
    flasher: PTFlash,
    inputs: Tensor,
    ki: Tensor,
    lnphis: Optional[Tensor] = None,
    vapour_like: bool = True,
    sa_max_nit1: int = 9,
    sa_max_nit2: int = 40,
    debug: bool = False,
):
    """Stability analysis"""

    res = flasher.ss_analyser(
        inputs,
        ki=ki,
        lnphis=lnphis,
        vapour_like=vapour_like,
        max_nit=sa_max_nit1,
        tol=1.0e-6,
    )
    stable, _, wi, tm, nits, unconverged = res

    phase = "vapour" if vapour_like else "liquid"
    if debug:
        ratio = (1 - unconverged.shape[0] / inputs.shape[0]) * 100
        print("Stability analysis with successive substitution")
        print(f"Using {phase}-like estimate:")
        print(f"# of samples: {inputs.shape[0]}")
        print("Convergence: {:.4f}%".format(ratio))
        print(f"Max iteration: {nits.max()} \n")

    if unconverged.numel() > 0:
        res2 = flasher.newton_analyser(
            inputs[unconverged], wi=wi[unconverged], max_nit=sa_max_nit2, tol=1.0e-6
        )
        stable2, _, wi2, tm2, nits2, unconverged2 = res2
        wi[unconverged] = wi2
        tm[unconverged] = tm2
        stable[unconverged] = stable2
        if debug:
            print("Stability analysis with the second-order method")
            print(f"Using {phase}-like estimate:")
            print(f"# of samples: {unconverged.shape[0]}")
            print(f"# of failures: {unconverged2.shape[0]}")
            print(f"Max iteration: {nits2.max()} \n")

    return stable, wi, tm


@torch.jit.script
def async_analyser(
    flasher: PTFlash,
    inputs: Tensor,
    ki: Tensor,
    lnphis: Optional[Tensor] = None,
    sa_max_nit1: int = 9,
    sa_max_nit2: int = 40,
    debug: bool = False,
):
    """
    Run vapour-like and liquid-like estimates in parallel
    Note that this function consumes much more memory.
    """
    fut_vapour = torch.jit.fork(
        analyser,
        flasher=flasher,
        inputs=inputs,
        ki=ki,
        lnphis=lnphis,
        vapour_like=True,
        sa_max_nit1=sa_max_nit1,
        sa_max_nit2=sa_max_nit2,
        debug=debug,
    )
    fut_liquid = torch.jit.fork(
        analyser,
        flasher=flasher,
        inputs=inputs,
        ki=ki,
        lnphis=lnphis,
        vapour_like=False,
        sa_max_nit1=sa_max_nit1,
        sa_max_nit2=sa_max_nit2,
        debug=debug,
    )
    return torch.jit.wait(fut_vapour), torch.jit.wait(fut_liquid)


def splitter(
    flasher: PTFlash,
    inputs: Tensor,
    ki: Tensor,
    lnphis: Optional[Tensor] = None,
    split_max_nit1: int = 9,
    split_max_nit2: int = 40,
    debug: bool = False,
):
    """Phase split calculations"""

    #if debug:
    #    start = default_timer()
    res = flasher.ss_splitter(
        inputs, ki=ki, lnphis=lnphis, max_nit=split_max_nit1, tol=1.0e-8
    )
    theta_v, xi, yi, ki, nits, _, unconverged = res

    '''
    if debug:
        time = default_timer() - start
        ratio = (1 - unconverged.shape[0] / inputs.shape[0]) * 100.0
        print("Phase split calculations with successive substitution")
        print(f"# of samples: {inputs.shape[0]}")
        print("Convergence: {:.4f}%".format(ratio))
        print(f"Max iteration: {nits.max()}")
        print("Time: {:.4f}s \n".format(time))
    '''
    if unconverged.numel() > 0:
        #if debug:
        #    start = default_timer()
        res2 = flasher.newton_splitter(
            inputs[unconverged], ki=ki[unconverged], max_nit=split_max_nit2, tol=1.0e-8
        )
        theta_v2, xi2, yi2, ki2, nits2, _, unconverged2 = res2
        theta_v[unconverged] = theta_v2
        xi[unconverged] = xi2
        yi[unconverged] = yi2
        ki[unconverged] = ki2

        '''
        if debug:
            time = default_timer() - start
            print("Phase split calculations with the second-order method")
            print(f"# of samples: {unconverged.shape[0]}")
            print(f"# of failures: {unconverged2.shape[0]}")
            print(f"Max iteration: {nits2.max()}")
            print("Time: {:.4f}s \n".format(time))
        '''
    return theta_v, xi, yi, ki

class PTVLE:
    """
    Vapour-liquid equilibrium based on PyTorch

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
        self.flasher = PTFlash(
            pcs, tcs, omegas, kij, kijt, kijt2, cubic_solver, dtype, device
        )
        self.dtype = dtype
        self.device = device

    def init_ki(self, inputs, init_engine="wilson", initializer=None, log=False):
        """
        Initialize ki using:
            (1) Wilson approximation if `init_engine` is wilson
            (2) neural network (initializer) if `init_engine` is nn
        """

        if init_engine == "nn":
            initializer.eval()
            initializer.to(self.device)
            with torch.no_grad():
                lnki = initializer(inputs)
        elif init_engine == "wilson":
            part1 = torch.log(self.pcs / inputs[:, :1])
            part2 = 5.373 * (1 + self.omegas) * (1 - self.tcs / inputs[:, 1:2])
            lnki = part1 + part2
        else:
            raise ValueError(f"wilson or nn, but gotten {init_engine}.")

        if log:
            return lnki
        else:
            return lnki.exp()

    def __call__(
        self,
        inputs,
        ki=None,
        init_engine="wilson",
        initializer=None,
        classifier=None,
        threshold=(0.05, 0.95),
        sa_max_nit1=9,
        sa_max_nit2=40,
        split_max_nit1=9,
        split_max_nit2=40,
        async_analysis=False,
        debug=False,
    ):
        """
        Run complete flash calculation for vapour-liquid equilibrium

        Parameters
        ----------
        inputs: array_like
            The first column is the pressure, the second is the temperature, and the others
            are the composition.

        ki: array_like, default=None
            Initial distribution coefficients. If not passed, call `init_ki` to initialize it.

        init_engine: str, default="wilson"
            The method for initializing distribution coefficients, which is "nn" or "wilson".

        initializer: module of pytorch, default=None
            A neural network used to initialize distribution coefficients ki.

        classifier: module of pytorch, default=None
            A neural network used to predict the stability of given mixtures.
            Note that the output of `classifier` should be logit and use sigmoid to get probabilities.

        threshold: tuple of floats, default=(0.05, 0.95)
            (pl, pr), thresholds for instability and stability. If p >= pr, the mixture is predicted as
            stable. If p <= pl, the mixture is unstable. If pl < p < pr, run stability analysis.

        sa_max_nit1: int, default=15
            Max number of iterations of successive substitution of stability analysis.

        sa_max_nit2: int, default=40
            Max number of iterations of second-order method of stability analysis.

        split_max_nit1: int, default=15
            Max number of iterations of successive substitution of phase split calculations.

        split_max_nit2: int, default=40
            Max number of iterations of second-order method of phase split calculations.

        async_analysis: bool, default=False
            If True, run stability analysis using vapour and liquid-like estimates in parallel.
            If False, run them in sequence.

        debug: bool, default=False
            If True, print the process results for debugging.
        """

        # if classifier is given, use it to predict stability
        if classifier is not None:
            classifier.eval()
            classifier = classifier.to(self.device)
            if debug:
                start = default_timer()
            with torch.no_grad():
                prob = classifier(inputs).sigmoid()
            stable = (prob >= threshold[1]).view(-1)

            if debug:
                time = default_timer() - start
                print(f"{stable.sum()} of {inputs.shape[0]} are predicted as stable.")
                print(f"Time: {time:.4f}s. \n")

            prob = prob[~stable]
            inputs = inputs[~stable]
            undetermined = (prob > threshold[0]).view(-1)
            sa_indices = torch.where(undetermined)[0]
            sa_inputs = inputs[sa_indices]
            lnphis = None

        # initialize ki if necessary
        if ki is None:
            if debug:
                start = default_timer()
            ki = self.init_ki(inputs, init_engine=init_engine, initializer=initializer)
            if debug:
                time = default_timer() - start
                print(f"Initialize ki with {init_engine} for {inputs.shape[0]}")
                print(f"Time: {time:.4f}s. \n")

        # if classifier is not given, first try `negative_tpd_detector`
        if classifier is None:
            if debug:
                start = default_timer()
            lnphis = self.flasher.eos(inputs)[0]
            stable, ki = negative_tpd_detector(
                self.flasher, inputs, ki, lnphis, max_nit=3
            )
            sa_indices = torch.where(stable)[0]
            sa_inputs = inputs[sa_indices]
            if debug:
                time = default_timer() - start
                ratio = ((~stable).sum() / inputs.shape[0]) * 100
                print(f"tpd < 0 for {ratio:.4f}% of {inputs.shape[0]}")
                print("Time: {:.4f}s. \n".format(time))

        # extract samples used to run stability analysis
        if sa_indices.numel() > 0:
            sa_ki = ki[sa_indices]
            if lnphis is None:
                sa_lnphis = None
            else:
                sa_lnphis = lnphis[sa_indices]

            # stability analysis
            if async_analysis:
                # run vapour and liquid-like estimates in parallel
                if debug:
                    start = default_timer()
                vapour_res, liquid_res = async_analyser(
                    self.flasher,
                    sa_inputs,
                    sa_ki,
                    sa_lnphis,
                    sa_max_nit1=sa_max_nit1,
                    sa_max_nit2=sa_max_nit2,
                    debug=debug,
                )
                if debug:
                    time = default_timer() - start
                    print(f"Asynchronous stability analysis {time:.4f}s. \n")
                vapour_stable, vapour_wi, vapour_tm = vapour_res
                liquid_stable, liquid_wi, liquid_tm = liquid_res
            else:
                # run vapour and liquid-like estimates in sequence
                if debug:
                    start = default_timer()
                vapour_stable, vapour_wi, vapour_tm = analyser(
                    self.flasher,
                    sa_inputs,
                    sa_ki,
                    sa_lnphis,
                    vapour_like=True,
                    sa_max_nit1=sa_max_nit1,
                    sa_max_nit2=sa_max_nit2,
                    debug=debug,
                )
                if debug:
                    time = default_timer() - start
                    print(
                        f"Stability analysis with vapour-like estimates: {time:.4f}s \n"
                    )

                if debug:
                    start = default_timer()
                liquid_stable, liquid_wi, liquid_tm = analyser(
                    self.flasher,
                    sa_inputs,
                    sa_ki,
                    sa_lnphis,
                    vapour_like=False,
                    sa_max_nit1=sa_max_nit1,
                    sa_max_nit2=sa_max_nit2,
                    debug=debug,
                )
                if debug:
                    time = default_timer() - start
                    print(
                        f"Stability analysis with liquid-like estimates: {time:.4f}s \n"
                    )

            # merge the results of stability analysis with vapour-like and liquid-like estimates
            sa_stable = (vapour_stable & liquid_stable).view(-1)
            sa_unstable = ~sa_stable
            # reinitialize ki based on stability analysis
            # if liquid_tm < vapour_tm, then ki = zi / liquid_wi
            # if liquid_tm > vapour_tm, then ki = vapour_wi / zi
            cond = liquid_tm[sa_unstable] < vapour_tm[sa_unstable]
            sa_inputs2 = sa_inputs[sa_unstable]
            liquid_ki = sa_inputs2[:, 2:] / liquid_wi[sa_unstable]
            vapour_ki = vapour_wi[sa_unstable] / sa_inputs2[:, 2:]
            estimated_ki = torch.where(cond.unsqueeze(-1), liquid_ki, vapour_ki)
            ki[sa_indices[sa_unstable]] = estimated_ki

            if classifier is None:
                stable[sa_indices[sa_unstable]] = False
                unstable = ~stable
                inputs = inputs[unstable]
                ki = ki[unstable]
            else:
                indices = torch.where(~stable)[0]
                stable[indices[sa_indices[sa_stable]]] = True
                unstable = ~stable
                mask = torch.ones(inputs.shape[0], dtype=torch.bool)
                mask[sa_indices[sa_stable]] = False
                inputs = inputs[mask]
                ki = ki[mask]
        else:
            if classifier is None:
                inputs = inputs[~stable]
                ki = ki[~stable]
            unstable = ~stable

        # Phase split calculations
        if lnphis is not None:
            lnphis = lnphis[unstable]
        theta_v, xi, yi, ki = splitter(
            self.flasher, inputs, ki, lnphis, split_max_nit1, split_max_nit2, debug=debug
        )

        return unstable, theta_v, xi, yi, ki

class SPTVLE:
    """
    Vapour-liquid equilibrium based on PyTorch

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

        self.sa_max_nit1 = 9
        self.sa_max_nit2 = 40
        self.split_max_nit1 = 9
        self.split_max_nit2 = 40

        self.pcs = pcs.to(dtype=dtype, device=device)
        self.tcs = tcs.to(dtype=dtype, device=device)
        self.omegas = omegas.to(dtype=dtype, device=device)
        self.flasher = PTFlash(
            pcs, tcs, omegas, kij, kijt, kijt2, cubic_solver, dtype, device
        )
        self.dtype = dtype
        self.device = device

    def init_ki(self, inputs):
        """
        Initialize ki using:
            (1) Wilson approximation if `init_engine` is wilson
            (2) neural network (initializer) if `init_engine` is nn
        """

        part1 = torch.log(self.pcs / inputs[:, :1])
        part2 = 5.373 * (1 + self.omegas) * (1 - self.tcs / inputs[:, 1:2])
        lnki = part1 + part2

        return lnki.exp()

    def __call__(
        self,
        inputs
    ):
        """
        Run complete flash calculation for vapour-liquid equilibrium

        Parameters
        ----------
        inputs: array_like
            The first column is the pressure, the second is the temperature, and the others
            are the composition.

        ki: array_like, default=None
            Initial distribution coefficients. If not passed, call `init_ki` to initialize it.

        init_engine: str, default="wilson"
            The method for initializing distribution coefficients, which is "nn" or "wilson".

        initializer: module of pytorch, default=None
            A neural network used to initialize distribution coefficients ki.

        classifier: module of pytorch, default=None
            A neural network used to predict the stability of given mixtures.
            Note that the output of `classifier` should be logit and use sigmoid to get probabilities.

        threshold: tuple of floats, default=(0.05, 0.95)
            (pl, pr), thresholds for instability and stability. If p >= pr, the mixture is predicted as
            stable. If p <= pl, the mixture is unstable. If pl < p < pr, run stability analysis.

        sa_max_nit1: int, default=15
            Max number of iterations of successive substitution of stability analysis.

        sa_max_nit2: int, default=40
            Max number of iterations of second-order method of stability analysis.

        split_max_nit1: int, default=15
            Max number of iterations of successive substitution of phase split calculations.

        split_max_nit2: int, default=40
            Max number of iterations of second-order method of phase split calculations.

        async_analysis: bool, default=False
            If True, run stability analysis using vapour and liquid-like estimates in parallel.
            If False, run them in sequence.

        debug: bool, default=False
            If True, print the process results for debugging.
        """
        threshold = (0.05, 0.95)
        # if classifier is given, use it to predict stability

        # initialize ki if necessary
        ki = self.init_ki(inputs)

        # if classifier is not given, first try `negative_tpd_detector`
        lnphis = self.flasher.eos(inputs)[0]

        stable, ki = negative_tpd_detector(
            self.flasher, inputs, ki, lnphis, max_nit=3
        )
        #print("STABLE",stable)
        sa_indices = torch.where(stable)[0]
        #print("INDICES NUMEL",sa_indices.numel())
        sa_inputs = inputs[sa_indices]

        # extract samples used to run stability analysis
        if sa_indices.numel() > 0:
            sa_ki = ki[sa_indices]
            if lnphis is None:
                sa_lnphis = None
            else:
                sa_lnphis = lnphis[sa_indices]

            # stability analysis
            # run vapour and liquid-like estimates in sequence
            vapour_stable, vapour_wi, vapour_tm = analyser(
                self.flasher,
                sa_inputs,
                sa_ki,
                sa_lnphis,
                vapour_like=True,
                sa_max_nit1=self.sa_max_nit1,
                sa_max_nit2=self.sa_max_nit2,
                debug=False,
            )
            liquid_stable, liquid_wi, liquid_tm = analyser(
                self.flasher,
                sa_inputs,
                sa_ki,
                sa_lnphis,
                vapour_like=False,
                sa_max_nit1=self.sa_max_nit1,
                sa_max_nit2=self.sa_max_nit2
            )

            # merge the results of stability analysis with vapour-like and liquid-like estimates
            sa_stable = (vapour_stable & liquid_stable).view(-1)
            sa_unstable = ~sa_stable
            # reinitialize ki based on stability analysis
            # if liquid_tm < vapour_tm, then ki = zi / liquid_wi
            # if liquid_tm > vapour_tm, then ki = vapour_wi / zi
            cond = liquid_tm[sa_unstable] < vapour_tm[sa_unstable]
            sa_inputs2 = sa_inputs[sa_unstable]
            liquid_ki = sa_inputs2[:, 2:] / liquid_wi[sa_unstable]
            vapour_ki = vapour_wi[sa_unstable] / sa_inputs2[:, 2:]
            estimated_ki = torch.where(cond.unsqueeze(-1), liquid_ki, vapour_ki)
            ki[sa_indices[sa_unstable]] = estimated_ki

            stable[sa_indices[sa_unstable]] = False
            unstable = ~stable
            inputs = inputs[unstable]
            ki = ki[unstable]
            #print("UNSTABLE",unstable)
        else:
            inputs = inputs[~stable]
            ki = ki[~stable]
            unstable = ~stable
            #print("UNSTABLE",unstable)

        # Phase split calculations
        if lnphis is not None:
            lnphis = lnphis[unstable]
        theta_v, xi, yi, ki = splitter(
            self.flasher, inputs, ki, lnphis, self.split_max_nit1, self.split_max_nit2, debug=False
        )

        return unstable, theta_v, xi, yi, ki
