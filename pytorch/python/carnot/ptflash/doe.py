import pandas as pd
from pyDOE2 import lhs
from scipy.stats.distributions import gamma


class DOE:
    """
    Generate Design of Experiments (DOE) using Latin Hypercube Sampling (LHS) algorithm.

    Parameters
    ----------
    n_samples: int
        Number of samples

    n_components: int
        Number of components

    doe_type: str, default=dirichlet
        linear or dirichlet

    pmin: float, default=1e6

    pmax: float, default=1e8

    tmin: float, default=200

    tmax: float, default=500

    alphas: array_like, default=ones(n_components)
        The concentration parameters of Dirichlet distribution.

    random_state: int

    lhs_kwargs: dict, default=None
        Other keyword arguments of pyDOE2.lhs except for `n` and `samples`.
    """

    def __init__(
        self,
        n_samples,
        n_components,
        pmin=1e6,
        pmax=1e8,
        tmin=200,
        tmax=500,
        doe_type="linear",
        alphas=None,
        random_state=None,
        lhs_kwargs=None,
    ):
        self.n_samples = n_samples
        self.n_components = n_components
        self.pmin = pmin
        self.pmax = pmax
        self.tmin = tmin
        self.tmax = tmax
        self.doe_type = doe_type
        self.alphas = [1] * self.n_components if alphas is None else alphas
        self.random_state = random_state
        self.lhs_kwargs = lhs_kwargs or {}

    def create_design(self, pt=True):
        """
        Create DOE

        Parameters
        ----------

        pt: bool, default=True
            If True, pressure and temperature are included.
        """

        zi_cols = [f"Z{i}" for i in range(1, self.n_components + 1)]
        if pt:
            n = self.n_components + 2
            cols = ["P", "T"] + zi_cols
        else:
            n = self.n_components
            cols = zi_cols

        design = pd.DataFrame(
            data=lhs(
                n=n,
                samples=self.n_samples,
                random_state=self.random_state,
                **self.lhs_kwargs,
            ),
            columns=cols,
        )

        if pt:
            design["P"] = self.pmin + (self.pmax - self.pmin) * design["P"]
            design["T"] = self.tmin + (self.tmax - self.tmin) * design["T"]

        if self.doe_type == "dirichlet":
            for i, alpha in enumerate(self.alphas):
                design[f"Z{i+1}"] = gamma(a=alpha, scale=1).ppf(design[f"Z{i+1}"])

        sums = design.loc[:, zi_cols].sum(axis=1)
        design.loc[:, zi_cols] = design.loc[:, zi_cols].divide(sums, axis=0)
        return design
