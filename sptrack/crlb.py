"""The Cramer-Rao Lower Bound: the theoretical precision floor no unbiased
estimator can beat, for this exact forward model.

WHY THIS REUSES gaussian_fit.py'S JACOBIAN, NOT A SEPARATE DERIVATION
---------------------------------------------------------------------------
Three things need to agree with each other, or the whole characterisation
that follows is meaningless: the SIMULATOR that renders frames, the
ESTIMATOR that fits a model to them, and the BOUND that says how well any
estimator could possibly do. If the bound were derived from a different
model than the one the fit actually uses, "the fit attains the bound" or
"the fit falls short of the bound" would both be comparing against a
fiction, not a real limit. So this module computes the bound from the
SAME Jacobian (`psf.pixel_response_1d_with_derivative`) and the SAME
Gaussian-approximated-Poisson variance model (`mu + read_var`) that
`gaussian_fit.py` already uses -- structurally, not just by coincidence.

THE MATH: FISHER INFORMATION, THEN INVERT
----------------------------------------------
For a set of independent, approximately-Gaussian-distributed pixel
measurements (variance var_i, mean mu_i(theta)), the Fisher Information
Matrix for the parameter vector theta = (x0, y0, flux, bg) is:

    FIM[j, k] = sum_i  ( d(mu_i)/d(theta_j) * d(mu_i)/d(theta_k) ) / var_i

This is EXACTLY the Gauss-Newton Hessian approximation already computed
inside `gaussian_fit_estimate` at every iteration (`jac.T @ (weight * jac)`)
-- which is not a coincidence, it is the whole point of Gauss-Newton: it IS
an approximation to the Fisher information, which is why a maximum-
likelihood fit's own Hessian at convergence is routinely used to report its
uncertainty in real systems.

The Cramer-Rao bound states that any UNBIASED estimator's covariance
matrix satisfies Cov >= inverse(FIM) (in the positive-semidefinite sense).
So the bound on x0's variance is simply the (0, 0) entry of inverse(FIM):

    Var[x0_hat] >= inverse(FIM)[0, 0]      for any unbiased estimator

WHY EVALUATED AT THE TRUE PARAMETERS, NOT AN ESTIMATE
-----------------------------------------------------------
The bound is a property of the MEASUREMENT -- how much information a frame
with this true position, flux, and noise budget actually contains -- not a
property of any particular estimator's guess. Evaluating the Jacobian and
variance at the true (known, since this is a simulation) parameters gives
the bound that describes the experiment itself, independent of which
estimator is later compared against it.

PIXELATION MAKES THIS BOUND SLIGHTLY LOOSER THAN THE CLASSICAL FORMULA --
CHECKED, NOT ASSUMED
------------------------------------------------------------------------------
A well-known closed-form approximation exists in the simplest limit
(continuous, non-pixelated sampling, no background, no read noise):
sigma_x ~= sigma_PSF / sqrt(N) (e.g. Thompson et al. 2002). Because this
bound is built from the PIXEL-INTEGRATED response (the same one that
renders and fits every spot in this project), it sits slightly above that
classical value -- verified directly by sweeping sigma at fixed flux: the
gap is 15.1% at sigma=0.5 px, 1.35% at sigma=1.75 px (this project's
nominal spot width), and 0.04% at sigma=10 px, shrinking monotonically as
sigma grows relative to the fixed 1-pixel sampling pitch. That is exactly
the signature of a genuine pixelation effect converging to the continuous-
sampling limit -- not an implementation bug -- and matches known results in
the localisation-microscopy literature (e.g. Mortensen et al. 2010's
pixel-size correction term).
"""

from __future__ import annotations

import numpy as np

from .psf import pixel_response_1d_with_derivative


def position_crlb(
    shape: tuple[int, int],
    x0: float,
    y0: float,
    flux: float,
    bg: float,
    sigma: float,
    read_var_e2: float = 0.0,
) -> tuple[float, float]:
    """Cramer-Rao lower bound on position standard deviation, (std_x, std_y).

    ``shape`` is the fitting window's size -- the bound depends on how many
    pixels' worth of information the estimator actually gets to use, the
    same as any real estimator's window would.
    """
    h, w = shape
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)

    px, dpx = pixel_response_1d_with_derivative(xs, x0, sigma)
    py, dpy = pixel_response_1d_with_derivative(ys, y0, sigma)
    outer_response = np.outer(py, px)
    mu = flux * outer_response + bg

    var = np.maximum(mu, 1e-6) + read_var_e2
    weight = 1.0 / var

    jac = np.empty((h, w, 4))
    jac[:, :, 0] = flux * np.outer(py, dpx)
    jac[:, :, 1] = flux * np.outer(dpy, px)
    jac[:, :, 2] = outer_response
    jac[:, :, 3] = 1.0

    jac_flat = jac.reshape(-1, 4)
    w_flat = weight.ravel()
    fisher = jac_flat.T @ (w_flat[:, None] * jac_flat)

    cov = np.linalg.inv(fisher)
    std_x = float(np.sqrt(cov[0, 0]))
    std_y = float(np.sqrt(cov[1, 1]))
    return std_x, std_y
