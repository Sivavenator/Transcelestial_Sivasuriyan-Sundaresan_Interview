"""2D Gaussian maximum-likelihood fit, via Poisson-weighted Gauss-Newton.

THE IDEA: FIT A MODEL, DON'T JUST WEIGH PIXELS
---------------------------------------------------
The centroid (centroid.py) is a weighted average -- simple, cheap, no model
of the spot's shape. This estimator instead fits an actual Gaussian model
to the window's pixel values and reads the position off the best-fit
model's centre. It uses exactly the same pixel-integrated response function
that RENDERS a spot (`psf.pixel_response_1d`) -- so the model being fit is
the same model doing the simulating, not an approximation of it. Fitting a
spot with a different model than the one that generated it would be testing
model mismatch, not the estimator.

WHY POISSON-WEIGHTED, NOT ORDINARY (UNWEIGHTED) LEAST SQUARES
-------------------------------------------------------------------
Ordinary least squares minimises sum((data - model)^2), implicitly treating
every pixel as equally noisy. That is wrong here: from the very first noise
source built in this project, brighter pixels have MORE absolute photon
noise than fainter ones (variance scales with signal). Weighting every
pixel equally means the fit gives a noisy bright pixel exactly as much say
as a much quieter faint one -- throwing away real information. Weighting
each pixel by the inverse of ITS OWN predicted variance,

    chi2 = sum( (data_i - model_i)^2 / var_i ),    var_i = model_i + read_var

makes this the (Gaussian-approximated) Poisson maximum-likelihood estimate
instead -- trusting each pixel exactly as much as its own statistics
justify, which is what lets this method approach the Cramer-Rao bound
(the theoretical precision floor, derived later in Characterization)
where the centroid cannot.

WHY THE WEIGHTS USE THE MODEL'S PREDICTION, NOT THE NOISY DATA
--------------------------------------------------------------------
It is tempting to set var_i from the observed pixel value instead of the
model's prediction -- but the data IS the noisy quantity being fit; using
it to also set how much to trust itself creates a feedback loop that
biases the fit (a pixel that happened to read low also gets weighted as if
it were expected to be low). Using the current MODEL prediction for the
variance breaks that loop. Since the model changes every iteration as the
fit refines, the weights are recomputed every iteration too -- this is an
iteratively reweighted least squares, converging to the same fixed point a
true Poisson MLE would.

WHY LEVENBERG-MARQUARDT, NOT PLAIN GAUSS-NEWTON
----------------------------------------------------
Gauss-Newton can overshoot or diverge when the current parameter guess is
still far from the optimum -- exactly the situation on early iterations.
Levenberg-Marquardt damps the step by blending toward gradient descent
(safer, slower) when a step doesn't improve chi2, and relaxes back toward
full Gauss-Newton (faster convergence) once it's working -- a standard,
well-tested way to get Gauss-Newton's speed without its fragility.

WHY SEEDED FROM THE CENTROID
---------------------------------
This is a local, iterative optimiser -- it refines a starting guess, it
does not search globally for one. The centroid estimator (already built,
already tested) is cheap and normally accurate to a fraction of a pixel,
making it a good, realistic starting point -- exactly what a real system
would do rather than starting from an arbitrary guess.

WHAT IS AND ISN'T FIT
--------------------------
Free parameters: x0, y0 (position), flux, and bg -- four parameters. sigma
is NOT fit; it is taken as given (the caller's assumed/calibrated PSF
width). This mirrors real systems, where the PSF width is typically
calibrated separately rather than re-estimated every frame, and it keeps
this estimator directly comparable to the centroid, which also assumes a
fixed window/shape rather than fitting one.
"""

from __future__ import annotations

import numpy as np

from ..psf import pixel_response_1d_with_derivative
from .base import Estimate, extract_window
from .centroid import centroid_estimate


def gaussian_fit_estimate(
    image: np.ndarray,
    half_width: int,
    sigma: float,
    read_var_e2: float = 0.0,
    prior: tuple[float, float] | None = None,
    max_iter: int = 20,
    tol_px: float = 1e-4,
) -> Estimate:
    """Estimate a spot's position via a Poisson-weighted 2-D Gaussian fit.

    ``sigma`` is the assumed (fixed, not fit) PSF width. ``read_var_e2`` is
    the read-noise variance added to the Poisson term when weighting each
    pixel -- pass the sensor's ``sigma_read_e**2`` for a realistic weighting
    scheme, or 0.0 to weight by photon statistics alone.
    """
    seed = centroid_estimate(image, half_width, prior=prior)
    if not seed.ok:
        return Estimate(float("nan"), float("nan"), ok=False)

    window, wx0, wy0 = extract_window(image, seed.x, seed.y, half_width)
    h, w = window.shape
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)

    # Parameters in the window's LOCAL coordinates; converted back to global
    # coordinates only in the returned Estimate.
    p = np.array(
        [seed.x - wx0, seed.y - wy0, max(seed.flux, 1.0), seed.bg], dtype=np.float64
    )

    lam = 1e-3
    converged = False
    for _ in range(max_iter):
        px, dpx = pixel_response_1d_with_derivative(xs, p[0], sigma)
        py, dpy = pixel_response_1d_with_derivative(ys, p[1], sigma)
        outer_response = np.outer(py, px)
        mu = p[2] * outer_response + p[3]

        resid = window - mu
        var = np.maximum(mu, 1e-6) + read_var_e2
        weight = 1.0 / var

        jac = np.empty((h, w, 4))
        jac[:, :, 0] = p[2] * np.outer(py, dpx)
        jac[:, :, 1] = p[2] * np.outer(dpy, px)
        jac[:, :, 2] = outer_response
        jac[:, :, 3] = 1.0

        jac_flat = jac.reshape(-1, 4)
        w_flat = weight.ravel()
        resid_flat = resid.ravel()

        grad = jac_flat.T @ (w_flat * resid_flat)
        hess = jac_flat.T @ (w_flat[:, None] * jac_flat)
        hess_damped = hess + lam * np.diag(np.diag(hess))

        try:
            step = np.linalg.solve(hess_damped, grad)
        except np.linalg.LinAlgError:
            return Estimate(float("nan"), float("nan"), ok=False)

        # Trust-region cap: never move the position by more than a pixel in
        # one step, even if the (locally linearised) solve suggests more --
        # the linearisation is least trustworthy exactly when it wants a
        # big jump.
        step[0] = np.clip(step[0], -1.0, 1.0)
        step[1] = np.clip(step[1], -1.0, 1.0)

        new_p = p.copy()
        new_p += step
        new_px, _ = pixel_response_1d_with_derivative(xs, new_p[0], sigma)
        new_py, _ = pixel_response_1d_with_derivative(ys, new_p[1], sigma)
        new_mu = new_p[2] * np.outer(new_py, new_px) + new_p[3]
        new_chi2 = float(np.sum((window - new_mu) ** 2 / np.maximum(new_mu, 1e-6)))
        old_chi2 = float(np.sum(resid_flat**2 * w_flat))

        if new_chi2 < old_chi2:
            p = new_p
            lam = max(lam * 0.4, 1e-7)
        else:
            lam *= 10.0

        if abs(step[0]) < tol_px and abs(step[1]) < tol_px:
            converged = True
            break

    return Estimate(x=p[0] + wx0, y=p[1] + wy0, flux=p[2], bg=p[3], ok=converged)
