import numpy as np
import pytest

from sptrack.psf import render_spot


def _centroid(img: np.ndarray) -> tuple[float, float]:
    h, w = img.shape
    xs = np.arange(w)
    ys = np.arange(h)
    total = img.sum()
    cx = (img.sum(axis=0) * xs).sum() / total
    cy = (img.sum(axis=1) * ys).sum() / total
    return cx, cy


def test_render_spot_conserves_flux():
    img = render_spot((25, 25), x0=12.3, y0=11.7, flux=1000.0, sigma=1.75)
    # A 25x25 window around a sigma=1.75 spot encloses it almost completely.
    assert img.sum() == pytest.approx(1000.0, rel=1e-6)


def test_render_spot_centroid_matches_injected_position():
    for x0, y0 in [(12.0, 12.0), (12.3, 11.7), (11.5, 12.5), (10.05, 13.95)]:
        img = render_spot((25, 25), x0=x0, y0=y0, flux=1000.0, sigma=1.75)
        cx, cy = _centroid(img)
        # 1e-7 is still four orders of magnitude tighter than the pixel-locking
        # bias (~1e-3 to 1e-1 px) this test would actually catch; the residual
        # here is float64 summation roundoff over a 625-pixel grid, not signal.
        assert cx == pytest.approx(x0, abs=1e-7)
        assert cy == pytest.approx(y0, abs=1e-7)
