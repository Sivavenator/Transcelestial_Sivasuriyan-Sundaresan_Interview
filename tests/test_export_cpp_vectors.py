"""Regression test for tools/export_cpp_vectors.py.

The first version of the exporter used repr() directly on numpy scalars.
Under numpy 2 that produces "np.float64(1.23)" instead of "1.23", which
is not a number any C++ parser accepts. The bug was silent in Python:
csv.writer happily wrote the string, and nothing failed until a C++
compiler tried to read it. This test reads the actual generated file
rather than testing the fix in isolation, so it would have caught the
original bug.
"""

from __future__ import annotations

import csv

from tools.export_cpp_vectors import run


def test_generated_csv_contains_no_numpy_repr_leakage(tmp_path, monkeypatch):
    import tools.export_cpp_vectors as mod

    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
    out = run()

    text = out.read_text()
    assert "np." not in text
    assert "float64" not in text


def test_generated_csv_fields_parse_as_plain_floats(tmp_path, monkeypatch):
    import tools.export_cpp_vectors as mod

    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
    out = run()

    with open(out, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert len(rows) > 0
    numeric_fields = ["sigma", "read_var", "prior_x", "prior_y", "true_x", "true_y",
                       "py_cx", "py_cy", "py_cflux", "py_cbg",
                       "py_gx", "py_gy", "py_gflux", "py_gbg"]
    for row in rows:
        for field in numeric_fields:
            float(row[field])  # raises ValueError if not a plain numeric literal
        pixels = [float(v) for v in row["pixels"].split()]
        assert len(pixels) == int(row["h"]) * int(row["w"])


def test_python_estimators_agree_with_themselves_on_every_exported_case(tmp_path, monkeypatch):
    """The vectors carry py_cx/py_cy and py_gx/py_gy as the values the
    C++ side must reproduce. This checks those recorded values are
    actually what the same estimators return on the same pixels, so a
    future refactor of the exporter cannot silently record the wrong
    numbers without a test catching it."""
    import numpy as np

    import tools.export_cpp_vectors as mod
    from sptrack.estimators.centroid import centroid_estimate
    from sptrack.estimators.gaussian_fit import gaussian_fit_estimate

    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
    out = run()

    with open(out, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    for row in rows:
        h, w = int(row["h"]), int(row["w"])
        pixels = np.array([float(v) for v in row["pixels"].split()], dtype=np.float64).reshape(h, w)
        sigma = float(row["sigma"])
        read_var = float(row["read_var"])
        half_width = int(row["half_width"])
        prior = (float(row["prior_x"]), float(row["prior_y"]))

        c = centroid_estimate(pixels, half_width, prior=prior)
        g = gaussian_fit_estimate(pixels, half_width, sigma, read_var, prior=prior)

        assert c.x == float(row["py_cx"])
        assert c.y == float(row["py_cy"])
        assert g.x == float(row["py_gx"])
        assert g.y == float(row["py_gy"])
