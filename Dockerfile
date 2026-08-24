# Reproducible build and validation environment for sub-pixel-tracker.
#
# Debian-based to match the Ubuntu 24.04 VirtualBox environment this
# project is otherwise validated against, and pinned to a specific Python
# so floating-point results are reproducible across machines for anything
# that is seeded (everything except the wall-clock timings in
# exp02_realtime and exp05e_latency_budget, which are Python timings on
# whatever machine runs them by construction).
#
#   docker build -t sptrack .
#   docker run --rm -v "$PWD/figures:/app/figures" -v "$PWD/results:/app/results" sptrack
#
# The volume mounts matter: the container's job is to produce artefacts,
# and artefacts that stay inside a stopped container are not
# deliverables.
#
# WHY THE C++ CROSS-VALIDATION RUNS INSIDE THE BUILD
# ----------------------------------------------------------
# cpp/tests/test_against_python.cpp checks the C++ estimators against
# vectors exported from the Python reference. Running that check as part
# of `docker build` means an image that builds successfully is an image
# where the two implementations are known to agree, not an image where
# they merely both compiled. If the check fails the build fails.
#
# STATUS
# ------
# This Dockerfile has not been built or run: the development machine used
# for the rest of this repository has no Docker installed. It is written
# to be run on a machine that has Docker, most directly inside the
# Ubuntu 24.04 VM already used to validate this project's dependencies.

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so this layer caches across source edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Matplotlib must not try to find a display.
ENV MPLBACKEND=Agg
# Single-threaded BLAS: the workload here is many small problems, not a
# few large ones, so threading adds contention and makes the wall-clock
# timings in exp02/exp05e noisier and less comparable across runs.
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Python side: the full test suite must pass before anything else runs.
RUN python -m pytest tests -q

# C++ side: regenerate the cross-validation vectors from the Python
# reference actually running in this image, then build and run the
# comparison. A stale committed CSV never gets a chance to hide a real
# regression, because it is regenerated fresh every build.
RUN python -m tools.export_cpp_vectors
RUN cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build cpp/build --parallel \
    && ctest --test-dir cpp/build --output-on-failure

CMD ["python", "run_all.py"]
