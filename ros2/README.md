# beacon_tracker ROS2 package

Wraps the position estimators in `sptrack/estimators/` as a ROS2 node:
subscribes to a camera image topic, publishes sub-pixel position,
validity, and flux. See `beacon_tracker/beacon_tracker_node.py` for the
design reasoning.

## Status

This package has not been built or run against a real ROS2 installation.
The development machine used for the rest of this repository has no C++
toolchain, no CMake, no Docker, and no WSL, so nothing in `cpp/` or
`ros2/` has been compiled or executed here. Every claim below about how
to build and run it is a set of instructions to follow on a machine that
does have ROS2 installed, not a report of having done so.

What has been checked here: `beacon_tracker_node.py`, `setup.py` and
`package.xml` all parse without error, and the estimator functions the
node calls (`centroid_estimate`, `gaussian_fit_estimate`,
`matched_filter_estimate`) are the same functions covered by the 136
tests in `tests/`. What has not been checked: that the node builds under
`colcon`, that the topic wiring matches a real camera driver's message
types, and that it behaves correctly against live data.

## Prerequisites

A ROS2 distribution with `rclpy`, `sensor_msgs`, `geometry_msgs`, and
`std_msgs`. Developed against ROS2 Jazzy on Ubuntu 24.04, matching the
VirtualBox environment used to validate the rest of this repository.

```bash
sudo apt update
sudo apt install ros-jazzy-desktop python3-colcon-common-extensions
source /opt/ros/jazzy/setup.bash
```

`sptrack` itself must be importable from the node's Python interpreter:

```bash
cd /path/to/sub-pixel-tracker
pip install -e .
```

This repository does not currently have a `pyproject.toml` or `setup.py`
at its root, so `pip install -e .` will not work until one is added. In
the meantime, put the repository root on `PYTHONPATH` before launching
the node:

```bash
export PYTHONPATH="/path/to/sub-pixel-tracker:$PYTHONPATH"
```

## Build

```bash
mkdir -p ~/ros2_ws/src
ln -s /path/to/sub-pixel-tracker/ros2/beacon_tracker ~/ros2_ws/src/beacon_tracker
cd ~/ros2_ws
colcon build --packages-select beacon_tracker
source install/setup.bash
```

## Run

```bash
ros2 run beacon_tracker beacon_tracker_node \
  --ros-args \
  -p estimator:=gaussian_fit \
  -p half_width:=9 \
  -p sigma_px:=1.75 \
  -p image_topic:=/camera/image_raw
```

Published topics:

- `~/position` (`geometry_msgs/PointStamped`): x, y in pixels, NaN when
  the estimate is not valid.
- `~/valid` (`std_msgs/Bool`): whether the estimate should be trusted.
- `~/flux` (`std_msgs/Float64`): the estimator's own background-
  subtracted flux, zero when invalid.

## A known limitation carried over from the estimator itself

`experiments/exp05d_low_photon_count.py` found that the centroid's
`ok` flag is a formal check, not a quality check: it reports `ok=True`
even when the measurement is dominated by noise. A downstream consumer
of this node should not treat `~/valid` alone as sufficient when
`estimator:=centroid` is selected; `~/flux` should be checked against an
expected range as well. The Gaussian fit and matched filter do not have
this problem, since both report `ok=False` on a genuine failure to
converge or lock onto a peak.

## What is intentionally not in this package

Gimbal actuation, image acquisition from a specific camera driver, and
any filtering of the published position. Temporal filtering is
implemented in `sptrack/tracking.py` and characterised in
`experiments/exp07_kalman_tracking.py`, which found it does not help at
the SNR this project's default operating point assumes and only becomes
useful once measurement noise approaches per-frame target motion. If
filtering is wanted for a specific deployment, it belongs in a separate
node downstream of this one, tuned to that deployment's actual SNR, not
folded into the estimator node.
