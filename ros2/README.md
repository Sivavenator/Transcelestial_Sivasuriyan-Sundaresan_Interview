# beacon_tracker ROS2 package

Wraps the position estimators in `sptrack/estimators/` as a ROS2 node:
subscribes to a camera image topic, publishes sub-pixel position,
validity, and flux. See `beacon_tracker/beacon_tracker_node.py` for the
design reasoning.

## Status

Built and run on an Ubuntu 24.04 VirtualBox VM with ROS2 Jazzy
(`ros-jazzy-ros-base`). `colcon build --packages-select beacon_tracker`
succeeds and `ros2 run beacon_tracker beacon_tracker_node` starts
cleanly, logging `beacon_tracker up: estimator=centroid, half_width=9,
sigma=1.75 px, topic=/camera/image_raw` and waiting on the image topic.
The development machine used for the rest of this repository still has
no ROS2, C++ toolchain, CMake, Docker, or WSL, so this verification was
done entirely on the VM.

Two real packaging bugs found and fixed only once an actual build was
possible, neither visible from reading the code:

- No `setup.cfg`. `ament_python` packages need one pointing
  `install_scripts` at `$base/lib/beacon_tracker`; without it, setuptools
  used its own default (`$base/bin`), so the built executable existed
  but `ros2 run` could not find it. This is boilerplate
  `ros2 pkg create --build-type ament_python` normally generates, and
  was simply missing since this package was hand-written.
- `package.xml` declared `python3-numpy` but not `python3-scipy`, even
  though `sptrack/psf.py` needs `scipy.special.erf`. `ros2 run` uses
  ROS2's system Python, not the project's own `.venv`, so the venv
  having scipy installed did not help; the node failed at import time
  with `ModuleNotFoundError: No module named 'scipy'` until
  `python3-scipy` was installed system-wide and declared as a dependency.

Building from a symlink into the shared VirtualBox folder
(`/media/sf_...`) also failed with `Operation not permitted` on an
internal ament_python development-mode symlink, the same `vboxsf`
limitation that affected the Python `venv` earlier in this project.
Fixed by copying the package into the ROS2 workspace instead of
symlinking it.

What has not been checked: the topic wiring against a real camera
driver's message types, and behaviour against live (non-synthetic)
data. No image publisher exists in this project to feed the node,
so it has only been confirmed to start, resolve its dependencies, and
subscribe correctly, not to produce a correct position estimate from a
real frame.

## Prerequisites

A ROS2 distribution with `rclpy`, `sensor_msgs`, `geometry_msgs`, and
`std_msgs`. Verified against ROS2 Jazzy on Ubuntu 24.04 in a VirtualBox
VM. The ROS2 apt repository is not preconfigured on a stock Ubuntu
24.04 image and must be added first:

```bash
sudo apt install -y software-properties-common curl
sudo add-apt-repository universe -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-jazzy-ros-base python3-colcon-common-extensions python3-scipy
source /opt/ros/jazzy/setup.bash
```

`ros-jazzy-ros-base` rather than `ros-jazzy-desktop`: this node has no
GUI dependency, and base is a much smaller install. `python3-scipy` is
required at the system level, not just in the project's own `.venv`,
because `ros2 run` uses ROS2's system Python interpreter (see the
scipy bug in Status above).

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

If `/path/to/sub-pixel-tracker` is on a VirtualBox shared folder
(`vboxsf`), copy the package into the workspace rather than symlinking
it; `vboxsf` does not support the symlinks `ament_python`'s build step
creates internally, and the failure is silent (`colcon build` reports
success even though no executable gets installed):

```bash
mkdir -p ~/ros2_ws/src
cp -r /path/to/sub-pixel-tracker/ros2/beacon_tracker ~/ros2_ws/src/beacon_tracker
cd ~/ros2_ws
colcon build --packages-select beacon_tracker
source install/setup.bash
```

On a native (non-shared-folder) filesystem, a symlink instead of a copy
works fine.

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
