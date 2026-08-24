#!/usr/bin/env python3
"""ROS2 node wrapping the spot estimator for a pointing control loop.

WHAT THIS NODE IS FOR
---------------------
It is the integration boundary between the estimator characterised in
this repository and a real pointing system: camera frames arrive on a
topic, sub-pixel positions leave on a topic, and a gimbal controller
elsewhere consumes them. Nothing in the estimation is new here. The point
of the node is to expose the parts a system integrator has to configure,
and to publish the diagnostics that make a failure legible from outside.

WHY THE ESTIMATOR IS SELECTABLE AT RUNTIME
------------------------------------------
experiments/exp02_realtime.py and experiments/exp01_snr_characterization.py
measure a real accuracy against cost tradeoff: the Gaussian fit is the
most accurate (efficiency 0.95) but its per-frame cost is data-dependent
and its worst observed frame exceeded the 1 ms budget. The matched filter
gives up some accuracy (0.84) for fixed, bounded cost. Which of those is
correct depends on the deployment, so the choice is a parameter rather
than a hard-coded import.

WHY ok=False IS PUBLISHED RATHER THAN DROPPED
---------------------------------------------
A controller needs to distinguish "the spot is here" from "no usable
measurement this frame". Silently skipping a failed frame looks identical
downstream to a frame that never arrived, and the two call for different
responses: one is a dropout to coast through, the other is a broken
pipeline. The node therefore publishes a validity flag every frame.

A caveat on the failure flag, from experiments/exp05d_low_photon_count.py:
the centroid's ok flag is a formal check (positive background-subtracted
flux) rather than a quality check, and it stays true even at 0.4 photons
where its answers are noise. Consumers should gate on the published flux
as well, not on validity alone.

WHY THE PRIOR IS HELD ACROSS FRAMES
-----------------------------------
The estimator is seeded from the previous frame's own output, never from
ground truth, matching sptrack/sequence.py. A failed frame does not
update the prior, so the next frame is still seeded from the last
known-good position. That is what lets the tracker coast through the
isolated dropouts measured in experiments/exp04a_scintillation.py rather
than losing lock on the first bad frame.

STATUS
------
This node has not been run against a live ROS2 installation. It is
written against the rclpy API and the message types named below, and the
estimation path it calls is the one covered by the repository's test
suite, but the ROS integration itself is unverified. See docs/DEPLOYMENT.md.
"""

from __future__ import annotations

import numpy as np

try:
    import rclpy
    from geometry_msgs.msg import PointStamped
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool, Float64
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "rclpy and the standard ROS2 message packages are required to run this node. "
        "Source a ROS2 environment first, for example:  source /opt/ros/jazzy/setup.bash"
    ) from exc

from sptrack.estimators.centroid import centroid_estimate
from sptrack.estimators.gaussian_fit import gaussian_fit_estimate
from sptrack.estimators.matched_filter import matched_filter_estimate


class BeaconTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("beacon_tracker")

        self.declare_parameter("estimator", "gaussian_fit")
        self.declare_parameter("half_width", 9)
        self.declare_parameter("sigma_px", 1.75)
        self.declare_parameter("read_noise_e", 5.0)
        self.declare_parameter("gain_e_per_dn", 10.0)
        self.declare_parameter("black_level_dn", 100.0)
        self.declare_parameter("image_topic", "/camera/image_raw")

        self.estimator_name = str(self.get_parameter("estimator").value)
        self.half_width = int(self.get_parameter("half_width").value)
        self.sigma_px = float(self.get_parameter("sigma_px").value)
        self.read_var = float(self.get_parameter("read_noise_e").value) ** 2
        self.gain = float(self.get_parameter("gain_e_per_dn").value)
        self.black_level = float(self.get_parameter("black_level_dn").value)

        if self.estimator_name not in ("centroid", "gaussian_fit", "matched_filter"):
            raise ValueError(
                f"unknown estimator {self.estimator_name!r}, "
                "expected centroid, gaussian_fit or matched_filter"
            )

        self._prior: tuple[float, float] | None = None
        self._frames = 0
        self._failures = 0

        topic = str(self.get_parameter("image_topic").value)
        # Depth 1 with the default reliable QoS: for a 1 kHz control loop a
        # stale frame is worse than a dropped one, so the queue is kept
        # short deliberately rather than buffering frames the controller
        # can no longer act on.
        self.create_subscription(Image, topic, self.on_image, 1)

        self.pub_position = self.create_publisher(PointStamped, "~/position", 1)
        self.pub_valid = self.create_publisher(Bool, "~/valid", 1)
        self.pub_flux = self.create_publisher(Float64, "~/flux", 1)

        self.get_logger().info(
            f"beacon_tracker up: estimator={self.estimator_name}, "
            f"half_width={self.half_width}, sigma={self.sigma_px} px, topic={topic}"
        )

    def _to_electrons(self, msg: Image) -> np.ndarray:
        """Decode the image and convert to electrons.

        The estimators' noise weighting is defined in electrons: the fit
        weights each pixel by 1/(model + read_variance), and read_variance
        is in electrons squared. Passing raw digital numbers through
        unconverted would put the signal and the variance term on
        different scales and mis-weight every pixel.
        """
        if msg.encoding == "mono16":
            raw = np.frombuffer(msg.data, dtype=np.uint16)
        elif msg.encoding == "mono8":
            raw = np.frombuffer(msg.data, dtype=np.uint8)
        else:
            raise ValueError(
                f"unsupported encoding {msg.encoding!r}, expected mono8 or mono16"
            )
        frame = raw.reshape(msg.height, msg.width).astype(np.float64)
        return (frame - self.black_level) * self.gain

    def on_image(self, msg: Image) -> None:
        try:
            frame = self._to_electrons(msg)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return

        prior = self._prior
        if prior is None:
            # No prior on the first frame. The estimators fall back to the
            # brightest pixel, which is vulnerable to a bright false
            # source: see experiments/exp04d_clutter.py. A deployment with
            # clutter in the field of view should acquire with
            # sptrack.acquisition.acquire_target instead.
            prior = None

        if self.estimator_name == "centroid":
            est = centroid_estimate(frame, self.half_width, prior=prior)
        elif self.estimator_name == "matched_filter":
            est = matched_filter_estimate(frame, self.half_width, self.sigma_px, prior=prior)
        else:
            est = gaussian_fit_estimate(
                frame, self.half_width, self.sigma_px, self.read_var, prior=prior
            )

        self._frames += 1
        if est.ok:
            # Only a good fit updates the prior. A failed frame leaves the
            # last known-good position in place so one bad frame cannot
            # drag the search window off the spot.
            self._prior = (est.x, est.y)
        else:
            self._failures += 1

        point = PointStamped()
        point.header = msg.header
        point.point.x = float(est.x) if est.ok else float("nan")
        point.point.y = float(est.y) if est.ok else float("nan")
        point.point.z = 0.0
        self.pub_position.publish(point)

        valid = Bool()
        valid.data = bool(est.ok)
        self.pub_valid.publish(valid)

        flux = Float64()
        flux.data = float(est.flux) if est.ok else 0.0
        self.pub_flux.publish(flux)

        if self._frames % 1000 == 0:
            rate = 100.0 * self._failures / self._frames
            self.get_logger().info(
                f"{self._frames} frames, {self._failures} failed ({rate:.2f}%)"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BeaconTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
