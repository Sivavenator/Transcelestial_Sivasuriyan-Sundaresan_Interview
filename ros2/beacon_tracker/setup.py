from setuptools import setup

package_name = "beacon_tracker"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    package_dir={package_name: "."},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="sub-pixel-tracker",
    maintainer_email="slr007suriya@gmail.com",
    description="ROS2 wrapper around the sub-pixel-tracker position estimators.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "beacon_tracker_node = beacon_tracker.beacon_tracker_node:main",
        ],
    },
)
