from setuptools import find_namespace_packages, setup

setup(
    name="rikai-ros",
    version="0.0.1",
    license="Apache License, Version 2.0",
    author="Rikai authors",
    author_email="rikai-dev@eto.ai",
    url="https://github.com/eto-ai/rikai",
    python_requires=">=3.7",
    install_requires=[
        "rikai",
        "numpy",
        "rosbag",
    ],
    extras_require={
        "dev": [
            "pytest",
            "isort",
            "black",
            "sensor-msgs",
            "diagnostic-msgs",
            "trajectory-msgs",
            "tf2-msgs",
        ]
    },
    packages=find_namespace_packages(include=["rikai.*"]),
)
