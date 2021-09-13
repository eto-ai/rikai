import pathlib

from setuptools import find_namespace_packages, setup

setup(
    name="rikai-ros",
    version="0.0.1",
    license="Apache License, Version 2.0",
    author="Rikai authors",
    author_email="rikai-dev@eto.ai",
    url="https://github.com/eto-ai/rikai",
    packages=find_namespace_packages(include="rikai.contrib.*"),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "pyspark>=3.1,<3.2",
        "rosbag",
        "rospy",
    ],
    dependency_links=[
        "https://rospypi.github.io/simple/",
    ],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
    ],
)
