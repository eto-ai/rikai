from setuptools import find_namespace_packages, setup

setup(
    name="rikai-yolov5",
    version="0.0.2",
    license="Apache License, Version 2.0",
    author="Rikai authors",
    author_email="rikai-dev@eto.ai",
    url="https://github.com/eto-ai/rikai",
    python_requires=">=3.7",
    install_requires=["rikai >= 0.0.17", "yolov5 >=5.0.0, <6.0.0"],
    extras_require={
        "dev": [
            "black",
            "isort",
            # for testing
            "pytest",
            "mlflow",
        ]
    },
    packages=find_namespace_packages(include=["rikai.*"]),
)
