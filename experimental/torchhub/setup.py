from setuptools import find_namespace_packages, setup

setup(
    name="rikai-torchhub",
    version="0.0.2",
    license="Apache License, Version 2.0",
    author="Rikai authors",
    author_email="rikai-dev@eto.ai",
    url="https://github.com/eto-ai/rikai",
    python_requires=">=3.7",
    install_requires=["rikai>=0.1.1", "torch", "torchvision"],
    extras_require={
        "dev": [
            "black",
            "isort",
            # for testing
            "pytest",
        ]
    },
    packages=find_namespace_packages(include=["rikai.*"]),
)
