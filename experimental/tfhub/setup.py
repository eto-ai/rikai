from setuptools import find_namespace_packages, setup

setup(
    name="rikai-tfhub",
    version="0.0.1",
    license="Apache License, Version 2.0",
    author="Rikai authors",
    author_email="rikai-dev@eto.ai",
    url="https://github.com/eto-ai/rikai",
    python_requires=">=3.7",
    install_requires=["rikai>=0.1.1", "tensorflow>=2.5.0", "tensorflow_hub"],
    extras_require={
        "dev": [
            "black",
            "isort",
            # for testing
            "pytest",
            "torch",
        ]
    },
    packages=find_namespace_packages(include=["rikai.*"]),
)
