import pathlib
import re
from setuptools import find_packages, setup

about = {}
with open(pathlib.Path("rikai") / "__version__.py", "r") as fh:
    exec(fh.read(), about)

# extras
test = ["pytest", "parameterized"]
torch = ["torch>=1.5.0", "torchvision"]
jupyter = ["matplotlib", "jupyterlab"]
aws = ["boto"]
docs = ["sphinx"]
youtube = ["pafy", "youtube_dl"]
all = test + torch + jupyter + aws + docs + youtube


setup(
    name="rikai",
    version=about["version"],
    license="Apache License, Version 2.0",
    author="Rikai authors",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "ipython",
        "numpy",
        "opencv-python",
        "pandas",
        "Pillow",
        "pyspark>=3",
        "pyarrow>=2.0",
    ],
    extras_require={
        "test": test,
        "pytorch": torch,
        "jupyter": jupyter,
        "aws": aws,
        "docs": docs,
        "youtube": youtube,
        "all": all,
    },
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
    ],
)
