import pathlib
import re
from setuptools import find_packages, setup

about = {}
with open(pathlib.Path("rikai") / "__version__.py", "r") as fh:
    exec(fh.read(), about)

with open(
    pathlib.Path(__file__).absolute().parent.parent / "README.md",
    "r",
) as fh:
    long_description = fh.read()

# extras
test = ["pytest"]
torch = ["torch>=1.5.0", "torchvision"]
jupyter = ["matplotlib", "jupyterlab"]
aws = ["boto"]
docs = ["sphinx"]
youtube = ["pafy", "youtube_dl", "ffmpeg-python"]
all = test + torch + jupyter + aws + docs + youtube


setup(
    name="rikai",
    version=about["version"],
    license="Apache License, Version 2.0",
    author="Rikai authors",
    author_email="rikai-dev@eto.ai",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eto-ai/rikai",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "antlr4-python3-runtime",
        "ipython",
        "jsonschema",
        "numpy",
        "opencv-python",
        "pandas",
        "Pillow",
        "pyarrow>=2.0",
        "pyspark>=3.1,<3.2",
        "pyyaml",
        "requests",
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
