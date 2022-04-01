import os
import pathlib

from setuptools import find_packages, setup

about = {}
with open(pathlib.Path("rikai") / "__version__.py", "r") as fh:
    exec(fh.read(), about)

with open(
    pathlib.Path(__file__).absolute().parent.parent / "README.md", "r"
) as fh:
    long_description = fh.read()

# extras
dev = [
    "black==22.1.0",
    "click<8.1",
    "bump2version",
    "flake8",
    "isort",
    "pylint",
    "pytest-timeout",
    "pytest",
    "requests-mock",
    "wheel",
]
sklearn = ["scikit-learn"]
torch = ["torch>=1.8.1", "torchvision>=0.10.0"]
tf = ["tensorflow>=2.5.0", "tensorflow_hub"]
jupyter = ["matplotlib", "jupyterlab"]
aws = ["boto3", "botocore"]
gcp = ["gcsfs"]
docs = ["sphinx"]
video = ["ffmpeg-python", "scenedetect<0.6"]
youtube = ["pafy", "youtube_dl"]
mlflow = ["mlflow==1.24"]
all = (
    sklearn
    + torch
    + tf
    + jupyter
    + gcp
    + docs
    + video
    + youtube
    + mlflow
    + aws
)

if os.environ.get("SPARK_VERSION", None):
    spark_version = os.environ["SPARK_VERSION"]
else:
    spark_version = "3.1.2"

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
        "antlr4-python3-runtime==4.8",
        "ipython>=7.31.1,!=8.0.0",
        "jsonschema",
        "numpy",
        "opencv-python-headless",
        "pandas",
        "Pillow",
        "pyarrow>=6.0",
        f"pyspark=={spark_version}",
        "pyyaml",
        "requests",
        "semver",
    ],
    extras_require={
        "all": all,
        "dev": dev,
        "docs": docs,
        "gcp": gcp,
        "jupyter": jupyter,
        "mlflow": mlflow,
        "sklearn": sklearn,
        "pytorch": torch,
        "tf": tf,
        "video": video,
        "youtube": youtube,
        "aws": aws,
    },
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
    ],
)
