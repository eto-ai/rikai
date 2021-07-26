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
    "black",
    "bump2version",
    "flake8",
    "isort",
    "pylint",
    "pytest-timeout",
    "pytest",
    "requests-mock",
    "wheel",
]
torch = ["torch>=1.4.0", "torchvision"]
jupyter = ["matplotlib", "jupyterlab"]
aws = ["boto3"]
gcp = ["gcsfs"]
docs = ["sphinx"]
video = ["ffmpeg-python", "scenedetect"]
youtube = ["pafy", "youtube_dl"]
mlflow = ["mlflow>=1.15"]
all = torch + jupyter + gcp + docs + video + youtube + mlflow + aws


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
        "semver",
    ],
    extras_require={
        "all": all,
        "dev": dev,
        "docs": docs,
        "gcp": gcp,
        "jupyter": jupyter,
        "mlflow": mlflow,
        "pytorch": torch,
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
