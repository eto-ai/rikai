import distutils.command.build
import glob
import os
import shutil
from subprocess import check_call

from setuptools import find_packages, setup


class BuildCommand(distutils.command.build.build):
    def run(self):
        jars_dir = os.path.join("rikai", "jars")
        if os.path.exists(jars_dir):
            shutil.rmtree(jars_dir)
        os.makedirs(jars_dir)
        check_call("sbt clean", cwd=os.pardir, shell=True)
        check_call("sbt package", cwd=os.pardir, shell=True)
        for jar_file in glob.glob("../target/scala-2.12/*.jar"):
            print(f"Copying {jar_file} to {jars_dir}")
            shutil.copy(jar_file, jars_dir)
        super().run()


setup(
    name="rikai",
    version="0.0.1",
    license="Apache License, Version 2.0",
    author="Rikai authors",
    packages=find_packages() + ["rikai.jars"],
    include_package_data=True,
    python_requires=">=3.6",
    package_data={"": ["*.jar"]},
    install_requires=[
        "numpy",
        "opencv-python",
        "pafy",
        "pandas",
        "Pillow",
        "pyspark>=3",
        "pyarrow>=2.0",
        "youtube_dl",
    ],
    extras_require={
        "test": ["pytest", "parameterized"],
        "torch": ["torch>=1.5.0", "torchvision"],
        "jupyter": ["matplotlib", "jupyterlab"],
        "aws": ["boto"],
        "docs": ["sphinx"],
    },
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
    ],
    cmdclass={"build": BuildCommand},
)
