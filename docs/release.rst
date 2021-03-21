.. toctree::
   :maxdepth: 1

How to Release
==============
This section is for package maintainers.
It's important to note that the Spark unit test in python and demo notebooks refer to the maven
jar so when making a release, make sure to update those references.

Release Maven Jar
-----------------
We release the rikai jar to the maven artifactory using`sbt-sonatype`_ in conjunction with
`sbt-release`_. You can follow
`these instructions <https://www.scala-sbt.org/1.x/docs/Using-Sonatype.html>`_ to setup
:code:`sbt-sonatype`.

Sanpshot builds can be published to Sonatype snapshots repo using :code:`sbt publishSigned`.
Release builds can be published to Sonatype staging repo then sync'd to maven using a command like:
:code:`sbt release release-version <0.0.1> next-version <0.0.2-SNAPSHOT>`

Release PyPI Package
--------------------
Once the maven jar is released, you can make a pypi package release by following `pypi`_
packaging instructions.

1. Update the `rikai/python/rikai/__version__.py` version file
2. Clean the build directories :code:`rm -rf ./build ./dist ./*egg-info`
3. Build the package via :code:`python setup.py sdist bdist_wheel`
4. Upload build to the testpypi repo: :code:`twine upload --repository testpypi dist/*`. You'll
   need to have `Twine` installed.
5. Check the test build:
   :code:`pip install --index-url https://test.pypi.org/simple/ --no-deps rikai`
6. Validate the package
7. When you're ready for the actual release, run :code:`twine upload dist/*` to upload to public
   pypi.
8. Check the release: :code:`pip install rikai=<new version number>`

.. _sbt-sonatype : https://github.com/xerial/sbt-sonatype
.. _sbt-release : https://github.com/sbt/sbt-release
.. _pypi : https://packaging.python.org/tutorials/packaging-projects/