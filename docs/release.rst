.. toctree::
   :maxdepth: 1

How to Release
==============
This section is for package maintainers.

Maven Jar
---------
We release the rikai jar to the maven artifactory using `sbt-sonatype`_ in conjunction with
`sbt-ci-release`_. For details see :code:`.github/workflows/scala-publish.yml` github action
workflow.

Python Package
--------------
We release the rikai python package to `pypi`_ via :code:`.github/workflows/python-publish.yml`_.

Release Process
---------------
0. Make sure top of master is green and make sure local repo is up to date
1. Locally, run `make release`. This creates a commit that bumps the version and a tag. The explicit
   version determines the version for python. The tag determines the jar version (via sbt dynver).
   The tag and the python version must be aligned (should be automatic).
2. Push the version bump commit (with the tag!) to remote: `git push origin master --tags`
3. Wait til the version bump is green on remote.
4. Via GH UI, create a new release with the new tag. This should trigger the two release workflows.
5. Check PYPI to make sure the latest version is up. Maven Central takes a few hours to sync but
   you can check sonatype to make sure the latest version is up before that.

.. _sbt-sonatype : https://github.com/xerial/sbt-sonatype
.. _sbt-release : https://github.com/olafurpg/sbt-ci-release
.. _pypi : https://packaging.python.org/tutorials/packaging-projects/