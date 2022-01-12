.. toctree::
   :maxdepth: 1

How to Release
==============

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
0. Make sure top of main is green
1. Run the bump version for release github action. This tests the head of main again, creates a version bump
   commit, and a commensurate tag.
2. Wait til the version bump commit is green on GH.
4. Via GH UI, create a new release with the new tag. This should trigger the two release workflows.
5. Check PYPI to make sure the latest version is up. Maven Central takes a few hours to sync but
   you can check sonatype to make sure the latest version is up before that.

.. _sbt-sonatype : https://github.com/xerial/sbt-sonatype
.. _sbt-release : https://github.com/olafurpg/sbt-ci-release
.. _pypi : https://packaging.python.org/tutorials/packaging-projects/