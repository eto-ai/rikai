.. toctree::
   :maxdepth: 1

Release
=======
This section is for package maintainers.

Release Maven Jar
-----------------
We release the rikai jar to the maven artifactory using
`sbt-sonatype <https://github.com/xerial/sbt-sonatype>`_ in conjunction with
`sbt-release <https://github.com/sbt/sbt-release>`_. You can follow `these instructions
<https://www.scala-sbt.org/1.x/docs/Using-Sonatype.html>`_ to setup :code:`sbt-sonatype`.

Sanpshot builds can be published to Sonatype snapshots repo using :code:`sbt publishSigned`.
Release builds can be published to Sonatype staging repo then sync'd to maven using
:code:`sbt release`