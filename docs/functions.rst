.. toctree::
   :maxdepth: 3

Rikai Spark UDFs
================

Rikai provides many Spark User-Defined-Functions (*UDF*) to work with unstructured semantic types,
like images, videos, and annotations.

Geometry UDFs.
--------------

area
^^^^^

``area(box: Box2d): float`` calculates the area of a 2-D bounding box (:py:class:`~rikai.types.geomotry.Box2d`).

**Example**

.. code-block:: sql

    SELECT * FROM dataset WHERE area(pred.box) > 2000


box2d
^^^^^

``box2d(coords: array<float>): Box2d`` builds a :py:class:`~rikai.types.geometry.Box2d`
from ``[xmin,ymin,xmax,ymax]`` array.

**Example**

.. code-block:: sql

    SELECT box2d(array(1, 2, 3.5, 4.0))

box2d_from_top_left
^^^^^^^^^^^^^^^^^^^

``box2d_from_top_left(coords: array<float>): Box2d`` builds a
:py:class:`~rikai.types.geometry.Box2d` from ``[x0, y0, width, height]`` array.

**Example**

.. code-block:: sql

    SELECT box2d_from_top_left(array(1, 2, 15, 20))


box2d_from_center
^^^^^^^^^^^^^^^^^

``box2d_from_center(coords: array<float>): Box2d`` builds a
:py:class:`~rikai.types.geometry.Box2d` from ``[center_x, center_y, width, height]`` array.


Computer Vision UDFs
--------------------

to_image
^^^^^^^^

Builds an :py:class:`~rikai.types.Image` from bytes or URI.

**Example**

.. code-block:: sql

    SELECT to_image("s3://bucket/to/image.png")


image_copy
^^^^^^^^^^

``image_copy(image: Image, uri: str): Image`` copies image to another URI and returns the new Image object.

**Example**

.. code-block:: sql

    SELECT image_copy(img, "s3://bucket/to/new_image.png") FROM images


I/O UDFs
----------------

copy
^^^^

``copy(src: str, dest: str): str`` copies a file from the ``source`` to ``dest``, and returns
the destination URI.

