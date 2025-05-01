.. meta::
   :description: |summary|
   :twitter:description: |summary|

pointtree
====================

.. rst-class:: lead

   |summary|

----

This is the documentation for version |current| of the |product| package.

Get started
-----------

#. Install `PyTorch <https://pytorch.org/get-started/locally>`__

#. Install `torch-scatter <https://github.com/rusty1s/pytorch_scatter>`__

#. Install `torch-cluster <https://pypi.org/project/torch-cluster>`__

#. Install `PyTorch3D <https://pytorch3d.org/#quickstart>`__ (optional)

#. Install the package:

   .. literalinclude:: how-to/install/includes/install.sh
      :language: sh

To avoid the manual installation of the package and it's dependencies, you can also use our Docker container, which
contains a ready-to-use installation of the pointtree package. To start the container you can run:

````bash
docker run --rm -it josafatburmeister/pointtree:latest
```

Inside the image, you can just start the Python interpreter and import the pointtree package.


Upgrade
-------

If you want to upgrade to version |current| of the package, see :doc:`changelog/index`.

Package Documentation
----------------------

.. toctree::
   :maxdepth: 4
   :titlesonly:

   pointtree
