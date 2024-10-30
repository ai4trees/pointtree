:orphan: true

.. meta::
   :description: Information on how to contribute to the package development.
   :twitter:description: Information on how to contribute to the package development.

Testing
------------------

To execute the tests and measure the code coverage, run:

.. code-block:: bash

   $ pytest --cov --cov-report=xml

Releasing
----------

Releases are published automatically when a tag is pushed to the main branch on GitHub.

.. code-block:: bash

   # Create a commit and tag it as release
   $ git commit -m "<commit message>"
   $ git tag -a "release"

   # Push
   $ git push upstream --tags
