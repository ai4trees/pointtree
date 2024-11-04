:orphan: true

.. meta::
   :description: Information on how to contribute to the package development.
   :twitter:description: Information on how to contribute to the package development.

Pre-Commit Hooks
------------------

To set up pre-commit hooks that automatically check your code when you commit, run the following in the root directory
of the repository

.. code-block:: bash

   $ pre-commit install

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
