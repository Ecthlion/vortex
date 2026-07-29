Input and Output
================

Vortex arrays support reading and writing to local and remote file systems, including plain-old
HTTP, S3, Google Cloud Storage, and Azure Blob Storage.

.. autosummary::
   :nosignatures:

   ~vortex.open
   ~vortex.VortexFile
   ~vortex.open_files
   ~vortex.VortexFiles
   ~vortex.RepeatedScan
   ~vortex.io.read_url
   ~vortex.io.write

.. raw:: html

   <hr>

.. autofunction:: vortex.open

.. autoclass:: vortex.VortexFile
   :members:

.. autofunction:: vortex.open_files

.. autoclass:: vortex.VortexFiles
   :members:

.. autoclass:: vortex.RepeatedScan
   :members:

.. automodule:: vortex.io
    :members:
    :imported-members:

