==========
Pseudo XRR
==========

pxrr
====

Pseudo-XRR / GIXOS analysis tools for processing grazing-incidence X-ray scattering data and exporting ORSO-compatible results.

This package provides utilities for:

- loading and processing GIXS / GIXOS data
- applying geometrical and background corrections
- extracting 1D cuts from 2D detector data
- computing pseudo-reflectivity (pXRR)
- exporting results in ORSO-compatible formats


Installation
------------

Install from GitHub::

    pip install "git+https://github.com/XYangXRay/pxrr.git"

Install a specific branch::

    pip install "git+https://github.com/XYangXRay/pxrr.git@p08-dev"


Recommended: virtual environment::

    python -m venv pxrr-env
    source pxrr-env/bin/activate   # Linux
    pxrr-env\Scripts\activate      # Windows

    pip install --upgrade pip
    pip install "git+https://github.com/XYangXRay/pxrr.git"


Dependencies
------------

Core dependencies:

- numpy
- scipy
- matplotlib
- pandas
- h5py
- ruamel.yaml
- joblib
- orsopy
- xray-general-io

Optional:

- p08-general (for PETRA III / P08 workflows)

Install with optional support::

    pip install "pxrr[p08] @ git+https://github.com/XYangXRay/pxrr.git"


Quick Start
-----------

.. code-block:: python

    from pseudo_xrr.data_io import load_metadata, load_gixos_from_meta
    from pseudo_xrr.GIXOS import GIXOS_th2q

    meta = load_metadata("metadata.yaml")
    data, bkg = load_gixos_from_meta("metadata.yaml")

    data_q = GIXOS_th2q(data)


Examples
--------

Two example scripts are included demonstrating typical workflows:

NSLS-II / OPLS (1D GIXOS input)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Script::

    OPLS_test_pXRR.py

This example demonstrates:

- loading 1D GIXOS cuts
- background correction
- pseudo-XRR calculation

Run::

    python OPLS_test_pXRR.py


PETRA III / P08 (2D GIXS input)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Script::

    p08_test_pXRR.py

This example demonstrates:

- loading 2D GIXS detector data
- extracting 1D GIXOS cuts
- applying geometrical correction
- performing pseudo-XRR analysis

Run::

    python p08_test_pXRR.py


Metadata
--------

Both examples require a YAML metadata file describing:

- instrument parameters (energy, geometry, etc.)
- scan numbers
- file paths
- processing parameters


Notes
-----

- ORSO-compatible export requires ``orsopy``
- P08-specific workflows require ``p08_general`` (optional dependency)
- Use compatible versions of ``orsopy`` across:
  - pxrr
  - xray_general_io
  - p08_general


HPC / Cluster usage
-------------------

To avoid conflicts with system Python::

    unset PYTHONPATH
    unset PYTHONHOME


Development
-----------

Install in editable mode::

    pip install -e .


Versioning
----------

Versions are derived from Git tags::

    git tag v1.0.0
    git push origin v1.0.0

Install a tagged version::

    pip install "git+https://github.com/XYangXRay/pxrr.git@v1.0.0"


License
-------

Add your license here.


Authors
-------

Developed for GIXS / GIXOS analysis workflows at synchrotron beamlines .

