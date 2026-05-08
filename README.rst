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

    pip install "git+https://github.com/XYangXRay/pxrr.git@ocko"


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

Download example folder::
	
	git clone --filter=blob:none --no-checkout -b ocko https://github.com/XYangXRay/pxrr.git
	cd pxrr
	git sparse-checkout init --cone
	git sparse-checkout set example
	git checkout

Two example scripts are included demonstrating typical workflows:

NSLS-II / OPLS (1D GIXOS input)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Script::

    example/opls_1d/pXRR_example_1dGIXOS_OPLS.py
	
Jupyter-notebook::
	
	example/opls_1d/pXRR_example_1dGIXOS_OPLS.ipynb

This example demonstrates:

- loading 1D GIXOS cuts
- background correction
- pseudo-XRR calculation

test data:

- DSPC data from OPLS: example/testing_data/opls_DSPC_data/
- CaCl2 data from OPLS: example/testing_data/opls_CaCl2_data/

Run::

    in spyder run pXRR_example_1dGIXOS_OPLS.py
	
Jupyter-notebook::
	
	pXRR_example_1dGIXOS_OPLS.ipynb


PETRA III / P08 (2D GIXS input)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Script::

    example/p08_2d/pXRR_example_2dGIXS_p08.py

This example demonstrates:

- loading 2D GIXS detector data
- extracting 1D GIXOS cuts
- applying geometrical correction
- performing pseudo-XRR analysis

test data:

- DPPC data from p08: example/testing_data/p08_DPPC_data/

Run::

    in spyder run pXRR_example_2dGIXS_p08.py


Metadata
--------

Both examples require a YAML metadata file describing:

- instrument parameters (energy, geometry, etc.)
- scan numbers
- file paths
- processing parameters

YAML file stored in the same folder with the scripts


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


Acknowledgement and citations
-------

This GIXOS-pseudo XRR processing library is jointly developed by Brookhaven National Laboratory and Deutsches Elektronen-Synchrotron DESY, and Dr. Benjamin Ocko, Mr. Alex Palomino, Dr. Chen Shen, Dr. Xiaogang Yang are acknowledged.

Please cite the two papers: (1) https://doi.org/10.1107/s1600576724002887; (2) https://doi.org/10.1103/znt1-fmx6
