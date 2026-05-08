==========
Pseudo XRR
==========

# pxrr

Pseudo-XRR / GIXOS analysis tools for processing grazing-incidence X-ray scattering data and exporting ORSO-compatible results.

This package provides utilities for:

* loading and processing GIXS / GIXOS data
* applying geometrical and background corrections
* extracting 1D cuts from 2D detector data
* computing pseudo-reflectivity (pXRR)
* exporting results in ORSO-compatible formats

---

## Installation

Install from GitHub::

```
pip install "git+https://github.com/XYangXRay/pxrr.git"
```

Install a specific version::

```
pip install "git+https://github.com/XYangXRay/pxrr.git@v1.0.0"
```

Recommended: virtual environment::

```
python -m venv pxrr-env
source pxrr-env/bin/activate   # Linux
pxrr-env\Scripts\activate      # Windows

pip install --upgrade pip
pip install "git+https://github.com/XYangXRay/pxrr.git"
```

---

## Dependencies

Core dependencies:

* numpy
* scipy
* matplotlib
* pandas
* h5py
* ruamel.yaml
* joblib
* orsopy
* xray-general-io

Optional:

* p08-general (for PETRA III / P08 workflows)

Install with optional support::

```
pip install "pxrr[p08] @ git+https://github.com/XYangXRay/pxrr.git"
```

---

## Quick Start

.. code-block:: python

```
from pseudo_xrr.data_io import load_metadata, load_gixos_from_meta
from pseudo_xrr.GIXOS import GIXOS_th2q

meta = load_metadata("metadata.yaml")
data, bkg = load_gixos_from_meta("metadata.yaml")

data_q = GIXOS_th2q(data)
```

---

## Examples

Two example scripts are included:

NSLS-II / OPLS (1D GIXOS)

```

Script::

    OPLS_test_pXRR.py

Run::

    python OPLS_test_pXRR.py

PETRA III / P08 (2D GIXS)
```

Script::

```
p08_test_pXRR.py
```

Run::

```
python p08_test_pXRR.py
```

---

## Metadata

Both examples require a YAML file (in their folders) defining:

* instrument parameters
* scan numbers
* data paths

---

## Notes

* ORSO export requires `xray_general_io`
* P08 workflows require `p08_general` (optional)
* Keep compatible versions across:

  * pxrr
  * xray_general_io
  * p08_general

---

## HPC usage

Avoid environment conflicts::

```
unset PYTHONPATH
unset PYTHONHOME
```

---

## Development

Install editable mode::

```
pip install -e .
```

---

## Install a tagged version

```
pip install "git+https://github.com/XYangXRay/pxrr.git@v1.0.0"
```

---

## License

Add your license here.

---

## Authors

Developed for GIXS / GIXOS analysis workflows at synchrotron beamlines.
