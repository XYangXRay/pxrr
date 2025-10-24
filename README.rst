==========
Pseudo XRR
==========

.. image:: https://github.com/XYangXRay/pxrr/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/XYangXRay/pxrr/actions/workflows/testing.yml

.. image:: https://img.shields.io/pypi/v/pxrr.svg
   :target: https://pypi.python.org/pypi/pxrr

Python package for pseudo X-ray reflectivity

* Free software: 3-clause BSD license

Installation
------------

To install the `pxrr` package in a local development environment, follow these steps:

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/XYangXRay/pxrr.git
    cd pxrr

2. Create the pixi environments (default and 'dev'):

.. code-block:: bash

    pixi install --all

3. That's all!

Features
--------

* TODO

Quick Start (Notebook Flow)
---------------------------

Minimal processing sequence (mirrors ``pxrr_step_example``)::

    from pxrr.data_io import load_inputs, binning_GIXOS_data, remove_negative_2theta, real_space_2theta, save_pseudo_reflectivity
    from pxrr.plots import GIXOS_data_plot_prep, GIXOS_data_plot, R_data_plot, R_pseudo_data_plot
    from pxrr.slit import rect_slit_function, conversion_to_reflectivity
    from pxrr.dependency import dependency_real_space_2theta, create_dependency_models

    metadata_file = "./testing_data/gixos_metadata.yaml"

    # Load raw data + metadata
    importGIXOSdata, importbkg, metadata = load_inputs(metadata_file)

    # Re-bin & trim negative 2θ
    importGIXOSdata, importbkg = binning_GIXOS_data(importGIXOSdata, importbkg)
    importGIXOSdata, importbkg, tt_step = remove_negative_2theta(importGIXOSdata, importbkg)

    # Geometry (single qxy0 workflow)
    metadata = real_space_2theta(metadata)

    # Prepare GIXOS signal
    GIXOS, DSbetaHW = GIXOS_data_plot_prep(importGIXOSdata, importbkg, metadata, tt_step)
    GIXOS_data_plot(GIXOS, metadata)

    # Fresnel + transmission + DS terms
    GIXOS = GIXOS_RF_and_SF(GIXOS, metadata, DSbetaHW)

    # Slit + reflectivity conversion
    xrr_config = rect_slit_function(GIXOS, metadata)
    GIXOS = conversion_to_reflectivity(GIXOS, xrr_config)

    # Plots
    R_data_plot(GIXOS, metadata, xrr_config)                 # R
    R_pseudo_data_plot(GIXOS, metadata, xrr_config)          # (R/R_F)/scale

    # Save pseudo reflectivity (qz ref dR dqz)
    outfile = save_pseudo_reflectivity(GIXOS, metadata)
    print("Saved:", outfile)

Multi‑curve pseudo reflectivity (different scalings + offsets)::

    R_pseudo_data_plot(
        GIXOS, metadata, xrr_config,
        rf_scalings=[metadata['RFscaling'], metadata['RFscaling']*2, metadata['RFscaling']*5],
        offsets=[0, 1, 2]   # multiply curves by 10^offset (visual separation)
    )

Dependency (qz-selected modeling)::

    metadata = dependency_real_space_2theta(metadata)   # multi-qxy geometry
    model, assume_model, CWM_model = create_dependency_models(GIXOS, metadata, DSbetaHW)
    dependency_plot(GIXOS, metadata, model, assume_model, CWM_model)

Outputs
-------
* Reflectivity: ``*_R_PYTHON_TEST.dat``
* DS/(R/RF): ``*_DS2RRF_PYTHON_TEST.dat``
* Structure factor: ``*_SF_PYTHON_TEST.dat``
* Pseudo XRR (portable): ``<sample>_pseudo_qx.txt`` (qz ref dR dqz)

Tips
----
* Override Fresnel scaling in plots: ``R_data_plot(..., rf_scaling=value)``
* Provide multiple scalings + offsets for publication layering.
* Set ``rf_scalings=None`` to default to ``metadata['RFscaling']``.
