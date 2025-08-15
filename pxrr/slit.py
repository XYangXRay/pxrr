from pxrr.data_io import (
    GIXOS_data_plot_prep,
    GIXOS_file_output,
    GIXOS_RF_and_SF,
    binning_GIXOS_data,
    conversion_to_reflectivity,
    load_data,
    load_metadata,
    real_space_2theta,
    rect_slit_function,
    remove_negative_2theta,
)
from pxrr.plots import GIXOS_data_plot, R_data_plot, R_pseudo_data_plot


def rect_slit_wrapper(metadata_file="./testing_data/gixos_metadata.yaml"):
    importGIXOSdata, importbkg = load_data(metadata_file)
    metadata = load_metadata(metadata_file)
    importGIXOSdata, importbkg = binning_GIXOS_data(importGIXOSdata, importbkg)
    importGIXOSdata, importbkg, tt_step = remove_negative_2theta(importGIXOSdata, importbkg)
    metadata = real_space_2theta(metadata)
    GIXOS, DSbetaHW = GIXOS_data_plot_prep(importGIXOSdata, importbkg, metadata, tt_step)
    GIXOS_data_plot(GIXOS, metadata)
    GIXOS = GIXOS_RF_and_SF(GIXOS, metadata, DSbetaHW)
    xrr_config = rect_slit_function(GIXOS, metadata)
    xrr_config = conversion_to_reflectivity(GIXOS, xrr_config)
    GIXOS_file_output(GIXOS, xrr_config, metadata, tt_step)
    R_data_plot(GIXOS, metadata, xrr_config)
    R_pseudo_data_plot(GIXOS, metadata, xrr_config)
