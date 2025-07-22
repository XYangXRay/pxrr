from pseudo_xrr.data_io import load_data, load_metadata, binning_GIXOS_data, remove_negative_2theta, real_space_2theta, GIXOS_data_plot_prep, GIXOS_RF_and_SF, rect_slit_function, conversion_to_reflectivity, GIXOS_file_output
from pseudo_xrr.plots import GIXOS_data_plot, R_data_plot, R_pseudo_data_plot
def rect_slit_wrapper(metadata_file='./testing_data/gixos_metadata.yaml'):
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