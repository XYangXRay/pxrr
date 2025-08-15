from pxrr.data_io import (
    GIXOS_data_plot_prep,
    GIXOS_file_output,
    GIXOS_RF_and_SF,
    Qxy_GIXOS_data_plot_prep,
    binning_GIXOS_data,
    conversion_to_reflectivity,
    create_dependency_models,
    dependency_real_space_2theta,
    load_data,
    load_metadata,
    rect_slit_function,
    remove_negative_2theta,
)
from pxrr.plots import GIXOS_vs_Qxy_plot, GIXOS_vs_Qz_plot, dependency_plot


def dependency_wrapper(metadata_file="./testing_data/gixos_metadata.yaml"):
    importGIXOSdata, importbkg = load_data(metadata_file)
    metadata = load_metadata(metadata_file)
    importGIXOSdata, importbkg = binning_GIXOS_data(importGIXOSdata, importbkg)
    importGIXOSdata, importbkg, tt_step = remove_negative_2theta(importGIXOSdata, importbkg)
    metadata = dependency_real_space_2theta(metadata)
    GIXOS, DSbetaHW = Qxy_GIXOS_data_plot_prep(importGIXOSdata, importbkg, metadata, tt_step)
    GIXOS_vs_Qz_plot(GIXOS, metadata)
    GIXOS_vs_Qxy_plot(GIXOS, metadata)
    model, assume_model, CWM_model = create_dependency_models(GIXOS, metadata, DSbetaHW)
    dependency_plot(GIXOS, metadata, model, assume_model, CWM_model)
    return


# load all data - also for slit
# input check
# models
# dependency plot
