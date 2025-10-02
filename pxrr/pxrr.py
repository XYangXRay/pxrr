from pxrr.data_io import load_data, load_metadata, GIXOS_file_output
from pxrr.dependency import dependency_real_space_2theta, create_dependency_models, dependency_plot
from pxrr.gixos import (binning_GIXOS_data, 
                        remove_negative_2theta, 
                        real_space_2theta, 
                        GIXOS_RF_and_SF, 
                        conversion_to_reflectivity)
from pxrr.plots import (Qxy_GIXOS_data_plot_prep,
                        GIXOS_data_plot_prep,
                        GIXOS_data_plot, 
                        GIXOS_vs_Qz_plot, 
                        GIXOS_vs_Qxy_plot, 
                        R_data_plot, 
                        R_pseudo_data_plot)
from pxrr.slit import rect_slit_function

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

def apply_slit(
    metadata_file="./testing_data/gixos_metadata.yaml",
):  # can make this a main function to run the whole code and have a parameter be the text file
    importGIXOSdata, importbkg = load_data(metadata_file)
    metadata = load_metadata(metadata_file)
    importGIXOSdata, importbkg = binning_GIXOS_data(importGIXOSdata, importbkg)
    importGIXOSdata, importbkg, tt_step = remove_negative_2theta(importGIXOSdata, importbkg)
    metadata = real_space_2theta(metadata)
    GIXOS, DSbetaHW = GIXOS_data_plot_prep(importGIXOSdata, importbkg, metadata, tt_step)
    GIXOS_data_plot(GIXOS, metadata)
    GIXOS = GIXOS_RF_and_SF(GIXOS, metadata, DSbetaHW)
    xrr_config = rect_slit_function(GIXOS, metadata)
    GIXOS = conversion_to_reflectivity(GIXOS, xrr_config)
    print("xrr_config keys:", xrr_config.keys())

    GIXOS_file_output(GIXOS, xrr_config, metadata, tt_step)
    R_data_plot(GIXOS, metadata, xrr_config)
    R_pseudo_data_plot(GIXOS, metadata, xrr_config)


