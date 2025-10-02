import numpy as np
import copy
from joblib import Parallel, delayed
from pxrr.gixos import (
    binning_GIXOS_data,
    remove_negative_2theta,
    calc_film_DS_RRF_integ,
)
from pxrr.data_io import (
    load_data,
    load_metadata,
)
from pxrr.plots import (GIXOS_vs_Qxy_plot, 
                        GIXOS_vs_Qz_plot, 
                        dependency_plot)


def create_dependency_models(GIXOS, metadata, DSbetaHW):
    model = {
        "tt": np.ones(len(metadata["qz_selected"])),
        "Qz": np.ones(len(metadata["qz_selected"])),
        "Qxy": np.zeros((len(metadata["qz_selected"]), GIXOS["Qxy"].shape[1])),
    }

    for idx in range(len(metadata["qz_selected"])):
        rowidx_arr = np.where(GIXOS["Qz"] <= metadata["qz_selected"][idx])[0]
        rowidx = rowidx_arr[-1]
        model["tt"][idx] = GIXOS["tt"][rowidx]
        model["Qz"][idx] = GIXOS["Qz"][rowidx]
        model["Qxy"][idx, :] = GIXOS["Qxy"][rowidx, :]

    assume_model = {"1": copy.deepcopy(model), "2": copy.deepcopy(model)}

    CWM_model = copy.deepcopy(model)

    model["DS_RRF"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    model["DS_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    model["RRF_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    assume_model["1"]["DS_RRF"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    assume_model["1"]["DS_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    assume_model["1"]["RRF_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    assume_model["2"]["DS_RRF"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    assume_model["2"]["DS_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    assume_model["2"]["RRF_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    CWM_model["DS_RRF"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    CWM_model["DS_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    CWM_model["RRF_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))

    metadata["energy"] = np.asarray(metadata["energy"])
    metadata["alpha_i"] = np.asarray(metadata["alpha_i"])
    metadata["RqxyHW"] = np.asarray(metadata["RqxyHW"])
    metadata["DSqxyHW_real"] = np.asarray(metadata["DSqxyHW_real"])
    # GIXOS["DSbetaHW"] = np.asarray(GIXOS["DSbetaHW"])        only add this and make changes if we say that DSbetaHW is part of GIXOS above
    DSbetaHW = np.asarray(DSbetaHW)
    metadata["tension"] = np.asarray(metadata["tension"])
    metadata["temperature"] = np.asarray(metadata["temperature"])
    metadata["kappa"] = np.asarray(metadata["kappa"])
    metadata["amin"] = np.asarray(metadata["amin"])

    def process(idx):
        show_last_plot = idx == GIXOS["GIXOS"].shape[1] - 1  # Only show plot on last iteration
        model_DS_RRF, model_DS_term, model_RRF_term = calc_film_DS_RRF_integ(
            model["tt"],
            metadata["qxy0"][idx],
            metadata["energy"] / 1000,
            metadata["alpha_i"],
            metadata["RqxyHW"],
            metadata["DSqxyHW_real"][idx],
            DSbetaHW,
            metadata["tension"],
            metadata["temperature"],
            metadata["kappa"],
            metadata["amin"],
            show_plot=False,
        )
        assume_model_1_DS_RRF, assume_model_1_DS_term, assume_model_1_RRF_term = calc_film_DS_RRF_integ(
            model["tt"],
            metadata["qxy0"][idx],
            metadata["energy"] / 1000,
            metadata["alpha_i"],
            metadata["RqxyHW"],
            metadata["DSqxyHW_real"][idx],
            DSbetaHW,
            metadata["tension"],
            metadata["temperature"],
            metadata["assume_kappa"][0],
            metadata["amin"],
            show_plot=False,
        )
        assume_model_2_DS_RRF, assume_model_2_DS_term, assume_model_2_RRF_term = calc_film_DS_RRF_integ(
            model["tt"],
            metadata["qxy0"][idx],
            metadata["energy"] / 1000,
            metadata["alpha_i"],
            metadata["RqxyHW"],
            metadata["DSqxyHW_real"][idx],
            DSbetaHW,
            metadata["tension"],
            metadata["temperature"],
            metadata["assume_kappa"][1],
            metadata["amin"],
            show_plot=False,
        )
        CWM_model_DS_RRF, CWM_model_DS_term, CWM_model_RRF_term = calc_film_DS_RRF_integ(
            model["tt"],
            metadata["qxy0"][idx],
            metadata["energy"] / 1000,
            metadata["alpha_i"],
            metadata["RqxyHW"],
            metadata["DSqxyHW_real"][idx],
            DSbetaHW,
            metadata["tension"],
            metadata["temperature"],
            0,
            metadata["amin"],
            show_plot=show_last_plot,
        )
        return (
            idx,
            model_DS_RRF,
            model_DS_term,
            model_RRF_term,
            assume_model_1_DS_RRF,
            assume_model_1_DS_term,
            assume_model_1_RRF_term,
            assume_model_2_DS_RRF,
            assume_model_2_DS_term,
            assume_model_2_RRF_term,
            CWM_model_DS_RRF,
            CWM_model_DS_term,
            CWM_model_RRF_term,
        )

    results = Parallel(n_jobs=-1, backend="loky")(delayed(process)(i) for i in range(GIXOS["GIXOS"].shape[1]))

    for (
        idx,
        model_DS_RRF,
        model_DS_term,
        model_RRF_term,
        assume_model_1_DS_RRF,
        assume_model_1_DS_term,
        assume_model_1_RRF_term,
        assume_model_2_DS_RRF,
        assume_model_2_DS_term,
        assume_model_2_RRF_term,
        CWM_model_DS_RRF,
        CWM_model_DS_term,
        CWM_model_RRF_term,
    ) in results:
        model["DS_RRF"][:, idx], model["DS_term"][:, idx], model["RRF_term"][:, idx] = (
            model_DS_RRF,
            model_DS_term,
            model_RRF_term,
        )
        (
            assume_model["1"]["DS_RRF"][:, idx],
            assume_model["1"]["DS_term"][:, idx],
            assume_model["1"]["RRF_term"][:, idx],
        ) = (assume_model_1_DS_RRF, assume_model_1_DS_term, assume_model_1_RRF_term)
        (
            assume_model["2"]["DS_RRF"][:, idx],
            assume_model["2"]["DS_term"][:, idx],
            assume_model["2"]["RRF_term"][:, idx],
        ) = (assume_model_2_DS_RRF, assume_model_2_DS_term, assume_model_2_RRF_term)
        CWM_model["DS_RRF"][:, idx], CWM_model["DS_term"][:, idx], CWM_model["RRF_term"][:, idx] = (
            CWM_model_DS_RRF,
            CWM_model_DS_term,
            CWM_model_RRF_term,
        )
    return model, assume_model, CWM_model

def dependency_real_space_2theta(metadata):
    metadata["tth"] = (
        np.degrees(np.arcsin(metadata["qxy0"] * metadata["wavelength"] / 4 / np.pi)) * 2
    )  # math.asin would work if not a list
    metadata["tth_roiHW_real"] = np.degrees(metadata["pixel"] * metadata["DSpxHW"] / metadata["Ddet"])
    metadata["DSqxyHW_real"] = (
        np.radians(metadata["tth_roiHW_real"])
        / 2
        * 4
        * np.pi
        / metadata["wavelength"]
        * np.cos(np.radians(metadata["tth"] / 2))
    )
    return metadata


def Qxy_GIXOS_data_plot_prep(importGIXOSdata, importbkg, metadata, tt_step):
    GIXOS = {
        "tt": importGIXOSdata["tt"],
    }
    DSbetaHW = np.mean(GIXOS["tt"][1:] - GIXOS["tt"][0:-1]) / 2
    qxy0_idx_arr = np.where(metadata["qxy0"] > metadata["qxy_bkg"])
    if len(qxy0_idx_arr) == 0:
        qxy0_idx = len(metadata["qxy0"]) + 1
    else:
        qxy0_idx = int(qxy0_idx_arr[0])

    GIXOS["Qxy"] = np.zeros((len(GIXOS["tt"]), qxy0_idx))
    for idx in range(qxy0_idx):
        GIXOS["Qxy"][:, idx] = (
            2
            * np.pi
            / metadata["wavelength"]
            * np.sqrt(
                (np.cos(np.radians(GIXOS["tt"])) * np.sin(np.radians(metadata["tth"][idx]))) ** 2
                + (
                    np.cos(np.radians(metadata["alpha_i"]))
                    - np.cos(np.radians(GIXOS["tt"])) * np.cos(np.radians(metadata["tth"][idx]))
                )
                ** 2
            )
        )
    GIXOS["Qz"] = (
        2
        * np.pi
        / metadata["wavelength"]
        * (np.sin(np.radians(GIXOS["tt"])) + np.sin(np.radians(metadata["alpha_i"])))
    )
    GIXOS["GIXOS_raw"] = importGIXOSdata["Intensity"][:, :qxy0_idx]
    GIXOS["GIXOS_bkg"] = importbkg["Intensity"][:, :qxy0_idx]
    if qxy0_idx <= len(metadata["qxy0"]):
        GIXOS_raw_largetth = np.mean(importGIXOSdata["Intensity"][:, qxy0_idx:], axis=1)
        GIXOS_bkg_largetth = np.mean(importbkg["Intensity"][:, qxy0_idx:], axis=1)
        bulkbkg = GIXOS_raw_largetth - GIXOS_bkg_largetth
    else:
        bulkbkg = np.zeros(len(metadata["qxy0"]), 1)

    fdtt = np.radians(tt_step) / (
        np.arctan((np.tan(np.radians(GIXOS["tt"])) * metadata["Ddet"] + metadata["pixel"] / 2) / metadata["Ddet"])
        - np.arctan(
            (np.tan(np.radians(GIXOS["tt"])) * metadata["Ddet"] - metadata["pixel"] / 2) / metadata["Ddet"]
        )
    )
    fdtt = fdtt / fdtt[0]
    fdtt = fdtt[:, np.newaxis]

    GIXOS["GIXOS"] = (GIXOS["GIXOS_raw"] - GIXOS["GIXOS_bkg"]) * fdtt - np.mean(bulkbkg[-1 - 10 :], axis=0) * fdtt
    GIXOS["error"] = (
        np.sqrt(np.sqrt(np.abs(GIXOS["GIXOS_raw"])) ** 2 + np.sqrt(np.abs(GIXOS["GIXOS_bkg"])) ** 2) * fdtt
    )
    GIXOS["bkg"] = 0
    return (
        GIXOS,
        DSbetaHW,
    ) 

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


