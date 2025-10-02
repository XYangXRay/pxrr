import numpy as np
import copy
from joblib import Parallel, delayed
from pxrr.gixos import calc_film_DS_RRF_integ



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


