import numpy as np
import pandas as pd
# from scipy.special import jv as besselj
from scipy.special import kv as besselk
from pxrr.gixos import film_integral_delta_beta_delta_phi
from scipy.integrate import dblquad
from joblib import Parallel, delayed
from pxrr.data_io import (
    GIXOS_file_output,
    load_data,
    load_metadata
    )
from pxrr.gixos import (GIXOS_RF_and_SF, 
                        binning_GIXOS_data, 
                        remove_negative_2theta, 
                        real_space_2theta, 
                        conversion_to_reflectivity)
from pxrr.plots import (GIXOS_data_plot, 
                        R_data_plot, 
                        R_pseudo_data_plot, 
                        GIXOS_data_plot_prep)

def rect_slit_function(GIXOS, metadata):
    xrr_data = pd.read_csv(metadata["path_xrr"] + metadata["xrr_datafile"], delim_whitespace=True)
    xrr_config = {
        "energy": 14400,
        "sdd": 1039.9,
        "slit_h": 1,
        "slit_v": 0.66,
    }

    xrr_config["wavelength"] = 12400 / xrr_config["energy"]
    xrr_config["wave_number"] = 2 * np.pi / xrr_config["wavelength"]
    xrr_config["Qz"] = GIXOS["Qz"]
    xrr_config["dataQz"] = (
        xrr_data.iloc[:, 0].astype(float).to_numpy()
    )  # see later bc accessing data !!!!!!!!!!!!!!!!!!!  might have bugs bc came out as strings and need to convert to float
    xrr_config["beta_xrr"] = np.degrees(np.arcsin(xrr_config["Qz"] / 2 / xrr_config["wave_number"]))
    xrr_config["beta_xrr"] = xrr_config["beta_xrr"].reshape(
        -1, 1
    )  # do this, otherwise beta_xrr has shape (46,) instead of (46, 1) which will mess up xrr_config_phi_array_for_qxy_slit_min and make it (46, 46) instead of (46, 1) like MATLAB code
    xrr_config["dataRF"] = (
        (xrr_config["dataQz"] - np.lib.scimath.sqrt(xrr_config["dataQz"] ** 2 - metadata["Qc"] ** 2))
        / (xrr_config["dataQz"] + np.lib.scimath.sqrt(xrr_config["dataQz"] ** 2 - metadata["Qc"] ** 2))
    ) * np.conj(
        (xrr_config["dataQz"] - np.lib.scimath.sqrt(xrr_config["dataQz"] ** 2 - metadata["Qc"] ** 2))
        / (xrr_config["dataQz"] + np.lib.scimath.sqrt(xrr_config["dataQz"] ** 2 - metadata["Qc"] ** 2))
    )  # needed for final file conversion
    xrr_config["RF"] = GIXOS["fresnel"][:, 1]
    xrr_config["kbT_gamma"] = metadata["kb"] * metadata["temperature"] / metadata["tension"] * 10**20
    xrr_config["eta"] = xrr_config["kbT_gamma"] / 2 / np.pi * xrr_config["Qz"] ** 2
    # maybe delete xrr_config for simplicity
    xrr_config["delta_phi_HW"] = np.degrees(
        np.arctan(xrr_config["slit_h"] / 2 / xrr_config["sdd"] / np.cos(np.radians(xrr_config["beta_xrr"])))
    )
    xrr_config["delta_beta_HW"] = np.degrees(
        np.arcsin(xrr_config["slit_v"] / 2 / xrr_config["sdd"] * np.cos(np.radians(xrr_config["beta_xrr"])))
    )

    xrr_config["slit_h_coord"] = np.arange(-xrr_config["slit_h"] / 2, xrr_config["slit_h"] / 2 + 0.005, 0.005)
    xrr_config["slit_v_coord"] = np.arange(-xrr_config["slit_v"] / 2, xrr_config["slit_v"] / 2 + 0.005, 0.005)
    xrr_config["slit_t"] = np.column_stack(
        (xrr_config["slit_h_coord"], np.ones(len(xrr_config["slit_h_coord"])) * xrr_config["slit_v"] / 2)
    )
    xrr_config["slit_b"] = np.column_stack(
        (xrr_config["slit_h_coord"], np.ones(len(xrr_config["slit_h_coord"])) * -xrr_config["slit_v"] / 2)
    )
    xrr_config["slit_l"] = np.column_stack(
        (np.ones(len(xrr_config["slit_v_coord"])) * -xrr_config["slit_h"] / 2, xrr_config["slit_v_coord"])
    )
    xrr_config["slit_r"] = np.column_stack(
        (np.ones(len(xrr_config["slit_v_coord"])) * xrr_config["slit_h"] / 2, xrr_config["slit_v_coord"])
    )
    xrr_config["slit_coord"] = np.concatenate(
        (
            xrr_config["slit_t"],
            xrr_config["slit_r"],
            np.flipud(xrr_config["slit_b"]),
            np.flipud(xrr_config["slit_l"]),
        ),
        axis=0,
    )
    xrr_config["qxy_slit"] = np.zeros((xrr_config["slit_coord"].shape[0], 2, xrr_config["beta_xrr"].shape[0]))
    xrr_config["qxy_slit_min"] = np.zeros((xrr_config["beta_xrr"].shape[0], 1))
    xrr_config["ang"] = np.arange(0, 2 * np.pi, 0.01)
    # xrr_config_ang = xrr_config_ang.reshape(-1, 1)  # transposing to make it a column vector
    xrr_config["qxy_slit_min_coord"] = np.zeros(
        (xrr_config["ang"].shape[0], 2, xrr_config["qxy_slit_min"].shape[0])
    )
    for idx in range(len(xrr_config["beta_xrr"])):
        xrr_config["qxy_slit"][:, :, idx] = xrr_config["wave_number"] * np.column_stack(
            [
                xrr_config["slit_coord"][:, 0] / xrr_config["sdd"],
                xrr_config["slit_coord"][:, 1]
                / xrr_config["sdd"]
                * np.sin(np.radians(xrr_config["beta_xrr"][idx])),
            ]
        )
        xrr_config["qxy_slit_min"][idx, 0] = np.min(
            np.sqrt(xrr_config["qxy_slit"][:, 0, idx] ** 2 + xrr_config["qxy_slit"][:, 1, idx] ** 2)
        )
        xrr_config["qxy_slit_min_coord"][:, :, idx] = xrr_config["qxy_slit_min"][idx] * np.column_stack(
            [np.cos(xrr_config["ang"]), np.sin(xrr_config["ang"])]
        )  # might not need np.array

    xrr_config["phi_max_qxy_slit_min"] = np.degrees(
        np.arctan(
            xrr_config["qxy_slit_min"] / xrr_config["wave_number"] / np.cos(np.radians(xrr_config["beta_xrr"]))
        )
    )

    xrr_config["phi_array_for_qxy_slit_min"] = xrr_config["phi_max_qxy_slit_min"] * np.array(
        [0, 1 / 5, 2 / 5, 3 / 5, 4 / 5]
    )
    xrr_config["delta_beta_array_for_qxy_slit_min"] = np.degrees(
        np.arcsin(
            (
                np.sqrt(
                    np.maximum(
                        xrr_config["qxy_slit_min"][:, 0:1] ** 2
                        - (
                            np.tan(np.radians(xrr_config["phi_array_for_qxy_slit_min"]))
                            * np.cos(np.radians(xrr_config["beta_xrr"]))
                            * xrr_config["wave_number"]
                        )
                        ** 2,
                        0,
                    )
                )
                / (xrr_config["wave_number"] * np.sin(np.radians(xrr_config["beta_xrr"])))
            )
            * np.cos(np.radians(xrr_config["beta_xrr"]))
        )
    )

    delta_beta_HW_1d = xrr_config["delta_beta_HW"][:, 0]
    for idx in range(xrr_config["delta_beta_array_for_qxy_slit_min"].shape[1]):
        repidx = xrr_config["delta_beta_array_for_qxy_slit_min"][:, idx] >= delta_beta_HW_1d
        xrr_config["delta_beta_array_for_qxy_slit_min"][repidx, idx] = delta_beta_HW_1d[
            repidx
        ]  # replace values that are greater than delta_beta_HW with delta_beta_HW

    xrr_config["phi_array_for_qxy_slit_min"] = np.hstack(
        [xrr_config["phi_array_for_qxy_slit_min"], xrr_config["phi_max_qxy_slit_min"]]
    )  # might not work with np.column_stack bc size mismatch --> column_hstack

    xrr_config["bkgoff"] = 1
    xrr_config["bkg_phi"] = np.degrees(
        np.arctan(xrr_config["bkgoff"] / (xrr_config["sdd"] * np.cos(np.radians(xrr_config["beta_xrr"]))))
    )  # off by ten thousandths place - supposed to get larger as index increases, but decreases instead?

    xrr_config["r_step"] = 0.001
    xrr_config["r"] = np.sqrt(
        np.maximum(
            np.arange(0.001, 8 * round(metadata["Lk"]) + xrr_config["r_step"], xrr_config["r_step"]) ** 2
            + metadata["amin"] ** 2,
            0,
        )
    )
    xrr_config["C_integrand"] = np.zeros((len(xrr_config["Qz"]), len(xrr_config["r"])))
    for idx in range(len(xrr_config["Qz"])):
        xrr_config["C_integrand"][idx, :] = (
            2
            * np.pi
            * xrr_config["r"] ** (1 - xrr_config["eta"][idx])
            * (np.exp(-xrr_config["eta"][idx] * besselk(0, xrr_config["r"] / metadata["Lk"])) - 1)
        )  # off by thousandths place
    # Matches up till here

    xrr_config["C"] = np.sum(xrr_config["C_integrand"], axis=1) * xrr_config["r_step"]
    xrr_config["qxy_slit_min_flat"] = xrr_config[
        "qxy_slit_min"
    ].flatten()  # (46,) so that RRF_term does not return a (46, 46) array since MATLAB returns a (46, 1)
    xrr_config["RRF_term"] = (
        (
            xrr_config["qxy_slit_min_flat"] ** xrr_config["eta"]
            + xrr_config["qxy_slit_min_flat"] ** 2 * xrr_config["C"] / 4 / np.pi
        )
        * (1 / metadata["qmax"]) ** xrr_config["eta"]
        * np.exp(xrr_config["eta"] * besselk(0, 1 / metadata["Lk"] / metadata["qmax"]))
    )
    xrr_config["specular_qxy_min"] = xrr_config["RF"] * xrr_config["RRF_term"]  # off by hundredths/thousandths

    xrr_config["region_around_radial_u_r"] = np.zeros(
        (len(xrr_config["beta_xrr"]), xrr_config["delta_beta_array_for_qxy_slit_min"].shape[1])
    )  # off by ~5-8 thousandths
    xrr_config["region_around_radial_l_r"] = np.zeros(
        (len(xrr_config["beta_xrr"]), xrr_config["delta_beta_array_for_qxy_slit_min"].shape[1])
    )  # off by ~5-8 thousandths
    xrr_config["diff_r"] = np.zeros((len(xrr_config["beta_xrr"]), 1))  # VERY OFF
    xrr_config["diff_r_bkgoff"] = np.zeros((len(xrr_config["beta_xrr"]), 1))  # VERY OFF

    xrr_config["diff_r"] = xrr_config["diff_r"].flatten()
    xrr_config["diff_r_bkgoff"] = xrr_config["diff_r_bkgoff"].flatten()

    angle_factor = (np.pi / 180) ** 2 * (9.42e-6) ** 2  # taking it outside of the loop decreases computation time

    def process_idx(idx):
        beta = xrr_config["beta_xrr"][idx]
        sin_beta = np.sin(np.radians(beta))
        Lk_idx = (
            metadata["Lk"] if np.isscalar(metadata["Lk"]) else metadata["Lk"][idx]
        )  # Handles array or scalar Lk

        fun_film = lambda tt, tth: film_integral_delta_beta_delta_phi(
            tt, tth, xrr_config["kbT_gamma"], xrr_config["wave_number"], beta, Lk_idx, metadata["amin"]
        )

        upper_vals = []
        lower_vals = []

        for phi_idx in range(xrr_config["delta_beta_array_for_qxy_slit_min"].shape[1]):
            # Upper
            upper, _ = dblquad(
                lambda tth, tt: fun_film(tt, tth),
                xrr_config["beta_xrr"][idx] + xrr_config["delta_beta_array_for_qxy_slit_min"][idx, phi_idx],
                xrr_config["beta_xrr"][idx] + xrr_config["delta_beta_HW"][idx],
                lambda _: xrr_config["phi_array_for_qxy_slit_min"][idx, phi_idx],
                lambda _: xrr_config["phi_array_for_qxy_slit_min"][idx, phi_idx + 1],
                epsabs=1e-12,
                epsrel=1e-10,
            )
            upper_vals.append(upper * angle_factor / sin_beta)

            # Lower
            lower, _ = dblquad(
                lambda tth, tt: fun_film(tt, tth),
                xrr_config["beta_xrr"][idx] - xrr_config["delta_beta_HW"][idx],
                xrr_config["beta_xrr"][idx] - xrr_config["delta_beta_array_for_qxy_slit_min"][idx, phi_idx],
                lambda _: xrr_config["phi_array_for_qxy_slit_min"][idx, phi_idx],
                lambda _: xrr_config["phi_array_for_qxy_slit_min"][idx, phi_idx + 1],
                epsabs=1e-12,
                epsrel=1e-10,
            )
            lower_vals.append(lower * angle_factor / sin_beta)

        # diff_r
        result, _ = dblquad(
            func=fun_film,
            a=xrr_config["phi_max_qxy_slit_min"][idx],
            b=xrr_config["delta_phi_HW"][idx],
            gfun=lambda _: xrr_config["beta_xrr"][idx] - xrr_config["delta_beta_HW"][idx],
            hfun=lambda _: xrr_config["beta_xrr"][idx] + xrr_config["delta_beta_HW"][idx],
            epsabs=1e-8,
            epsrel=1e-6,
        )
        diff_r = result * angle_factor / sin_beta

        # diff_r_bkgoff
        result2, _ = dblquad(
            func=fun_film,
            a=xrr_config["bkg_phi"][idx] - xrr_config["delta_phi_HW"][idx],
            b=xrr_config["bkg_phi"][idx] + xrr_config["delta_phi_HW"][idx],
            gfun=lambda _: xrr_config["beta_xrr"][idx] - xrr_config["delta_beta_HW"][idx],
            hfun=lambda _: xrr_config["beta_xrr"][idx] + xrr_config["delta_beta_HW"][idx],
            epsabs=1e-8,
            epsrel=1e-6,
        )
        diff_r_bkgoff = result2 * angle_factor / sin_beta

        upper_vals = np.array(upper_vals, dtype=np.float64).flatten()
        lower_vals = np.array(lower_vals, dtype=np.float64).flatten()

        return idx, upper_vals, lower_vals, diff_r, diff_r_bkgoff

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_idx)(i) for i in range(len(xrr_config["beta_xrr"]))
    )

    # Fill in results
    for idx, upper_vals, lower_vals, diff_r, diff_r_bkgoff in results:
        xrr_config["region_around_radial_u_r"][idx, :] = upper_vals
        xrr_config["region_around_radial_l_r"][idx, :] = lower_vals
        xrr_config["diff_r"][idx] = diff_r
        xrr_config["diff_r_bkgoff"][idx] = diff_r_bkgoff

    xrr_config["Rterm_rect_slit"] = xrr_config["specular_qxy_min"] + 2 * (
        np.sum(xrr_config["region_around_radial_u_r"] + xrr_config["region_around_radial_u_r"], axis=1)
        + xrr_config["diff_r"]
    )
    xrr_config["bkgterm_rect_slit"] = xrr_config["diff_r_bkgoff"]
    return xrr_config





