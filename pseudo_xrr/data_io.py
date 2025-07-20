import yaml
import numpy as np
import pandas as pd
import math
import os
from pseudo_xrr.gixos import GIXOS_fresnel, GIXOS_Tsqr, GIXOS_dQz, calc_film_DS_RRF_integ, film_integral_delta_beta_delta_phi
from scipy.integrate import dblquad

def load_metadata(yaml_path: str):
    """
    Load metadata from a YAML file and return all parameters 
    and derived quantities matching the original script.
    """
    # Load YAML
    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)

    # Raw parameters
    colorset               = meta["colorset"]
    gamma_E                = meta["gamma_E"]
    Qc                     = meta["Qc"]
    energy                 = meta["energy"]
    alpha_i                = meta["alpha_i"]
    Ddet                   = meta["Ddet"]
    pixel                  = meta["pixel"]
    footprint              = meta["footprint"]

    # Derived directly in original code
    wavelength             = 12404 / energy

    qxy0                   = np.array(meta["qxy0"])
    qxy0_select_idx        = meta["qxy0_select_idx"]
    qxy_bkg                = meta["qxy_bkg"]

    RqxyHW                 = meta["RqxyHW"]
    DSresHW                = meta["DSresHW"]
    DSpxHW                 = meta["DSpxHW"]

    # Paths and data loading
    path_xrr               = meta["paths"]["path_xrr"]
    xrr_datafile           = meta["paths"]["xrr_datafile"]
    xrr_data               = pd.read_csv(
        path_xrr + xrr_datafile,
        delim_whitespace=True
    )

    path                   = meta["paths"]["path"]
    path_out               = meta["paths"]["path_out"]

    sample                 = meta["sample"]
    scan                   = np.array(meta["scan"])
    bkgsample              = meta["bkgsample"]
    bkgscan                = np.array(meta["bkgscan"])

    I0ratio_sample2bkg     = meta["I0ratio_sample2bkg"]

    # Physical constants
    kb                     = meta["physical_constants"]["kb"]
    tension                = meta["physical_constants"]["tension"]
    temperature            = meta["physical_constants"]["temperature"]

    # Derived physics quantities
    kappa                  = meta["derived"]["kappa"]
    Lk                     = math.sqrt(kappa * kb * temperature / tension) * 1e10
    amin                   = meta["derived"]["amin"]
    qmax                   = math.pi / amin

    # RFscaling exactly as in the original script
    RFscaling = (
        I0ratio_sample2bkg
        * 1e11
        * 55 / 3
        / 6.8
        * (math.pi / 180) ** 2
        * (9.42e-6) ** 2
        / math.sin(math.radians(alpha_i))
        * 4
    )

    # Return all variables in a dictionary
    return {
        "colorset": colorset,
        "gamma_E": gamma_E,
        "Qc": Qc,
        "energy": energy,
        "alpha_i": alpha_i,
        "Ddet": Ddet,
        "pixel": pixel,
        "footprint": footprint,
        "wavelength": wavelength,
        "qxy0": qxy0,
        "qxy0_select_idx": qxy0_select_idx,
        "qxy_bkg": qxy_bkg,
        "RqxyHW": RqxyHW,
        "DSresHW": DSresHW,
        "DSpxHW": DSpxHW,
        "path_xrr": path_xrr,
        "xrr_datafile": xrr_datafile,
        "xrr_data": xrr_data,
        "path": path,
        "path_out": path_out,
        "sample": sample,
        "scan": scan,
        "bkgsample": bkgsample,
        "bkgscan": bkgscan,
        "I0ratio_sample2bkg": I0ratio_sample2bkg,
        "kb": kb,
        "tension": tension,
        "kappa": kappa,
        "temperature": temperature,
        "Lk": Lk,
        "amin": amin,
        "qmax": qmax,
        "RFscaling": RFscaling
    }
    


def load_data(yaml_path: str):
    """
    Load GIXOS sample and background data according to metadata in a YAML file.
    Returns two dicts: importGIXOSdata and importbkg, each containing:
      - "Intensity": 2D array (rows x len(qxy0))
      - "tt_qxy0": 2D array
      - "error": 2D array
      - "tt":  1D array (mean of tt_qxy0)
    """
    # Load metadata
    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)
    
    # Extract parameters
    sample      = meta["sample"]
    bkgsample   = meta["bkgsample"]
    path        = meta["paths"]["path"]
    qxy0        = np.array(meta["qxy0"])
    scan        = np.array(meta["scan"], dtype=int)
    bkgscan     = np.array(meta["bkgscan"], dtype=int)

    importGIXOSdata = None
    importbkg       = None

    # Loop over each qxy0 index
    for idx in range(len(qxy0)):
        # Construct filenames
        fileprefix       = f"{sample}-id{scan[idx]}"
        GIXOSfilename    = f"{path}{fileprefix}.txt"
        importGIXOS_qxy0 = np.loadtxt(GIXOSfilename, skiprows=16)

        # Initialize storage dicts on first iteration
        if importGIXOSdata is None:
            nrows = importGIXOS_qxy0.shape[0]
            ncols = len(qxy0)
            importGIXOSdata = {
                "Intensity": np.zeros((nrows, ncols)),
                "tt_qxy0":   np.zeros((nrows, ncols)),
                "error":     np.zeros((nrows, ncols)),
                "tt":        None
            }
        # Fix bad pixel row 269 by averaging rows 268 & 270
        mean_row = np.mean(importGIXOS_qxy0[[268, 270], :], axis=0)
        importGIXOS_qxy0[269, :] = mean_row

        # Populate sample data
        importGIXOSdata["Intensity"][:, idx] = importGIXOS_qxy0[:, 2]
        importGIXOSdata["tt_qxy0"][:, idx]   = importGIXOS_qxy0[:, 1] - 0.01
        importGIXOSdata["error"][:, idx]     = np.sqrt(importGIXOS_qxy0[:, 2])

        # Background file
        bkgprefix        = f"{bkgsample}-id{bkgscan[idx]}"
        bkgfilename      = f"{path}{bkgprefix}.txt"
        importbkg_qxy0   = np.loadtxt(bkgfilename, skiprows=16)

        if importbkg is None:
            nrows_bkg = importbkg_qxy0.shape[0]
            importbkg = {
                "Intensity": np.zeros((nrows_bkg, ncols)),
                "tt_qxy0":   np.zeros((nrows_bkg, ncols)),
                "error":     np.zeros((nrows_bkg, ncols)),
                "tt":        None
            }
        # Fix bad pixel
        importbkg_qxy0[269, :] = np.mean(importbkg_qxy0[[268, 270], :], axis=0)

        importbkg["Intensity"][:, idx] = importbkg_qxy0[:, 2]
        importbkg["tt_qxy0"][:, idx]   = importGIXOS_qxy0[:, 1]
        importbkg["error"][:, idx]     = np.sqrt(importbkg_qxy0[:, 2])

        # Scale for specific scan range
        if 29646 <= scan[idx] <= 29682:
            importGIXOSdata["Intensity"][:, idx] *= 2
            importGIXOSdata["error"][:, idx]     = np.sqrt(importGIXOSdata["Intensity"][:, idx])
            importbkg["Intensity"][:, idx]      *= 2
            importbkg["error"][:, idx]          = np.sqrt(importbkg["Intensity"][:, idx])

        print(f"{qxy0[idx]:f}", end="\t")

    # Compute mean tt over qxy0 for both dicts
    importGIXOSdata["tt"] = np.mean(importGIXOSdata["tt_qxy0"], axis=1)
    importbkg["tt"]       = np.mean(importbkg["tt_qxy0"], axis=1)

    return importGIXOSdata, importbkg

# Example usage:
# importGIXOSdata, importbkg = load_data("metadata.yaml")


def binning_GIXOS_data(importGIXOSdata, importbkg):
    binsize = 10
    groupnumber =  math.floor(importGIXOSdata ["Intensity"].shape[0] / binsize)      # look at the first row with .shape[0]
    num_columns = importGIXOSdata["Intensity"].shape[1]

    binneddata = None
    binnedbkg = None

    for groupidx in range(groupnumber): # why can't we just round up before if we are adding 1 to it?
        start = groupidx * binsize
        end = (groupidx + 1) * binsize
    
        if binneddata is None:
            binneddata = {
            "Intensity": np.zeros((groupnumber, num_columns)),
            "error": np.zeros((groupnumber, num_columns)),
            "tt": np.zeros(groupnumber),
        } 

        binneddata ["Intensity"][groupidx, :] = np.sum(importGIXOSdata ["Intensity"][start:end, :], axis=0)
        binneddata ["error"][groupidx, :] = np.sqrt(np.sum(importGIXOSdata ["error"][start:end, :]**2, axis=0))
        binneddata ["tt"][groupidx] = np.mean(importGIXOSdata ["tt"][start:end])

        if binnedbkg is None:
            binnedbkg = {
            "Intensity": np.zeros((groupnumber, num_columns)),
            "error": np.zeros((groupnumber, num_columns)),
            "tt": np.zeros((groupnumber, 1)),
        }
        binnedbkg ["Intensity"][groupidx, :] = np.sum(importbkg ["Intensity"][start:end, :], axis=0)
        binnedbkg ["error"][groupidx, :] = np.sqrt(np.sum(importbkg ["error"][start:end, :]**2, axis=0))
        binnedbkg ["tt"][groupidx, :] = np.mean(importbkg ["tt"][start:end], axis=0)    # no [ , :] because would index through all columns but we only have 1 in ["tt"]

    importGIXOSdata = binneddata
    importbkg = binnedbkg
    return importGIXOSdata, importbkg

def remove_negative_2theta(importGIXOSdata, importbkg):
    tt_step = np.mean(importGIXOSdata ["tt"][1:] - importGIXOSdata ["tt"][0:-1])  # calculating the step size of tt for future function
    indices =  np.where(importGIXOSdata ["tt"] < 0)[0] # finding indices where value stored is less than 0
    tt_start_idx = indices[-1] if len(indices) > 0 else None  # taking the last  value of indices, and checking if indices is a valid list to take from

    importGIXOSdata["Intensity"] = importGIXOSdata["Intensity"][tt_start_idx+1 :,:]
    importGIXOSdata["error"] = importGIXOSdata["error"][tt_start_idx+1 :,:]
    importGIXOSdata["tt"] = importGIXOSdata["tt"][tt_start_idx+1 :]
    importbkg["Intensity"] = importbkg["Intensity"][tt_start_idx+1 :,:]
    importbkg["error"] = importbkg["error"][tt_start_idx+1 :,:]
    importbkg["tt"] = importbkg["tt"][tt_start_idx+1 :]  # in essence, we are removing the first rows of the data that have negative tt values
    return importGIXOSdata, importbkg, tt_step

def real_space_2theta(metadata):
    metadata["tth"] = np.degrees(np.arcsin(metadata["qxy0"][ metadata["qxy0_select_idx"] ] * metadata["wavelength"] / 4 / np.pi)) * 2          # math.asin would work if not a list
    metadata["tth_roiHW_real"] = np.degrees(metadata["pixel"] * metadata["DSpxHW"] / metadata["Ddet"])
    metadata["DSqxyHW_real"]= np.radians(metadata["tth_roiHW_real"]) / 2 * 4 * np.pi / metadata["wavelength"] * np.cos(np.radians(metadata["tth"]/2))
    return metadata    # we add tth, tth_roiHW_real, and DSqxyHW_real to the metadata dict so we can use it later in the plotting function

# will have here for now, but might need to break up into computation and plotting functions & will need to rewrite parts to make more simple inputs
def GIXOS_data_plot_prep(importGIXOSdata, importbkg, metadata, tt_step, wide_angle = True):
    qxy_bkg = 0.3
    GIXOS = {
        "tt": importGIXOSdata["tt"],
        "GIXOS_raw": importGIXOSdata["Intensity"] [:, metadata["qxy0_select_idx"]],
        "GIXOS_bkg": importbkg["Intensity"] [:, metadata["qxy0_select_idx"]]
    }
    DSbetaHW = np.mean(GIXOS["tt"][1:] - GIXOS["tt"][0:-1]) / 2    # for later use in upcoming functions
    qxy0_idx = np.where(metadata["qxy0"] > qxy_bkg)
    qxy0_idx = qxy0_idx[0]
    if len(qxy0_idx) == 0:
        qxy0_idx = [len(metadata["qxy0"]) + 1]
    GIXOS["Qxy"] = 2 * np.pi / metadata["wavelength"] * np.sqrt((np.cos(np.radians(GIXOS["tt"])) * np.sin(np.radians(metadata["tth"])))**2 + (np.cos(np.radians(metadata["alpha_i"])) - np.cos(np.radians(GIXOS["tt"])) * np.cos(np.radians(metadata["tth"])))**2)
    GIXOS["Qz"] = 2 * np.pi / metadata["wavelength"] * (np.sin(np.radians(GIXOS["tt"])) + np.sin(np.radians(metadata["alpha_i"])))
    GIXOS["GIXOS_raw"] = importGIXOSdata["Intensity"] [:, metadata["qxy0_select_idx"]]
    GIXOS["GIXOS_bkg"] = importbkg["Intensity"] [:, metadata["qxy0_select_idx"]]
    if qxy0_idx[0] <= len(metadata["qxy0"]):
        GIXOS["raw_largetth"] = np.mean(importGIXOSdata["Intensity"][:, int(qxy0_idx):], axis = 1)
        GIXOS["bkg_largetth"] = np.mean(importbkg["Intensity"][:, int(qxy0_idx):], axis = 1)
        bulkbkg = GIXOS["raw_largetth"] - GIXOS["bkg_largetth"]
    else:
        bulkbkg = np.zeros(len(metadata["qxy0"]))

    fdtt = np.radians(tt_step) / (np.arctan((np.tan(np.radians(GIXOS ["tt"]))*metadata["Ddet"] + metadata["pixel"]/2)/metadata["Ddet"]) - np.arctan((np.tan(np.radians(GIXOS ["tt"]))*metadata["Ddet"] - metadata["pixel"] / 2) / metadata["Ddet"]))
    fdtt = fdtt / fdtt [0]


    # add if statement here for no wide angle/wide angle
    if wide_angle:
        GIXOS["GIXOS"] = (GIXOS ["GIXOS_raw"] - GIXOS ["GIXOS_bkg"]) * fdtt - bulkbkg * fdtt
        GIXOS["error"] = np.sqrt(importGIXOSdata["error"][:, metadata["qxy0_select_idx"]]**2 + importbkg["error"][:, metadata["qxy0_select_idx"]]**2) * fdtt
    else:
        GIXOS["GIXOS"]= (GIXOS["GIXOS_raw"] - GIXOS["GIXOS_bkg"]  ) * fdtt - np.mean(bulkbkg[-1-10:-1],1) * fdtt 
        GIXOS["GIXOS"] = GIXOS["GIXOS"] - np.mean(GIXOS["GIXOS"][-1-5:-1]) * 0.5     # GIXOS ["error"] does not exist if we use this, so it will result in errors later; maybe ask Chen?
    
    return GIXOS, DSbetaHW


def GIXOS_RF_and_SF(GIXOS, metadata, DSbetaHW):
    GIXOS["fresnel"] = GIXOS_fresnel(GIXOS ["Qz"], metadata["Qc"]) # check if fresnel == GIXOS_fresnel      SAME
    GIXOS["Qz_array"] = np.asarray(GIXOS ["Qz"]).reshape(-1, 1) # done to convert GIXOS ["Qz"] from a row vetor to a column vector for GIXOS_Tsqr
    GIXOS["transmission"] = GIXOS_Tsqr(GIXOS["Qz_array"], metadata["Qc"], metadata["energy"], metadata["alpha_i"], metadata["Ddet"], metadata["footprint"])  #  Mostly the same, but the 4th column starts to deviate from the MATLAB output by hundredths
    GIXOS["dQz"] = GIXOS_dQz(GIXOS["Qz"], metadata["energy"], metadata["alpha_i"], metadata["Ddet"], metadata["footprint"])     # Almost same, just not iterating through enough times(?) --> missing last row      SAME now

    GIXOS["DS_RRF_integ"], GIXOS["DS_term_integ"], GIXOS["RRF_term_integ"] = calc_film_DS_RRF_integ(GIXOS["tt"], metadata["qxy0"][ metadata["qxy0_select_idx"] ], metadata["energy"]/1000, metadata["alpha_i"], metadata["RqxyHW"], metadata["DSqxyHW_real"], DSbetaHW, metadata["tension"], metadata["temperature"], metadata["kappa"], metadata["amin"], use_approx=False)
    # DS = Diffuse Scatter; RRF = Specular Reflectivity Normalized by Fresnel Reflectivity
    # Approx form is derived from Taylor expansion, which is dependent on being close to 0 angle --> higher deviations at high angles

    # computes reflectivity
    GIXOS["refl"] = np.column_stack([
        GIXOS["Qz"],
        GIXOS["GIXOS"] / GIXOS["DS_RRF_integ"] * GIXOS["fresnel"][:, 1] / GIXOS["transmission"][:, 3],
        GIXOS["error"] / GIXOS["DS_RRF_integ"] * GIXOS["fresnel"][:, 1] / GIXOS["transmission"][:, 3],
        GIXOS["dQz"][:, 4]
    ])

    # computes structure factor 
    GIXOS["SF"] = np.column_stack([
        GIXOS["Qz"],
        GIXOS["GIXOS"] / GIXOS["DS_term_integ"] / GIXOS["transmission"][:, 3],
        GIXOS["error"] / GIXOS["DS_term_integ"] / GIXOS["transmission"][:, 3],
        GIXOS["dQz"][:, 4]
    ])
    return GIXOS # outputs GIXOS with reflectivity and structure factor added as new columns


def rect_slit_function(GIXOS, metadata):
    xrr_data = pd.read_csv(metadata["path_xrr"] + metadata["xrr_datafile"], delim_whitespace=True)
    xrr_config = {
        "energy" : 14400,
        "sdd" : 1039.9,
        "slit_h" : 1,
        "slit_v" : 0.66, 
    }

    xrr_config["wavelength"] = 12400/xrr_config["energy"]
    xrr_config["wave_number"] = 2 * np.pi / xrr_config["wavelength"]
    xrr_config["Qz"] = GIXOS["Qz"]
    xrr_config["dataQz"] = xrr_data.iloc[:, 0].astype(float).to_numpy()  # see later bc accessing data !!!!!!!!!!!!!!!!!!!  might have bugs bc came out as strings and need to convert to float
    xrr_config["beta_xrr"] = np.degrees(np.arcsin(xrr_config["Qz"] / 2 / xrr_config["wave_number"]))
    xrr_config["beta_xrr"] = xrr_config["beta_xrr"].reshape(-1, 1) # do this, otherwise beta_xrr has shape (46,) instead of (46, 1) which will mess up xrr_config_phi_array_for_qxy_slit_min and make it (46, 46) instead of (46, 1) like MATLAB code
    xrr_config["dataRF"] = (( xrr_config["dataQz"] - np.lib.scimath.sqrt(xrr_config["dataQz"]**2 - metadata["Qc"]**2)) / (xrr_config["dataQz"] + np.lib.scimath.sqrt(xrr_config["dataQz"]**2 - metadata["Qc"]**2))) * np.conj((xrr_config["dataQz"] - np.lib.scimath.sqrt(xrr_config["dataQz"]**2 - metadata["Qc"]**2)) / (xrr_config["dataQz"] + np.lib.scimath.sqrt(xrr_config["dataQz"]**2 - metadata["Qc"]**2)))   # needed for final file conversion
    xrr_config["RF"] = GIXOS["fresnel"][:, 1]
    xrr_config["kbT_gamma"] = metadata["kb"] * metadata["temperature"] / metadata["tension"] * 10 ** 20
    xrr_config["eta"] = xrr_config["kbT_gamma"] / 2 / np.pi * xrr_config["Qz"] ** 2
    # maybe delete xrr_config for simplicity
    xrr_config["delta_phi_HW"] = np.degrees(np.arctan(xrr_config["slit_h"] / 2 / xrr_config["sdd"] / np.cos(np.radians(xrr_config["beta_xrr"]))))
    xrr_config["delta_beta_HW"] =   np.degrees(np.arcsin(xrr_config["slit_v"] / 2 / xrr_config["sdd"] * np.cos(np.radians(xrr_config["beta_xrr"]))))

    xrr_config["slit_h_coord"] = np.arange(-xrr_config["slit_h"] / 2, xrr_config["slit_h"] / 2 + 0.005, 0.005)
    xrr_config["slit_v_coord"] = np.arange(-xrr_config["slit_v"] / 2, xrr_config["slit_v"] / 2 + 0.005, 0.005)
    xrr_config["slit_t"] = np.column_stack((xrr_config["slit_h_coord"], np.ones(len(xrr_config["slit_h_coord"])) * xrr_config["slit_v"] / 2))  
    xrr_config["slit_b"] = np.column_stack((xrr_config["slit_h_coord"], np.ones(len(xrr_config["slit_h_coord"])) * -xrr_config["slit_v"] / 2))
    xrr_config["slit_l"] = np.column_stack((np.ones(len(xrr_config["slit_v_coord"])) * -xrr_config["slit_h"] / 2, xrr_config["slit_v_coord"]))
    xrr_config["slit_r"] = np.column_stack((np.ones(len(xrr_config["slit_v_coord"])) * xrr_config["slit_h"] / 2, xrr_config["slit_v_coord"]))
    xrr_config["slit_coord"] = np.concatenate((xrr_config["slit_t"], xrr_config["slit_r"], np.flipud(xrr_config["slit_b"]), np.flipud(xrr_config["slit_l"])), axis = 0)
    xrr_config["qxy_slit"] = np.zeros( (xrr_config["slit_coord.shape"][0], 2, xrr_config["beta_xrr"].shape[0]) )
    xrr_config["qxy_slit_min"] = np.zeros( (xrr_config["beta_xrr"].shape[0], 1) )
    xrr_config["ang"] = np.arange(0, 2 * np.pi, 0.01)
    #xrr_config_ang = xrr_config_ang.reshape(-1, 1)  # transposing to make it a column vector
    xrr_config["qxy_slit_min_coord"] = np.zeros( (xrr_config["ang"].shape[0], 2, xrr_config["qxy_slit_min"].shape[0]))
    for idx in range(len(xrr_config["beta_xrr"])):
        xrr_config["qxy_slit"][:, :, idx] = xrr_config["wave_number"] * np.column_stack([xrr_config["slit_coord"][:, 0] / xrr_config["sdd"], xrr_config["slit_coord"][:, 1] / xrr_config["sdd"] * np.sin(np.radians(xrr_config["beta_xrr"][idx]))])
        xrr_config["qxy_slit_min"][idx, 0] = np.min(np.sqrt(xrr_config["qxy_slit"][:, 0, idx]**2 + xrr_config["qxy_slit"][:, 1, idx]**2))
        xrr_config["qxy_slit_min_coord"][:, :, idx] = xrr_config["qxy_slit_min"][idx] * np.column_stack([np.cos(xrr_config["ang"]), np.sin(xrr_config["ang"])])  # might not need np.array

    xrr_config["phi_max_qxy_slit_min"] = np.degrees(np.arctan(xrr_config["qxy_slit_min"] / xrr_config["wave_number"] / np.cos(np.radians(xrr_config["beta_xrr"]))))

    xrr_config["phi_array_for_qxy_slit_min"] = xrr_config["phi_max_qxy_slit_min"] * np.array([0, 1/5, 2/5, 3/5, 4/5])
    xrr_config["delta_beta_array_for_qxy_slit_min"] = np.degrees(
        np.arcsin(
            (np.sqrt(
                np.maximum(xrr_config["qxy_slit_min"][:, 0:1]**2 - 
                        (np.tan(np.radians(xrr_config["phi_array_for_qxy_slit_min"])) * 
                            np.cos(np.radians(xrr_config["beta_xrr"])) * 
                            xrr_config["wave_number"]) ** 2, 0))
            / (xrr_config["wave_number"] * np.sin(np.radians(xrr_config["beta_xrr"]))))
            * np.cos(np.radians(xrr_config["beta_xrr"]))
        )
    )

    delta_beta_HW_1d = xrr_config["delta_beta_HW"][:, 0]
    for idx in range(xrr_config["delta_beta_array_for_qxy_slit_min"].shape[1]):
        repidx = xrr_config["delta_beta_array_for_qxy_slit_min"][:, idx] >= delta_beta_HW_1d
        xrr_config["delta_beta_array_for_qxy_slit_min"][repidx, idx] = delta_beta_HW_1d[repidx]  # replace values that are greater than delta_beta_HW with delta_beta_HW

    xrr_config["phi_array_for_qxy_slit_min"] = np.hstack([xrr_config["phi_array_for_qxy_slit_min"], xrr_config["phi_max_qxy_slit_min"]]) # might not work with np.column_stack bc size mismatch --> column_hstack

    xrr_config["bkgoff"] = 1
    xrr_config["bkg_phi"] = np.degrees(np.arctan(xrr_config["bkgoff"] / (xrr_config["sdd"] * np.cos(np.radians(xrr_config["beta_xrr"])))))   # off by ten thousandths place - supposed to get larger as index increases, but decreases instead?


    xrr_config["r_step"] = 0.001
    xrr_config["r"] = np.sqrt(np.maximum(np.arange(0.001, 8*round(metadata["Lk"]) + xrr_config["r_step"], xrr_config["r_step"]) ** 2 + metadata["amin"] ** 2, 0))
    xrr_config["C_integrand"] = np.zeros((len(xrr_config["Qz"]), len(xrr_config["r"])))
    for idx in range(len(xrr_config["Qz"])):
        xrr_config["C_integrand"][idx, :] = 2 * np.pi * xrr_config["r"]**(1 - xrr_config["eta"][idx]) * (np.exp(-xrr_config["eta"][idx] * besselk(0, xrr_config["r"] / metadata["Lk"])) - 1)    # off by thousandths place
    # Matches up till here

    xrr_config["C"] = np.sum(xrr_config["C_integrand"], axis = 1) * xrr_config["r_step"]
    xrr_config["qxy_slit_min_flat"] = xrr_config["qxy_slit_min"].flatten()  # (46,) so that RRF_term does not return a (46, 46) array since MATLAB returns a (46, 1)
    xrr_config["RRF_term"] = (xrr_config["qxy_slit_min_flat"] ** xrr_config["eta"] + xrr_config["qxy_slit_min_flat"] ** 2 * xrr_config["C"] / 4 / np.pi) * (1/metadata["qmax"]) ** xrr_config["eta"] * np.exp(xrr_config["eta"] * besselk(0, 1 / metadata["Lk"] / metadata["qmax"]))
    xrr_config["specular_qxy_min"] = xrr_config["RF"] * xrr_config["RRF_term"] # off by hundredths/thousandths

    xrr_config["region_around_radial_u_r"] = np.zeros((len(xrr_config["beta_xrr"]), xrr_config["delta_beta_array_for_qxy_slit_min"].shape[1]))    # off by ~5-8 thousandths
    xrr_config["region_around_radial_l_r"] = np.zeros((len(xrr_config["beta_xrr"]), xrr_config["delta_beta_array_for_qxy_slit_min"].shape[1]))    # off by ~5-8 thousandths
    xrr_config["diff_r"] = np.zeros((len(xrr_config["beta_xrr"]), 1))    # VERY OFF
    xrr_config["diff_r_bkgoff"] = np.zeros((len(xrr_config["beta_xrr"]), 1))    # VERY OFF

    xrr_config["diff_r"] = xrr_config["diff_r"].flatten()
    xrr_config["diff_r_bkgoff"] = xrr_config["diff_r_bkgoff"].flatten()

    for idx in range(len(xrr_config["beta_xrr"])):
        fun_film = lambda tt, tth: film_integral_delta_beta_delta_phi(tt, tth, xrr_config["kbT_gamma"], xrr_config["wave_number"], xrr_config["beta_xrr"][idx], metadata["Lk"], metadata["amin"])
        for phi_idx in range(xrr_config["delta_beta_array_for_qxy_slit_min"].shape[1]):
            upper, _ = dblquad(lambda tth, tt: fun_film(tt, tth), (xrr_config["beta_xrr"][idx] + xrr_config["delta_beta_array_for_qxy_slit_min"][idx, phi_idx]), (xrr_config["beta_xrr"][idx] + xrr_config["delta_beta_HW"][idx]), lambda _: xrr_config["phi_array_for_qxy_slit_min"][idx, phi_idx], lambda _: xrr_config["phi_array_for_qxy_slit_min"][idx, phi_idx + 1],  epsabs=1e-12, epsrel=1e-10)
            xrr_config["region_around_radial_u_r"][idx, phi_idx] = upper * (np.pi/180) ** 2 * (9.42e-6)**2 / np.sin(np.radians(xrr_config["beta_xrr"][idx]))

            lower, _ = dblquad(lambda tth, tt: fun_film (tt, tth), (xrr_config["beta_xrr"][idx] - xrr_config["delta_beta_HW"][idx]), (xrr_config["beta_xrr"][idx] - xrr_config["delta_beta_array_for_qxy_slit_min"][idx, phi_idx]), lambda _: xrr_config["phi_array_for_qxy_slit_min"][idx, phi_idx], lambda _: xrr_config["phi_array_for_qxy_slit_min"][idx, phi_idx + 1],  epsabs=1e-12, epsrel=1e-10)
            xrr_config["region_around_radial_l_r"][idx, phi_idx] = lower * (np.pi/180)**2 * (9.42e-6)**2 / np.sin(np.radians(xrr_config["beta_xrr"][idx]))

        # Run the integral
        result, _ = dblquad(
        func=fun_film,
        a = xrr_config["beta_xrr"][idx] - xrr_config["delta_beta_HW"][idx],
        b = xrr_config["beta_xrr"][idx] + xrr_config["delta_beta_HW"][idx],
        gfun=lambda beta: xrr_config["phi_max_qxy_slit_min"][idx],
        hfun=lambda beta: xrr_config["delta_phi_HW"][idx] + xrr_config["delta_phi_HW"][idx],  epsabs=1e-12, epsrel=1e-10
        )

    # Store the result (second column, index 0 in Python)
        xrr_config["diff_r"][idx] = (result * (np.pi / 180)**2 * (9.42e-6)**2 / np.sin(np.radians(xrr_config["beta_xrr"][idx])))
            
        result2, _ = dblquad(func=fun_film,
        a=xrr_config["beta_xrr"][idx] - xrr_config["delta_beta_HW"][idx],
        b=xrr_config["beta_xrr"][idx] + xrr_config["delta_beta_HW"][idx],
        gfun=lambda beta: xrr_config["bkg_phi"][idx] - xrr_config["delta_phi_HW"][idx],
        hfun=lambda beta: xrr_config["bkg_phi"][idx] + xrr_config["delta_phi_HW"][idx],  epsabs=1e-12, epsrel=1e-10
        )
        xrr_config["diff_r_bkgoff"][idx] = (result2 * (np.pi / 180)**2 * (9.42e-6)**2 / np.sin(np.radians(xrr_config["beta_xrr"][idx])))



    xrr_config["Rterm_rect_slit"] = xrr_config["specular_qxy_min"] + 2*(np.sum(xrr_config["region_around_radial_u_r"] + xrr_config["region_around_radial_u_r"], axis = 1) + xrr_config["diff_r"])
    xrr_config["bkgterm_rect_slit"] = xrr_config["diff_r_bkgoff"]
    return xrr_config






def conversion_to_reflectivity(GIXOS, xrr_config):
    numerator_scaling = (xrr_config["Rterm_rect_slit"] - xrr_config["bkgterm_rect_slit"])
    denominator = GIXOS["DS_term_integ"] * GIXOS["transmission"][:, 3]  # Element-wise multiplication

    GIXOS["refl_recSlit"] = np.column_stack([
        GIXOS["Qz"],
        GIXOS["GIXOS"] * numerator_scaling / denominator,    # off by a bit
        GIXOS["error"] * numerator_scaling / denominator,    # off by a bit - see how to fix these 2
        GIXOS["dQz"][:, 4]                                               # issues are caused by numerator_scaling for sure (and it is both xrr_config_Rterm_rect_slit and xrr_config_bkgterm_rect_slit) 
    ])

    GIXOS["refl_roughness_term"] = (xrr_config["Rterm_rect_slit"] - xrr_config["bkgterm_rect_slit"]) / xrr_config["RF"]
    GIXOS["refl_roughness"] = np.sqrt(-np.log(GIXOS["refl_roughness_term"]) / GIXOS["Qz"]**2)
    GIXOS["SF"] = np.column_stack([GIXOS["SF"], GIXOS["refl_roughness"], GIXOS["refl_roughness_term"]])
    return GIXOS









def GIXOS_file_output(GIXOS, xrr_config, metadata, tt_step):
    xrrfilename = f"{metadata['path_out']}{metadata['sample']}_{metadata['scan'][ metadata['qxy0_select_idx'] ]:05d}_R_PYTHON_TEST.dat" # becomes "instrument_46392_R_PYTHON.dat" - qz and dqz columns are very accurate, but R and dR start off semi-accurate but increasingly deviate after ~15th value
    with open(xrrfilename, 'w') as f:
        f.write(f"# files\n")
        f.write(f"sample file: {metadata['sample']}-id{metadata['scan'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"background file: {metadata['bkgsample']}-id{metadata['bkgscan'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"wide angle bkg at qxy0 = {metadata['qxy_bkg']:.6f} /A\n")
        f.write(f"# geometry\n")
        f.write(f"energy [eV]: {metadata['energy']:.2f}\n")
        f.write(f"incidence [deg]: {metadata['alpha_i']}\n")
        f.write(f"footprint [mm]: {metadata['footprint']:.1f}\n")
        f.write(f"sdd [mm]: {metadata['Ddet']:.2f}\n")
        f.write(f"qxy resolution HWHM at specular [A^-1]: {metadata['DSresHW']}\n")
        f.write(f"phi_opening [deg]: {metadata['tth_roiHW_real']}\n")
        f.write(f"beta_step [deg]: {tt_step}\n")
        f.write(f"# DS-XRR conversion optics setting\n")
        f.write(f"phi [deg]: {metadata['tth']}\n")
        f.write(f"qxy(beta=0) [A^-1]: {metadata['qxy0'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"phi integration HW [deg]: {metadata['tth_roiHW_real']}\n")
        f.write(f"corresponding qxy HW [A^-1]: {metadata['DSqxyHW_real']}\n")
        f.write(f"R slit: {xrr_config['slit_v']} mm (v) {xrr_config['slit_h']} mm (h) at {xrr_config['sdd']} mm distance, {xrr_config['energy']} eV beam energy\n")
        f.write(f"scaling: {metadata['RFscaling']}\n")
        f.write(f"# DS-XRR conversion sample setting\n")
        f.write(f"tension [N/m]: {metadata['tension']}\n")
        f.write(f"temperature [K]: {metadata['temperature']:.1f}\n")
        f.write(f"kappa [kbT]: {metadata['kappa']:.1f}\n")
        f.write(f"CW short cutoff [A]: {metadata['amin']}\n")
        f.write(f"CW and Kapa roughness [A]: {GIXOS['refl_roughness'][0]} to {GIXOS['refl_roughness'][-1]}\n")
        f.write("# data\nqz\tR\tdR\tdqz\n[A^-1]\t[a.u.]\t[a.u.]\t[A^-1]\n")

    # Save reflectivity data
    with open(xrrfilename, 'a') as f:
        np.savetxt(f, GIXOS["refl_recSlit"], delimiter='\t', fmt='%.6e')
    #    np.savetxt(f, refl_recSlit, delimiter='\t', fmt='%.6e', comments='', header='', encoding='utf-8', newline='\n', append=True)


    # ---- FILE 2: DS/(R/RF) ----
    ds2rrf_filename = f"{metadata['path_out']}{metadata['sample']}_{metadata['scan'][ metadata['qxy0_select_idx'] ]:05d}_DS2RRF_PYTHON_TEST.dat" # becomes "instrument_46392_DS2RRF_PYTHON.dat" - first column, less than 0.1% error ; if we approx at the same decimal point that MATLAB appears to round off at, would be the same values - second column: starts semi close (less than 0.1% error), but deviates heavily by the end (~x2.5 the actual value it is supposed to have)
    with open(ds2rrf_filename, 'w') as f:
        f.write(f"# files\n")
        f.write(f"sample file: {metadata['sample']}-id{metadata['scan'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"background file: {metadata['bkgsample']}-id{metadata['bkgscan'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"wide angle bkg at qxy0 = {metadata['qxy_bkg']:.6f} /A\n")
        f.write(f"# geometry\n")
        f.write(f"energy [eV]: {metadata['energy']:.2f}\n")
        f.write(f"incidence [deg]: {metadata['alpha_i']}\n")
        f.write(f"footprint [mm]: {metadata['footprint']:.1f}\n")
        f.write(f"sdd [mm]: {metadata['Ddet']:.2f}\n")
        f.write(f"qxy resolution HWHM at specular [A^-1]: {metadata['DSresHW']}\n")
        f.write(f"phi_opening [deg]: {metadata['tth_roiHW_real']}\n")
        f.write(f"beta_step [deg]: {tt_step}\n")
        f.write(f"# DS-XRR conversion optics setting\n")
        f.write(f"phi [deg]: {metadata['tth']}\n")   
        f.write(f"qxy(beta=0) [A^-1]: {metadata['qxy0'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"phi integration HW [deg]: {metadata['tth_roiHW_real']}\n")
        f.write(f"corresponding qxy HW [A^-1]: {metadata['DSqxyHW_real']}\n")
        f.write(f"R slit: {xrr_config['slit_v']} mm (v) {xrr_config['slit_h']} mm (h) at {xrr_config['sdd']} mm distance, {xrr_config['energy']} eV beam energy\n")
        f.write(f"scaling: {metadata['RFscaling']}\n")
        f.write(f"# DS-XRR conversion sample setting\n")
        f.write(f"tension [N/m]: {metadata['tension']}\n")
        f.write(f"temperature [K]: {metadata['temperature']:.1f}\n")
        f.write(f"kappa [kbT]: {metadata['kappa']:.1f}\n")
        f.write(f"CW short cutoff [A]: {metadata['amin']}\n")
        f.write(f"CW and Kapa roughness [A]: {GIXOS['refl_roughness'][0]} to {GIXOS['refl_roughness'][-1]}\n")
        f.write("# data\nqz\tDS/(R/RF)\n[A^-1]\t[a.u.]\n")

    ds_over_rrf = GIXOS["DS_term_integ"] / (xrr_config["Rterm_rect_slit"] / xrr_config["RF"])
    with open(ds2rrf_filename, 'a') as f:
        np.savetxt(f, np.column_stack((GIXOS["Qz"], ds_over_rrf)), delimiter='\t', fmt='%.6e')
    #     np.savetxt(f, np.column_stack((GIXOS["Qz"], ds_over_rrf)), delimiter='\t', fmt='%.6e', comments='', header='', encoding='utf-8', newline='\n', append=True)


    # ---- FILE 3: Structure Factor ----
    sf_filename = f"{metadata['path_out']}{metadata['sample']}_{metadata['scan'][ metadata['qxy0_select_idx'] ]:05d}_SF_PYTHON_TEST.dat" # becomes "instrument_46392_SF_PYTHON.dat" - first four columns are largely accurate; last 2 columns deviate (first is off by ~8%, second is off by a larger margin but both start semi-close and then increasingly deviate as index increases)
    with open(sf_filename, 'w') as f:
        f.write(f"# pure structure factor and kapa/cw roughness with its decay term under given XRR resolution\n")
        f.write(f"# files\nsample file: {metadata['sample']}-id{metadata['scan'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"background file: {metadata['bkgscan']}-id{metadata['bkgscan'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"wide angle bkg at qxy0 = {metadata['qxy_bkg']:.6f} /A\n")    
        f.write(f"# geometry\n")
        f.write(f"energy [eV]: {metadata['energy']:.2f}\n")
        f.write(f"incidence [deg]: {metadata['alpha_i']}\n")
        f.write(f"footprint [mm]: {metadata['footprint']:.1f}\n")
        f.write(f"sdd [mm]: {metadata['Ddet']:.2f}\n")
        f.write(f"qxy resolution HWHM at specular [A^-1]: {metadata['DSresHW']}\n")
        f.write(f"phi_opening [deg]: {metadata['tth_roiHW_real']}\n")
        f.write(f"beta_step [deg]: {tt_step}\n")
        f.write(f"# DS-XRR conversion optics setting\n")
        f.write(f"phi [deg]: {metadata['tth']}\n")   
        f.write(f"qxy(beta=0) [A^-1]: {metadata['qxy0'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"phi integration HW [deg]: {metadata['tth_roiHW_real']}\n")
        f.write(f"corresponding qxy HW [A^-1]: {metadata['DSqxyHW_real']}\n")
        f.write(f"R slit: {xrr_config['slit_v']} mm (v) {xrr_config['slit_h']} mm (h) at {xrr_config['sdd']} mm distance, {xrr_config['energy']} eV beam energy\n")
        f.write(f"scaling: {metadata['RFscaling']}\n")
        f.write(f"# DS-XRR conversion sample setting\n")
        f.write(f"tension [N/m]: {metadata['tension']}\n")
        f.write(f"temperature [K]: {metadata['temperature']:.1f}\n")
        f.write(f"kappa [kbT]: {metadata['kappa']:.1f}\n")
        f.write(f"CW short cutoff [A]: {metadata['amin']}\n")
        f.write(f"CW and Kapa roughness [A]: {GIXOS['refl_roughness'][0]} to {GIXOS['refl_roughness'][-1]}\n")
        f.write("# data\nqz\tSF\tdSF\tdQz\tsigma_R\texp(-qz2sigma2)\n[A^-1]\t[a.u.]\t[a.u.]\t[A^-1]\t[A^-1]\t[a.u.]\n")

    with open(sf_filename, 'a') as f:
        np.savetxt(f, GIXOS["SF"], delimiter='\t', fmt='%.6e')



from pseudo_xrr.plots import GIXOS_data_plot, R_data_plot
def rectangular_slit():
    importGIXOSdata, importbkg = load_data('./testing_data/gixos_metadata.yaml')
    metadata = load_metadata('./testing_data/gixos_metadata.yaml')
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