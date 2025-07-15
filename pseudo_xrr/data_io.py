import yaml
import numpy as np
import pandas as pd
import math

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
    kappa                  = meta["physical_constants"]["kappa"]
    temperature            = meta["physical_constants"]["temperature"]

    # Derived physics quantities
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



def load_GIXOS_data(path, qxy0, scan):
    # put clear and close all commands up here
    import numpy as np
    sample = "instrument"
    bkgsample = 'instrument'
    bkgscan = scan+1

    importGIXOSdata = None
    importbkg = None

    for idx in range(len(qxy0)):
        fileprefix = f"{sample}-id{int(scan[idx])}"        # sample + "-id" + int(scan[idx],'%d')  # fix later
        GIXOSfilename = path + fileprefix + ".txt"
        importGIXOSdata_qxy0 = np.loadtxt(GIXOSfilename, skiprows = 16)
        #importGIXOSdata_qxy0 = importGIXOSdata_qxy0.to_numpy() 
        if importGIXOSdata is None:
            importGIXOSdata = {
            "Intensity": np.zeros((importGIXOSdata_qxy0.shape[0], len(qxy0))),
            "tt_qxy0": np.zeros((importGIXOSdata_qxy0.shape[0], len(qxy0))),
            "error": np.zeros((importGIXOSdata_qxy0.shape[0], len(qxy0))),
            "tt": np.zeros((importGIXOSdata_qxy0.shape[0], len(qxy0))),
        }  

        # Compute element-wise mean
        mean_row = np.mean(importGIXOSdata_qxy0[[268, 270], :], axis=0)

        # Assign the mean to row 269
        importGIXOSdata_qxy0[269, :] = mean_row   # .data is the name of the column, so need to convert that to Python list access

        importGIXOSdata ["Intensity"][:, idx] = importGIXOSdata_qxy0[:,2]
        importGIXOSdata ["tt_qxy0"][:,idx] = importGIXOSdata_qxy0 [:,1] - 0.01
        importGIXOSdata ["error"][:,idx] = np.sqrt(importGIXOSdata_qxy0 [:,2]) # might run into errors bc math only works for scalars & not arrays   

        #file name of the bkg GISAXS to be imported
        bkgfileprefix = f"{bkgsample}-id{int(bkgscan[idx])}"
        bkgGIXOSfilename = path + bkgfileprefix + ".txt"
        importbkg_qxy0 = np.loadtxt(bkgGIXOSfilename, skiprows = 16)
        #importbkg_qxy0 = importbkg_qxy0.to_numpy()    # ~~~~~~~~~~~~~~~~~~~ Use to import using pandas with:   importbkg_qxy0 = pd.read_csv(filename, delimiter=' ', skiprows=16)    but did not read well; would skip a line at first can likely rewrite with pandas if needed
        if importbkg is None:
            importbkg = {
                "Intensity": np.zeros((importbkg_qxy0.shape[0], len(qxy0) )),
                "tt_qxy0": np.zeros((importbkg_qxy0.shape[0], len(qxy0))),
                "error": np.zeros((importbkg_qxy0.shape[0], len(qxy0))),
                "tt": np.zeros((importbkg_qxy0.shape[0], len(qxy0))),
            }

        importbkg_qxy0 [269,:] = np.mean(importbkg_qxy0 [[268,270],:], axis=0); # bad pixel
        importbkg ["Intensity"][:,idx] = importbkg_qxy0 [:,2]
        importbkg ["tt_qxy0"][:,idx] = importGIXOSdata_qxy0 [:,1]     # why not use importbkg_qxy0[:,1] ?
        importbkg ["error"][:,idx] = np.sqrt(importbkg_qxy0 [:,2])

        if (scan[idx]>=29646 and scan[idx]<=29682):
            importGIXOSdata ["Intensity"][:,idx] = importGIXOSdata ["Intensity"][:,idx]*2
            importGIXOSdata ["error"][:,idx] = np.sqrt(importGIXOSdata ["Intensity"][:,idx])
            importbkg ["Intensity"][:,idx] = importbkg ["Intensity"][:,idx]*2
            importbkg ["error"][:,idx] = np.sqrt(importbkg ["Intensity"][:,idx])
        print(f"{qxy0[idx]:f}", end='\t')     #      fprintf('%f/t', qxy0[idx])        # what i had initially was:   print('/t'.join(f"{qxy0[idx]:f}"))
        
    importGIXOSdata ["tt"] = np.mean(importGIXOSdata ["tt_qxy0"], axis = 1)
    importbkg ["tt"] = np.mean(importbkg ["tt_qxy0"], axis = 1)
    return importGIXOSdata, importbkg