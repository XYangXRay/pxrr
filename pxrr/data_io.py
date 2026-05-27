from ruamel.yaml import YAML
import json
import h5py

import numpy as np
import pandas as pd
from scipy.constants import pi, Boltzmann as kb
import os
import platform
from pathlib import Path

from orsopy import fileio
from orsopy.fileio import Reduction, Software, File

try:
    from pxrr.OrsoIO import OrsoIO
except Exception:
    OrsoIO = None
try:
    from p08_general.P08OrsoIO import P08OrsoIO
except Exception:
    P08OrsoIO = None

from pxrr.helpers import *
from pxrr.gixos import *

'''
author shenc
everything related to data input and output
'''



#%% 
# -----------------------------------------------------------------------------
# input block
# -----------------------------------------------------------------------------

def load_metadata(yaml_path: str):
    """
    Load metadata from a YAML file and return all parameters 
    and derived quantities matching the original script.
    """
    yaml = YAML(typ='safe')
    
    # Load YAML
    with open(yaml_path, "r") as f:
        meta = yaml.load(f)
    
    
    # linux and windows path
    if platform.system() == "Windows":
        meta["paths"]["gixs_path"] = meta["paths"]["gixs_path"].replace("/","\\")
        meta["paths"]["path_out"] = meta["paths"]["path_out"].replace("/","\\")
    else:
        meta["paths"]["gixs_path"] = meta["paths"]["gixs_path"].replace("\\","/")
        meta["paths"]["path_out"] = meta["paths"]["path_out"].replace("\\","/")
        
    # for dictionary return
    meta["measurements"]["scan"]  = np.array(meta["measurements"]["scan"], dtype=int)
    meta["measurements"]["bkgscan"] = np.array(meta["measurements"]["bkgscan"], dtype=int)
    meta["instrument"]["wavelength"] = 12404/meta["instrument"]["energy"]
    meta["qxy0"] = np.array(meta["qxy0"])
    meta['tth']= np.degrees(np.arcsin(meta["qxy0"] * meta["instrument"]["wavelength"] / 4 / pi)) * 2
    meta["sample_params"]["rho_b"] = meta["sample_params"]["Qc"]**2/16/pi
    # RFscaling exactly as in the original script
    meta['I0'] = meta['measurements']['flux'] * meta['measurements']['cttime_sample']
    
    return meta

def load_gixos_from_meta(yaml_path: str):
    """
    Load GIXOS sample and background data according to metadata in a YAML file.
    Returns two dicts: importGIXOSdata and importbkg, each containing at least:
      - "Intensity": 2D array (rows x len(qxy0))
      - "error": 2D array
      - "tt":  1D array (mean of tt_qxy0)
      - "tth": 1D array
    for 1D data, it generate "tt_qxy0": 2D array, and calculate tt from it
    """
    # Load metadata
    meta = load_metadata(yaml_path)
    
    # Extract parameters
    datatype = meta["datatype"]
    path        = meta["paths"]["gixs_path"]
    sample      = meta["measurements"]["sample"]
    bkgsample   = meta["measurements"]["bkgsample"]
    scan        = meta["measurements"]["scan"]
    bkgscan     = meta["measurements"]["bkgscan"]
    qxy0        = meta["qxy0"]
    tth         = meta["tth"]

    importGIXOSdata = None
    importbkg       = None
    print("indicator")
    
    if "2d gixs" in datatype.lower():
        # checking facility otherwise raise error
        if meta["facility"] == "PETRA III/P08":
            filepattern = f"{sample}_{scan:05d}_angle"
            bkgfilepattern = f"{bkgsample}_{bkgscan:05d}_angle"
            importGIXOSdata = load_data(filepattern, path, metadata = yaml_path, datatype=datatype)
            importbkg = load_data(bkgfilepattern, path, metadata = yaml_path, datatype=datatype)
        else:
            raise ValueError(
                "2d gixs mode has only been implemented for PETRA III/P08 'Langmuir GID setup'"
            )
        
        print("load 2d gixs and generate sets of 1d gixos from qxy0 list in meta, using DSpxHW binning")
        if meta["geometrical_correction"] and importGIXOSdata["HWtth"].shape == (1,1) and importGIXOSdata["HWtt"].shape == (1,):
            print("geometrical correction")
            importGIXOSdata = geometrical_corr(importGIXOSdata, Ddet = meta["instrument"]["Ddet"], det_px = meta["instrument"]["pixel"], HWtth = importGIXOSdata["HWtth"][0,0], HWtt = importGIXOSdata["HWtt"][0])
            importbkg = geometrical_corr(importbkg, Ddet = meta["instrument"]["Ddet"], det_px = meta["instrument"]["pixel"], HWtth = importGIXOSdata["HWtth"][0,0], HWtt = importGIXOSdata["HWtt"][0])
        else:
            print("geometrical correction cannot be performed when HWtth and HWtt are not unified across data")
        print("extract 1d gixos curve for qxy0 list, each binning %d pixels" %(2*meta["instrument"]["DSpxHW"]))
        importGIXOSdata = extract_1dGIXOS(importGIXOSdata, tth, HWpx_h = meta["instrument"]["DSpxHW"])
        importbkg = extract_1dGIXOS(importbkg, tth, HWpx_h = meta["instrument"]["DSpxHW"])       
            
    elif "1d gixos" in datatype.lower():
        if meta["facility"] == "NSLS-II/12ID":
            # build one prefix per scan id:
            # <sample>-id<id>
            sample_prefix_list = [f"{sample}-id{int(scan_id)}" for scan_id in scan]
            bkg_prefix_list = []
            for scan_id in bkgscan:
                sid = int(scan_id)
                preferred = f"{bkgsample}-id{sid}"
                fallback = f"{sample}-id{sid}"

                preferred_file = f"{path}{preferred}.txt"
                fallback_file = f"{path}{fallback}.txt"

                if os.path.exists(preferred_file):
                    bkg_prefix_list.append(preferred)
                elif os.path.exists(fallback_file):
                    bkg_prefix_list.append(fallback)
                else:
                    # keep preferred name so downstream error message reflects metadata expectation
                    bkg_prefix_list.append(preferred)

            print("load 1d gixos cuts from NSLS-II/12ID")

            importGIXOSdata = load_data(
                sample_prefix_list,
                path,
                metadata=yaml_path,
                datatype=datatype
            )

            importbkg = load_data(
                bkg_prefix_list,
                path,
                metadata=yaml_path,
                datatype=datatype
            )
        else:
            raise ValueError(
                "1d gixos mode has only been implemented for NSLS-II/12ID."
            )
    
    else:
        print("only 2d gixs or 1d gixos cut is supported")
    
    return importGIXOSdata, importbkg

# Example usage:
# importGIXOSdata, importbkg = load_data_from_meta("metadata.yaml")

def load_data(gixosdataprefix, path, metadata = None, datatype = "2d gixs"):
    """
    this loads the data directly through data and path
    2d gixs data file: <prefix>_I.dat, <prefix>_tt.dat, <prefix>_tth.dat
    1d gixos cut file: <prefix>.txt
    Parameters
    ----------
    gixosdataprefix : str
        file name prefix (see above).
    path : str
        file directory, ends with /.
    metadata : str, optional
        yaml metadata file path. The default is None.
    datatype : str, optional
        either "1d gixos" or "2d gixs". The default is "2d gixs".

    Returns
    -------
    importeddata : dictionary
        field: 
            'Intensity':    intensity map, 2d or one line cut,
            ('error':       only for 1d gixos at this stage, error column)
            'tth':          tth axis in deg, 
            'tt':           tt axis in deg, 
            'HWtth':        half width of each column in tth in deg, 
            'HWtt':         half width of each row in tt in deg, 
            'HWpx_h' = .5:  horizontal half width in pixel, 
            'HWpx_v' = .5:  vertical half width in pixel.
            'metadata':     metadata, if no entry it will be None

    """
    importeddata = None
    if "2d gixs" in datatype.lower():
        '''
        currently implemented for P08 GIXS file _angle, ascii format
        by p08_GIXD.read_2D a dictionary of 'mat', 'tth', 'tt' is created
        '''
        print("load 2d image")
        
        # create dictionary for the data
        importeddata = {"tth": [], "tt": [], 'Intensity': []}
        
        # load files    
        file_intensity = path+gixosdataprefix+"_I.dat"
        file_xaxis = path+gixosdataprefix+"_tth.dat"
        file_yaxis = path+gixosdataprefix+"_tt.dat"
        importeddata['Intensity'] = np.loadtxt(file_intensity)
        importeddata['tth'] = np.loadtxt(file_xaxis, ndmin = 2)
            
        tt_tmp =  np.loadtxt(file_yaxis)
        if importeddata["tth"].shape[0]>1 and tt_tmp.ndim==1:
            importeddata["tt"] = np.zeros([importeddata['Intensity'].shape[0],importeddata['Intensity'].shape[1]])
            for i in range(importeddata["tt"].shape[1]):
                importeddata["tt"][:,i] = tt_tmp
        else:
            importeddata["tt"]=tt_tmp
        del tt_tmp
        
        # other derived quantity for GIXOS
        importeddata["HWtth"] = mean1d_if_within_percent(np.diff(importeddata["tth"], axis = 1)/2)
        importeddata["HWpx_h"] = .5
        importeddata["HWtt"] = mean1d_if_within_percent(np.diff(importeddata["tt"], axis = 0)/2)
        importeddata["HWpx_v"] = .5
    
    elif "1d gixos" in datatype.lower():
        """
        implemented for NSLS-II/12ID 1d GIXOS cuts

        expected input:
        - gixosdataprefix can be either:
            * a single string prefix, or
            * a list/tuple/ndarray of string prefixes
        - each file is: <path><prefix>.txt

        expected file format:
        four columns: idx, tt(beta), intensity, qz
        """
        # normalize input to a list of file prefixes
        if isinstance(gixosdataprefix, str):
            prefix_list = [gixosdataprefix]
        else:
            prefix_list = list(gixosdataprefix)

        if len(prefix_list) == 0:
            raise ValueError("For '1d gixos', gixosdataprefix must contain at least one file prefix.")

        # load metadata early because we need tth / HWtth
        meta_loaded = None
        if metadata is not None:
            try:
                meta_loaded = load_metadata(metadata)
            except Exception:
                meta_loaded = None
                print("cannot find metadata")

        ncols = len(prefix_list)
        
        # optional consistency checks against metadata
        if meta_loaded is not None:
            if "tth" not in meta_loaded:
                raise ValueError("metadata must contain 'tth' for '1d gixos' loading.")

            if len(meta_loaded["tth"]) != ncols:
                raise ValueError(
                    "Number of 1d gixos files must match len(metadata['tth']) / len(metadata['qxy0'])."
                )

            if "qxy0" in meta_loaded and len(meta_loaded["qxy0"]) != ncols:
                raise ValueError(
                    "Number of 1d gixos files must match number of qxy0 entries in metadata."
                )        
        
        importeddata = None

        for idx, prefix in enumerate(prefix_list):
            GIXOSfilename = f"{path}{prefix}.txt"
            dataread = np.loadtxt(GIXOSfilename, skiprows=16)

            # expected columns: idx, tt(beta), intensity, qz
            tt_col = np.asarray(dataread[:, 1], dtype=float)
            inten_col = np.asarray(dataread[:, 2], dtype=float)

            if importeddata is None:
                nrows = len(tt_col)

                importeddata = {
                    "Intensity": np.zeros((nrows, ncols), dtype=float),
                    "error": np.zeros((nrows, ncols), dtype=float),
                    "tt": tt_col.copy(),                 # final tt should be 1D: (n,)
                    "tth": np.zeros((1, ncols), dtype=float),
                    "HWtth": None,
                    "HWpx_h": 0.5,
                    "HWtt": None,
                    "HWpx_v": 0.5,
                }

                # HWtth from metadata['HWtth'] -> shape (1,1)
                if meta_loaded is not None and "tth" in meta_loaded:
                    importeddata["tth"] = meta_loaded["tth"][np.newaxis, :]
                else:
                    raise ValueError(
                        "For '1d gixos', metadata must provide 'qxy0', 'energy', so tth can be calculated and the output matches extract_1dGIXOS."
                    )

                # HWtt from averaged step size of the tt column -> shape (1,)
                importeddata["HWtt"] = np.array([np.mean(np.diff(tt_col)) / 2], dtype=float)

                # HWtth from metadata['HWtth'] -> shape (1,1)
                if meta_loaded is not None and "HWtth" in meta_loaded["instrument"]:
                    importeddata["HWtth"] = np.array([[float(meta_loaded["instrument"]["HWtth"])]], dtype=float)
                else:
                    raise ValueError(
                        "For '1d gixos', metadata must provide 'HWtth' so the output matches extract_1dGIXOS."
                    )

                # optional consistency check for later files
                tt_ref = tt_col.copy()
            else:
                if len(tt_col) != importeddata["Intensity"].shape[0]:
                    raise ValueError(
                        f"1d gixos files do not have the same number of rows: "
                        f"{prefix} has {len(tt_col)}, expected {importeddata['Intensity'].shape[0]}"
                    )

                # require same tt grid
                if not np.allclose(tt_col, tt_ref, rtol=0, atol=1e-8):
                    raise ValueError(
                        f"1d gixos files do not share the same tt axis: {prefix}"
                    )

            importeddata["Intensity"][:, idx] = inten_col
            importeddata["error"][:, idx] = np.sqrt(np.maximum(inten_col, 0.0))

            # tth comes from metadata qxy0 list, shape must be (1, m)
            if metadata is not None:
                try:
                    if "meta_loaded" in locals() and meta_loaded is not None:
                        importeddata["metadata"] = meta_loaded
                    else:
                        importeddata["metadata"] = load_metadata(metadata)
                except Exception:
                    importeddata["metadata"] = None
                    print("cannot find metadata")
            else:
                raise ValueError("metadata is required for '1d gixos' loading.")


    else:
        print("only 2d gixs (PETRA III/P08) or 1d gixos cut (NSLS-II/12ID) is supported")
    
    if metadata is not None:
        try:
            importeddata["metadata"] = load_metadata(metadata)
        except:
            importeddata["metadata"] = None
            print("cannot find metadata")
    else:
        importeddata["metadata"] = None
        print("metadata = None")
    
    if importeddata is None:
        print("no data is loaded")
    else:
        print('%s is loaded'% datatype)
    return importeddata

#%%
# ----------------------------------------------------------------------------
#   output block
# ----------------------------------------------------------------------------

def save_metadata_yaml(metadata_dict, filename):
    """
    Save metadata as YAML, preserving inline list style for selected fields.
    """
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.sort_base_mapping_type_on_output = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096

    clean_dict = yamlify(metadata_dict)
    clean_dict = set_flow_style_lists(clean_dict)

    with open(filename, "w", encoding="utf-8") as f:
        yaml.dump(clean_dict, f)


def update_metadata(
    yaml_path,
    updates=None,
    out_path=None,
    create_missing=False,
    **natural_updates,
):
    """Update values in an existing metadata YAML file.

    Parameters
    ----------
    yaml_path : str or Path
        Path to the existing metadata YAML file.
    updates : dict, optional
        Updates to apply. Supports two forms:
        1) nested dictionaries, e.g. ``{"sample_params": {"kappa": 5}}``
        2) flat key paths, e.g. ``{"sample_params.kappa": 5}``
    out_path : str or Path, optional
        Output path for the updated YAML. If ``None``, overwrite ``yaml_path``.
    create_missing : bool, optional
        If ``False`` (default), only existing keys can be updated and missing
        keys raise ``KeyError``. If ``True``, missing keys are created.
    **natural_updates
        Additional key-value updates using natural key paths. Use ``__`` as a
        separator for keyword-safe nested keys. Example:
        ``sample_params__kappa=5`` is treated as ``sample_params.kappa``.
        For bare keys (e.g. ``kappa=5``), the function auto-resolves to a
        unique nested match if one exists.

    Returns
    -------
    str
        Path to the written YAML file.
    """
    if updates is None:
        updates = {}
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dictionary if provided")

    yaml_path = str(yaml_path)
    if out_path is None:
        out_path = yaml_path
    out_path = str(out_path)

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"metadata file not found: {yaml_path}")

    yaml = YAML(typ="safe")
    with open(yaml_path, "r", encoding="utf-8") as f:
        meta = yaml.load(f)

    if not isinstance(meta, dict):
        raise ValueError("metadata file must contain a top-level mapping")

    def _update(dst, src, path_keys):
        for key, value in src.items():
            key_path = path_keys + [str(key)]
            dotted = ".".join(key_path)

            if isinstance(value, dict):
                if key not in dst:
                    if create_missing:
                        dst[key] = {}
                    else:
                        raise KeyError(f"Missing key in metadata: {dotted}")
                elif not isinstance(dst[key], dict):
                    raise TypeError(f"Cannot update nested key under non-dict: {dotted}")
                _update(dst[key], value, key_path)
            else:
                if key not in dst and not create_missing:
                    raise KeyError(f"Missing key in metadata: {dotted}")
                dst[key] = value

    def _set_by_path(dst, dotted_key, value):
        if not isinstance(dotted_key, str) or dotted_key.strip() == "":
            raise ValueError("Update keys must be non-empty strings")

        parts = [p for p in dotted_key.split(".") if p]
        if not parts:
            raise ValueError(f"Invalid update key: {dotted_key}")

        cur = dst
        for part in parts[:-1]:
            if part not in cur:
                if create_missing:
                    cur[part] = {}
                else:
                    raise KeyError(f"Missing key in metadata: {'.'.join(parts)}")
            elif not isinstance(cur[part], dict):
                raise TypeError(f"Cannot update nested key under non-dict: {'.'.join(parts)}")
            cur = cur[part]

        leaf = parts[-1]
        if leaf not in cur and not create_missing:
            raise KeyError(f"Missing key in metadata: {'.'.join(parts)}")
        cur[leaf] = value

    def _find_key_paths(node, target_key, prefix=None):
        if prefix is None:
            prefix = []
        matches = []
        if isinstance(node, dict):
            for k, v in node.items():
                new_prefix = prefix + [str(k)]
                if str(k) == target_key:
                    matches.append(".".join(new_prefix))
                if isinstance(v, dict):
                    matches.extend(_find_key_paths(v, target_key, new_prefix))
        return matches

    # 1) apply nested-dict updates from `updates`
    nested_updates = {}
    flat_updates = {}
    for k, v in updates.items():
        if isinstance(k, str) and "." in k:
            flat_updates[k] = v
        else:
            nested_updates[k] = v

    if nested_updates:
        _update(meta, nested_updates, [])

    # 2) apply dotted-path updates from `updates`
    for k, v in flat_updates.items():
        _set_by_path(meta, k, v)

    # 3) apply keyword natural updates (double underscore as separator)
    for k, v in natural_updates.items():
        resolved_key = k.replace("__", ".")
        if "." in resolved_key:
            _set_by_path(meta, resolved_key, v)
            continue

        # bare key path: update top-level key if present
        if resolved_key in meta:
            _set_by_path(meta, resolved_key, v)
            continue

        # otherwise try unique nested-key auto-resolution
        matches = _find_key_paths(meta, resolved_key)
        if len(matches) == 1:
            _set_by_path(meta, matches[0], v)
        elif len(matches) == 0:
            if create_missing:
                _set_by_path(meta, resolved_key, v)
            else:
                raise KeyError(
                    f"Missing key in metadata: {resolved_key}. "
                    f"Use dotted path (e.g. sample_params.{resolved_key}) or set create_missing=True."
                )
        else:
            raise KeyError(
                f"Ambiguous key '{resolved_key}'. Matches found: {matches}. "
                "Use an explicit dotted path."
            )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    save_metadata_yaml(meta, out_path)
    return out_path


# ----------------------------------------------------------------------------
# export to Orso format
# ----------------------------------------------------------------------------

def _get_export_qsel(GIXOS):
    """
    Resolve which qxy0 tracks should be exported.

    Rules
    -----
    - If metadata["PseudoR"]["qxy0_select_idx"] exists:
        * scalar -> export that one track
        * list   -> export all listed tracks
    - If it does not exist:
        * if GIXOS contains multi-track output -> export all tracks
        * otherwise -> export the single available track

    Returns
    -------
    qsel : np.ndarray
        1D integer array of raw qxy0 indices.
    """
    meta = GIXOS["metadata"]
    pr = meta.get("PseudoR", {})

    # explicit selection from metadata
    if "qxy0_select_idx" in pr and pr["qxy0_select_idx"] is not None:
        qsel_raw = pr["qxy0_select_idx"]
        if np.ndim(qsel_raw) == 0:
            return np.array([int(qsel_raw)], dtype=int)
        return np.asarray(qsel_raw, dtype=int).ravel()

    # no explicit selection: export all available tracks
    if "refl" in GIXOS and np.asarray(GIXOS["refl"]).ndim == 3:
        nsel = np.asarray(GIXOS["refl"]).shape[0]
        qxy0_all = np.asarray(meta["qxy0"], dtype=float).ravel()
        if nsel > len(qxy0_all):
            raise ValueError("Number of exported tracks exceeds number of qxy0 values in metadata.")
        return np.arange(nsel, dtype=int)

    return np.array([0], dtype=int)


def _extract_table_track(arr, selected_pos):
    """
    Extract one selected track from a table-like field.

    Supported shapes
    ----------------
    single-qxy0:
        (n, m)

    multi-qxy0:
        (idx, n, m)

    Parameters
    ----------
    arr : array-like
        Table-like data array.

    selected_pos : int
        Position in the selected qxy0 list.

    Returns
    -------
    out : np.ndarray
        Selected table with shape (n, m).
    """
    a = np.asarray(arr)

    if a.ndim == 3:
        return np.asarray(a[selected_pos], dtype=float)

    if a.ndim == 2:
        return np.asarray(a, dtype=float)

    raise ValueError(
        f"Expected table-like array with ndim 2 or 3, got shape {a.shape}."
    )


def _extract_vector_track(arr, selected_pos):
    """
    Extract one selected track from a vector-like field.

    Supported shapes
    ----------------
    single-qxy0:
        (n,)

    multi-qxy0:
        (idx, n)

    Parameters
    ----------
    arr : array-like
        Vector-like data array.

    selected_pos : int
        Position in the selected qxy0 list.

    Returns
    -------
    out : np.ndarray
        Selected vector with shape (n,).
    """
    a = np.asarray(arr)

    if a.ndim == 2:
        return np.asarray(a[selected_pos], dtype=float)

    if a.ndim == 1:
        return np.asarray(a, dtype=float)

    raise ValueError(
        f"Expected vector-like array with ndim 1 or 2, got shape {a.shape}."
    )


def add_chamber_bkg_to_additional_files(io, GIXOS):
    """
    Add chamber background scan number(s) to ORSO additional_files.

    Parameters
    ----------
    io : P08OrsoIO-like object
        ORSO writer object.

    GIXOS : dict
        GIXOS analysis dictionary containing metadata.
    """
    meta = GIXOS.get("metadata", {})
    meas = meta.get("measurements", {})

    bkg_scan = meas.get("bkgscan", None)
    if bkg_scan is None:
        return

    bkg_scan_arr = np.asarray(bkg_scan).ravel()
    bkg_entries = [f"{int(v):05d}" for v in bkg_scan_arr]

    if io.header.DataSource.measurement["additional_files"] is None:
        io.header.DataSource.measurement["additional_files"] = []

    for entry in bkg_entries:
        if entry not in io.header.DataSource.measurement["additional_files"]:
            io.header.DataSource.measurement["additional_files"].append(entry)

def _fmt_optional_float(val, fmt=".4g"):
    """
    Format a scalar value if possible, otherwise return as string.
    """
    if val is None:
        return "None"
    try:
        return format(float(val), fmt)
    except Exception:
        return str(val)


def _build_bulkbkg_correction_text(GIXOS):
    """
    Build one-line bulk background correction description
    for ORSO reduction metadata.
    """
    correction_str = "bulk scattering background subtraction: none"

    if "bulkbkg" not in GIXOS or GIXOS["bulkbkg"] is None:
        return correction_str

    bulk = GIXOS["bulkbkg"]

    # fitting mode
    if "fit_params" in bulk and bulk["fit_params"] is not None:
        fitp = bulk["fit_params"]
        y0 = fitp.get("y0", None)
        F = fitp.get("F", None)
        t = fitp.get("t", None)

        bulk_tth = None
        if "tth" in bulk and bulk["tth"] is not None:
            bulk_tth_arr = np.asarray(bulk["tth"]).ravel()
            if bulk_tth_arr.size > 0:
                bulk_tth = float(np.mean(bulk_tth_arr))

        eq_str = (
            f"I_bulk = {_fmt_optional_float(y0)} + "
            f"{_fmt_optional_float(F)}*exp(-Q/{_fmt_optional_float(t)})"
        )

        if bulk_tth is None:
            correction_str = (
                "bulk scattering background subtraction: "
                f"fit wide-angle data, {eq_str}"
            )
        else:
            correction_str = (
                "bulk scattering background subtraction: "
                f"fit wide-angle data at tth = {bulk_tth:.3f} deg, {eq_str}"
            )
        return correction_str

    # constant modes
    const_mode = bulk.get("const_mode", None)

    if const_mode == 0:
        val = bulk.get("const_value", None)
        correction_str = (
            "bulk scattering background subtraction: "
            f"constant background from user-provided value, "
            f"value = {_fmt_optional_float(val)}"
        )

    elif const_mode == 1:
        qzlb = bulk.get("const_qz_lb", None)
        correction_str = (
            "bulk scattering background subtraction: "
            "constant background from high-Qz average, "
            f"Qz >= {_fmt_optional_float(qzlb)} /angstrom"
        )

    elif const_mode == 2:
        correction_str = (
            "bulk scattering background subtraction: "
            "constant background from average of 3 minimum points with Qz > 3Qc"
        )

    elif "Intensity_at_GIXOS" in bulk:
        correction_str = (
            "bulk scattering background subtraction: "
            "direct subtraction of wide-angle data"
        )

    return correction_str


def build_gixos2r_reduction_metadata(GIXOS, which="refl", qidx=None, selected_pos=None):
    """
    Build ORSO Reduction metadata for GIXOS2R outputs.

    Detailed settings are appended as extra entries in `corrections`.
    The `call` field follows the actual workflow using function names.

    Parameters
    ----------
    GIXOS : dict
        Output dictionary from GIXOS analysis / GIXOS2R.

    which : {"refl", "SF", "GIXOS"}, optional
        Which exported dataset this reduction metadata is intended for.

    qidx : int, optional
        Raw qxy0 column index in the original GIXOS arrays. If None, the first
        selected qxy0 index from metadata["PseudoR"]["qxy0_select_idx"] is used.

    selected_pos : int, optional
        Position within the selected qxy0 list for multi-qxy0 GIXOS2R output.
        This is used for fields stacked along axis 0. If None, single-track
        behavior is assumed.

    Returns
    -------
    reduction : orsopy.fileio.Reduction
        ORSO Reduction metadata object.
    """
    meta = GIXOS["metadata"]
    pr = meta.get("PseudoR", {})
    samp = meta.get("sample_params", {})
    inst = meta.get("instrument", {})

    # ------------------------------------------------------------
    # resolve qidx (raw qxy0 column index) and selected_pos
    # ------------------------------------------------------------
    qsel_raw = pr.get("qxy0_select_idx", 0)
    if np.ndim(qsel_raw) == 0:
        qsel = np.array([int(qsel_raw)], dtype=int)
    else:
        qsel = np.asarray(qsel_raw, dtype=int).ravel()

    if qsel.size == 0:
        raise ValueError("metadata['PseudoR']['qxy0_select_idx'] must not be empty.")

    if qidx is None:
        qidx = int(qsel[0])

    if selected_pos is None:
        if len(qsel) == 1:
            selected_pos = 0
        else:
            matches = np.where(qsel == int(qidx))[0]
            if matches.size > 0:
                selected_pos = int(matches[0])
            else:
                selected_pos = 0

    # selected GIXOS point from raw metadata
    tth_val = None
    if "tth" in meta and meta["tth"] is not None:
        tth_arr = np.asarray(meta["tth"]).ravel()
        if qidx < len(tth_arr):
            tth_val = float(tth_arr[qidx])

    qxy0_val = None
    if "qxy0" in meta and meta["qxy0"] is not None:
        qxy0_arr = np.asarray(meta["qxy0"]).ravel()
        if qidx < len(qxy0_arr):
            qxy0_val = float(qxy0_arr[qidx])

    # diffuse scattering resolution used in GIXOS2R for this exported track
    ds_phi_hw = None
    ds_beta_hw = None

    if "HWtth" in GIXOS and GIXOS["HWtth"] is not None:
        hw_tth = np.asarray(GIXOS["HWtth"], dtype=float)

        if hw_tth.ndim == 2:
            if hw_tth.shape[0] == 1 and qidx < hw_tth.shape[1]:
                ds_phi_hw = float(hw_tth[0, qidx])
            elif hw_tth.shape[0] == 1 and selected_pos < hw_tth.shape[1]:
                ds_phi_hw = float(hw_tth[0, selected_pos])
            elif hw_tth.size > 0:
                ds_phi_hw = float(hw_tth.ravel()[0])
        elif hw_tth.size > 0:
            ds_phi_hw = float(hw_tth.ravel()[0])

    if "HWtt" in GIXOS and GIXOS["HWtt"] is not None:
        hw_tt = np.asarray(GIXOS["HWtt"], dtype=float).ravel()

        if hw_tt.size == 1:
            ds_beta_hw = float(hw_tt[0])
        else:
            if qidx < hw_tt.size:
                ds_beta_hw = float(hw_tt[qidx])
            elif selected_pos < hw_tt.size:
                ds_beta_hw = float(hw_tt[selected_pos])
            elif hw_tt.size > 0:
                ds_beta_hw = float(hw_tt[0])

    # reflectivity settings
    resolution_mode = pr.get("resolution_mode", None)
    resolution_hw = pr.get("resolution_HW", None)
    virtual_energy = pr.get("energy", None)
    virtual_ddet = pr.get("Ddet", None)

    if resolution_mode == 0:
        res_str = f"circular, {_fmt_optional_float(resolution_hw)} /angstrom"
        refl_setting_lines = [
            "reflectivity settings",
            f"resolution = {res_str}",
        ]

    elif resolution_mode == 1:
        if isinstance(resolution_hw, (list, tuple, np.ndarray)):
            res_hw_arr = np.asarray(resolution_hw).ravel()
            if res_hw_arr.size >= 2:
                res_str = (
                    "slit, "
                    f"[{_fmt_optional_float(res_hw_arr[0])}, "
                    f"{_fmt_optional_float(res_hw_arr[1])}] mm"
                )
            else:
                res_str = f"slit, {_fmt_optional_float(resolution_hw)} mm"
        else:
            res_str = f"slit, {_fmt_optional_float(resolution_hw)} mm"

        refl_setting_lines = [
            "reflectivity settings",
            f"virtual xrr energy = {_fmt_optional_float(virtual_energy,fmt='.1f')} eV",
            f"virtual xrr detector distance = {_fmt_optional_float(virtual_ddet)} mm",
            f"resolution = {res_str}",
        ]

    else:
        res_str = "unknown"
        refl_setting_lines = [
            "reflectivity settings",
            f"resolution = {res_str}",
        ]

    bkg_mode = pr.get("bkg_mode", None)
    bkg_off = pr.get("bkg_off", None)

    if bkg_mode is None:
        bkg_str = "none"
    elif bkg_mode == 0:
        bkg_str = f"phi offset, {_fmt_optional_float(bkg_off)} mm"
    elif bkg_mode == 1:
        bkg_str = f"beta offset, {_fmt_optional_float(bkg_off)} mm"
    else:
        bkg_str = str(bkg_mode)

    # kappa string with optional error
    kappa = samp.get("kappa", None)
    kappa_err = samp.get("kappa_err", None)
    if kappa is None:
        kappa_str = "None"
    elif kappa_err is None:
        kappa_str = f"{float(kappa):.2f} kBT"
    else:
        kappa_str = f"{float(kappa):.2f} +/- {float(kappa_err):.2f} kBT"

    kappa_is_fitted = (kappa is not None and kappa_err is not None)

    bulk_corr_str = _build_bulkbkg_correction_text(GIXOS)

    # qxy-dependence fit settings
    qxy_fit_lines = []
    qxy_ana = GIXOS.get("qxy_dependence_ana", None)

    if kappa_is_fitted and qxy_ana is not None:
        target_qz = np.asarray(qxy_ana.get("target_qz", []), dtype=float).ravel()
        row_index = np.asarray(qxy_ana.get("row_index", []), dtype=int).ravel()

        qxy0_meta = np.asarray(meta.get("qxy0", []), dtype=float).ravel()
        qxy_res = np.asarray(qxy_ana.get("Qxy", []), dtype=float)
        if qxy_res.ndim == 2 and qxy_res.shape[1] > 0:
            ncols_fit = qxy_res.shape[1]
        else:
            ncols_fit = len(qxy0_meta)

        qxy0_use = qxy0_meta[:ncols_fit]

        qz_str = ", ".join(_fmt_optional_float(v) for v in target_qz)
        row_str = ", ".join(str(int(v)) for v in row_index)
        qxy0_str = ", ".join(_fmt_optional_float(v) for v in qxy0_use)

        qxy_fit_lines = [
            f"qz = [{qz_str}] /angstrom (row [{row_str}])",
            f"qxy0 range = [{qxy0_str}] /angstrom",
        ]

    # detect earlier processing steps
    datatype = str(meta.get("datatype", "")).lower()
    geometrical_correction = bool(meta.get("geometrical_correction", False))

    hwpx_h = float(np.asarray(GIXOS.get("HWpx_h", 0.5)).ravel()[0]) if "HWpx_h" in GIXOS else 0.5
    hwpx_v = float(np.asarray(GIXOS.get("HWpx_v", 0.5)).ravel()[0]) if "HWpx_v" in GIXOS else 0.5

    # beta transmission correction: tbeta_sqr not all ones
    beta_transmission_done = False
    tbeta_sqr = GIXOS.get("tbeta_sqr", None)
    if tbeta_sqr is not None:
        tb = np.asarray(tbeta_sqr, dtype=float)
        if tb.size > 0 and not np.allclose(tb, 1.0, rtol=0, atol=1e-12, equal_nan=False):
            beta_transmission_done = True

    # footprint broadening correction: dQz not all nan
    footprint_done = False
    refl = GIXOS.get("refl", None)
    if refl is not None:
        refl_arr = np.asarray(refl, dtype=float)
        if refl_arr.ndim == 2 and refl_arr.shape[1] > 3:
            dqz = refl_arr[:, 3]
            if np.any(np.isfinite(dqz)):
                footprint_done = True
        elif refl_arr.ndim == 3 and refl_arr.shape[2] > 3:
            dqz = refl_arr[selected_pos, :, 3]
            if np.any(np.isfinite(dqz)):
                footprint_done = True

    # corrections list
    corrections = []

    # raw-data pre-processing steps
    if datatype == "2d gixs":
        corrections.append("rebin 2D map in angular space (pre-processed)")

    if geometrical_correction:
        corrections.append("pixel geometrical correction (solid angle)")

    if datatype == "2d gixs":
        extract_line = "extract 1D GIXOS"
        if hwpx_h > 0.5:
            extract_line += ", bin phi"
        corrections.append(extract_line)

    if hwpx_v > 0.5:
        corrections.append("bin beta")

    corrections.append("angle to Q space")

    # background and qxy-fit
    corrections.extend([
        "chamber background subtraction using 'additional_files'",
        bulk_corr_str,
    ])

    if kappa_is_fitted:
        corrections.append("fit qxy dependence to obtain kappa")
        corrections.extend(qxy_fit_lines)

    # GIXOS2R and later corrections
    corrections.append("conversion from GIXOS diffuse scattering (I0 R*) to pseudo-reflectivity / structure factor using GIXOS2R")

    if beta_transmission_done:
        corrections.append("beta transmission correction")

    if footprint_done:
        corrections.append(
            f"footprint broadening correction, footprint = {_fmt_optional_float(inst.get('footprint', None))} mm"
        )

    corrections.extend([
        f"I0 = {_fmt_optional_float(meta.get('I0', None))}",
        "GIXOS diffuse scattering settings",
        f"GIXOS phi = {_fmt_optional_float(tth_val)} deg (qxy0 = {_fmt_optional_float(qxy0_val)} /angstrom)",
        f"delta_phi (HWHM) = {_fmt_optional_float(ds_phi_hw)} deg",
        f"delta_beta (HWHM) = {_fmt_optional_float(ds_beta_hw)} deg",
        *refl_setting_lines,
        f"background = {bkg_str}",
        "sample settings",
        f"Qc = {_fmt_optional_float(samp.get('Qc', None))} /angstrom",
        f"tension = {_fmt_optional_float(samp.get('tension', None))} N/m",
        f"temperature = {_fmt_optional_float(samp.get('temperature', None))} K",
        f"kappa = {kappa_str}",
        f"amin = {_fmt_optional_float(samp.get('amin', None))} angstrom",
    ])

    # compact workflow-style call
    call_parts = []

    if geometrical_correction:
        call_parts.append("geometrical_corr()")

    if datatype == "2d gixs":
        call_parts.append("extract_1dGIXOS()")

    if hwpx_v > 0.5:
        call_parts.append("binning_GIXOS_tt()")

    call_parts.append("th2q()")
    call_parts.append("GIXOS_background_corr()")

    if kappa_is_fitted:
        call_parts.append("GIXOS_qxy_dependence()")

    call_parts.append(
        "GIXOS2R("
        f"qxy0_select_idx={pr.get('qxy0_select_idx', None)}, "
        f"resolution_mode={pr.get('resolution_mode', None)}, "
        f"resolution_HW={pr.get('resolution_HW', None)}, "
        f"bkg_mode={pr.get('bkg_mode', None)}, "
        f"bkg_off={pr.get('bkg_off', None)}, "
        f"tension={_fmt_optional_float(samp.get('tension', None))}, "
        f"temp={_fmt_optional_float(samp.get('temperature', None))}, "
        f"kappa={_fmt_optional_float(samp.get('kappa', None))}, "
        f"amin={_fmt_optional_float(samp.get('amin', None))}"
        ")"
    )
    call_str = "; ".join(call_parts)

    reduction = Reduction(
        software=[Software(name="pxrr")],
        corrections=corrections,
        call=call_str,
        comment=None,
    )

    return reduction

def export_orso(
    GIXOS,
    which="refl",
    exportpath=None,
    json_path=None,
    fio_path=None,
):
    """
    Export processed GIXOS-derived data to one or more ORSO files.

    This exporter supports:
    - pseudo-reflectivity ("refl")
    - structure factor ("SF")
    - background-corrected GIXOS ("GIXOS")

    Export selection behavior
    -------------------------
    - If metadata["PseudoR"]["qxy0_select_idx"] is a scalar:
        export only that one qxy0 track.
    - If it is a list / array:
        export all listed qxy0 tracks.
    - If it is not set:
        export all available tracks when the GIXOS2R output is multi-track.

    Output filenames
    ----------------
    Default output filenames are generated with `make_filename(...)` using:
    - phi<idx>_R.ort
    - phi<idx>_SF.ort
    - phi<idx>_GIXOS.ort

    where <idx> is the raw qxy0 column index.

    Parameters
    ----------
    GIXOS : dict
        Processed GIXOS dictionary.

    which : {"refl", "SF", "GIXOS"}, optional
        Which dataset to export.

    exportpath : str or None, optional
        Explicit output ORSO filename. If provided, it is only used when a
        single track is exported. For multiple tracks, default filenames are
        generated automatically with `make_filename(...)`.

    json_path : str or None, optional
        Path to beamtime metadata JSON file.

    fio_path : str or None, optional
        Path to scan .fio file.

    Returns
    -------
    result : tuple or list of tuple
        Single export:
            (io, dataset)

        Multiple export:
            [(io_0, dataset_0), (io_1, dataset_1), ...]
    """
    if "metadata" not in GIXOS or GIXOS["metadata"] is None:
        raise ValueError("GIXOS['metadata'] is required for ORSO export.")

    meta = GIXOS["metadata"]
    pr = meta.get("PseudoR", {})
    inst = meta.get("instrument", {})
    meas = meta.get("measurements", {})

    qsel = _get_export_qsel(GIXOS)
    multi_export = len(qsel) > 1

    results = []

    for selected_pos, qidx in enumerate(qsel):
        # ------------------------------------------------------------------
        # create a fresh IO object for each file
        # ------------------------------------------------------------------
        if (
            meta.get("facility") == "PETRA III/P08"
            and P08OrsoIO is not None
            and ((json_path is not None) or (fio_path is not None))
        ):
            io = P08OrsoIO()
        else:
            io = OrsoIO()

        add_chamber_bkg_to_additional_files(io, GIXOS)
        reduction = build_gixos2r_reduction_metadata(
                                                        GIXOS,
                                                        which=which,
                                                        qidx=int(qidx),
                                                        selected_pos=selected_pos
                                                    )

        # ------------------------------------------------------------
        # try loading external metadata files, but do not fail if missing
        # ------------------------------------------------------------
        if hasattr(io, "load_metadata_from_json"):
            if json_path is not None and os.path.isfile(json_path):
                try:
                    io.load_metadata_from_json(json_path)
                except Exception as e:
                    print(f"Could not load JSON metadata: {e}")
            else:
                if json_path is not None:
                    print(f"JSON metadata file not found: {json_path}")

        if hasattr(io, "load_metadata_from_scan"):
            if fio_path is not None and os.path.isfile(fio_path):
                try:
                    io.load_metadata_from_scan(fio_path)
                except Exception as e:
                    print(f"Could not load FIO metadata: {e}")
            else:
                if fio_path is not None:
                    print(f"FIO scan file not found: {fio_path}")

        if hasattr(io, "populate_metadata_to_header"):
            try:
                io.populate_metadata_to_header()
            except Exception:
                pass
        else:
            io.set_basic_header(
                title="",
                sample_name=meas.get("sample", ""),
                data_type="",
            )

        # ------------------------------------------------------------
        # fill / overwrite header information from runtime metadata
        # ------------------------------------------------------------
        sample_name = meas.get("sample", "")
        scan_val = meas.get("scan", None)

        io.header.SampleName = sample_name
        io.header.DataReduction = reduction

        if which == "refl":
            io.header.DataType = "PseudoR(Qz)"
        elif which == "SF":
            io.header.DataType = "|Phi(Qz)|^2"
        elif which == "GIXOS":
            io.header.DataType = "GIXOS(beta)"
        else:
            raise ValueError(f"Unsupported ORSO export type: {which}")

        io.header.DataSource.sample["name"] = sample_name

        if "alpha" in inst and inst["alpha"] is not None:
            io.header.DataSource.instrument_settings["incident_angle"] = fileio.base.Value(
                inst["alpha"], unit="deg"
            )
            io.header.DataSource.instrument_settings["incident_angle"].movement = "fixed"

        if "wavelength" in inst and inst["wavelength"] is not None:
            io.header.DataSource.instrument_settings["wavelength"] = fileio.base.Value(
                inst["wavelength"], unit="angstrom"
            )

        if "Ddet" in inst and inst["Ddet"] is not None:
            io.header.DataSource.instrument_settings["sample_detector_distance"] = fileio.base.Value(
                float(inst["Ddet"]),
                unit="mm",
                comment="GIXOS/GIXS detector"
            )

        if scan_val is not None:
            scan_arr = np.asarray(scan_val).ravel()
            io.scanmetadata["scan_no"] = np.char.zfill(scan_arr.astype(int).astype(str), 5)

        # ------------------------------------------------------------
        # reflectivity-specific instrument settings
        # ------------------------------------------------------------
        if which == "refl":
            if pr.get("resolution_mode", None) == 0:
                io.header.DataSource.instrument_settings["roi_specular"] = fileio.base.Value(
                    float(pr["resolution_HW"]),
                    unit="1/angstrom"
                )
                io.header.DataSource.instrument_settings["roi_specular"].definition = "HWHM"
                io.header.DataSource.instrument_settings["roi_specular"].configuration = "circular"
                io.header.DataSource.instrument_settings["roi_specular"].orientation_normal = "Qxy"

            elif pr.get("resolution_mode", None) == 1:
                res_hw = np.asarray(pr["resolution_HW"], dtype=float).ravel()
                io.header.DataSource.instrument_settings["roi_specular"] = fileio.base.ValueVector(
                    res_hw[0], res_hw[1], 0.0, unit="mm"
                )
                io.header.DataSource.instrument_settings["roi_specular"].definition = "HWHM"
                io.header.DataSource.instrument_settings["roi_specular"].configuration = "rectangular vxh"
                io.header.DataSource.instrument_settings["roi_specular"].orientation_normal = "beta"

                virtual_energy = pr.get("energy", None)
                virtual_ddet = pr.get("Ddet", None)
                io.header.DataSource.instrument_settings["roi_specular"].comment = (
                    f"virtual xrr energy: {virtual_energy} eV, "
                    f"xrr detector distance: {virtual_ddet} mm"
                )

            if pr.get("bkg_mode", None) is not None and pr.get("bkg_off", None) is not None:
                if pr["bkg_mode"] == 0:
                    bkg_comment = "phi off"
                elif pr["bkg_mode"] == 1:
                    bkg_comment = "beta off"
                else:
                    bkg_comment = ""

                io.header.DataSource.instrument_settings["roi_bkg_offspec"] = fileio.base.Value(
                    float(pr["bkg_off"]),
                    unit="mm",
                    comment=bkg_comment
                )
                io.header.DataSource.instrument_settings["roi_bkg_offspec"].use = True
            else:
                io.header.DataSource.instrument_settings["roi_bkg_offspec"] = fileio.base.Value(
                    [], unit="mm"
                )
                io.header.DataSource.instrument_settings["roi_bkg_offspec"].use = False

        # ------------------------------------------------------------
        # choose dataset + ORSO column descriptions
        # ------------------------------------------------------------
        io.header.ColDescription = fileio.orso.Orso.empty().columns[:]

        if which == "refl":
            refl_i = _extract_table_track(GIXOS["refl"], selected_pos)
            psi_r_i = _extract_vector_track(GIXOS["Psi_R"], selected_pos)
            sigma_i = _extract_vector_track(GIXOS["sigma_CW"], selected_pos)
            rred_i = _extract_vector_track(GIXOS["r_reduced"], selected_pos)
            
            n_refl = len(refl_i)
            if len(psi_r_i) != n_refl or len(sigma_i) != n_refl or len(rred_i) != n_refl:
                raise ValueError("Psi_R, sigma_CW or r_reduced must match the selected refl track length.")

            io.header.ColDescription[0] = fileio.base.Column(
                name="Qz",
                unit="1/angstrom",
                physical_quantity="wavevector transfer"
            )
            io.header.ColDescription[1] = fileio.base.Column(
                name="R",
                unit=None,
                physical_quantity="(pseudo)reflectivity, calculated from GIXOS diffuse scattering (R=R*/r)"
            )
            io.header.ColDescription.append(
                fileio.base.ErrorColumn(
                    error_of="R",
                    error_type="uncertainty",
                    value_is="sigma"
                )
            )
            io.header.ColDescription.append(
                fileio.base.ErrorColumn(
                    error_of="Qz",
                    error_type="resolution",
                    value_is="sigma"
                )
            )
            io.header.ColDescription.append(
                fileio.base.Column(
                    name="Psi_R",
                    unit=None,
                    physical_quantity="specular roughness factor"
                )
            )
            io.header.ColDescription.append(
                fileio.base.Column(
                    name="sigma_CW",
                    unit="angstrom",
                    physical_quantity="capillary wave roughness at reflectivity resolution, sqrt(-ln(Psi_R)/Qz^2)"
                )
            )
            io.header.ColDescription.append(
                fileio.base.Column(
                    name="r_reduced",
                    unit=None,
                    physical_quantity="reduced r (r_reduced = Psi_DS/Psi_R)"
                )
            )

            io.Dataset = np.column_stack([
                np.asarray(refl_i[:, 0], dtype=float),
                np.asarray(refl_i[:, 1], dtype=float),
                np.asarray(refl_i[:, 2], dtype=float),
                np.asarray(refl_i[:, 3], dtype=float),
                np.asarray(psi_r_i, dtype=float),
                np.asarray(sigma_i, dtype=float),
                np.asarray(rred_i, dtype=float),
            ])

        elif which == "SF":
            sf_i = _extract_table_track(GIXOS["SF"], selected_pos)
            fresnel_i = _extract_table_track(GIXOS["fresnel"], selected_pos)
            psi_ds_i = _extract_vector_track(GIXOS["Psi_DS"], selected_pos)
            pref_ds_i = _extract_vector_track(GIXOS["prefactor_DS"], selected_pos)

            n_sf = len(sf_i)
            if len(fresnel_i[:, 1]) != n_sf:
                raise ValueError("Selected fresnel track must have the same length as selected SF track.")
            if len(psi_ds_i) != n_sf:
                raise ValueError("Selected Psi_DS track must have the same length as selected SF track.")
            if len(pref_ds_i) != n_sf:
                raise ValueError("Selected prefactor_DS track must have the same length as selected SF track.")

            io.header.ColDescription[0] = fileio.base.Column(
                name="Qz",
                unit="1/angstrom",
                physical_quantity="wavevector transfer"
            )
            io.header.ColDescription[1] = fileio.base.Column(
                name="SF*RF",
                unit=None,
                physical_quantity="structure factor times Fresnel reflectivity (|Phi|^2*RF)"
            )
            io.header.ColDescription.append(
                fileio.base.ErrorColumn(
                    error_of="SF*RF",
                    error_type="uncertainty",
                    value_is="sigma"
                )
            )
            io.header.ColDescription.append(
                fileio.base.ErrorColumn(
                    error_of="Qz",
                    error_type="resolution",
                    value_is="sigma"
                )
            )
            io.header.ColDescription.append(
                fileio.base.Column(
                    name="SF",
                    unit=None,
                    physical_quantity="structure factor (|Phi|^2 = R*/Psi_DS/prefactor_DS)"
                )
            )
            io.header.ColDescription.append(
                fileio.base.ErrorColumn(
                    error_of="SF",
                    error_type="uncertainty",
                    value_is="sigma"
                )
            )
            io.header.ColDescription.append(
                fileio.base.Column(
                    name="Psi_DS",
                    unit=None,
                    physical_quantity="roughness factor of the GIXOS diffuse scattering R*"
                )
            )
            io.header.ColDescription.append(
                fileio.base.Column(
                    name="prefactor_DS",
                    unit=None,
                    physical_quantity="diffuse scattering prefactor (ta^2*tb^2*(Qc/2/Qz)^4)"
                )
            )

            RF = np.asarray(fresnel_i[:, 1], dtype=float)
            io.Dataset = np.column_stack([
                np.asarray(sf_i[:, 0], dtype=float),
                np.asarray(sf_i[:, 1], dtype=float) * RF,
                np.asarray(sf_i[:, 2], dtype=float) * RF,
                np.asarray(sf_i[:, 3], dtype=float),
                np.asarray(sf_i[:, 1], dtype=float),
                np.asarray(sf_i[:, 2], dtype=float),
                np.asarray(psi_ds_i, dtype=float),
                np.asarray(pref_ds_i, dtype=float),
            ])

        elif which == "GIXOS":
            io.header.ColDescription[0] = fileio.base.Column(
                name="beta", unit="deg", physical_quantity="angle"
            )
            io.header.ColDescription[1] = fileio.base.Column(
                name="Qz",
                unit="1/angstrom",
                physical_quantity="wavevector transfer"
            )
            io.header.ColDescription.append(
                fileio.base.Column(
                                    name="I", 
                                    unit=None, 
                                    physical_quantity="diffuse scattering I0 R* (bkg subtracted GIXOS)")
            )
            io.header.ColDescription.append(
                fileio.base.ErrorColumn(error_of="I", error_type="uncertainty", value_is="sigma")
            )

            io.Dataset = np.column_stack([
                np.asarray(GIXOS["tt"], dtype=float).ravel(),
                np.asarray(GIXOS["Qz"][:, qidx], dtype=float).ravel(),
                np.asarray(GIXOS["Intensity"][:, qidx], dtype=float).ravel(),
                np.asarray(GIXOS["error"][:, qidx], dtype=float).ravel(),
                
            ])

        # ------------------------------------------------------------
        # generate default export path
        # ------------------------------------------------------------
        if exportpath is not None and not multi_export:
            exportpath_i = exportpath
        else:
            suffix_map = {
                "refl": f"phi{int(qidx)}_R.ort",
                "SF": f"phi{int(qidx)}_SF.ort",
                "GIXOS": f"phi{int(qidx)}_GIXOS.ort",
            }
            exportpath_i = make_filename(meta, suffix=suffix_map[which])

        # ------------------------------------------------------------
        # convert to ORSO datasource and write file
        # ------------------------------------------------------------
        io.populate_metadata_to_orsodatasource()
        dataset = io.create_orsodataset(exportpath=exportpath_i)
        results.append((io, dataset))

    if len(results) == 1:
        return results[0]

    return results

#%%
# ----------------------------------------------------------------------------
# helper for recursive HDF5 storage of Python / NumPy objects
# ----------------------------------------------------------------------------

def _to_hdf5_compatible(obj):
    """
    Convert Python / NumPy objects into HDF5-storable equivalents.
    """
    if isinstance(obj, dict):
        return {k: _to_hdf5_compatible(v) for k, v in obj.items()}

    if isinstance(obj, np.ndarray):
        return obj

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, (str, bytes, int, float, bool)):
        return obj

    if obj is None:
        return None

    if isinstance(obj, tuple):
        obj = list(obj)

    if isinstance(obj, list):
        # try homogeneous numeric array first
        try:
            arr = np.asarray(obj)
            if arr.dtype != object:
                return arr
        except Exception:
            pass
        return [_to_hdf5_compatible(v) for v in obj]

    # fallback
    return json.dumps(obj)


def _write_obj_to_hdf5(h5group, name, obj):
    """
    Recursively write an object into an HDF5 group.
    """
    obj = _to_hdf5_compatible(obj)

    if isinstance(obj, dict):
        subgrp = h5group.create_group(name)
        subgrp.attrs["_py_type"] = "dict"
        for k, v in obj.items():
            _write_obj_to_hdf5(subgrp, str(k), v)
        return

    if obj is None:
        ds = h5group.create_dataset(name, data=np.array([], dtype=float))
        ds.attrs["_py_type"] = "none"
        return

    if isinstance(obj, list):
        subgrp = h5group.create_group(name)
        subgrp.attrs["_py_type"] = "list"
        for i, v in enumerate(obj):
            _write_obj_to_hdf5(subgrp, f"{i:08d}", v)
        return

    if isinstance(obj, str):
        ds = h5group.create_dataset(name, data=np.bytes_(obj))
        ds.attrs["_py_type"] = "str"
        return

    if isinstance(obj, bytes):
        ds = h5group.create_dataset(name, data=np.bytes_(obj))
        ds.attrs["_py_type"] = "bytes"
        return

    if isinstance(obj, bool):
        ds = h5group.create_dataset(name, data=np.bool_(obj))
        ds.attrs["_py_type"] = "bool"
        return

    if isinstance(obj, (int, float)):
        ds = h5group.create_dataset(name, data=obj)
        ds.attrs["_py_type"] = type(obj).__name__
        return

    if isinstance(obj, np.ndarray):
        ds = h5group.create_dataset(name, data=obj)
        ds.attrs["_py_type"] = "ndarray"
        return

    ds = h5group.create_dataset(name, data=np.bytes_(str(obj)))
    ds.attrs["_py_type"] = "str_fallback"


def _read_obj_from_hdf5(node):
    """
    Recursively reconstruct an object from HDF5.
    """
    if isinstance(node, h5py.Group):
        py_type = node.attrs.get("_py_type", None)

        if py_type == "list":
            keys = sorted(node.keys())
            return [_read_obj_from_hdf5(node[k]) for k in keys]

        out = {}
        for k in node.keys():
            out[k] = _read_obj_from_hdf5(node[k])
        return out

    py_type = node.attrs.get("_py_type", None)
    data = node[()]

    if py_type == "none":
        return None

    if py_type in ("str", "bytes", "str_fallback"):
        if isinstance(data, bytes):
            return data.decode("utf-8")
        if isinstance(data, np.bytes_):
            return bytes(data).decode("utf-8")
        return str(data)

    if py_type == "bool":
        return bool(data)

    if py_type == "int":
        return int(data)

    if py_type == "float":
        return float(data)

    if py_type == "ndarray":
        return np.array(data)

    arr = np.array(data)

    if arr.shape == ():
        val = arr.item()
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return val

    return arr


# ----------------------------------------------------------------------------
# Nexus / NXsas export and import for GIXOS
# ----------------------------------------------------------------------------

def export_gixos_nxs(
    GIXOS,
    filename,
    *,
    entry_name="entry",
    store_full_dict=True,
    compression="gzip"
):
    """
    Export background-corrected GIXOS data to a Nexus/HDF5 file with an NXsas-style core.

    Stored NXsas-style data
    -----------------------
    /entry
      definition = "NXsas"
      data (NXdata)
        img_gid_q : 3D array, shape (1, n_beta, n_qxy)
        q_z       : 2D array, shape (n_beta, n_qxy)
        q_xy      : 2D array, shape (n_beta, n_qxy)
        tt        : beta axis
        tth       : tth axis

    Additional round-trip storage
    -----------------------------
    /entry/gixos_dict/root
        full recursively stored GIXOS dictionary

    Parameters
    ----------
    GIXOS : dict
        Background-corrected GIXOS dictionary.

    filename : str or pathlib.Path
        Output .nxs / .h5 filename.

    entry_name : str, optional
        Name of the NXentry group. Default is "entry".

    store_full_dict : bool, optional
        If True, store a full recursive copy of GIXOS under
        /entry/gixos_dict/root for exact reconstruction.

    compression : str or None, optional
        Compression for HDF5 datasets, e.g. "gzip". Default is "gzip".

    Returns
    -------
    None
    """
    filename = Path(filename)

    intensity = np.asarray(GIXOS["Intensity"], dtype=float)
    qz = np.asarray(GIXOS["Qz"], dtype=float)
    qxy = np.asarray(GIXOS["Qxy"], dtype=float)

    # NXsas-like main signal with leading image dimension
    img_gid_q = intensity[np.newaxis, :, :]

    tt = np.asarray(GIXOS.get("tt", []), dtype=float)
    tth = np.asarray(GIXOS.get("tth", []), dtype=float)

    with h5py.File(filename, "w") as f:
        entry = f.create_group(entry_name)
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("definition", data=np.bytes_("NXsas"))

        # optional title
        title = ""
        try:
            title = GIXOS.get("metadata", {}).get("measurements", {}).get("sample", "")
        except Exception:
            title = ""
        if title:
            entry.create_dataset("title", data=np.bytes_(str(title)))

        data_grp = entry.create_group("data")
        data_grp.attrs["NX_class"] = "NXdata"

        ds_img = data_grp.create_dataset(
            "img_gid_q",
            data=img_gid_q,
            compression=compression
        )
        ds_qz = data_grp.create_dataset(
            "q_z",
            data=qz,
            compression=compression
        )
        ds_qxy = data_grp.create_dataset(
            "q_xy",
            data=qxy,
            compression=compression
        )
        ds_tt = data_grp.create_dataset("tt", data=tt)
        ds_tth = data_grp.create_dataset("tth", data=tth)

        data_grp.attrs["signal"] = "img_gid_q"

        ds_qz.attrs["units"] = "/angstrom"
        ds_qxy.attrs["units"] = "/angstrom"
        ds_tt.attrs["units"] = "deg"
        ds_tth.attrs["units"] = "deg"

        ds_img.attrs["long_name"] = "background corrected GIXOS intensity"
        ds_qz.attrs["long_name"] = "Qz"
        ds_qxy.attrs["long_name"] = "Qxy"
        ds_tt.attrs["long_name"] = "beta"
        ds_tth.attrs["long_name"] = "tth"

        if store_full_dict:
            dict_grp = entry.create_group("gixos_dict")
            dict_grp.attrs["NX_class"] = "NXcollection"
            _write_obj_to_hdf5(dict_grp, "root", GIXOS)


def load_gixos_nxs(
    filename,
    *,
    entry_name="entry",
    prefer_full_dict=True
):
    """
    Load a GIXOS dictionary from a Nexus/HDF5 file created by export_gixos_nxs().

    Parameters
    ----------
    filename : str or pathlib.Path
        Input .nxs / .h5 filename.

    entry_name : str, optional
        NXentry group name. Default is "entry".

    prefer_full_dict : bool, optional
        If True and /entry/gixos_dict/root exists, reconstruct and return
        the full stored dictionary.
        Otherwise return a minimal dictionary built from /entry/data.

    Returns
    -------
    GIXOS : dict
        Reconstructed GIXOS dictionary.
    """
    filename = Path(filename)

    with h5py.File(filename, "r") as f:
        entry = f[entry_name]

        if prefer_full_dict and "gixos_dict" in entry:
            dict_grp = entry["gixos_dict"]
            if "root" in dict_grp:
                return _read_obj_from_hdf5(dict_grp["root"])

        # fallback minimal reconstruction
        data_grp = entry["data"]

        img_gid_q = np.asarray(data_grp["img_gid_q"])
        qz = np.asarray(data_grp["q_z"])
        qxy = np.asarray(data_grp["q_xy"])
        tt = np.asarray(data_grp["tt"])
        tth = np.asarray(data_grp["tth"])

        GIXOS = {
            "Intensity": img_gid_q[0],
            "Qz": qz,
            "Qxy": qxy,
            "tt": tt,
            "tth": tth,
        }

        return GIXOS