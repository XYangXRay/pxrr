# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:25:50 2026

@author: shenc

general helper functions (some specific ones are in the respective modules)

"""
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
import numpy as np
import os
import numbers


def check_keys_numeric(keys, *dicts):
    """
    new function from Chen
    keys  – iterable of required keys (e.g., ['flux','cttime'])
    dicts – any number of dictionaries to check
    """
    for key in keys:
        for d in dicts:
            if key not in d:
                return False
            if not isinstance(d[key], numbers.Number):
                return False
    return True    

def mean1d_if_within_percent(arr, tol=0.01):
    """
    If all values differ from the mean by less than tol (relative),
    replace by the mean while preserving a compatible shape:
      (1,n) -> (1,1)
      (n,)  -> (1,)
    Otherwise return the original array.
    """
    a = np.asarray(arr, dtype=float)
    flat = a.ravel()
    m = flat.mean()

    scale = max(abs(m), 1e-12)
    max_rel_dev = np.max(np.abs(flat - m)) / scale

    if max_rel_dev < tol:
        print("array values vary < tolerance %.2f%%, average" % (tol * 100))
        if a.ndim == 2 and a.shape[0] == 1:
            return np.array([[m]], dtype=float)   # (1,1)
        elif a.ndim == 1:
            return np.array([m], dtype=float)     # (1,)
        else:
            # fallback for unexpected shapes
            return np.full_like(a, m)
    else:
        print("array values vary > tolerance %.2f%%, keep as array" % (tol * 100))
        return a
    
def yamlify(obj):
    """
    Recursively convert numpy objects into YAML-safe Python objects.
    """
    if isinstance(obj, dict):
        return {k: yamlify(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [yamlify(v) for v in obj]
    elif isinstance(obj, tuple):
        return [yamlify(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return yamlify(obj.item())
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj  



def _to_flow_seq(seq):
    """
    Convert a sequence into a ruamel flow-style sequence: [a, b, c]
    """
    cs = CommentedSeq(seq)
    cs.fa.set_flow_style()
    return cs


def set_flow_style_lists(meta):
    """
    Force selected list-like fields to be written in inline YAML style.
    """
    # top-level qxy0
    if "qxy0" in meta and isinstance(meta["qxy0"], list):
        meta["qxy0"] = _to_flow_seq(meta["qxy0"])

    # top-level tth
    if "tth" in meta and isinstance(meta["tth"], list):
        meta["tth"] = _to_flow_seq(meta["tth"])

    # measurements.scan / bkgscan
    if "measurements" in meta and isinstance(meta["measurements"], dict):
        if "scan" in meta["measurements"] and isinstance(meta["measurements"]["scan"], list):
            meta["measurements"]["scan"] = _to_flow_seq(meta["measurements"]["scan"])

        if "bkgscan" in meta["measurements"] and isinstance(meta["measurements"]["bkgscan"], list):
            meta["measurements"]["bkgscan"] = _to_flow_seq(meta["measurements"]["bkgscan"])

    # PseudoR.qxy0_select_idx
    if (
        "PseudoR" in meta
        and isinstance(meta["PseudoR"], dict)
        and "qxy0_select_idx" in meta["PseudoR"]
        and isinstance(meta["PseudoR"]["qxy0_select_idx"], list)
    ):
        meta["PseudoR"]["qxy0_select_idx"] = _to_flow_seq(meta["PseudoR"]["qxy0_select_idx"])
        
    # dependency.qz_selected
    if (
        "dependency" in meta
        and isinstance(meta["dependency"], dict)
        and "qz_selected" in meta["dependency"]
        and isinstance(meta["dependency"]["qz_selected"], list)
    ):
        meta["dependency"]["qz_selected"] = _to_flow_seq(meta["dependency"]["qz_selected"])

    # PseudoR.resolution_HW
    if (
        "PseudoR" in meta
        and isinstance(meta["PseudoR"], dict)
        and "resolution_HW" in meta["PseudoR"]
        and isinstance(meta["PseudoR"]["resolution_HW"], list)
    ):
        meta["PseudoR"]["resolution_HW"] = _to_flow_seq(meta["PseudoR"]["resolution_HW"])

    return meta

def normalize_qxy0_select_idx_inplace(GIXOS):
    """
    Normalize metadata['PseudoR']['qxy0_select_idx'] in place.

    Supports either:
    - a scalar integer index
    - a list / array of indices

    If metadata['qxy_bkg'] exists and is numeric, any selected qxy0 position
    with qxy0 >= qxy_bkg is removed. If metadata['qxy_bkg'] is missing or None,
    no chopping is applied.

    Parameters
    ----------
    GIXOS : dict
        GIXOS dictionary containing metadata['PseudoR'] and metadata['qxy0'].

    Returns
    -------
    qsel : np.ndarray
        1D integer array of the retained selected qxy0 indices.

    Notes
    -----
    The normalized selection is written back into:

        GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]

    as a scalar integer if one index remains, or as a list of integers if
    multiple indices remain.
    """
    meta = GIXOS["metadata"]
    pr = meta["PseudoR"]

    if "qxy0_select_idx" not in pr or pr["qxy0_select_idx"] is None:
        raise ValueError("metadata['PseudoR']['qxy0_select_idx'] is required.")

    qsel_raw = pr["qxy0_select_idx"]

    if np.ndim(qsel_raw) == 0:
        qsel = np.array([int(qsel_raw)], dtype=int)
    else:
        qsel = np.asarray(qsel_raw, dtype=int).ravel()

    if qsel.size == 0:
        raise ValueError("metadata['PseudoR']['qxy0_select_idx'] must not be empty.")

    qxy0_all = np.asarray(meta["qxy0"], dtype=float).ravel()
    ncols = len(qxy0_all)

    if np.any(qsel < 0) or np.any(qsel >= ncols):
        raise ValueError(
            f"metadata['PseudoR']['qxy0_select_idx'] contains index outside 0..{ncols-1}."
        )

    # only chop by qxy_bkg if it actually exists
    qxy_bkg = meta.get("qxy_bkg", None)
    if qxy_bkg is not None:
        try:
            qxy_bkg = float(qxy_bkg)
            keep_mask = qxy0_all[qsel] < qxy_bkg
            qsel = qsel[keep_mask]
        except (TypeError, ValueError):
            raise ValueError("metadata['qxy_bkg'] must be numeric if provided.")

        if qsel.size == 0:
            raise ValueError(
                "No selected qxy0 positions remain after filtering with qxy_bkg."
            )

    # write back normalized form
    if qsel.size == 1:
        pr["qxy0_select_idx"] = int(qsel[0])
    else:
        pr["qxy0_select_idx"] = qsel.tolist()

    return qsel


def make_filename(metadata, suffix=None):
    """
    Build a general output filename for data or plots from metadata.

    Naming:
    - single scan -> <sample>_<id>_<suffix>
    - scan array  -> <sample>_<id_min>_<id_max>_<suffix>

    where each id is formatted as at least 5 digits with leading zeros.

    Examples
    --------
    suffix = "SF.dat"    -> sample_00123_SF.dat
    suffix = "R.png"     -> sample_00123_R.png
    suffix = "Qxy.png"   -> sample_00123_00130_Qxy.png

    Special case
    ------------
    If metadata["facility"] indicates PETRA III/P08 and the sample name starts
    with "GID_", that prefix is removed from the filename.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary containing at least:
        - metadata["paths"]["path_out"]
        - metadata["measurements"]["sample"]
        - metadata["measurements"]["scan"]

    suffix : str
        Required output suffix, e.g. "SF.dat", "R.dat", "R.png", "Qxy.png".

    Returns
    -------
    filename : str
        Full output path including directory and filename.
    """
    if metadata is None:
        raise ValueError("metadata is required.")
    if suffix is None:
        raise ValueError("suffix must be provided.")

    if "paths" not in metadata or metadata["paths"] is None:
        raise ValueError("metadata['paths'] is required.")
    if "measurements" not in metadata or metadata["measurements"] is None:
        raise ValueError("metadata['measurements'] is required.")

    path_out = metadata["paths"].get("path_out", None)
    sample = metadata["measurements"].get("sample", None)
    scan = metadata["measurements"].get("scan", None)
    facility = metadata.get("facility", None)

    if path_out is None:
        raise ValueError("metadata['paths']['path_out'] is required.")
    if sample is None:
        raise ValueError("metadata['measurements']['sample'] is required.")
    if scan is None:
        raise ValueError("metadata['measurements']['scan'] is required.")

    # facility-specific sample-name cleanup
    if facility is not None:
        facility_str = str(facility)
        if ("PETRA III" in facility_str) and ("P08" in facility_str):
            if str(sample).startswith("GID_"):
                sample = str(sample)[4:]

    scan_arr = np.asarray(scan).ravel()
    if scan_arr.size == 0:
        raise ValueError("metadata['measurements']['scan'] must not be empty.")

    if scan_arr.size == 1:
        scan_part = f"{int(scan_arr[0]):05d}"
    else:
        scan_part = f"{int(np.min(scan_arr)):05d}_{int(np.max(scan_arr)):05d}"

    filename = f"{sample}_{scan_part}_{suffix}"
    os.makedirs(path_out, exist_ok=True)
    return os.path.join(path_out, filename)
