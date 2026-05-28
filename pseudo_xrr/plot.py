# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:46:59 2026

@author: shenc

everything related to plotting
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker, DrawingArea
from matplotlib.lines import Line2D

def _get_selected_qidx_array(metadata):
    """
    Return the selected qxy0 indices as a 1D integer NumPy array.

    This helper normalizes ``metadata["PseudoR"]["qxy0_select_idx"]`` so that
    downstream plotting code can treat both cases uniformly:

    - a single selected qxy0 index stored as an integer
    - multiple selected qxy0 indices stored as a list / array

    Parameters
    ----------
    metadata : dict
        Metadata dictionary containing a ``"PseudoR"`` section with the field
        ``"qxy0_select_idx"``.

    Returns
    -------
    qsel : np.ndarray
        One-dimensional integer array of selected qxy0 indices.

    Notes
    -----
    This helper is used by:
    - `_get_qxy0_value()`
    - `GIXOS_RRF_plot()`
    - `GIXOS_R_plot()`
    - `GIXOS_RRF_multi_plot()`
    - `GIXOS_raw_plot()`
    """
    pr = metadata["PseudoR"]
    qsel = pr["qxy0_select_idx"]
    if np.ndim(qsel) == 0:
        return np.array([int(qsel)], dtype=int)
    return np.asarray(qsel, dtype=int).ravel()


def _get_qxy0_value(metadata, selected_pos):
    """
    Return the raw qxy0 column index and its qxy0 value for one selected track.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary containing:
        - ``metadata["PseudoR"]["qxy0_select_idx"]``
        - ``metadata["qxy0"]``

    selected_pos : int
        Position within the selected-qxy0 list. For example:
        - 0 for the first selected qxy0 track
        - 1 for the second selected qxy0 track

    Returns
    -------
    qidx : int
        Raw column index in the original GIXOS arrays.

    qxy0_val : float
        Corresponding qxy0 value in /Å.

    Notes
    -----
    This helper is used whenever a plot needs to:
    - slice the original raw GIXOS arrays by column index
    - show the physical qxy0 value in the title
    """
    qsel = _get_selected_qidx_array(metadata)
    qidx = int(qsel[selected_pos])
    qxy0_all = np.asarray(metadata["qxy0"], dtype=float).ravel()
    return qidx, float(qxy0_all[qidx])

def _format_rrf_legend_metadata(metadata, sigma_cw=None):
    """
    Format legend metadata text for R/RF-style plots.

    This helper extracts reflectivity resolution and background metadata from
    ``metadata["PseudoR"]`` and formats them consistently for both:

    - single-qxy0 plots, where a simple legend title string is needed
    - multi-qxy0 plots, where the same text is used inside a custom legend box

    Parameters
    ----------
    metadata : dict
        Metadata dictionary containing a ``"PseudoR"`` section.

    sigma_cw : float or None, optional
        Capillary-wave roughness value to include in the legend metadata.
        If None, only reflectivity resolution and background mode are returned.

    Returns
    -------
    info : dict
        Dictionary with the following string fields:

        - ``"res_str"`` :
          formatted reflectivity resolution string
        - ``"bkg_str"`` :
          formatted background-mode string
        - ``"legend_title_single"`` :
          multiline legend title for single-qxy0 plots
        - ``"sigma_str"`` :
          formatted sigma_CW string if ``sigma_cw`` is provided, otherwise None

    Notes
    -----
    Resolution formatting:
    - circular resolution (mode 0): scalar, formatted in /Å to 5 decimals
    - slit resolution (mode 1): two-element array, formatted in mm to 2 decimals
    """
    pr = metadata["PseudoR"]

    resolution_mode = pr.get("resolution_mode", None)
    resolution_hw = pr.get("resolution_HW", None)
    bkg_mode = pr.get("bkg_mode", None)

    if resolution_mode == 0:
        res_str = f"{float(resolution_hw):.5f} /Å"

    elif resolution_mode == 1:
        res_hw = np.asarray(resolution_hw, dtype=float).ravel()
        if res_hw.size >= 2:
            res_str = f"[{res_hw[0]:.2f}, {res_hw[1]:.2f}] mm"
        elif res_hw.size == 1:
            res_str = f"{res_hw[0]:.2f} mm"
        else:
            res_str = "unknown"

    else:
        res_str = "unknown"

    if bkg_mode is None:
        bkg_str = "w/o bkg"
    else:
        bkg_str = "with bkg"

    sigma_str = None
    if sigma_cw is not None and np.isfinite(sigma_cw):
        sigma_str = f"{float(sigma_cw):.2f} Å"

    legend_title_single = rf"$\delta Q_{{xy,R}}$: {res_str}" + "\n" + f"{bkg_str}"

    return {
        "res_str": res_str,
        "bkg_str": bkg_str,
        "sigma_str": sigma_str,
        "legend_title_single": legend_title_single,
    }

def _make_rrf_multi_legend_box(metadata, qsel, sigma_avg, max_entries_per_line=4):
    """
    Create one combined boxed legend for the multi-qxy0 RRF comparison plot.

    The box contains:

    - reflectivity resolution information
    - background-mode information
    - average sigma_CW value at Qz ~ 0.6 /Å
    - marker meaning for:
      * open circle : |Phi|^2
      * x marker    : R/R_F
    - a wrapped color-code block for the selected qxy0 values

    Parameters
    ----------
    metadata : dict
        Metadata dictionary containing:
        - ``metadata["PseudoR"]``
        - ``metadata["qxy0"]``

    qsel : array-like
        Selected qxy0 indices.

    sigma_avg : float
        Average capillary-wave roughness value at Qz ~ 0.6 /Å across all
        selected qxy0 tracks.

    max_entries_per_line : int, optional
        Maximum number of qxy0 values shown per wrapped line in the color-code
        section of the legend box. Default is 4.

    Returns
    -------
    anchored : matplotlib.offsetbox.AnchoredOffsetbox
        Combined boxed legend artist that can be added to an axes by
        ``ax.add_artist(...)``.

    Notes
    -----
    This helper is used by `GIXOS_RRF_multi_plot()`.
    """
    pr = metadata["PseudoR"]
    qxy0_all = np.asarray(metadata["qxy0"], dtype=float).ravel()

    # ------------------------------------------------------------
    # title text: 3 lines
    # ------------------------------------------------------------
    info = _format_rrf_legend_metadata(metadata, sigma_cw=sigma_avg)
    res_str = info["res_str"]
    bkg_str = info["bkg_str"]
    sigma_str = info["sigma_str"]

    title_area = TextArea(
        rf"$\delta Q_{{xy,R}}$: {res_str}" + "\n"
        + f"{bkg_str}" + "\n"
        + rf"$\sigma_{{CW}}(0.6/\AA)$: {sigma_str}",
        textprops=dict(size=9)
    )

    # ------------------------------------------------------------
    # marker rows
    # ------------------------------------------------------------
    def make_marker_row(marker, label, open_circle=False):
        da = DrawingArea(20, 12, 0, 0)
        if open_circle:
            artist = Line2D(
                [10], [6],
                marker=marker,
                markersize=6,
                markerfacecolor="none",
                markeredgecolor="k",
                linestyle="None",
                color="k"
            )
        else:
            artist = Line2D(
                [10], [6],
                marker=marker,
                markersize=6,
                linestyle="None",
                color="k"
            )
        da.add_artist(artist)

        txt = TextArea(label, textprops=dict(size=9))
        return HPacker(children=[da, txt], align="center", pad=0, sep=4)

    row_sf = make_marker_row("o", r"$|\Phi|^2$", open_circle=True)
    row_rrf = make_marker_row("x", r"$R/R_F$")

    # ------------------------------------------------------------
    # wrapped colored qxy0 lines (no indent, label on its own line)
    # ------------------------------------------------------------
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)
    if color_cycle is None or len(color_cycle) == 0:
        color_cycle = ["k"]

    qxy_lines = []

    # label line
    label_line = TextArea(r"$Q_{xy,0}$ color code:", textprops=dict(size=9, color="k"))
    qxy_lines.append(label_line)

    # create colored value elements
    qxy_text_areas = []
    for isel, qidx in enumerate(qsel):
        color_i = color_cycle[isel % len(color_cycle)]
        qxy0_val = float(qxy0_all[qidx])

        txt = f"{qxy0_val:.3f}/Å"
        if isel < len(qsel) - 1:
            txt += ", "

        qxy_text_areas.append(
            TextArea(txt, textprops=dict(size=9, color=color_i))
        )

    # wrap into rows WITHOUT indentation
    for start in range(0, len(qxy_text_areas), max_entries_per_line):
        chunk = qxy_text_areas[start:start + max_entries_per_line]

        row = HPacker(
            children=chunk,
            align="baseline",
            pad=0,
            sep=1
        )

        qxy_lines.append(row)

    # ------------------------------------------------------------
    # final packed box
    # ------------------------------------------------------------
    packed_children = [title_area, row_sf, row_rrf] + qxy_lines

    packed = VPacker(
        children=packed_children,
        align="left",
        pad=0,
        sep=3
    )

    anchored = AnchoredOffsetbox(
        loc="lower left",
        child=packed,
        frameon=True,
        borderpad=0.6,
        pad=0.4
    )
    return anchored


def GIXOS_raw_plot(
    GIXOSdata_q,
    GIXOSbkg_q,
    GIXOS_ana,
    *,
    metadata=None,
    selected_pos=0,
    title=None,
    show=True,
    add_top_qz_axis=True
):
    """
    Plot the effect of GIXOS background correction for one selected qxy0 track.

    This function compares, at one selected qxy0 position:

    - raw GIXOS data
    - chamber background
    - chamber-background-subtracted data
    - bulk background used for subtraction
    - final corrected data with error bars

    The x-axis is beta (deg), taken from the ``"tt"`` field. Optionally, a
    top axis labelled by Qz is added.

    Parameters
    ----------
    GIXOSdata_q : dict
        Raw sample data dictionary after conversion to q-space.

    GIXOSbkg_q : dict
        Chamber background data dictionary after conversion to q-space.

    GIXOS_ana : dict
        Output dictionary from `GIXOS_background_corr()` or later processing.

    metadata : dict, optional
        Metadata dictionary. If None, ``GIXOS_ana["metadata"]`` is used.

    selected_pos : int, optional
        Position within ``metadata["PseudoR"]["qxy0_select_idx"]`` specifying
        which selected qxy0 track to plot. Default is 0.

    title : str, optional
        Optional user-supplied extra title line.

    show : bool, optional
        If True, call ``plt.show()``. Default is True.

    add_top_qz_axis : bool, optional
        If True and Qz is available, add a top x-axis labelled by Qz.
        Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.

    ax : matplotlib.axes.Axes
        Main axes.
    """
    if metadata is None:
        metadata = GIXOS_ana.get("metadata", None)

    if metadata is None:
        raise ValueError("metadata is required either explicitly or in GIXOS_ana['metadata'].")

    if "PseudoR" not in metadata or metadata["PseudoR"] is None:
        raise ValueError("metadata['PseudoR'] is required.")

    col_idx, qxy0_val = _get_qxy0_value(metadata, selected_pos)

    # x-axis: beta / tt
    beta = np.asarray(GIXOSdata_q["tt"], dtype=float).ravel()

    # raw sample and chamber background
    y_raw = np.asarray(GIXOSdata_q["Intensity"][:, col_idx], dtype=float).ravel()
    y_bkg = np.asarray(GIXOSbkg_q["Intensity"][:, col_idx], dtype=float).ravel()

    # chamber-subtracted data
    y_chamber_sub = y_raw - y_bkg

    # bulk background at selected column
    if "bulkbkg" not in GIXOS_ana or GIXOS_ana["bulkbkg"] is None:
        y_bulk = np.full_like(beta, np.nan, dtype=float)
    else:
        if "Intensity_at_GIXOS" in GIXOS_ana["bulkbkg"] and GIXOS_ana["bulkbkg"]["Intensity_at_GIXOS"] is not None:
            bulk_arr = np.asarray(GIXOS_ana["bulkbkg"]["Intensity_at_GIXOS"], dtype=float)

            if bulk_arr.ndim == 2:
                if bulk_arr.shape[1] > col_idx:
                    y_bulk = bulk_arr[:, col_idx].ravel()
                else:
                    y_bulk = bulk_arr[:, -1].ravel()
            else:
                y_bulk = bulk_arr.ravel()
        else:
            y_bulk = np.full_like(beta, np.nan, dtype=float)

    # corrected data and error
    y_corr = np.asarray(GIXOS_ana["Intensity"][:, col_idx], dtype=float).ravel()
    err_corr = np.asarray(GIXOS_ana["error"][:, col_idx], dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(5.8, 4.5))

    ax.plot(beta, y_raw, color="k", linewidth=1.4, label="raw GIXOS")
    ax.plot(beta, y_bkg, color="gray", linewidth=1.4, label="chamber bkg")
    ax.plot(beta, y_chamber_sub, color="b", linewidth=1.4, label="raw - chamber bkg")

    if np.any(np.isfinite(y_bulk)):
        ax.plot(beta, y_bulk, color="orange", linewidth=1.6, label="bulk bkg")

    ax.errorbar(
        beta, y_corr, yerr=err_corr,
        fmt="o", markersize=4,
        color="r", ecolor="r",
        elinewidth=1, capsize=2,
        label="corrected"
    )

    ax.axhline(0, color="k", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_xlabel(r"$\beta\;[\mathrm{deg}]$")
    ax.set_ylabel("intensity")

    # title
    title_lines = []

    if title is not None:
        title_lines.append(title)

    meas = metadata.get("measurements", None)
    if meas is not None:
        scan_val = meas.get("scan", None)
        if scan_val is not None:
            scan_arr = np.asarray(scan_val).ravel()
            if scan_arr.size == 1:
                scan_str = f"{int(scan_arr[0])}"
            else:
                scan_str = f"{int(np.min(scan_arr))} - {int(np.max(scan_arr))}"
            title_lines.append(f"scan Id = {scan_str}, " + rf"$q_{{xy,0}} = {qxy0_val:.3f}\,/\,\AA$")
    else:
        title_lines.append("scan Id = ?, " + rf"$q_{{xy,0}} = {qxy0_val:.3f}\,/\,\AA$")

    ax.set_title("\n".join(title_lines))

    ax.legend()

    # set y-limits using only data above 2*Qc
    Qc = None
    if "sample_params" in metadata and metadata["sample_params"] is not None:
        Qc = metadata["sample_params"].get("Qc", None)

    if Qc is not None and "Qz" in GIXOSdata_q:
        qz_arr = np.asarray(GIXOSdata_q["Qz"][:, col_idx], dtype=float).ravel()
        mask_ylim = qz_arr > 2.0 * float(Qc)

        if np.count_nonzero(mask_ylim) > 3:
            y_for_ylim = []

            for arr in [y_raw, y_bkg, y_chamber_sub, y_bulk, y_corr]:
                arr = np.asarray(arr, dtype=float).ravel()
                if arr.size == qz_arr.size:
                    vals = arr[mask_ylim]
                    vals = vals[np.isfinite(vals)]
                    if vals.size > 0:
                        y_for_ylim.append(vals)

            if len(y_for_ylim) > 0:
                y_for_ylim = np.concatenate(y_for_ylim)

                ymin = np.min(y_for_ylim)
                ymax = np.max(y_for_ylim)

                if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                    pad = 0.05 * (ymax - ymin)
                    ax.set_ylim(ymin - pad, ymax + pad)

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(left=0, right=xmax)
    fig.tight_layout()

    # optional top axis: Qz
    if add_top_qz_axis and "Qz" in GIXOSdata_q:
        qz_arr = np.asarray(GIXOSdata_q["Qz"][:, col_idx], dtype=float).ravel()

        if qz_arr.size == beta.size and beta.size > 1:
            ax_top = ax.twiny()
            ax_top.set_xlim(ax.get_xlim())

            beta_ticks = ax.get_xticks()
            qz_ticklabels = []
            for bt in beta_ticks:
                idx = int(np.argmin(np.abs(beta - bt)))
                qz_ticklabels.append(f"{qz_arr[idx]:.2f}")

            ax_top.set_xticks(beta_ticks)
            ax_top.set_xticklabels(qz_ticklabels)
            ax_top.set_xlabel(r"$Q_z\;[\AA^{-1}]$")

    if show:
        plt.show()

    return fig, ax


# def GIXOS_raw_plot(
#     GIXOSdata_q,
#     GIXOSbkg_q,
#     GIXOS_ana,
#     *,
#     metadata=None,
#     title=None,
#     show=True,
#     add_top_qz_axis=True
# ):
#     """
#     Plot the effect of GIXOS background correction for the selected qxy0 column.

#     This function compares, at the selected qxy0 column:

#     - raw GIXOS data
#     - chamber background
#     - chamber-background-subtracted data
#     - bulk background used for subtraction
#     - final corrected data with error bars

#     The x-axis is beta (deg), taken from the "tt" field. Optionally, a top
#     axis showing Qz is added if Qz information is available.

#     Parameters
#     ----------
#     GIXOSdata_q : dict
#         Raw sample data dictionary after conversion to q-space.

#     GIXOSbkg_q : dict
#         Chamber background data dictionary after conversion to q-space.

#     GIXOS_ana : dict
#         Output dictionary from `GIXOS_background_corr()`.

#     metadata : dict, optional
#         Metadata dictionary. If None, `GIXOS_ana["metadata"]` is used.

#     title : str, optional
#         Plot title.

#     show : bool, optional
#         If True, show the figure. Default is True.

#     add_top_qz_axis : bool, optional
#         If True and Qz is available, add a top x-axis labelled by Qz.
#         Default is True.

#     Returns
#     -------
#     fig : matplotlib.figure.Figure
#         The generated figure.

#     ax : matplotlib.axes.Axes
#         Main axes.
#     """
#     if metadata is None:
#         metadata = GIXOS_ana.get("metadata", None)

#     if metadata is None:
#         raise ValueError("metadata is required either explicitly or in GIXOS_ana['metadata'].")

#     if "PseudoR" not in metadata or metadata["PseudoR"] is None:
#         raise ValueError("metadata['PseudoR'] is required.")
#     if "qxy0_select_idx" not in metadata["PseudoR"]:
#         raise ValueError("metadata['PseudoR']['qxy0_select_idx'] is required.")

#     col_idx = int(metadata["PseudoR"]["qxy0_select_idx"])

#     # x-axis: beta / tt
#     beta = np.asarray(GIXOSdata_q["tt"], dtype=float).ravel()

#     # raw sample and chamber background
#     y_raw = np.asarray(GIXOSdata_q["Intensity"][:, col_idx], dtype=float).ravel()
#     y_bkg = np.asarray(GIXOSbkg_q["Intensity"][:, col_idx], dtype=float).ravel()

#     # chamber-subtracted data
#     y_chamber_sub = y_raw - y_bkg

#     # bulk background at selected column
#     if "bulkbkg" not in GIXOS_ana or GIXOS_ana["bulkbkg"] is None:
#         y_bulk = np.full_like(beta, np.nan, dtype=float)
#     else:
#         if "Intensity_at_GIXOS" in GIXOS_ana["bulkbkg"] and GIXOS_ana["bulkbkg"]["Intensity_at_GIXOS"] is not None:
#             bulk_arr = np.asarray(GIXOS_ana["bulkbkg"]["Intensity_at_GIXOS"], dtype=float)

#             if bulk_arr.ndim == 2:
#                 # if corrected data has had bulk columns removed, shapes may differ
#                 if bulk_arr.shape[1] > col_idx:
#                     y_bulk = bulk_arr[:, col_idx].ravel()
#                 else:
#                     # fallback: use the last available column if selected col was removed
#                     y_bulk = bulk_arr[:, -1].ravel()
#             else:
#                 y_bulk = bulk_arr.ravel()
#         else:
#             y_bulk = np.full_like(beta, np.nan, dtype=float)

#     # corrected data and error
#     y_corr = np.asarray(GIXOS_ana["Intensity"][:, col_idx], dtype=float).ravel()
#     err_corr = np.asarray(GIXOS_ana["error"][:, col_idx], dtype=float).ravel()

#     fig, ax = plt.subplots(figsize=(5.8, 4.5))

#     ax.plot(beta, y_raw, color="k", linewidth=1.4, label="raw GIXOS")
#     ax.plot(beta, y_bkg, color="gray", linewidth=1.4, label="chamber bkg")
#     ax.plot(beta, y_chamber_sub, color="b", linewidth=1.4, label="raw - chamber bkg")

#     if np.any(np.isfinite(y_bulk)):
#         ax.plot(beta, y_bulk, color="orange", linewidth=1.6, label="bulk bkg")

#     ax.errorbar(
#         beta, y_corr, yerr=err_corr,
#         fmt="o", markersize=4,
#         color="r", ecolor="r",
#         elinewidth=1, capsize=2,
#         label="corrected"
#     )

#     ax.axhline(0, color="k", linestyle="--", linewidth=1.0, alpha=0.7)

#     ax.set_xlabel(r"$\beta\;[\mathrm{deg}]$")
#     ax.set_ylabel("intensity")

#     # title
#     title_lines = []
#     if title is not None:
#         title_lines.append(title)
#     else:
#         title_lines.append("GIXOS background correction")

#     if "measurements" in metadata and metadata["measurements"] is not None:
#         scan_val = metadata["measurements"].get("scan", None)
#         if scan_val is not None:
#             scan_arr = np.asarray(scan_val).ravel()
#             if scan_arr.size == 1:
#                 title_lines.append(f"scan Id = {int(scan_arr[0])}")
#             elif scan_arr.size > 1:
#                 title_lines.append(f"scan Id = {int(np.min(scan_arr))} - {int(np.max(scan_arr))}")

#     ax.set_title("\n".join(title_lines))

#     ax.legend()
    
#     # ------------------------------------------------------------
#     # set y-limits using only data above 2*Qc
#     # ------------------------------------------------------------
#     if "sample_params" in metadata and metadata["sample_params"] is not None:
#         Qc = metadata["sample_params"].get("Qc", None)
#     else:
#         Qc = None
    
#     if Qc is not None and "Qz" in GIXOSdata_q:
#         qz_arr = np.asarray(GIXOSdata_q["Qz"][:, col_idx], dtype=float).ravel()
#         mask_ylim = qz_arr > 2.0 * float(Qc)
    
#         if np.count_nonzero(mask_ylim) > 3:
#             y_for_ylim = []
    
#             for arr in [y_raw, y_bkg, y_chamber_sub, y_bulk, y_corr]:
#                 arr = np.asarray(arr, dtype=float).ravel()
#                 if arr.size == qz_arr.size:
#                     vals = arr[mask_ylim]
#                     vals = vals[np.isfinite(vals)]
#                     if vals.size > 0:
#                         y_for_ylim.append(vals)
    
#             if len(y_for_ylim) > 0:
#                 y_for_ylim = np.concatenate(y_for_ylim)
    
#                 ymin = np.min(y_for_ylim)
#                 ymax = np.max(y_for_ylim)
    
#                 if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
#                     pad = 0.05 * (ymax - ymin)
#                     ax.set_ylim(ymin - pad, ymax + pad)
    
#     xmin, xmax = ax.get_xlim()
#     ax.set_xlim(left=0, right=xmax)
#     fig.tight_layout()

#     # optional top axis: Qz
#     if add_top_qz_axis and "Qz" in GIXOSdata_q:
#         qz_arr = np.asarray(GIXOSdata_q["Qz"][:, col_idx], dtype=float).ravel()

#         if qz_arr.size == beta.size and beta.size > 1:
#             ax_top = ax.twiny()
#             ax_top.set_xlim(ax.get_xlim())

#             # choose a few tick positions from bottom axis and map nearest beta->Qz
#             beta_ticks = ax.get_xticks()
#             qz_ticklabels = []
#             for bt in beta_ticks:
#                 idx = int(np.argmin(np.abs(beta - bt)))
#                 qz_ticklabels.append(f"{qz_arr[idx]:.2f}")

#             ax_top.set_xticks(beta_ticks)
#             ax_top.set_xticklabels(qz_ticklabels)
#             ax_top.set_xlabel(r"$Q_z\;[\AA^{-1}]$")

#     if show:
#         plt.show()

#     return fig, ax


# -----------------------------------------------------------------------------
# qxy dependence fit / predict and kappa value
# -----------------------------------------------------------------------------

def GIXOS_qxy_dependence_plot(
    results,
    *,
    metadata=None,
    show_refs=True,
    show_err=True,
    title=None
    ):
    """
    Plot extracted Qxy dependences together with CWM / eCWM reference curves.

    Parameters
    ----------
    results : dict
        Output dictionary from GIXOS_qxy_dependence().

    metadata : dict, optional
        Metadata dictionary. If provided and if
        metadata["sample_params"] contains the fields
        "tension", "temperature", and "amin", these are displayed
        on the plot.

    show_refs : bool, optional
        If True, plot the CWM / eCWM reference curves.

    show_err : bool, optional
        If True, plot the upper/lower eCWM error curves if available.

    title : str, optional
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(5, 7))

    for idx, qxy in enumerate(results['Qxy']):

        ax.plot(
            qxy,
            results['I_sum_offset'][idx, :],
            marker="o",
            linestyle="None",
            label=f"$Q_{{z}}≈{results['target_qz'][idx]:.2f}\\,\\AA^{{-1}}$"
        )

        if show_refs and results["ref_CWM"] is not None:
            ax.plot(
                qxy,
                results["ref_CWM"][idx, :],
                linestyle=":",
                color="k",
                linewidth=1.4
            )

        if show_refs and results["ref_eCWM"] is not None:
            ax.plot(
                qxy,
                results["ref_eCWM"][idx, :],
                linestyle="-",
                color="k",
                linewidth=1.8
            )

        if show_err and "ref_eCWM_err_u" in results and "ref_eCWM_err_l" in results:
            ax.plot(
                qxy,
                results["ref_eCWM_err_u"][idx, :],
                linestyle="--",
                color="k",
                linewidth=1.4
            )
            ax.plot(
                qxy,
                results["ref_eCWM_err_l"][idx, :],
                linestyle="--",
                color="k",
                linewidth=1.4
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$Q_{xy}\,(\AA^{-1})$")
    ax.set_ylabel("(GI) diffuse intensity (offset)")

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(r"$Q_{xy}$ dependence")

    # sample-parameter + scan annotation
    if metadata is not None:
        text_lines = []
        
        # --- scan id ---
        if "measurements" in metadata and metadata["measurements"] is not None:
            meas = metadata["measurements"]
            scan_val = meas.get("scan", None)

            if scan_val is not None:
                scan_arr = np.asarray(scan_val).ravel()

                if scan_arr.size == 1:
                    text_lines.append(f"scan Id = {int(scan_arr[0])}")
                elif scan_arr.size > 1:
                    text_lines.append(
                        f"scan Id = {int(np.min(scan_arr))} - {int(np.max(scan_arr))}"
                    )
        # --- sample parameters ---
        if "sample_params" in metadata and metadata["sample_params"] is not None:
            sp = metadata["sample_params"]

            tension = sp.get("tension", None)
            temperature = sp.get("temperature", None)
            amin = sp.get("amin", None)

            if tension is not None:
                text_lines.append(rf"$\gamma = {float(tension)*1000:.1f}\,\mathrm{{mN/m}}$")
            if temperature is not None:
                text_lines.append(rf"$T = {float(temperature):.1f}\,\mathrm{{K}}$")
            if amin is not None:
                text_lines.append(rf"$a_{{\min}} = {float(amin):.1f}\,\AA$")

        if len(text_lines) > 0:
            ax.text(
                0.98, 0.98,
                "\n".join(text_lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="none"
                )
            )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

    fig.tight_layout()
    plt.show()

    return fig, ax

def GIXOS_RRF_plot(
    GIXOS,
    *,
    metadata=None,
    selected_pos=0,
    title=None,
    show=True
):
    """
    Plot pseudo-reflectivity / roughness-factor analysis for one selected qxy0 track.

    This function visualizes, for one selected qxy0 position:

    - normalized raw GIXOS intensity
    - intrinsic structure factor |Phi(Qz)|^2
    - pseudo-reflectivity normalized by Fresnel reflectivity, R/R_F
    - specular roughness factor Psi_R(Qz)

    The figure is shown on a semi-log scale (log intensity, linear Qz).

    Parameters
    ----------
    GIXOS : dict
        Output dictionary from `GIXOS2R()`.

    metadata : dict, optional
        Metadata dictionary. If None, ``GIXOS["metadata"]`` is used.

    selected_pos : int, optional
        Position within ``metadata["PseudoR"]["qxy0_select_idx"]`` specifying
        which selected qxy0 track to plot. Default is 0.

    title : str, optional
        Optional user-supplied extra title line. If None, only the standard
        title lines are shown.

    show : bool, optional
        If True, call ``plt.show()``. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.

    ax : matplotlib.axes.Axes
        Axes containing the plot.

    Notes
    -----
    This function supports both:
    - single-qxy0 output from `GIXOS2R()`
    - multi-qxy0 output stacked along axis 0
    """
    
    if metadata is None:
        metadata = GIXOS.get("metadata", None)

    if metadata is None:
        raise ValueError("metadata is required either explicitly or in GIXOS['metadata'].")

    if "PseudoR" not in metadata or metadata["PseudoR"] is None:
        raise ValueError("metadata['PseudoR'] is required.")
    if "sample_params" not in metadata or metadata["sample_params"] is None:
        raise ValueError("metadata['sample_params'] is required.")

    pr = metadata["PseudoR"]
    samp = metadata["sample_params"]

    qidx_raw, qxy0_val = _get_qxy0_value(metadata, selected_pos)
    Qc = float(samp["Qc"])

    # ------------------------------------------------------------
    # select one output track
    # ------------------------------------------------------------
    if np.asarray(GIXOS["SF"]).ndim == 3:
        sf = np.asarray(GIXOS["SF"][selected_pos], dtype=float)
        refl = np.asarray(GIXOS["refl"][selected_pos], dtype=float)
        fresnel = np.asarray(GIXOS["fresnel"][selected_pos], dtype=float)
        psi_r = np.asarray(GIXOS["Psi_R"][selected_pos], dtype=float)
    else:
        sf = np.asarray(GIXOS["SF"], dtype=float)
        refl = np.asarray(GIXOS["refl"], dtype=float)
        fresnel = np.asarray(GIXOS["fresnel"], dtype=float)
        psi_r = np.asarray(GIXOS["Psi_R"], dtype=float)

    # ------------------------------------------------------------
    # 1) structure factor |Phi|^2  (black scatter)
    # ------------------------------------------------------------
    qz_sf = np.asarray(sf[:, 0], dtype=float)
    y_sf = np.asarray(sf[:, 1], dtype=float)
    err_sf = np.asarray(sf[:, 2], dtype=float)

    # ------------------------------------------------------------
    # 2) pseudo-reflectivity / Fresnel  (blue scatter)
    # ------------------------------------------------------------
    qz_refl = np.asarray(refl[:, 0], dtype=float)
    y_refl_norm = np.asarray(refl[:, 1], dtype=float) / np.asarray(fresnel[:, 1], dtype=float)
    err_refl_norm = np.asarray(refl[:, 2], dtype=float) / np.asarray(fresnel[:, 1], dtype=float)

    # ------------------------------------------------------------
    # 3) normalized raw GIXOS intensity (red scatter)
    # ------------------------------------------------------------
    qz_raw = np.asarray(GIXOS["Qz"][:, qidx_raw], dtype=float)
    y_raw = np.asarray(GIXOS["Intensity"][:, qidx_raw], dtype=float)
    err_raw = np.asarray(GIXOS["error"][:, qidx_raw], dtype=float)

    q_target = 2.0 * Qc
    idx_norm_raw = int(np.argmin(np.abs(qz_raw - q_target)))
    idx_norm_sf = int(np.argmin(np.abs(qz_sf - q_target)))

    if y_raw[idx_norm_raw] <= 0 or y_sf[idx_norm_sf] <= 0:
        raw_scale = 1.0
    else:
        raw_scale = y_sf[idx_norm_sf] / y_raw[idx_norm_raw]

    y_raw_norm = y_raw * raw_scale
    err_raw_norm = err_raw * raw_scale

    # ------------------------------------------------------------
    # 4) Psi_R (blue line)
    # ------------------------------------------------------------
    psi_r = np.asarray(psi_r, dtype=float)

    # ------------------------------------------------------------
    # legend title from resolution metadata
    # keep old style, but remove deltaQ and sigma_CW
    # ------------------------------------------------------------
    legend_title = _format_rrf_legend_metadata(metadata)["legend_title_single"]

    # ------------------------------------------------------------
    # plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 4.8))

    # 1) Psi_R
    ax.plot(
        qz_refl, psi_r,
        linestyle="-", linewidth=1.8,
        color="b",
        label=r"$\Psi_R$",
        zorder=1
    )

    # 2) normalized GIXOS
    ax.errorbar(
        qz_raw, y_raw_norm, yerr=err_raw_norm,
        fmt="o", markersize=4,
        color="r", ecolor="r",
        elinewidth=1, capsize=2,
        label="GIXOS (norm.)",
        zorder=2
    )

    # 3) SF
    ax.errorbar(
        qz_sf, y_sf, yerr=err_sf,
        fmt="o", markersize=4,
        color="k", ecolor="k",
        elinewidth=1, capsize=2,
        label=r"$|\Phi(Q_z)|^2$",
        zorder=3
    )

    # 4) R/RF
    ax.errorbar(
        qz_refl, y_refl_norm, yerr=err_refl_norm,
        fmt="o", markersize=4,
        color="b", ecolor="b",
        elinewidth=1, capsize=2,
        label=r"$R/R_F$",
        zorder=4
    )
    
    # eta = 2 line and label
    if "Qz_eta2" in GIXOS and GIXOS["Qz_eta2"] is not None:
        qz_eta2 = float(GIXOS["Qz_eta2"])

        ax.axvline(
            qz_eta2,
            linestyle="--",
            color="k",
            linewidth=1.2
        )

        ax.text(
            qz_eta2 * 0.98, 0.95,
            r"$\eta = 2$",
            transform=ax.get_xaxis_transform(),
            ha="right", va="top"
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"$Q_z\,(\AA^{-1})$")
    ax.set_ylabel("intensity")

    # ------------------------------------------------------------
    # title: keep old style, but add qxy0
    # ------------------------------------------------------------
    title_lines = []

    if title is not None:
        title_lines.append(title)

    meas = metadata.get("measurements", None)
    if meas is not None:
        scan_val = meas.get("scan", None)
        if scan_val is not None:
            scan_arr = np.asarray(scan_val).ravel()
            if scan_arr.size == 1:
                scan_str = f"{int(scan_arr[0])}"
            else:
                scan_str = f"{int(np.min(scan_arr))} - {int(np.max(scan_arr))}"
            title_lines.append(f"scan Id = {scan_str}, " + rf"$q_{{xy,0}} = {qxy0_val:.3f}\,/\,\AA$")
    else:
        title_lines.append("scan Id = ?, " + rf"$q_{{xy,0}} = {qxy0_val:.3f}\,/\,\AA$")

    param_parts = []
    tension = samp.get("tension", None)
    temperature = samp.get("temperature", None)
    kappa = samp.get("kappa", None)

    if tension is not None:
        param_parts.append(rf"$\gamma={float(tension)*1000:.1f}\,\mathrm{{mN/m}}$")
    if temperature is not None:
        param_parts.append(rf"$T={float(temperature):.1f}\,\mathrm{{K}}$")
    if kappa is not None:
        param_parts.append(rf"$\kappa={float(kappa):.0f}\,k_{{\mathrm{{B}}}}T$")

    if len(param_parts) > 0:
        title_lines.append(", ".join(param_parts))

    ax.set_title("\n".join(title_lines))

    ax.legend(title=legend_title, title_fontsize=10)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(left=0, right=xmax)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax



def GIXOS_RRF_multi_plot(
    GIXOS,
    *,
    metadata=None,
    title=None,
    show=True
):
    """
    Plot |Phi(Qz)|^2 and R/R_F for all selected qxy0 positions in one figure.

    In this combined comparison plot:

    - open circles represent |Phi(Qz)|^2
    - x markers represent R/R_F
    - color indicates qxy0 position

    The legend box contains reflectivity resolution metadata, average sigma_CW,
    marker meaning, and a wrapped color-code block for the selected qxy0 values.

    Parameters
    ----------
    GIXOS : dict
        Multi-qxy0 output dictionary from `GIXOS2R()`. The fields ``"refl"``,
        ``"SF"``, ``"fresnel"``, and ``"sigma_CW"`` must be stacked along axis 0.

    metadata : dict, optional
        Metadata dictionary. If None, ``GIXOS["metadata"]`` is used.

    title : str, optional
        Optional figure title.

    show : bool, optional
        If True, call ``plt.show()``. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.

    ax : matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    ValueError
        If the supplied GIXOS object does not contain multi-qxy0 output.
    """
    
    if metadata is None:
        metadata = GIXOS.get("metadata", None)

    if metadata is None:
        raise ValueError("metadata is required either explicitly or in GIXOS['metadata'].")

    if "PseudoR" not in metadata or metadata["PseudoR"] is None:
        raise ValueError("metadata['PseudoR'] is required.")
    if "sample_params" not in metadata or metadata["sample_params"] is None:
        raise ValueError("metadata['sample_params'] is required.")
    
    pr = metadata["PseudoR"]
    samp = metadata["sample_params"]
        
    qsel = _get_selected_qidx_array(metadata)

    if np.asarray(GIXOS["refl"]).ndim != 3:
        raise ValueError(
            "GIXOS_RRF_multi_plot requires multi-qxy0 GIXOS2R output "
            "(GIXOS['refl'] and GIXOS['SF'] must be 3D)."
        )

    fig, ax = plt.subplots(figsize=(6.0, 5.4))

    # default matplotlib color cycle
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)
    if color_cycle is None or len(color_cycle) == 0:
        color_cycle = [None]

    sigma_vals = []

    for isel, qidx in enumerate(qsel):
        color_i = color_cycle[isel % len(color_cycle)]
        qxy0_val = float(np.asarray(metadata["qxy0"], dtype=float).ravel()[qidx])

        sf = np.asarray(GIXOS["SF"][isel], dtype=float)
        refl = np.asarray(GIXOS["refl"][isel], dtype=float)
        fresnel = np.asarray(GIXOS["fresnel"][isel], dtype=float)
        sigma_cw = np.asarray(GIXOS["sigma_CW"][isel], dtype=float)

        qz_sf = sf[:, 0]
        y_sf = sf[:, 1]
        err_sf = sf[:, 2]

        qz_refl = refl[:, 0]
        y_rrf = refl[:, 1] / fresnel[:, 1]
        err_rrf = refl[:, 2] / fresnel[:, 1]

        # sigma_CW at Qz ~ 0.6 /angstrom
        idx_sig = int(np.argmin(np.abs(refl[:, 0] - 0.6)))
        sigma_vals.append(float(sigma_cw[idx_sig]))

        # SF: open circle
        ax.errorbar(
            qz_sf,
            y_sf,
            yerr=err_sf,
            fmt="o",
            markersize=4,
            linestyle="None",
            markerfacecolor="none",
            color=color_i,
            ecolor=color_i,
            elinewidth=1,
            capsize=2
        )

        # R/RF: x marker
        ax.errorbar(
            qz_refl,
            y_rrf,
            yerr=err_rrf,
            fmt="x",
            markersize=5,
            linestyle="None",
            color=color_i,
            ecolor=color_i,
            elinewidth=1,
            capsize=2
        )

    # eta = 2 line
    if "Qz_eta2" in GIXOS and GIXOS["Qz_eta2"] is not None:
        qz_eta2 = float(GIXOS["Qz_eta2"])
        ax.axvline(qz_eta2, linestyle="--", linewidth=1.0, color="k")
        ax.text(
            qz_eta2 * 0.98, 0.95,
            r"$\eta = 2$",
            transform=ax.get_xaxis_transform(),
            ha="right", va="top"
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"$Q_z\,(\AA^{-1})$")
    ax.set_ylabel("intensity")

    # ------------------------------------------------------------
    # title: keep old style, but add qxy0
    # ------------------------------------------------------------
    title_lines = []

    if title is not None:
        title_lines.append(title)

    meas = metadata.get("measurements", None)
    if meas is not None:
        scan_val = meas.get("scan", None)
        if scan_val is not None:
            scan_arr = np.asarray(scan_val).ravel()
            if scan_arr.size == 1:
                scan_str = f"{int(scan_arr[0])}"
            else:
                scan_str = f"{int(np.min(scan_arr))} - {int(np.max(scan_arr))}"
            title_lines.append(f"scan Id = {scan_str}")
    else:
        title_lines.append("scan Id = ?")

    param_parts = []
    tension = samp.get("tension", None)
    temperature = samp.get("temperature", None)
    kappa = samp.get("kappa", None)

    if tension is not None:
        param_parts.append(rf"$\gamma={float(tension)*1000:.1f}\,\mathrm{{mN/m}}$")
    if temperature is not None:
        param_parts.append(rf"$T={float(temperature):.1f}\,\mathrm{{K}}$")
    if kappa is not None:
        param_parts.append(rf"$\kappa={float(kappa):.0f}\,k_{{\mathrm{{B}}}}T$")

    if len(param_parts) > 0:
        title_lines.append(", ".join(param_parts))

    ax.set_title("\n".join(title_lines))

    # ------------------------------------------------------------
    # one combined boxed legend
    # ------------------------------------------------------------
    sigma_avg = float(np.mean(sigma_vals)) if len(sigma_vals) > 0 else np.nan
    
    legend_box = _make_rrf_multi_legend_box(
        metadata,
        qsel,
        sigma_avg,
        max_entries_per_line=2
    )
    ax.add_artist(legend_box)

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(left=0, right=xmax)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def GIXOS_R_plot(
    GIXOS,
    *,
    metadata=None,
    selected_pos=0,
    title=None,
    show=True
):
    """
    Plot reflectivity-style quantities for one selected qxy0 track.

    This function visualizes, for one selected qxy0 position:

    - |Phi(Qz)|^2 * R_F
    - pseudo-reflectivity R(Qz)
    - Psi_R(Qz) * R_F

    The figure is shown on a semi-log scale (log intensity, linear Qz).

    Parameters
    ----------
    GIXOS : dict
        Output dictionary from `GIXOS2R()`.

    metadata : dict, optional
        Metadata dictionary. If None, ``GIXOS["metadata"]`` is used.

    selected_pos : int, optional
        Position within ``metadata["PseudoR"]["qxy0_select_idx"]`` specifying
        which selected qxy0 track to plot. Default is 0.

    title : str, optional
        Optional user-supplied extra title line. If None, only the standard
        title lines are shown.

    show : bool, optional
        If True, call ``plt.show()``. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.

    ax : matplotlib.axes.Axes
        Axes containing the plot.

    Notes
    -----
    This function supports both:
    - single-qxy0 output from `GIXOS2R()`
    - multi-qxy0 output stacked along axis 0
    """
    
    if metadata is None:
        metadata = GIXOS.get("metadata", None)

    if metadata is None:
        raise ValueError("metadata is required either explicitly or in GIXOS['metadata'].")

    if "PseudoR" not in metadata or metadata["PseudoR"] is None:
        raise ValueError("metadata['PseudoR'] is required.")
    if "sample_params" not in metadata or metadata["sample_params"] is None:
        raise ValueError("metadata['sample_params'] is required.")

    pr = metadata["PseudoR"]
    samp = metadata["sample_params"]

    qidx_raw, qxy0_val = _get_qxy0_value(metadata, selected_pos)

    # ------------------------------------------------------------
    # select one output track
    # ------------------------------------------------------------
    if np.asarray(GIXOS["refl"]).ndim == 3:
        sf = np.asarray(GIXOS["SF"][selected_pos], dtype=float)
        refl = np.asarray(GIXOS["refl"][selected_pos], dtype=float)
        fresnel = np.asarray(GIXOS["fresnel"][selected_pos], dtype=float)
        psi_r = np.asarray(GIXOS["Psi_R"][selected_pos], dtype=float)
    else:
        sf = np.asarray(GIXOS["SF"], dtype=float)
        refl = np.asarray(GIXOS["refl"], dtype=float)
        fresnel = np.asarray(GIXOS["fresnel"], dtype=float)
        psi_r = np.asarray(GIXOS["Psi_R"], dtype=float)

    # ------------------------------------------------------------
    # data
    # ------------------------------------------------------------
    qz_sf = np.asarray(sf[:, 0], dtype=float)
    y_sf_rf = np.asarray(sf[:, 1], dtype=float) * np.asarray(fresnel[:, 1], dtype=float)
    err_sf_rf = np.asarray(sf[:, 2], dtype=float) * np.asarray(fresnel[:, 1], dtype=float)

    qz_refl = np.asarray(refl[:, 0], dtype=float)
    y_refl = np.asarray(refl[:, 1], dtype=float)
    err_refl = np.asarray(refl[:, 2], dtype=float)

    y_psi_rf = np.asarray(psi_r, dtype=float) * np.asarray(fresnel[:, 1], dtype=float)

    # ------------------------------------------------------------
    # legend title from resolution metadata
    # keep old style, but remove deltaQ and sigma_CW
    # ------------------------------------------------------------
    legend_title = _format_rrf_legend_metadata(metadata)["legend_title_single"]

    # ------------------------------------------------------------
    # plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 4.8))

    # 1) SF * RF
    ax.errorbar(
        qz_sf, y_sf_rf, yerr=err_sf_rf,
        fmt="o", markersize=4,
        color="k", ecolor="k",
        elinewidth=1, capsize=2,
        label=r"$|\Phi(Q_z)|^2 R_F$",
        zorder=2
    )

    # 2) R
    ax.errorbar(
        qz_refl, y_refl, yerr=err_refl,
        fmt="o", markersize=4,
        color="b", ecolor="b",
        elinewidth=1, capsize=2,
        label=r"$R$",
        zorder=3
    )

    # 3) Psi_R * RF
    ax.plot(
        qz_refl, y_psi_rf,
        linestyle="-", linewidth=1.8,
        color="b",
        label=r"$\Psi_R R_F$",
        zorder=1
    )

    # eta = 2 line and label
    if "Qz_eta2" in GIXOS and GIXOS["Qz_eta2"] is not None:
        qz_eta2 = float(GIXOS["Qz_eta2"])

        ax.axvline(
            qz_eta2,
            linestyle="--",
            color="k",
            linewidth=1.2
        )

        ax.text(
            qz_eta2 * 0.98, 0.95,
            r"$\eta = 2$",
            transform=ax.get_xaxis_transform(),
            ha="right", va="top"
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"$Q_z\,(\AA^{-1})$")
    ax.set_ylabel("intensity")

    # ------------------------------------------------------------
    # title: keep old style, but add qxy0
    # ------------------------------------------------------------
    title_lines = []

    if title is not None:
        title_lines.append(title)
    
    meas = metadata.get("measurements", None)
    if meas is not None:
        scan_val = meas.get("scan", None)
        if scan_val is not None:
            scan_arr = np.asarray(scan_val).ravel()
            if scan_arr.size == 1:
                scan_str = f"{int(scan_arr[0])}"
            else:
                scan_str = f"{int(np.min(scan_arr))} - {int(np.max(scan_arr))}"
            title_lines.append(f"scan Id = {scan_str}, " + rf"$q_{{xy,0}} = {qxy0_val:.3f}\,/\,\AA$")
    else:
        title_lines.append("scan Id = ?, " + rf"$q_{{xy,0}} = {qxy0_val:.3f}\,/\,\AA$")

    param_parts = []
    tension = samp.get("tension", None)
    temperature = samp.get("temperature", None)
    kappa = samp.get("kappa", None)

    if tension is not None:
        param_parts.append(rf"$\gamma={float(tension)*1000:.1f}\,\mathrm{{mN/m}}$")
    if temperature is not None:
        param_parts.append(rf"$T={float(temperature):.1f}\,\mathrm{{K}}$")
    if kappa is not None:
        param_parts.append(rf"$\kappa={float(kappa):.0f}\,k_{{\mathrm{{B}}}}T$")

    if len(param_parts) > 0:
        title_lines.append(", ".join(param_parts))

    ax.set_title("\n".join(title_lines))

    ax.legend(title=legend_title, title_fontsize=10)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(left=0, right=xmax)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


