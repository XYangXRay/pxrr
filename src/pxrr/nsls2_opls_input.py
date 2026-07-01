import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import os
import time
from pathlib import Path
from PIL import Image
from ruamel.yaml import YAML
from pxrr.preprocess import *
from pxrr.data_io import save_metadata_yaml as _save_metadata_yaml


# Packaged default metadata template, reused from the shipped example config.
DEFAULT_CONFIG_YAML = (
    Path(__file__).resolve().parents[2]
    / "example"
    / "opls_full"
    / "gixos-process_config_1d.yaml"
)


def _load_base_metadata(metadata_input=None):
    """Load a base metadata template.

    Parameters
    ----------
    metadata_input : str or Path, optional
        Path to a YAML template. If ``None``, the packaged default
        (``DEFAULT_CONFIG_YAML``) is used.

    Returns
    -------
    dict
        Parsed metadata dictionary to be used as a base template.
    """
    src = metadata_input if metadata_input is not None else DEFAULT_CONFIG_YAML
    yaml = YAML(typ="safe")
    with open(src, "r") as f:
        return yaml.load(f)


class NSLS2OPLSInput:
    """Single object to process GIXOS data from NSLS-II OPLS beamline."""

    def __init__(
        self,
        path,
        sdd,
        pxsize,
        bad_pixel,
        roi_y,
        roi_x,
        roi_dy,
    ):
        self.path = path
        self.sdd = sdd
        self.pxsize = pxsize
        self.bad_pixel = bad_pixel
        self.roi_y = roi_y
        self.roi_x = roi_x
        self.roi_dy = roi_dy
        self.solidangle_ref = pxsize * (roi_dy * 2 + 1) * pxsize * 1 / sdd**2
        self.mg_ais = []
        self.imgs = []
        self.result_lst = []
        self.monitor_lst = []
        self.expo_time_lst = []
        self.sample_id_set = None
        self.scan_sample_name_map = {}

    @staticmethod
    def _sanitize_sample_name(name):
        """Return a filesystem-safe sample label."""
        name = str(name).strip()
        if not name:
            return "instrument"
        safe = []
        for ch in name:
            if ch.isalnum() or ch in ("-", "_"):
                safe.append(ch)
            elif ch in (" ", "."):
                safe.append("_")
            else:
                safe.append("_")
        label = "".join(safe).strip("_")
        return label or "instrument"

    def _get_sample_name_from_scan_id(self, sample_id, run=None):
        """Resolve sample_name for a scan_id from tiled metadata."""
        # 1) tiled metadata from the provided run (preferred)
        if run is not None:
            try:
                start = run.metadata.get("start", {})
                name = start.get("sample_name")
                if name is not None and str(name).strip() != "":
                    return str(name)
            except Exception:
                pass

        # 2) tiled lookup by scan_id
        try:
            start = get_run(int(sample_id)).metadata.get("start", {})
            name = start.get("sample_name")
            if name is not None and str(name).strip() != "":
                return str(name)
        except Exception:
            pass

        return "instrument"

    def search_scans(self, since, plan_name="gid_soller"):
        from tiled.queries import Key
        results = c.search(Key("plan_name") == plan_name)
        for uid, run in results.items():
            start = run.metadata["start"]
            if start.get("time", 0) >= np.datetime64(since).astype("datetime64[s]").astype("float64"):
                print(f"scan_id = {start['scan_id']}, sample_name = {start['sample_name']}")

    def load_data(self, sample_id_set):
        self.sample_id_set = np.asarray(sample_id_set)
        self.mg_ais = []
        self.imgs = []
        self.result_lst = []
        self.monitor_lst = []
        self.expo_time_lst = []
        self.scan_sample_name_map = {}

        print(f"loading scans: {self.sample_id_set}")

        for sample_id in self.sample_id_set:
            sample_id = int(sample_id)
            run = get_run(sample_id)
            sample_name = self._get_sample_name_from_scan_id(sample_id, run=run)
            primary_data = run["primary"]["data"]
            h_sample_monitor = np.mean(
                np.array(primary_data["monitor_3"])
            )
            h_sample_expo_time = np.sum(
                np.array(primary_data["expo_time"])
            )
            print(f"scan {sample_id}: monitor_3 average = {h_sample_monitor}")
            result, ai_lst = loadgixos_ai(
                sample_id,
                mode="sum",
                roi_y=self.roi_y,
                roi_dy=self.roi_dy,
                roi_x=self.roi_x,
                pxsize=self.pxsize,
                sdd=self.sdd,
            )
            self.mg_ais = self.mg_ais + ai_lst
            self.imgs = self.imgs + result["gixos_1d"]
            self.monitor_lst.append(h_sample_monitor)
            self.expo_time_lst.append(h_sample_expo_time)
            self.scan_sample_name_map[sample_id] = sample_name
            # geometry
            result["height"] = np.mean(np.array(primary_data["geo_sh"]))
            result["sample_name"] = sample_name
            result["px_beta"] = (
                np.rad2deg(
                    np.arctan(
                        (np.arange((result["gixos_1d"][0].shape)[1]) - self.roi_x)
                        * self.pxsize
                        / self.sdd
                    )
                )
                + result["beta"]
            )
            result["px_qz"] = (
                (
                    np.sin(np.deg2rad(result["alpha"]))
                    + np.sin(np.deg2rad(result["px_beta"]))
                )
                * 2
                * np.pi
                / result["wavelength"]
            )
            print(
                f"scan {result['id']}: qxy0 = {result['qxy']} /A, "
                f"sample height = {result['height']}"
            )
            self.result_lst.append(result)
            del result, ai_lst

    def plot(self, savefig=True):
        fig, ax = plt.subplots(figsize=[10, 5])
        xmin, xmax = 0.005, 0.9
        ymins, ymaxs = [], []
        for idx in range(len(self.imgs)):
            qz = self.result_lst[idx]["px_qz"]
            intensity = (
                self.imgs[idx][0]
                / self.monitor_lst[idx]
                * np.mean(self.monitor_lst)
            )
            ax.plot(
                qz,
                intensity,
                "-",
                label="%d" % self.result_lst[idx]["id"],
            )
            # collect positive intensities within the plotted x-range for ylim
            in_range = (qz >= xmin) & (qz <= xmax) & (intensity > 0)
            vals = intensity[in_range]
            if vals.size:
                ymins.append(np.min(vals))
                ymaxs.append(np.max(vals))
        ax.set_yscale("log")
        ax.set_xlim([xmin, xmax])
        if ymins and ymaxs:
            ax.set_ylim([0.5 * min(ymins), 2.0 * max(ymaxs)])
        ax.set_ylabel(r"Intensity")
        ax.set_xlabel(r"$q_{z}$ (${\rm \AA}^{-1}$)")
        ax.legend(loc="upper right")
        ax.axvline(x=0.0217, ls="--", c="k")
        if savefig and self.sample_id_set is not None:
            out_dir = os.path.join(self.path, "gixos")
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(
                os.path.join(
                    out_dir,
                    str(int(self.sample_id_set[0]))
                    + "_"
                    + str(int(self.sample_id_set[-1]))
                    + ".png",
                ),
                dpi=300,
            )
        return fig, ax

    def save_gixos_1d(self):
        print(f"saving 1D GIXOS .txt files under: {self.path}")
        os.makedirs(os.path.join(self.path, "gixos"), exist_ok=True)
        os.makedirs(os.path.join(self.path, "gixos2"), exist_ok=True)
        # Old method
        for idx in range(len(self.imgs)):
            fileheader = (
                "header\nenergy = %.1f eV\nsdd = %f m\nalpha = %f deg\n"
                "beta = %f deg\ntth = %f deg\nqxy0 = %f /A\nexpo_time = %d sec\n"
                "monitor_3_ave = %f /sec\npx_size = %f m\nroi_x = %d\nroi_y = %d\n"
                "roi_dy = %d\nchamber_id = %d\ndata\nidx\tbeta\tintensity\tqz"
            ) % (
                self.result_lst[idx]["energy"],
                self.sdd,
                self.result_lst[idx]["alpha"],
                self.result_lst[idx]["beta"],
                self.result_lst[idx]["tth"],
                self.result_lst[idx]["qxy"],
                self.expo_time_lst[idx],
                np.mean(self.monitor_lst),
                self.pxsize,
                self.roi_x,
                self.roi_y,
                self.roi_dy,
                self.result_lst[idx]["bkg_id"],
            )
            np.savetxt(
                self.path
                + "gixos/"
                + self._sanitize_sample_name(
                    self.result_lst[idx].get("sample_name", "instrument")
                )
                + "-id"
                + str(self.result_lst[idx]["id"])
                + ".txt",
                np.column_stack(
                    (
                        np.arange(len(self.result_lst[idx]["px_beta"])),
                        self.result_lst[idx]["px_beta"],
                        self.imgs[idx][0]
                        / self.monitor_lst[idx]
                        * np.mean(self.monitor_lst),
                        self.result_lst[idx]["px_qz"],
                    )
                ),
                fmt="%f",
                header=fileheader,
            )

        # New method, Pandas compatible for multiplotting
        for idx in range(len(self.imgs)):
            fileheader = (
                "{0}header\n{0}energy = %.1f eV\n{0}sdd = %f m\n{0}alpha = %f deg\n"
                "{0}beta = %f deg\n{0}tth = %f deg\n{0}qxy0 = %f /A\n"
                "{0}expo_time = %d sec\n{0}monitor_3_ave = %f /sec\n"
                "{0}px_size = %f m\n{0}roi_x = %d\n{0}roi_y = %d\n"
                "{0}roi_dy = %d\n{0}chamber_id = %d\n{0}data\n"
                "idx\tbeta\tintensity\tqz"
            ).format("# ") % (
                self.result_lst[idx]["energy"],
                self.sdd,
                self.result_lst[idx]["alpha"],
                self.result_lst[idx]["beta"],
                self.result_lst[idx]["tth"],
                self.result_lst[idx]["qxy"],
                self.expo_time_lst[idx],
                np.mean(self.monitor_lst),
                self.pxsize,
                self.roi_x,
                self.roi_y,
                self.roi_dy,
                self.result_lst[idx]["bkg_id"],
            )
            outpath = (
                self.path
                + "gixos2/"
                + self._sanitize_sample_name(
                    self.result_lst[idx].get("sample_name", "instrument")
                )
                + "-id"
                + str(self.result_lst[idx]["id"])
                + ".txt"
            )
            np.savetxt(
                outpath,
                np.column_stack(
                    (
                        np.arange(len(self.result_lst[idx]["px_beta"])),
                        self.result_lst[idx]["px_beta"],
                        self.imgs[idx][0]
                        / self.monitor_lst[idx]
                        * np.mean(self.monitor_lst),
                        self.result_lst[idx]["px_qz"],
                    )
                ),
                fmt="%f",
                header=fileheader,
                comments="",
                delimiter="\t",
            )
            print(f"saved GIXOS 1D: {outpath}")

    def save_metadata_yaml(
        self,
        yaml_path,
        sample_scans,
        bkg_scans,
        sample_name=None,
        bkgsample_name=None,
        gixs_path=None,
        path_out=None,
        qxy0=None,
        metadata_input=None,
        **overrides,
    ):
        """Write a YAML config compatible with ``load_gixos_from_meta``.

        Parameters
        ----------
        yaml_path : str
            Output YAML file path.
        sample_scans, bkg_scans : sequence of int
            Sample scan IDs and matching chamber-background scan IDs.
            Must have the same length; entries pair element-wise.
        sample_name : str, optional
            File prefix used by ``save_gixos_1d`` (``<name>-id<scanid>.txt``).
            If ``None``, infer from tiled ``sample_name`` metadata of
            ``sample_scans``. If mixed names are found, use the first one.
        bkgsample_name : str, optional
            Defaults to ``sample_name``.
        gixs_path, path_out : str, optional
            Override the data and output directories. Default to
            ``<self.path>/gixos2/`` and ``<self.path>/output/``.
        qxy0 : sequence of float, optional
            Per-scan qxy0 in 1/A. If None, read from ``result_lst`` entries
            matching ``sample_scans``.
        metadata_input : str, optional
            Path to a YAML template used as the base config. Scan-specific
            and computed fields (scans, sample names, energy, alpha, qxy0,
            paths, ...) are overwritten on top of it. If ``None``, the
            packaged default (``DEFAULT_CONFIG_YAML``) is used.
        **overrides
            Nested dict overrides (e.g. ``PseudoR={"qxy0_select_idx": [0, 1]}``)
            merged into the generated YAML.

        Returns
        -------
        str
            ``yaml_path``.
        """
        sample_scans = [int(s) for s in sample_scans]
        bkg_scans = [int(s) for s in bkg_scans]
        if len(sample_scans) != len(bkg_scans):
            raise ValueError("sample_scans and bkg_scans must have equal length")
        if sample_name is None:
            inferred = [
                self.scan_sample_name_map.get(s)
                for s in sample_scans
                if self.scan_sample_name_map.get(s)
            ]
            if inferred:
                sample_name = inferred[0]
            else:
                sample_name = "instrument"
        if bkgsample_name is None:
            bkgsample_name = f"{sample_name}_bkg"

        sample_name = self._sanitize_sample_name(sample_name)
        bkgsample_name = self._sanitize_sample_name(bkgsample_name)

        # look up per-scan results
        by_id = {int(r["id"]): r for r in self.result_lst}
        missing = [s for s in sample_scans if s not in by_id]
        if missing:
            raise ValueError(
                f"sample scans not loaded: {missing}. Run load_data() with these IDs first."
            )

        if qxy0 is None:
            qxy0 = [float(by_id[s]["qxy"]) for s in sample_scans]
        qxy0 = [float(v) for v in qxy0]

        ref = by_id[sample_scans[0]]
        energy = float(ref["energy"])
        alpha = float(np.mean(np.atleast_1d(ref["alpha"])))
        wavelength = 12404.0 / energy
        # HWtth from pixel half-width tangent in degrees
        HWtth = float(np.degrees(np.arctan(self.pxsize / 2.0 / self.sdd)))

        if gixs_path is None:
            gixs_path = os.path.join(self.path, "gixos2") + os.sep
        if path_out is None:
            path_out = os.path.join(self.path, "output") + os.sep

        # start from a base template (user-provided or packaged default) and
        # overwrite only the scan-specific / computed fields
        meta = _load_base_metadata(metadata_input)
        computed = {
            "paths": {
                "gixs_path": gixs_path,
                "path_out": path_out,
            },
            "measurements": {
                "sample": sample_name,
                "scan": sample_scans,
                "bkgsample": bkgsample_name,
                "bkgscan": bkg_scans,
                "flux": float(np.mean(self.monitor_lst)) if self.monitor_lst else 1.0,
            },
            "instrument": {
                "energy": energy,
                "alpha": alpha,
                "Ddet": self.sdd * 1000.0,
                "pixel": self.pxsize * 1000.0,
                "HWtth": HWtth,
            },
            "qxy0": qxy0,
            "PseudoR": {
                "qxy0_select_idx": [0, 1] if len(qxy0) >= 2 else [0],
                "energy": energy,
            },
        }

        # merge overrides
        def _merge(dst, src):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    _merge(dst[k], v)
                else:
                    dst[k] = v

        _merge(meta, computed)
        _merge(meta, overrides)

        os.makedirs(os.path.dirname(os.path.abspath(yaml_path)) or ".", exist_ok=True)
        _save_metadata_yaml(meta, yaml_path)
        print(f"config written: {yaml_path}")
        return yaml_path


def process_opls(
    start_scan,
    n_scan,
    path,
    *,
    sdd=680 / 1000,
    pxsize=172e-6,
    bad_pixel=(100, 269),
    roi_y=195 - 111,
    roi_x=41 + 9,
    roi_dy=3,
    metadata_input=None,
    metadata_output=None,
    gixs_path=None,
    path_out=None,
    sample_name=None,
    bkgsample_name=None,
    tension=0.072,
    kappa=5,
    plot=True,
    save_gixos=True,
    **yaml_overrides,
):
    """Run OPLS preprocessing for a contiguous block of scans.

    Pairs the scans as alternating sample/chamber-bkg (even index = sample,
    odd index = background), loads them, optionally plots, saves the 1D
    GIXOS .txt files, and writes a YAML config compatible with
    ``load_gixos_from_meta``.

    Parameters
    ----------
    start_scan : int
        First scan ID.
    n_scan : int
        Total number of scans (sample + bkg combined). Must be even.
    path : str
        Working directory (passed to ``NSLS2OPLSInput``); also default for
        ``gixs_path`` / ``path_out``.
    sdd, pxsize, bad_pixel, roi_y, roi_x, roi_dy
        Detector geometry / ROI parameters.
    metadata_input : str, optional
        Path to a YAML template used as the base config. If provided it is
        filled in with the scan-specific / computed values; otherwise the
        packaged default (``DEFAULT_CONFIG_YAML``) is used.
    metadata_output : str, optional
        Output *directory* for the generated YAML config. The filename is
        always auto-generated as ``gixos-process_config_1d_<first_scan>.yaml``.
        If ``None``, the data output directory (``path_out``) is used.
        Parent directories are created if needed.
    gixs_path, path_out : str, optional
        Overrides for the corresponding YAML fields. ``path`` is the raw
        data working directory; ``path_out`` is the processed-output
        directory (also the default location for ``metadata_output``).
    sample_name, bkgsample_name : str, optional
        Values written to ``metadata['measurements']['sample']`` and
        ``metadata['measurements']['bkgsample']``. If ``sample_name`` is
        ``None``, infer from tiled ``sample_name`` by ``scan_id``.
    tension : float
        Surface tension [N/m], written into ``sample_params.tension``.
    kappa : float
        Bending modulus [kbT], written into ``sample_params.kappa``.
    plot : bool
        Show / save the qz overview plot.
    save_gixos : bool
        Write per-scan 1D GIXOS .txt files.
    **yaml_overrides
        Additional nested dict overrides forwarded to
        ``NSLS2OPLSInput.save_metadata_yaml``.

    Returns
    -------
    opls : NSLS2OPLSInput
        The loaded preprocessor instance.
    metadata_file : str
        Path to the written YAML config.
    """
    if n_scan % 2 != 0:
        raise ValueError("n_scan must be even (alternating sample / bkg pairs)")

    ids = np.arange(int(start_scan), int(start_scan) + int(n_scan))
    sample_scans = ids[0::2].tolist()
    bkg_scans = ids[1::2].tolist()

    opls = NSLS2OPLSInput(
        path=path,
        sdd=sdd,
        pxsize=pxsize,
        bad_pixel=bad_pixel,
        roi_y=roi_y,
        roi_x=roi_x,
        roi_dy=roi_dy,
    )
    opls.load_data(sample_id_set=ids)
    if plot:
        opls.plot()
    if save_gixos:
        opls.save_gixos_1d()

    if gixs_path is None:
        gixs_path = os.path.join(path, "gixos2") + os.sep
    if path_out is None:
        path_out = os.path.join("./testing_data/output/opls_full") + os.sep

    sp_override = yaml_overrides.pop("sample_params", {}) or {}
    sp_override.setdefault("tension", float(tension))
    sp_override.setdefault("kappa", float(kappa))

    # metadata_output is a directory; the filename is always auto-generated
    metadata_dir = path_out if metadata_output is None else metadata_output
    metadata_output = os.path.join(
        metadata_dir, f"gixos-process_config_1d_{int(ids[0])}.yaml"
    )
    metadata_file = opls.save_metadata_yaml(
        yaml_path=metadata_output,
        sample_scans=sample_scans,
        bkg_scans=bkg_scans,
        sample_name=sample_name,
        bkgsample_name=bkgsample_name,
        gixs_path=gixs_path,
        path_out=path_out,
        metadata_input=metadata_input,
        sample_params=sp_override,
        **yaml_overrides,
    )
    return opls, metadata_file
