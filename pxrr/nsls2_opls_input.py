import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import os
import time
from PIL import Image
from pxrr.preprocess import *


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

    def search_scans(self, since, plan_name="gid_soller"):
        from tiled.queries import Key
        results = c.search(Key("plan_name") == plan_name)
        for uid, run in results.items():
            start = run.metadata["start"]
            if start.get("time", 0) >= np.datetime64(since).astype("datetime64[s]").astype("float64"):
                print(start["scan_id"], start["sample_name"])

    def load_data(self, sample_id_set):
        self.sample_id_set = np.asarray(sample_id_set)
        self.mg_ais = []
        self.imgs = []
        self.result_lst = []
        self.monitor_lst = []
        self.expo_time_lst = []

        print(self.sample_id_set)

        for sample_id in self.sample_id_set:
            print(self.roi_y, self.roi_dy)
            sample_id = int(sample_id)
            run = lookup_run(sample_id)
            primary_data = run["primary"]["data"]
            h_sample_monitor = np.mean(
                np.array(primary_data["monitor_3"])
            )
            h_sample_expo_time = np.sum(
                np.array(primary_data["expo_time"])
            )
            print(h_sample_monitor)
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
            # geometry
            result["height"] = np.mean(np.array(primary_data["geo_sh"]))
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
            print("id = ", result["id"])
            print("qxy0 = ", result["qxy"])
            print("sample height = ", result["height"])
            self.result_lst.append(result)
            del result, ai_lst

    def plot(self, savefig=True):
        fig, ax = plt.subplots(figsize=[10, 5])
        for idx in range(len(self.imgs)):
            ax.plot(
                self.result_lst[idx]["px_qz"],
                self.imgs[idx][0]
                / self.monitor_lst[idx]
                * np.mean(self.monitor_lst),
                "-",
                label="%d" % self.result_lst[idx]["id"],
            )
        ax.set_yscale("log")
        ax.set_xlim([0.005, 0.9])
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
        print(self.path)
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
                + "gixos/instrument-id"
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
        print(self.path)
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
                + "gixos2/instrument-id"
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
            print(outpath)
