# from data_io import load_metadata, etc., etc. maybe
import matplotlib as plt
import os
import numpy as np

def GIXOS_data_plot(GIXOS, metadata):
    plt.figure(figsize = (10, 6))
    plt.plot(GIXOS["Qz"], GIXOS["GIXOS_raw"] - GIXOS["GIXOS_bkg"], 'ko', markersize=3, label='corrected data by chamber transmission bkg')
    plt.plot(GIXOS["Qz"], GIXOS["GIXOS"], 'go', markersize=3, label='with large Qz subtraction')
    plt.plot(GIXOS["Qz"], GIXOS["GIXOS_raw"], 'r-', linewidth=1.5, label='raw data')
    plt.plot(GIXOS["Qz"], GIXOS["GIXOS_bkg"], 'b-', linewidth=1.5, label='bkg data at same tth')
    plt.plot(GIXOS["Qz"], GIXOS["raw_largetth"], 'r:', linewidth=1.5, label='raw data wide angle')
    plt.plot(GIXOS["Qz"], GIXOS["bkg_largetth"], 'b:', linewidth=1.5, label='bkg data wide angle')
    plt.axhline(0, color='k', linestyle='-.', linewidth=1.5, label='0-line')

    # --- Show legend and display
    plt.legend()
    plt.xlabel("Qz")
    plt.ylabel("Intensity")
    plt.title("GIXOS Data")
    plt.grid(True)
    plt.tight_layout()

    plt.xlim([0, 1.05])
    ax = plt.gca()

    ax.set_xlabel(r'$Q_z$ [$\mathrm{\AA}^{-1}$]', fontsize=12)  # Qz with angstrom symbol
    ax.set_ylabel('GIXOS', fontsize=12)

    ax.set_xticks(np.arange(0, 1.01, 0.2))  # same as 0:0.2:1
    ax.tick_params(axis='both', labelsize=12, width=2, direction='out')

    plt.legend(loc='upper right', frameon=False)  # 'NorthEast' => 'upper right'

    # --- Save figure
    filename = f"{metadata['sample']}_{metadata['qxy0_select_idx']:05d}_GIXOS.jpg"
    save_path = os.path.join(metadata["path_out"], filename)
    plt.savefig(save_path, dpi=300)
    plt.show()



def R_data_plot(GIXOS, metadata, xrr_config): # want to have xrr_config be a dict that the xrr_config function will return w/ vars we need
    GIXOS["refl_recSlit"] = np.array(GIXOS["refl_recSlit"])

    # Close any existing figure named 'refl'
    plt.close('refl')

    # Create new figure
    fig, ax = plt.subplots(num='refl', figsize=(6, 5))
    fig.canvas.manager.set_window_title('refl')

    # Plot error bars
    # Commented line corresponds to the MATLAB commented errorbar line
    # ax.errorbar(GIXOS['refl'][:,0], GIXOS['refl'][:,1]/RFscaling, yerr=GIXOS['refl'][:,2]/RFscaling,
    #             fmt='o', markersize=3,
    #             label=f"δQ_xy= {geo['RqxyHW']:.1e} Å⁻¹")

    ax.errorbar(
        GIXOS["refl_recSlit"][:, 0],
        GIXOS["refl_recSlit"][:, 1] / metadata["RFscaling"],
        yerr=GIXOS["refl_recSlit"][:, 2] / metadata["RFscaling"],
        fmt='o',
        markersize=5,
        label=(
            f"slit: {xrr_config["slit_v"]:.2f} mm (v) x {xrr_config["slit_h"]:.2f} mm (h); "
            f"{xrr_config["sdd"]:.1f} mm, {xrr_config["energy"] / 1000:.1f} keV"
        )
    )

    ax.set_xlim(0, 0.8)
    ax.set_xlabel(r'$Q_z \; [\AA^{-1}]$', fontsize=14)
    ax.set_ylabel('R', fontsize=14)

    ax.tick_params(axis='both', which='major', labelsize=14, direction='out')
    ax.set_xticks(np.arange(0, 0.81, 0.2))
    ax.set_yscale('log')
    ax.legend(loc='lower right', frameon=False)  # 'Southeast' → 'lower right' in matplotlib

    plt.tight_layout()

    # Save figure
    filename = f"{metadata['path_out']}{metadata['sample']}_{metadata['scan'][ metadata['qxy0_select_idx'] ]:05d}_R_PYTHON.jpg"
    plt.savefig(filename, dpi=300)
    plt.show()






def R_pseudo_data_plot(GIXOS, metadata, xrr_config):
    GIXOS["refl_recSlit"] = np.array(GIXOS["refl_recSlit"])
    GIXOS["fresnel"] = np.array(GIXOS["fresnel"])
    xrr_data_data = np.array(metadata["xrr_data"])   # maybe add xrr_data to xrr_config dict

    # Close existing figure named 'RRF'
    plt.close('RRF')

    fig, ax = plt.subplots(num='RRF', figsize=(6, 5))
    fig.canvas.manager.set_window_title('RRF')

    # Plot first errorbar: refl_recSlit normalized by fresnel and RFscaling
    ax.errorbar(
        GIXOS["refl_recSlit"][:, 0],
        GIXOS["refl_recSlit"][:, 1] / GIXOS["fresnel"][:, 1] / metadata["RFscaling"],
        yerr=GIXOS["refl_recSlit"][:, 2] / GIXOS["fresnel"][:, 1] / metadata["RFscaling"],
        fmt='ro',
        markersize=5,
        linewidth=1.5,
        label=(
            f"slit: {xrr_config["slit_v"]:.2f} mm (v) x {xrr_config["slit_h"]:.2f} mm (h); "
            f"{xrr_config["sdd"]:.1f} mm, {xrr_config["energy"] / 1000:.1f} keV"
        )
    )

    # Plot second errorbar: xrr_data normalized by dataRF
    ax.errorbar(
        xrr_data_data[:, 0],
        xrr_data_data[:, 1] / xrr_config["dataRF"],
        yerr=xrr_data_data[:, 2] / xrr_config["dataRF"],
        fmt='ko',
        markersize=5,
        linewidth=1.5,
        capsize=4,
        label='XRR 0.66mm slit'
    )

    ax.set_xlim(0, 0.8)
    ax.set_ylim(1e-5, 10)
    ax.set_xlabel(r'$Q_z \; [\AA^{-1}]$', fontsize=14)
    ax.set_ylabel('R/R_F [a.u.]', fontsize=14)

    ax.tick_params(axis='both', which='major', labelsize=14, direction='out')
    ax.set_xticks(np.arange(0, 0.81, 0.2))
    ax.set_yscale('log')
    ax.legend(loc='lower left', frameon=False)  # MATLAB 'SouthWest' ≈ matplotlib 'lower left'

    plt.tight_layout()

    filename = f"{metadata['path_out']}{metadata['sample']}_{metadata['scan'][ metadata['qxy0_select_idx'] ]:05d}_RRF_PYTHON.jpg"
    plt.savefig(filename, dpi=300)
    plt.show()