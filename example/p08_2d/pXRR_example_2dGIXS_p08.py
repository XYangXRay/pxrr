# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 16:39:16 2025

@author: shenc

example routine for P08 data (2D GIXS image (angular rebinned from detector image))
- a series of 1D GIXOS linecuts needs to be extracted and this is executed during data loading
according to the datatype lable in the yaml file
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
from pseudo_xrr.eCWM import *
from pseudo_xrr.data_io import *
from pseudo_xrr.GIXOS import *

#%% directly load data from meta and GIXOS will be automatically extracted:
# alternatively, load_data, geometrical correction, extract_1dGIXOS, and provide metadata into this field
GIXOSdata, GIXOSbkg = load_gixos_from_meta('./gixos-process_config_2d.yaml') 

#%% info
# from here all meta has entered GIXOSdata and GIXOSbkg in ["metadata"] field
# yaml file is no longer needed
#%% corrections
# binning in tt
GIXOSdata= binning_GIXOS_tt(GIXOSdata)
GIXOSbkg= binning_GIXOS_tt(GIXOSbkg)
#% remove negative 2theta
GIXOSdata= remove_negative_2theta(GIXOSdata)
GIXOSbkg= remove_negative_2theta(GIXOSbkg)
#% 2theta to q
GIXOSdata_q = GIXOS_th2q(GIXOSdata)
GIXOSbkg_q = GIXOS_th2q(GIXOSbkg)
# background subtraction
GIXOS_ana = GIXOS_background_corr(GIXOSdata_q, GIXOSbkg_q, bulkbkg_mode = 2, bulkbkg_offset_lb=0.02, plot = True)

#%% export corrected GIXOS as h5 file. 
# This stores all the info + data and can be loaded back to continue
outgixosfile = make_filename(GIXOS_ana["metadata"], suffix="gixos.h5")
export_gixos_nxs(GIXOS_ana, outgixosfile)

#%% load back GIXOS that can be used to continue
GIXOS_back = load_gixos_nxs(outgixosfile)

#%% info
# we still use GIXOS_ana insteaded of the loaded back one. The GIXOS_back is only for demo
# from here on the operation will directly add results into the original dictionary variable (shared memory)

#%% qxy dependence
_, qxy_dependence_fit = GIXOS_qxy_dependence(GIXOS_ana, GIXOS_ana['metadata']['dependency']['qz_selected'], fit_kappa = True)

#%% processing pseudo
_ = GIXOS2R(GIXOS_ana, transmission_corr = True, footprint_effect=True, use_approx=True)

#%% ---- export configuration ----
# this gives back the exact yaml file structure but with newly generated values from analysis
# particularly include the fitted bending modulus
configfilename = make_filename(GIXOS_ana["metadata"], suffix="cfg.yaml")
save_metadata_yaml(GIXOS_ana["metadata"], configfilename)

#%% ------export orso -----------------
# at P08 we can fetch metadata for proposal and instrument from these files
jsonfilename = "../testing_data/p08_DPPC_data/beamtime-metadata-11024557.json"
sample      = GIXOS_ana['metadata']["measurements"]["sample"][4:]
scan      = GIXOS_ana['metadata']["measurements"]["scan"]
fiofilename = f"../testing_data/p08_DPPC_data/raw/{sample}_{scan:05d}.fio"

# pseudoreflectivity
_ = export_orso(
    GIXOS_ana,
    which="refl",
    json_path=jsonfilename,
    fio_path=fiofilename,
)

# structure factor
_ = export_orso(
    GIXOS_ana,
    which="SF",
    json_path=jsonfilename,
    fio_path=fiofilename,
)

# I0 R* (background subtracted GIXOS)
_ = export_orso(
    GIXOS_ana,
    which="GIXOS",
    json_path=jsonfilename,
    fio_path=fiofilename,
)

