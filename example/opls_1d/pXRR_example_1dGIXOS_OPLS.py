# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 16:39:16 2025

@author: shenc

example routine for OPLS data (a series of 1D GIXOS linecuts)

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
from pseudo_xrr.eCWM import *
from pseudo_xrr.data_io import *
from pseudo_xrr.GIXOS import *


#%% directly load data from meta and GIXOS will be automatically extracted:
GIXOSdata, GIXOSbkg = load_gixos_from_meta('./gixos-process_config_1d.yaml') 

#%% info
# from here all meta has entered GIXOSdata and GIXOSbkg in ["metadata"] field
# yaml file is no longer needed
#%% corrections
# binning in tt
GIXOSdata= binning_GIXOS_tt(GIXOSdata)
GIXOSbkg= binning_GIXOS_tt(GIXOSbkg)
# remove negative 2theta
GIXOSdata= remove_negative_2theta(GIXOSdata)
GIXOSbkg= remove_negative_2theta(GIXOSbkg)
# 2theta to q
GIXOSdata_q = GIXOS_th2q(GIXOSdata)
GIXOSbkg_q = GIXOS_th2q(GIXOSbkg)
# background subtraction
GIXOS_ana = GIXOS_background_corr(GIXOSdata_q, GIXOSbkg_q, bulkbkg_mode = 0, bulkbkg_const_mode= 1, bulkbkg_const_qz_lb= 0.7, plot = True)

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
_, qxy_dependence_fit = GIXOS_qxy_dependence(GIXOS_ana, GIXOS_ana['metadata']['dependency']['qz_selected'], row_window=3, fit_kappa = True)

#%% processing pseudo
_ = GIXOS2R(GIXOS_ana, transmission_corr = True, footprint_effect=False, use_approx=False)

#%% ---- export configuration ----
# this gives back the exact yaml file structure but with newly generated values from analysis
# particularly include the fitted bending modulus
configfilename = make_filename(GIXOS_ana["metadata"], suffix="cfg.yaml")
save_metadata_yaml(GIXOS_ana["metadata"], configfilename)

#%% ------export orso -----------------

# pseudoreflectivity
_ = export_orso(
    GIXOS_ana,
    which="refl"
)

# structure factor
_ = export_orso(
    GIXOS_ana,
    which="SF"
)

# I0 R* (background subtracted GIXOS)
_ = export_orso(
    GIXOS_ana,
    which="GIXOS"
)