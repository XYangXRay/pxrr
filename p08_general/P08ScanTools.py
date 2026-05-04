# -*- coding: utf-8 -*-
"""
P08-specific scan tools built on top of fsScanTools.BaseScan.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from fsScanTools import BaseScan, ScanAnalyzer, ScanFitter, ScanPlot
# comment from Chen: did you forget p08_general. ? 
from p08_general.fsScanTools import BaseScan, ScanAnalyzer, ScanFitter, ScanPlot


class Scan(BaseScan):

    DETECTOR_CONFIG = {
        "p100k": {
            "type": "Pilatus",
            "identify_columns": ["p100k"],
            "subfolder": "p100k",
            "index_column": "p100k_index",
            "loader": "_load_pilatus",
            "loader_kwargs": {"detector": "p100k"},
        },
        "p300k": {
            "type": "Pilatus",
            "identify_columns": ["p300k"],
            "subfolder": "p300k",
            "index_column": "p300k_index",
            "loader": "_load_pilatus",
            "loader_kwargs": {"detector": "p300k"},
        },
        "eiger": {
            "type": "Eiger",
            "identify_columns": ["eiger"],
            "exclude_columns": ["eiger4m"],
            "subfolder": "eiger1m",
            "index_column": "eiger_index",
            "loader": "_load_eiger",
            "loader_kwargs": {"detector": "eiger"},
        },
        "eiger4m": {
            "type": "Eiger",
            "identify_columns": ["eiger4m"],
            "subfolder": "eiger4m",
            "index_column": "eiger4m_index",
            "loader": "_load_eiger",
            "loader_kwargs": {"detector": "eiger4m"},
        },
        "mythen": {
            "type": "Mythen",
            "identify_columns": ["mythint", "mythmax", "mythroi1", "mythroi2", "mythen"],
            "subfolder": "mythen",
            "loader": "_load_mythen",
            "loader_kwargs": {"detector": "mythen"},
        },
        "lambda": {
            "type": "Lambda",
            "identify_columns": ["lambda"],
            "subfolder": "lambda",
            "index_column": "lambda_index",
            "loader": "_load_lambda",
            "loader_kwargs": {"detector": "lambda"},
        },
        "amptek": {
            "type": "Amptek",
            "identify_columns": ["amptek_roi", "amptek"],
            "subfolder": "",
            "loader": "_load_amptek",
            "loader_kwargs": {},
        },
        "pe": {
            "type": "PerkinElmer",
            "identify_columns": ["pe_roi", "pe_trigger"],
            "subfolder": "pe",
            "index_column": "pe_index",
            "loader": "_load_pe",
            "loader_kwargs": {"detector": "pe"},
        },
    }
