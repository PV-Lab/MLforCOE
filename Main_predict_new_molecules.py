#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:11:09 2021

Predict log2(MIC) value of a new molecule based on the fully trained RF model.

@author: armi tiihonen
"""

from Functions_downselection_training_RF import predict_plot_RF
from Main_downselection import fetch_csv, fetch_pickle
import numpy as np
import pandas as pd

def RF_predict_newdata(X, y, regressor, saveas='Newdata_prediction'): 
    
    if y is None:
        plotting = False
    else:
        plotting = True
    
    R2_newdata, RMSE_newdata, y_pred_newdata = predict_plot_RF(regressor, X, y, plotting=plotting, title=None, groups = None, saveas = './Results/' + saveas)
    results = [R2_newdata, RMSE_newdata, y_pred_newdata]
    
    print('New data mean RMSE=', RMSE_newdata, ' and R2=', R2_newdata)
    print('New data predictions:\n', y_pred_newdata.values)

    return results

if __name__ == "__main__":
    
    ##########################################################################
    # INPUT VALUES
    
    # Insert here a csv that has the same format than the other csv datafiles have
    # (i.e, molecules on rows, molecular descriptors in columns, no index in the
    # the csv file, and molecular descriptor names as headers in the first row).
    # The order or number of descriptors doesn't matter as long as all the Opt.
    # descriptors are somewhere in the file.
    X_newdata_path = './Data/Downselection_data_files/x_init_newdata'
    
    # Insert here log2mic csv that does not have nans or other invalid values.
    # The shape is the same than in the other csv datafiles.
    # If MIC has not been measured, insert None.
    #y_newdata_path = './Data/Downselection_data_files/y_init_newdata'
    y_newdata_path = None
    
    # Opt fingerprint.
    opt_descr_names = ['O%', 'MSD', 'MaxDD', 'CIC3', 'TI2_L', 'MATS8m', 'MATS2v', 'MATS7e',
       'P_VSA_MR_5', 'VE1sign_G', 'VE1sign_G/D', 'Mor22u', 'Mor20m', 'Mor25m',
       'Mor26m', 'Mor31m', 'Mor10v', 'Mor20v', 'Mor25v', 'R4i', 'H-047',
       'CATS2D_03_AL', 'SHED_AA', 'MLOGP2', 'ALOGP']
    #opt_descr_names = ['O%', 'MSD', 'MaxDD', 'CIC4', 'TI2_L', 'MATS2p', 'P_VSA_LogP_2',
    #   'P_VSA_MR_5', 'TDB07s', 'TDB10s', 'Mor06m', 'Mor20m', 'Mor25m',
    #   'Mor31m', 'Mor10v', 'Mor20v', 'Mor25v', 'Mor13s', 'H7u', 'R7u', 'R4i',
    #   'H-047', 'F02[C-N]', 'ALOGP']    
    
    ##############################################################################
    
    X_newdata = fetch_csv(X_newdata_path, index_col=None)
    X_newdata = X_newdata.loc[:,opt_descr_names]    
    
    if y_newdata_path is None:
        y_newdata = None
    else:
        y_newdata = fetch_csv(y_newdata_path).iloc[:,[-1]] # Dropping smiles
    
    rf_regressor = fetch_pickle('./Results/RF_regressor_opt_seed3')
    
    newdata_results = RF_predict_newdata(X_newdata, y_newdata, rf_regressor, saveas='Newdata_prediction')
