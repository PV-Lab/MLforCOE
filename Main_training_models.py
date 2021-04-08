#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:25:28 2020

@author: armi tiihonen
"""

from Functions_downselection_training_RF import define_groups_yvalue
from Main_downselection import RF_train_test_newdata, fetch_csv
from Downselection_xgb import XGB_train_test_newdata
from Downselection_gp import GP_train_test_newdata

import pandas as pd
from set_figure_defaults import FigureDefaults

def fetch_refdata(path_x = './Data/Reference_data_files/Fingerprints/',
                  path_y = './Data/Downselection_data_files/'):
    
    y_refdata_train = fetch_csv(path_y + 'y_opt_train_seed3').iloc[:,[-1]] # Dropping smiles.
    y_refdata_test = fetch_csv(path_y + 'y_opt_test_seed3').iloc[:,[-1]]
    y_refdata_newdata = fetch_csv(path_y + 'y_opt_newdata').iloc[:,[-1]]
    
    y_smiles_train = fetch_csv(path_x +'y_smiles_train_seed3').iloc[:,[-1]]
    y_smiles_test = fetch_csv(path_x +'y_smiles_test_seed3').iloc[:,[-1]]
    y_smiles_newdata = fetch_csv(path_x +'y_smiles_newdata').iloc[:,[-1]]
    y_selfies_train = fetch_csv(path_x +'y_selfies_train_seed3').iloc[:,[-1]]
    y_selfies_test = fetch_csv(path_x +'y_selfies_test_seed3').iloc[:,[-1]]
    y_selfies_newdata = fetch_csv(path_x +'y_selfies_newdata').iloc[:,[-1]]
    
    X_morgan_train = fetch_csv(path_x + 'x_morganfromdmpnn_train_seed3')
    X_morgan_train.index = y_refdata_train.index
    X_morgancount_train = fetch_csv(path_x + 'x_morgancountfromdmpnn_train_seed3')
    X_morgancount_train.index = y_refdata_train.index
    X_rdkit_train = fetch_csv(path_x + 'x_rdkitfromdmpnn_train_seed3')
    X_rdkit_train.index = y_refdata_train.index
    X_opt2_train = fetch_csv(path_x + 'x_optfromdmpnn_train_seed3')
    X_opt2_train.index = y_refdata_train.index
    
    X_selfies_train = fetch_csv(path_x + 'x_selfies_train_seed3')
    X_smiles_train = fetch_csv(path_x + 'x_smiles_train_seed3')

    X_morgan_test = fetch_csv(path_x + 'x_morganfromdmpnn_test_seed3')
    X_morgancount_test = fetch_csv(path_x + 'x_morgancountfromdmpnn_test_seed3')
    X_rdkit_test = fetch_csv(path_x + 'x_rdkitfromdmpnn_test_seed3')
    X_opt2_test = fetch_csv(path_x + 'x_optfromdmpnn_test_seed3')
    X_selfies_test = fetch_csv(path_x + 'x_selfies_test_seed3')
    X_smiles_test = fetch_csv(path_x + 'x_smiles_test_seed3')

    X_morgan_newdata = fetch_csv(path_x + 'x_morganfromdmpnn_newdata_seed3')
    X_morgancount_newdata = fetch_csv(path_x + 'x_morgancountfromdmpnn_newdata_seed3')
    X_rdkit_newdata = fetch_csv(path_x + 'x_rdkitfromdmpnn_newdata_seed3')
    X_opt2_newdata = fetch_csv(path_x + 'x_optfromdmpnn_newdata_seed3')
    X_selfies_newdata = fetch_csv(path_x + 'x_selfies_newdata')
    X_smiles_newdata = fetch_csv(path_x + 'x_smiles_newdata')
    
    # Drop one mol with nans in morgan and rdkit datasets.    
    y_refdata_train = y_refdata_train.drop(X_morgan_train[(X_morgan_train=='Invalid SMILES').any(axis=1)].index)
    X_morgancount_train=X_morgancount_train.drop(X_morgan_train[(X_morgan_train=='Invalid SMILES').any(axis=1)].index)
    X_rdkit_train=X_rdkit_train.drop(X_morgan_train[(X_morgan_train=='Invalid SMILES').any(axis=1)].index)
    #X_smiles_train=X_smiles_test.drop(X_morgan_test[X_morgan_test.isnull().any(axis=1)].index)
    #X_selfies_train=X_selfies_test.drop(X_morgan_test[X_morgan_test.isnull().any(axis=1)].index)
    X_opt2_train=X_opt2_train.drop(X_morgan_train[(X_morgan_train=='Invalid SMILES').any(axis=1)].index)
    X_morgan_train=X_morgan_train.drop(X_morgan_train[(X_morgan_train=='Invalid SMILES').any(axis=1)].index)
    
    groups_refdata_train =  define_groups_yvalue(y_refdata_train)
    groups_smiles_train =  define_groups_yvalue(y_smiles_train)
    groups_selfies_train =  define_groups_yvalue(y_selfies_train)
    
    all_refdata_x = [X_morgan_train, X_morgan_test, X_morgan_newdata,
                    X_morgancount_train, X_morgancount_test, X_morgancount_newdata,
                    X_rdkit_train, X_rdkit_test, X_rdkit_newdata,
                    X_selfies_train, X_selfies_test, X_selfies_newdata,
                    X_smiles_train, X_smiles_test, X_smiles_newdata,
                    X_morgancount_train, X_morgancount_test, X_morgancount_newdata,
                    X_opt2_train, X_opt2_test, X_opt2_newdata,
                    ]
    
    all_refdata_y = [y_refdata_train, y_refdata_test, y_refdata_newdata, groups_refdata_train,
                     y_smiles_train, y_smiles_test, y_smiles_newdata, groups_smiles_train,
                     y_selfies_train, y_selfies_test, y_selfies_newdata, groups_selfies_train
                     ]
    
    return all_refdata_x, all_refdata_y

def fetch_downselected_data(path_x = './Data/Downselection_data_files/',
                  path_y = './Data/Downselection_data_files/'):
    
    X_init_train = fetch_csv(path_x + 'x_init_train_seed3')
    X_var_train = fetch_csv(path_x + 'x_var_train_seed3')
    X_cor_train = fetch_csv(path_x + 'x_cor_train_seed3')
    X_opt_train = fetch_csv(path_x + 'x_opt_train_seed3')

    X_init_test = fetch_csv(path_x + 'x_init_test_seed3')
    X_var_test = fetch_csv(path_x + 'x_var_test_seed3')
    X_cor_test = fetch_csv(path_x + 'x_cor_test_seed3')
    X_opt_test = fetch_csv(path_x + 'x_opt_test_seed3')

    X_init_newdata = fetch_csv(path_x + 'x_init_newdata')
    X_var_newdata = fetch_csv(path_x + 'x_var_newdata')
    X_cor_newdata = fetch_csv(path_x + 'x_cor_newdata')
    X_opt_newdata = fetch_csv(path_x + 'x_opt_newdata')
    
    # All the files have the same y data.
    y_train = fetch_csv(path_y + 'y_opt_train_seed3').iloc[:,[-1]] # Dropping smiles
    y_test = fetch_csv(path_y + 'y_opt_test_seed3').iloc[:,[-1]]
    y_newdata = fetch_csv(path_y + 'y_opt_newdata').iloc[:,[-1]]
    
    groups_train =  define_groups_yvalue(y_train)
    
    all_downseldata_x = [X_init_train, X_init_test, X_init_newdata,
                    X_var_train, X_var_test, X_var_newdata,
                    X_cor_train, X_cor_test, X_cor_newdata,
                    X_opt_train, X_opt_test, X_opt_newdata
                    ]
    
    all_downseldata_y = [y_train, y_test, y_newdata, groups_train]
    
    return all_downseldata_x, all_downseldata_y

###############################################################################
if __name__ == "__main__":
    
    ##########################################################################
    # INPUT VALUES
    
    # Reference fingerprint data.
    path_refdata_x = './Data/Reference_data_files/Fingerprints/'
    path_refdata_y = './Data/Downselection_data_files/'
    
    path_downseldata_x = './Data/Downselection_data_files/'
    path_downseldata_y = './Data/Downselection_data_files/'
    
    # Hyperparameters. Have been optimized separately.
    
    ho_rf_init = {'bootstrap': True, 'max_depth': 11, 'max_features': 'sqrt', 'max_samples': 0.99, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 117}
    ho_rf_var = {'bootstrap': True, 'max_depth': 11, 'max_features': 0.3, 'max_samples': 0.99, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 545}
    ho_rf_cor = {'bootstrap': True, 'max_depth': 13, 'max_features': 0.5, 'max_samples': 0.99, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 140}
    ho_rf_opt = {'bootstrap': True, 'max_depth': 18, 'max_features': 'log2', 'max_samples': 0.99, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 236}
    ho_rf_smiles = {'bootstrap': True, 'max_depth': 11, 'max_features': 0.3, 'max_samples': 0.5790898638079025, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 245}
    ho_rf_selfies = {'bootstrap': True, 'max_depth': 12, 'max_features': 0.3, 'max_samples': 0.8593574927176049, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 448}
    ho_rf_morgan = {'bootstrap': True, 'max_depth': 17, 'max_features': 'log2', 'max_samples': 0.9598746066998797, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 163}
    ho_rf_morgancount = {'bootstrap': True, 'max_depth': 16, 'max_features': 'sqrt', 'max_samples': 0.9890381756230884, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 82}
    ho_rf_rdkit = {'bootstrap': True, 'max_depth': 9, 'max_features': 0.3, 'max_samples': 0.99, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 334}
    ho_rf_opt = {'bootstrap': True, 'max_depth': 18, 'max_features': 'log2', 'max_samples': 0.99, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 236}
    
    ho_xgb_init = {'eta': 0.16303971225231562, 'gamma': 0.10020848581862363, 'max_depth': 3, 'n_estimators': 227}
    ho_xgb_var = {'eta': 0.17153129326979943, 'gamma': 2.078168135370535, 'max_depth': 4, 'n_estimators': 379}
    ho_xgb_cor = {'eta': 0.3054092191428568, 'gamma': 1.9386571606670655, 'max_depth': 5, 'n_estimators': 357}
    ho_xgb_opt = {'eta': 0.47182303948199855, 'gamma': 3.969100460481385, 'max_depth': 7, 'n_estimators': 496}
    ho_xgb_smiles = {'eta': 0.18823544508702517, 'gamma': 5.515296912481177, 'max_depth': 8, 'n_estimators': 243}
    ho_xgb_selfies = {'eta': 0.37168367127553065, 'gamma': 1.264729879182438, 'max_depth': 9, 'n_estimators': 481}
    ho_xgb_morgan = {'eta': 0.3958674027273652, 'gamma': 8.492167513146274, 'max_depth': 3, 'n_estimators': 402}
    ho_xgb_morgancount = {'eta': 0.43107847212832207, 'gamma': 5.003826686513141, 'max_depth': 3, 'n_estimators': 377}
    ho_xgb_rdkit = {'eta': 0.3359144357414168, 'gamma': 2.0511001578641586, 'max_depth': 6, 'n_estimators': 82}
    
    # Random state that in this run affects only model training (not train/test divisions). 
    random_state = 3
    
    ##########################################################################
    # CODE EXECUTION STARTS
    
    
    ##########################################################################
    # Fetch data from files.
    
    all_refdata_x, all_refdata_y = fetch_refdata(path_refdata_x, path_refdata_y)
    
    [X_morgan_train, X_morgan_test, X_morgan_newdata,
                    X_morgancount_train, X_morgancount_test, X_morgancount_newdata,
                    X_rdkit_train, X_rdkit_test, X_rdkit_newdata,
                    X_selfies_train, X_selfies_test, X_selfies_newdata,
                    X_smiles_train, X_smiles_test, X_smiles_newdata,
                    X_morgancount_train, X_morgancount_test, X_morgancount_newdata,
                    X_opt2_train, X_opt2_test, X_opt2_newdata] = all_refdata_x
    
    [y_refdata_train, y_refdata_test, y_refdata_newdata, groups_refdata_train,
                     y_smiles_train, y_smiles_test, y_smiles_newdata, groups_smiles_train,
                     y_selfies_train, y_selfies_test, y_selfies_newdata, groups_selfies_train] = all_refdata_y

    all_downseldata_x, all_downseldata_y = fetch_downselected_data(path_downseldata_x, path_downseldata_y)
    
    [X_init_train, X_init_test, X_init_newdata,
                    X_var_train, X_var_test, X_var_newdata,
                    X_cor_train, X_cor_test, X_cor_newdata,
                    X_opt_train, X_opt_test, X_opt_newdata] = all_downseldata_x
    
    [y_train, y_test, y_newdata, groups_train] = all_downseldata_y
    ##########################################################################
    # Train RF models.
    mystyle = FigureDefaults('nature_comp_mat_tc')
    
    print('\nRF and Morgan:\n')
    cv_results_rf_morgan, test_rf_results_morgan, val_rf_results_morgan = RF_train_test_newdata(
            [X_morgan_train, X_morgan_test, X_morgan_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, ho_rf_morgan,
            saveas='./Results/rf_morgan_seed' + str(random_state),
            random_state=random_state)
    print('\nRF and Morgan count:\n')
    cv_results_rf_morgancount, test_results_rf_morgancount, val_results_rf_morgancount = RF_train_test_newdata(
            [X_morgancount_train, X_morgancount_test, X_morgancount_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, ho_rf_morgancount,
            saveas='./Results/rf_morgancount_seed'  + str(random_state),
            random_state=random_state)
    print('\nRF and Rdkit:\n')
    cv_results_rf_rdkit, test_results_rf_rdkit, val_results_rf_rdkit = RF_train_test_newdata(
            [X_rdkit_train, X_rdkit_test, X_rdkit_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, ho_rf_rdkit,
            saveas='./Results/rf_rdkit_seed'  + str(random_state),
            random_state=random_state)
    print('\nRF and Opt2:\n')
    cv_results_rf_opt2, test_results_rf_opt2, val_results_rf_opt2 = RF_train_test_newdata(
            [X_opt2_train, X_opt2_test, X_opt2_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, ho_rf_opt,
            saveas='./Results/rf_optfromdmpnn__seed'  + str(random_state),
            random_state=random_state)
    print('\nRF and Selfies:\n')
    cv_results_rf_selfies, test_results_rf_selfies, val_results_rf_selfies = RF_train_test_newdata(
            [X_selfies_train, X_selfies_test, X_selfies_newdata],
            [y_selfies_train, y_selfies_test, y_selfies_newdata], groups_selfies_train, ho_rf_selfies,
            saveas='./Results/rf_selfies_seed' + str(random_state),
            random_state=random_state)
    print('\nRF and Smiles:\n')
    cv_results_rf_smiles, test_results_rf_smiles, val_results_rf_smiles = RF_train_test_newdata(
            [X_smiles_train, X_smiles_test, X_smiles_newdata],
            [y_smiles_train, y_smiles_test, y_smiles_newdata], groups_smiles_train, ho_rf_smiles,
            saveas='./Results/rf_smiles_seed' + str(random_state),
            random_state=random_state)
    ###
    print('\nRF and Init:\n')
    cv_results_rf_init, test_results_rf_init, val_results_rf_init = RF_train_test_newdata(
            [X_init_train, X_init_test, X_init_newdata],
            [y_train, y_test, y_newdata], groups_train, ho_rf_init,
            saveas='./Results/rf_init_seed' + str(random_state),
            random_state=random_state)
    print('\nRF and Var:\n')
    cv_results_rf_var, test_results_rf_var, val_results_rf_var = RF_train_test_newdata(
            [X_var_train, X_var_test, X_var_newdata],
            [y_train, y_test, y_newdata], groups_train, ho_rf_var,
            saveas='./Results/rf_var_seed'  + str(random_state),
            random_state=random_state)
    print('\nRF and Cor:\n')
    cv_results_rf_cor, test_results_rf_cor, val_results_rf_cor = RF_train_test_newdata(
            [X_cor_train, X_cor_test, X_cor_newdata],
            [y_train, y_test, y_newdata], groups_train, ho_rf_cor,
            saveas='./Results/rf_cor_seed'  + str(random_state),
            random_state=random_state)
    print('\nRF and Opt:\n')
    cv_results_rf_opt, test_results_rf_opt, val_results_rf_opt = RF_train_test_newdata(
            [X_opt_train, X_opt_test, X_opt_newdata],
            [y_train, y_test, y_newdata], groups_train, ho_rf_opt,
            saveas='./Results/rf_opt_seed'  + str(random_state),
            random_state=random_state)
    ##########################################################################
    # Train XGB models.
    
    print('\nXGB and Morgan:\n')
    cv_results_xgb_morgan, test_xgb_results_morgan, val_xgb_results_morgan = XGB_train_test_newdata(
            [X_morgan_train, X_morgan_test, X_morgan_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, ho_xgb_morgan,
            saveas='./Results/xgb_morgan_seed' + str(random_state),
            random_state=random_state)
    print('\nXGB and Morgan count:\n')
    cv_results_xgb_morgancount, test_results_xgb_morgancount, val_results_xgb_morgancount = XGB_train_test_newdata(
            [X_morgancount_train, X_morgancount_test, X_morgancount_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, ho_xgb_morgancount,
            saveas='./Results/xgb_morgancount_seed'  + str(random_state),
            random_state=random_state)
    print('\nXGB and Rdkit:\n')
    cv_results_xgb_rdkit, test_results_xgb_rdkit, val_results_xgb_rdkit = XGB_train_test_newdata(
            [X_rdkit_train, X_rdkit_test, X_rdkit_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, ho_xgb_rdkit,
            saveas='./Results/xgb_rdkit_seed'  + str(random_state),
            random_state=random_state)
    print('\nXGB and Opt2:\n')
    cv_results_xgb_opt2, test_results_xgb_opt2, val_results_xgb_opt2 = XGB_train_test_newdata(
            [X_opt2_train, X_opt2_test, X_opt2_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, ho_xgb_opt,
            saveas='./Results/xgb_optfromdmpnn__seed'  + str(random_state),
            random_state=random_state)
    print('\nXGB and Selfies:\n')
    cv_results_xgb_selfies, test_results_xgb_selfies, val_results_xgb_selfies = XGB_train_test_newdata(
            [X_selfies_train, X_selfies_test, X_selfies_newdata],
            [y_selfies_train, y_selfies_test, y_selfies_newdata], groups_selfies_train, ho_xgb_selfies,
            saveas='./Results/xgb_selfies_seed' + str(random_state),
            random_state=random_state)
    print('\nXGB and Smiles:\n')
    cv_results_xgb_smiles, test_results_xgb_smiles, val_results_xgb_smiles = XGB_train_test_newdata(
            [X_smiles_train, X_smiles_test, X_smiles_newdata],
            [y_smiles_train, y_smiles_test, y_smiles_newdata], groups_smiles_train, ho_xgb_smiles,
            saveas='./Results/xgb_smiles_seed' + str(random_state),
            random_state=random_state)
    ###
    print('\nXGB and Init:\n')
    cv_results_xgb_init, test_results_xgb_init, val_results_xgb_init = XGB_train_test_newdata(
            [X_init_train, X_init_test, X_init_newdata],
            [y_train, y_test, y_newdata], groups_train, ho_xgb_init,
            saveas='./Results/xgb_init_seed' + str(random_state),
            random_state=random_state)
    print('\nXGB and Var:\n')
    cv_results_xgb_var, test_results_xgb_var, val_results_xgb_var = XGB_train_test_newdata(
            [X_var_train, X_var_test, X_var_newdata],
            [y_train, y_test, y_newdata], groups_train, ho_xgb_var,
            saveas='./Results/xgb_var_seed'  + str(random_state),
            random_state=random_state)
    print('\nXGB and Cor:\n')
    cv_results_xgb_cor, test_results_xgb_cor, val_results_xgb_cor = XGB_train_test_newdata(
            [X_cor_train, X_cor_test, X_cor_newdata],
            [y_train, y_test, y_newdata], groups_train, ho_xgb_cor,
            saveas='./Results/xgb_cor_seed'  + str(random_state),
            random_state=random_state)
    print('\nXGB and Opt:\n')
    cv_results_xgb_opt, test_results_xgb_opt, val_results_xgb_opt = XGB_train_test_newdata(
            [X_opt_train, X_opt_test, X_opt_newdata],
            [y_train, y_test, y_newdata], groups_train, ho_xgb_opt,
            saveas='./Results/xgb_opt_seed'  + str(random_state),
            random_state=random_state)
    
    ##############################################################################
    # Train GP models.
    
    print('Running GP takes longer because kernel optimization is integrated into the implementation.\n')
    print('\nGP and Morgan:\n')
    cv_results_gp_morgan, test_gp_results_morgan, val_gp_results_morgan = GP_train_test_newdata(
            [X_morgan_train, X_morgan_test, X_morgan_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, None,
            saveas='./Results/gp_morgan_seed' + str(random_state),
            random_state=random_state)
    print('\nGP and Morgan count:\n')
    cv_results_gp_morgancount, test_results_gp_morgancount, val_results_gp_morgancount = GP_train_test_newdata(
            [X_morgancount_train, X_morgancount_test, X_morgancount_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, None,
            saveas='./Results/gp_morgancount_seed'  + str(random_state),
            random_state=random_state)
    print('\nGP and Rdkit:\n')
    cv_results_gp_rdkit, test_results_gp_rdkit, val_results_gp_rdkit = GP_train_test_newdata(
            [X_rdkit_train, X_rdkit_test, X_rdkit_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, None,
            saveas='./Results/gp_rdkit_seed'  + str(random_state),
            random_state=random_state)
    print('\nGP and Opt2:\n')
    cv_results_gp_opt2, test_results_gp_opt2, val_results_gp_opt2 = GP_train_test_newdata(
            [X_opt2_train, X_opt2_test, X_opt2_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, None,
            saveas='./Results/gp_optfromdmpnn__seed'  + str(random_state),
            random_state=random_state)
    print('\nGP and Selfies:\n')
    cv_results_gp_selfies, test_results_gp_selfies, val_results_gp_selfies = GP_train_test_newdata(
            [X_selfies_train, X_selfies_test, X_selfies_newdata],
            [y_selfies_train, y_selfies_test, y_selfies_newdata], groups_selfies_train, None,
            saveas='./Results/gp_selfies_seed' + str(random_state),
            random_state=random_state)
    print('\nGP and Smiles:\n')
    cv_results_gp_smiles, test_results_gp_smiles, val_results_gp_smiles = GP_train_test_newdata(
            [X_smiles_train, X_smiles_test, X_smiles_newdata],
            [y_smiles_train, y_smiles_test, y_smiles_newdata], groups_smiles_train, None,
            saveas='./Results/gp_smiles_seed' + str(random_state),
            random_state=random_state)
    ###
    print('\nGP and Init:\n')
    cv_results_gp_init, test_results_gp_init, val_results_gp_init = GP_train_test_newdata(
            [X_init_train, X_init_test, X_init_newdata],
            [y_train, y_test, y_newdata], groups_train, None,
            saveas='./Results/gp_init_seed' + str(random_state),
            random_state=random_state)
    print('\nGP and Var:\n')
    cv_results_gp_var, test_results_gp_var, val_results_gp_var = GP_train_test_newdata(
            [X_var_train, X_var_test, X_var_newdata],
            [y_train, y_test, y_newdata], groups_train, None,
            saveas='./Results/gp_var_seed'  + str(random_state),
            random_state=random_state)
    print('\nGP and Cor:\n')
    cv_results_gp_cor, test_results_gp_cor, val_results_gp_cor = GP_train_test_newdata(
            [X_cor_train, X_cor_test, X_cor_newdata],
            [y_train, y_test, y_newdata], groups_train, None,
            saveas='./Results/gp_cor_seed'  + str(random_state),
            random_state=random_state)
    print('\nGP and Opt:\n')
    cv_results_gp_opt, test_results_gp_opt, val_results_gp_opt = GP_train_test_newdata(
            [X_opt_train, X_opt_test, X_opt_newdata],
            [y_train, y_test, y_newdata], groups_train, None,
            saveas='./Results/gp_opt_seed'  + str(random_state),
            random_state=random_state)
    
'''
Print-outs from the code:

RF and Morgan:

R2 and RMSE for dataset  0 :  [0.28423206 0.13934164 0.25014359 0.20500426 0.48860361 0.34716933
 0.6244426  0.17822921 0.4835597  0.25097172 0.53328562 0.63116263
 0.32134946 0.32103329 0.41610592 0.13271096 0.31976102 0.58936074
 0.03083938 0.32237079] [1.86123749 1.96028189 2.00760509 2.01773331 1.44966187 1.713239
 1.38523051 1.86539156 1.39163614 1.81024825 1.40254348 1.26973332
 1.64680028 1.8816762  1.2876987  1.83969018 1.79457694 1.27157071
 2.22781561 1.9084698 ]
Mean:  0.3434838779022107 1.6996420167690225
Std:  0.1669223277288744 0.2827840126275429
Min:  0.030839382466800846 1.2697333218834381
Max:  0.6311626332193971 2.227815607539234
Test set RMSE= 1.7242489747564358  and R2= 0.12814236746374796
Exp. validation set RMSE= 1.9739896630790787  and R2= 0.18403793031106963

RF and Morgan count:

R2 and RMSE for dataset  0 :  [0.33340535 0.12838825 0.29806282 0.15198674 0.4126414  0.18807379
 0.64564197 0.15855963 0.3906628  0.42193635 0.5516372  0.59472185
 0.24412865 0.28261535 0.31614689 0.0262934  0.29297546 0.5729984
 0.03384305 0.39413659] [1.79616646 1.97271647 1.9423986  2.08392788 1.55360122 1.91062722
 1.34556607 1.88758418 1.51162644 1.59029195 1.37469238 1.33098038
 1.73796801 1.93417912 1.39356951 1.94929164 1.82956806 1.29665677
 2.22436066 1.8045819 ]
Mean:  0.3219427967999069 1.7235177461239495
Std:  0.17516946046708756 0.27134290375695935
Min:  0.026293396311242834 1.2966567716486213
Max:  0.6456419744327226 2.2243606585002977
Test set RMSE= 1.7819865058967466  and R2= 0.06877539378354958
Exp. validation set RMSE= 1.9752655956949587  and R2= 0.18298275853302615

RF and Rdkit:

R2 and RMSE for dataset  0 :  [ 0.2916046  -0.02627706  0.22686819  0.11107311  0.4266443   0.20843057
  0.69695729  0.02153023  0.22317641  0.38920799  0.51965736  0.6101691
  0.33513849  0.48276909  0.24302797 -0.11291249  0.22626622  0.58411686
  0.01311847  0.35223936] [1.85162713 2.14059961 2.03852484 2.13360679 1.53497022 1.88652335
 1.24433068 2.03548715 1.70677721 1.63469112 1.42287356 1.30536868
 1.62998434 1.6423393  1.46617948 2.08397867 1.91393465 1.27966396
 2.24809094 1.865935  ]
Mean:  0.2911403036027812 1.7532743344624664
Std:  0.21718979775311273 0.3035418267703119
Min:  -0.11291249051561358 1.244330679635574
Max:  0.6969572911985655 2.2480909360311774
Test set RMSE= 1.7958527540445488  and R2= 0.054226652724518876
Exp. validation set RMSE= 1.9141503344243531  and R2= 0.2327581041193748

RF and Opt2:

R2 and RMSE for dataset  0 :  [0.35921487 0.11886675 0.21745748 0.17580169 0.46338352 0.30012746
 0.63889401 0.1254761  0.38959441 0.30669245 0.48843676 0.52572281
 0.29697762 0.30322855 0.41681686 0.1714826  0.2473586  0.58834552
 0.10584077 0.31368083] [1.7610508  1.9834622  2.05089397 2.05445779 1.48497756 1.77389192
 1.3583173  1.92433421 1.51295108 1.74161434 1.46838657 1.43982987
 1.67610959 1.90618841 1.28691453 1.79809891 1.88766698 1.27314159
 2.13987692 1.92066799]
Mean:  0.3276699823635971 1.7221416270208447
Std:  0.15227546658998334 0.26294678773839153
Min:  0.10584077319812424 1.2731415931870214
Max:  0.6388940135484285 2.139876918862372
Test set RMSE= 1.6685193009281207  and R2= 0.18359042300009842
Exp. validation set RMSE= 1.88730979894085  and R2= 0.25412403597551514

RF and Selfies:

R2 and RMSE for dataset  0 :  [-0.48091443  0.08831358 -0.06885233 -0.34651163 -0.05649507 -0.10580171
  0.0689284  -0.32643273 -0.0814278  -0.02667372 -0.09583492  0.24797071
 -0.17597877 -0.01062965 -0.15523678 -0.15198319  0.1225508   0.01945467
  0.0053727   0.09868318] [2.25624541 1.97473953 2.12742134 2.24240482 2.08759243 2.40945201
 2.22974816 2.36993957 2.07881863 2.05694401 2.12207367 1.8130611
 2.06357199 2.00961957 1.86164184 2.12024394 1.98622603 1.97172743
 1.9858353  1.95358988]
Mean:  -0.07157493400918459 2.0860448324533998
Std:  0.1686095655016993 0.15142109642737686
Min:  -0.4809144344579115 1.8130610965164276
Max:  0.24797070762864537 2.40945200673015
Test set RMSE= 1.8965890756361765  and R2= -0.054853408159087014
Exp. validation set RMSE= 2.06597396903641  and R2= 0.10622148035868895

RF and Smiles:

R2 and RMSE for dataset  0 :  [ 0.38474958 -0.07691696  0.21115456  0.14289028  0.26466289  0.21865286
  0.45600758 -0.00667808  0.29219747  0.29157744  0.42390205  0.47011671
  0.24075051  0.38061353  0.28765784 -0.16701964  0.34066829  0.47768353
  0.11712096  0.34104305] [1.72560595 2.19277586 2.05913677 2.095075   1.73832561 1.87430254
 1.66717163 2.06461922 1.62918983 1.76049673 1.55825634 1.52189669
 1.74184734 1.7972211  1.42230111 2.1340365  1.7667834  1.43409243
 2.12633637 1.88199192]
Mean:  0.2545417217614492 1.8095731162798212
Std:  0.1743271652321735 0.23254473024977007
Min:  -0.16701963602193848 1.4223011086969821
Max:  0.47768352765022226 2.192775856312539
Test set RMSE= 1.7900208080929367  and R2= 0.060359386098037926
Exp. validation set RMSE= 1.8320702428448985  and R2= 0.2971470625591003

RF and Init:

R2 and RMSE for dataset  0 :  [0.38221578 0.58671488 0.36498201 0.21691427 0.35915033 0.206016
 0.53444352 0.24553445 0.22873512 0.51854006 0.32620831 0.43331238
 0.27214572 0.36328262 0.16952127 0.24789722 0.33920274 0.65264306
 0.24380261 0.43979684] [1.45726911 1.3295723  1.63978788 1.71006894 1.62588588 2.0416699
 1.57670506 1.7873696  1.7555759  1.4085956  1.66399    1.57386299
 1.62346191 1.59511201 1.5784284  1.71317354 1.72366077 1.17354846
 1.73153241 1.54016566]
Mean:  0.3565529601026409 1.612471815797814
Std:  0.13243881072136185 0.17790195373711923
Min:  0.16952126606855034 1.1735484598372683
Max:  0.6526430612589904 2.0416699004963803
Test set RMSE= 1.1298118151545105  and R2= 0.6256672323569707
Exp. validation set RMSE= 2.20316474785942  and R2= -0.01642226668509661

RF and Var:

R2 and RMSE for dataset  0 :  [0.41219084 0.6406651  0.37038921 0.3225591  0.28636127 0.29002312
 0.52192951 0.28405205 0.1803601  0.41950964 0.41149777 0.42049064
 0.26009531 0.4323679  0.12826108 0.31927647 0.36691936 0.67397996
 0.35114641 0.43299735] [1.42147599 1.23975759 1.63279153 1.59054028 1.71573883 1.93064203
 1.5977552  1.74114677 1.80979498 1.54669169 1.55511296 1.59156832
 1.63684582 1.50609133 1.61716318 1.6298519  1.68712474 1.13693383
 1.60393363 1.54948438]
Mean:  0.37625361022998344 1.5870222505663647
Std:  0.12976369639892063 0.17131563570728
Min:  0.12826108230788613 1.1369338305198957
Max:  0.6739799557093158 1.9306420290393183
Test set RMSE= 1.2029036317004402  and R2= 0.5756665257600855
Exp. validation set RMSE= 2.0319847350486975  and R2= 0.13538830679454983

RF and Cor:

R2 and RMSE for dataset  0 :  [0.34979529 0.63291554 0.30652309 0.35393034 0.43105521 0.28218975
 0.55346519 0.31003569 0.14756295 0.46816861 0.40250109 0.46855071
 0.33886773 0.4487754  0.26840845 0.36483199 0.33981247 0.6822132
 0.34810203 0.4965054 ] [1.49501796 1.25305483 1.71360477 1.55327601 1.53195863 1.94126346
 1.5441587  1.70925937 1.84564849 1.48044826 1.56695472 1.52414391
 1.54726273 1.48416481 1.48147718 1.57437095 1.72286536 1.12248609
 1.607692   1.46013182]
Mean:  0.39971050711523404 1.557962002718001
Std:  0.12392283392717036 0.1764095939228251
Min:  0.1475629505466991 1.1224860853016365
Max:  0.6822131990205685 1.9412634625546972
Test set RMSE= 1.2536685451308631  and R2= 0.5390953603957376
Exp. validation set RMSE= 1.9344140939933017  and R2= 0.21642762194463083

RF and Opt:

R2 and RMSE for dataset  0 :  [0.60328733 0.68948775 0.53154234 0.40915748 0.59184708 0.43183789
 0.64342348 0.51144908 0.48178058 0.56437424 0.57953469 0.57828467
 0.5479287  0.54741098 0.34949024 0.50291659 0.38153875 0.74816704
 0.49773393 0.67321919] [1.16777558 1.15246156 1.40841237 1.48540484 1.29754799 1.72709325
 1.37987786 1.43830032 1.439046   1.33987111 1.31447694 1.35770293
 1.2794501  1.34483759 1.39697147 1.39276356 1.66753103 0.99923888
 1.41117202 1.17631309]
Mean:  0.5432206021056821 1.3588124239388706
Std:  0.10050148286436646 0.16158915542086635
Min:  0.34949023867662554 0.9992388837207261
Max:  0.7481670376696578 1.7270932523790081
Test set RMSE= 1.2309551764243842  and R2= 0.5556449717401799
Exp. validation set RMSE= 1.815837060507216  and R2= 0.3095472338236629

XGB and Morgan:

R2 and RMSE for dataset  0 :  [ 0.2574078  -0.10204311  0.17351146 -0.07264744  0.42251476  0.25692786
  0.69825974 -0.19242455  0.29578338  0.34202889  0.36021536  0.64042172
  0.33259799  0.58543555  0.70832035 -0.53424554  0.34567599  0.73101645
 -0.03048451  0.11295443] [1.89579277 2.21820881 2.10769445 2.34374233 1.54048804 1.82781892
 1.24165379 2.24703642 1.62505764 1.6966507  1.64213041 1.25369469
 1.63309553 1.47033631 0.91012342 2.44686605 1.76006114 1.02913688
 2.2972174  2.18354527]
Mean:  0.26656132935440535 1.7685175482478386
Std:  0.3250003795434959 0.43794763401416115
Min:  -0.5342455352322943 0.9101234212157076
Max:  0.7310164506789927 2.446866051785252
Test set RMSE= 1.484788002131712  and R2= 0.3534910817377477
Exp. validation set RMSE= 2.0140171615825073  and R2= 0.15061114858848668

XGB and Morgan count:

R2 and RMSE for dataset  0 :  [ 0.18276478 -0.21730487  0.25356875  0.11264106  0.27699464  0.3059227
  0.46950137  0.12153561  0.57992395  0.14970757  0.13450717  0.58238277
  0.36382378  0.3039978  -0.14064176  0.10291951  0.4601955   0.46160421
 -0.34633287  0.24226228] [1.98879128 2.33132495 2.00301472 2.13172426 1.72368795 1.76653237
 1.64636461 1.92866473 1.25510366 1.92873777 1.9099521  1.35108996
 1.59443401 1.90513588 1.79979009 1.87102017 1.59863597 1.45599912
 2.62577547 2.01812791]
Mean:  0.2199986970321533 1.8416953493066643
Std:  0.24184446150491923 0.3126783722557248
Min:  -0.34633286875989233 1.2551036626874352
Max:  0.5823827676259901 2.625775473955428
Test set RMSE= 2.248879473575345  and R2= -0.48312577321657457
Exp. validation set RMSE= 1.8094996447856322  and R2= 0.3143582937628754

XGB and Rdkit:

R2 and RMSE for dataset  0 :  [ 0.19093598 -0.23793234  0.14519426 -0.08795036  0.16649493  0.09225555
  0.34271413 -0.26300414 -0.1236766   0.06319282  0.17181838  0.48153675
  0.12120838  0.39243947  0.03302411 -0.56237584  0.01713692  0.466916
 -0.1778161   0.02791457] [1.97882375 2.35099436 2.14349734 2.36040162 1.85072551 2.02022379
 1.83257196 2.31258149 2.05275114 2.024483   1.86832977 1.50540741
 1.87396203 1.77998124 1.65712415 2.46919574 2.15713755 1.4487989
 2.45595332 2.28581697]
Mean:  0.06300134245472233 2.0214380516397354
Std:  0.25325927851194363 0.29144541701572757
Min:  -0.5623758437934823 1.4487989008782975
Max:  0.48153675355060854 2.4691957413656924
Test set RMSE= 2.1212821726299524  and R2= -0.31960060290840175
Exp. validation set RMSE= 2.059586152399538  and R2= 0.11173991094597979

XGB and Opt2:

R2 and RMSE for dataset  0 :  [ 0.5018841  -0.08012988  0.46705804  0.16888411  0.68423089  0.30229122
  0.47670315  0.18566038  0.02710588  0.05480704  0.17998771  0.59008743
  0.20738714  0.50438975  0.30038211 -0.05407304  0.25523745  0.39213165
 -0.24915825  0.13316745] [1.55267612 2.19604443 1.692502   2.06306142 1.13912848 1.77114767
 1.63515132 1.85693816 1.91006621 2.03352384 1.85909216 1.33856874
 1.77970651 1.60764735 1.40954088 2.02814086 1.87776069 1.54708812
 2.5292403  2.15852379]
Mean:  0.25240171700094116 1.7992774524360589
Std:  0.2372492750487427 0.3179690325642528
Min:  -0.24915825217730636 1.139128481289249
Max:  0.6842308874501959 2.5292403045155925
Test set RMSE= 2.1482314084909655  and R2= -0.3533425760783222
Exp. validation set RMSE= 2.2163142600358667  and R2= -0.028591436165291917

XGB and Selfies:

R2 and RMSE for dataset  0 :  [-0.81471437 -0.00383182 -0.17053653 -0.85604914 -0.3836653  -0.18761113
  0.23067319 -0.71229124  0.02098976 -0.27419491 -0.42616664  0.0597053
 -0.12907656 -0.12072741 -0.75582407 -0.4058117   0.06608721  0.01739211
  0.07939694  0.06383   ] [2.49761499 2.07213266 2.22631772 2.6327141  2.38906245 2.49698987
 2.02684113 2.69267213 1.97793243 2.29152278 2.42087867 2.02734217
 2.022002   2.11625419 2.29509743 2.34221274 2.04913627 1.97380008
 1.91050969 1.99100347]
Mean:  -0.2351213156804413 2.2226018496281172
Std:  0.324503128643944 0.23291232472118203
Min:  -0.8560491445217038 1.9105096914683404
Max:  0.23067318529944358 2.6926721310596
Test set RMSE= 2.004569029342637  and R2= -0.1783862150732205
Exp. validation set RMSE= 2.8888070088413302  and R2= -0.7475003879581577

XGB and Smiles:

R2 and RMSE for dataset  0 :  [ 0.3929155  -0.03083802  0.32813973  0.28257756  0.20804465  0.2286269
  0.54612148  0.07170333  0.40727222  0.30483451  0.46117561  0.57184541
  0.2104038   0.3499984   0.48319175 -0.42951909  0.41850305  0.69192428
  0.23175764  0.38234122] [1.71411613 2.14535095 1.90032865 1.91676487 1.80400708 1.8623012
 1.522838   1.98261348 1.49088201 1.7439464  1.50700386 1.36802918
 1.77631653 1.84110206 1.21146798 2.36187952 1.65922411 1.10138465
 1.98349215 1.82206366]
Mean:  0.3055509963049215 1.7357556232899953
Std:  0.23575292267469147 0.2965946784122653
Min:  -0.42951909064381155 1.1013846497008384
Max:  0.6919242802294252 2.361879524632779
Test set RMSE= 1.8599130352670108  and R2= -0.014450586145497102
Exp. validation set RMSE= 2.1271444309600263  and R2= 0.0525109911194509

XGB and Init:

R2 and RMSE for dataset  0 :  [ 0.27183543  0.58154211  0.07897737  0.24880151  0.28826405 -0.01808532
  0.37391035  0.27603054  0.00395453  0.40428951  0.29778918  0.56942526
  0.10850721  0.13897994  0.38536966  0.18317177  0.3038131   0.31763374
  0.18802074  0.39681417] [1.58210799 1.33786705 1.97483042 1.6748901  1.71344997 2.31191434
 1.82844767 1.75087355 1.99506669 1.56683718 1.6987194  1.37189015
 1.79671338 1.85491694 1.35789949 1.78536957 1.76921496 1.644833
 1.79426054 1.59815988]
Mean:  0.26995224270610024 1.720413114489863
Std:  0.15775187516926095 0.22403041977204155
Min:  -0.018085320298312002 1.3378670478307844
Max:  0.5815421106448464 2.311914343475151
Test set RMSE= 1.4323199414610988  and R2= 0.39837524495392207
Exp. validation set RMSE= 1.7722635119465222  and R2= 0.342286410969769

XGB and Var:

R2 and RMSE for dataset  0 :  [ 0.24735212  0.66144213  0.19420119  0.14969735  0.18962551  0.21255106
  0.55089121  0.29808873 -0.08490908 -0.12317     0.30560495  0.40377801
  0.15395874  0.14629869 -0.54226893  0.16854217  0.41411477  0.64783675
  0.10608418  0.22485449] [1.60848596 1.20338206 1.84717469 1.78195088 1.82833115 2.03325034
 1.54860285 1.72399406 2.08216195 2.15143879 1.68923937 1.61435505
 1.75031275 1.84701664 2.15100134 1.80128684 1.62302031 1.18163965
 1.88261429 1.81169996]
Mean:  0.2162287016299392 1.758047947067023
Std:  0.26425044431962136 0.2532351687415212
Min:  -0.5422689286417588 1.181639653669323
Max:  0.661442133355618 2.1514387891434787
Test set RMSE= 1.6801497095332045  and R2= 0.1721691945910523
Exp. validation set RMSE= 1.808688780400518  and R2= 0.3149726490897906

XGB and Cor:

R2 and RMSE for dataset  0 :  [ 0.4483376   0.70893073  0.25405261  0.14997166  0.25202948  0.24011015
  0.46389685  0.12445379  0.15847088  0.34326257  0.4847995   0.45819188
  0.11535587 -0.06487964  0.21545966  0.20259956  0.45087026  0.45566908
  0.34482122  0.57336186] [1.37707644 1.11579717 1.77725096 1.78166343 1.75652453 1.99735368
 1.69195363 1.92545864 1.83380186 1.64513723 1.45504365 1.53892627
 1.78979869 2.06285212 1.53415156 1.76400981 1.57128571 1.46907694
 1.61173245 1.34407808]
Mean:  0.3189882778291639 1.652148643046085
Std:  0.1815042847377386 0.22775851017510063
Min:  -0.06487963611471326 1.115797166896015
Max:  0.7089307312153311 2.0628521174287506
Test set RMSE= 1.6562465069637813  and R2= 0.19555645400887778
Exp. validation set RMSE= 1.833115627819717  and R2= 0.2963447335774064

XGB and Opt:

R2 and RMSE for dataset  0 :  [0.19961171 0.64184391 0.36959412 0.35727388 0.04770854 0.43104562
 0.46853979 0.33012618 0.23496533 0.16993462 0.50301581 0.62693655
 0.11716157 0.32690663 0.230346   0.35892936 0.46160392 0.52280693
 0.22771984 0.55108903] [1.65871478 1.23772238 1.63382218 1.54925155 1.98196929 1.72829699
 1.68461108 1.68419029 1.74847084 1.84953434 1.42908867 1.27698681
 1.78797111 1.64004397 1.5195269  1.58166938 1.55585323 1.37549844
 1.74984861 1.37871584]
Mean:  0.358857967258983 1.6025893336929138
Std:  0.16304404376681042 0.18711819624742324
Min:  0.04770854260548374 1.237722381886474
Max:  0.6418439143165132 1.9819692888015141
Test set RMSE= 1.9139468246908349  and R2= -0.07424998467569766
Exp. validation set RMSE= 2.0246886602812486  and R2= 0.14158613511751106
Running GP takes longer because kernel optimization is integrated into the implementation.


GP and Morgan:

R2 and RMSE for dataset  0 :  [0.40591478 0.19460439 0.3099221  0.22646185 0.51018318 0.36592405
 0.54710072 0.38802125 0.54245998 0.30114053 0.50831158 0.67761321
 0.37078093 0.42828559 0.427201   0.28464278 0.45419297 0.70592346
 0.21666873 0.43761955] [1.69566496 1.89630313 1.92592022 1.9903169  1.41874627 1.68845044
 1.52119435 1.60976629 1.30987598 1.74857376 1.43957975 1.18709009
 1.58569177 1.72667349 1.27540568 1.67079914 1.60749966 1.07606987
 2.00287615 1.73861869]
Mean:  0.4151486315350512 1.6057558305116024
Std:  0.13730750625988206 0.2536751588192528
Min:  0.19460438689915904 1.076069874098172
Max:  0.7059234605862461 2.0028761529796264
Test set RMSE= 1.9408868017567749  and R2= -0.10470427484857536
Exp. validation set RMSE= 1.8830539435328648  and R2= 0.2574841215449615

GP and Morgan count:

R2 and RMSE for dataset  0 :  [0.41528776 0.16630752 0.28239281 0.13356939 0.48297204 0.25321549
 0.54170538 0.34008633 0.5529457  0.2334789  0.52021401 0.63736823
 0.38791146 0.40933098 0.4169603  0.26444415 0.4097904  0.66923961
 0.17281004 0.45010256] [1.68223539 1.92932803 1.96395994 2.10643594 1.45762196 1.8323791
 1.53022844 1.67162255 1.29477937 1.83126454 1.42204886 1.25900654
 1.56395754 1.75506314 1.28675625 1.69422306 1.67160808 1.14121384
 2.05818302 1.71921457]
Mean:  0.3870066535412907 1.6435565078724195
Std:  0.15098187546926167 0.26619310789417766
Min:  0.13356939357561648 1.1412138414130621
Max:  0.6692396109631091 2.10643593858646
Test set RMSE= 1.8926358153090224  and R2= -0.05046050715262407
Exp. validation set RMSE= 1.8853238763696625  and R2= 0.25569290674501277

GP and Rdkit:

R2 and RMSE for dataset  0 :  [0.49354502 0.2635573  0.35841027 0.28973678 0.52742358 0.28080462
 0.57248274 0.32311609 0.47241595 0.3088277  0.5368186  0.68361393
 0.43957599 0.44561201 0.43700669 0.27265502 0.41613681 0.68133422
 0.21752022 0.48338215] [1.56561903 1.81331232 1.8570258  1.90717711 1.39355434 1.79821296
 1.47795332 1.69297971 1.40657036 1.73893037 1.39722483 1.1759903
 1.4964982  1.70030784 1.26444179 1.68474037 1.66259654 1.12015469
 2.00178728 1.66637959]
Mean:  0.42519878521845794 1.591072837540907
Std:  0.13195975200986593 0.23760697364984987
Min:  0.21752021881413885 1.120154685041349
Max:  0.6836139282500426 2.0017872806400487
Test set RMSE= 1.9644301131309572  and R2= -0.13166735172308064
Exp. validation set RMSE= 1.9909021219629164  and R2= 0.16999627477527457

GP and Opt2:

R2 and RMSE for dataset  0 :  [0.41045941 0.26264445 0.30209277 0.16299101 0.50491655 0.26964774
 0.54857708 0.36809305 0.48378196 0.28316182 0.54495687 0.65988377
 0.425123   0.36789469 0.49908462 0.16619007 0.39817403 0.67650627
 0.16803484 0.42166733] [1.68916676 1.81443581 1.93681474 2.07036266 1.42635323 1.81210712
 1.51871293 1.63576617 1.39133665 1.77092265 1.38489555 1.21929484
 1.51567226 1.81557991 1.19269502 1.80383285 1.68797802 1.12860824
 2.0641152  1.76310468]
Mean:  0.39619406735154133 1.6320877654847308
Std:  0.14860072253700568 0.2690728792851789
Min:  0.1629910063033445 1.1286082390968286
Max:  0.6765062711473665 2.0703626557474735
Test set RMSE= 2.0047297174469327  and R2= -0.17857514369937233
Exp. validation set RMSE= 1.884907464254564  and R2= 0.2560216611481796

GP and Selfies:

R2 and RMSE for dataset  0 :  [-0.09483575 -0.00462788 -0.02177605 -0.10347526 -0.03709864 -0.04815326
 -0.04393795 -0.09124944 -0.02694282 -0.02623992 -0.00213878 -0.02463137
 -0.00815833 -0.01305072 -0.10040483 -0.00506757 -0.02394691 -0.02332202
 -0.04895209 -0.03897601] [1.93997368 2.07295412 2.08004398 2.02997301 2.06834037 2.34580575
 2.36103061 2.14959516 2.02577387 2.0565094  2.02932601 2.11630805
 1.91066372 2.01202526 1.81692446 1.98043408 2.14563774 2.01427702
 2.03934576 2.09747802]
Mean:  -0.039349279758311775 2.0646210035908927
Std:  0.03208077253826873 0.12296782426347982
Min:  -0.10347525743646391 1.8169244580425676
Max:  -0.0021387832699619747 2.3610306075023644
Test set RMSE= 1.8466185312615795  and R2= 3.892441924335799e-13
Exp. validation set RMSE= 2.224217538662914  and R2= -0.03594033891235027

GP and Smiles:

R2 and RMSE for dataset  0 :  [-1.80613664e-01 -1.14110822e-01  2.00905418e-02  1.17746298e-01
 -1.74485545e-03 -5.76237574e-02 -2.38570370e-02 -1.99400402e-02
 -1.56087409e-02 -1.90623609e-01 -8.67955935e-06 -1.52925756e-02
 -1.67276560e-02  5.75789172e-03 -3.06080407e-02 -2.07971690e-01
 -2.57467033e-02 -1.33789208e-03 -2.00663401e-02 -4.48903187e-03] [2.39039446 2.23032077 2.29499746 2.12558319 2.02892712 2.1806365
 2.28719688 2.07817434 1.95154625 2.28231862 2.05301737 2.10664165
 2.01567334 2.27701938 1.7107793  2.17115655 2.20369452 1.98564044
 2.2855755  2.32360249]
Mean:  -0.03913882010965143 2.1491448073255706
Std:  0.07614727447880323 0.15804418262459866
Min:  -0.20797168981559566 1.7107792969773907
Max:  0.1177462984890163 2.3903944572583464
Test set RMSE= 1.846634552800752  and R2= -1.7352374672263693e-05
Exp. validation set RMSE= 2.25770639482441  and R2= -0.06737038502701154

GP and Init:

R2 and RMSE for dataset  0 :  [0.14925404 0.66245183 0.3375792  0.15912747 0.54367339 0.14117016
 0.59310867 0.05717029 0.55028602 0.52129818 0.57624708 0.53702942
 0.26552682 0.43235519 0.31586686 0.38421405 0.39747306 0.74599691
 0.47868053 0.39208571] [1.71009919 1.20158627 1.67479493 1.77204214 1.37198661 2.12340685
 1.4740204  1.99807271 1.34055929 1.40455511 1.31960586 1.42256382
 1.63082686 1.50610819 1.43261978 1.55016401 1.64590936 1.00353503
 1.4376892  1.60441176]
Mean:  0.4120297442620832 1.5312278686219587
Std:  0.18341271072820997 0.2501853654033293
Min:  0.05717028908516586 1.0035350325950994
Max:  0.7459969105603097 2.123406848368374
Test set RMSE= 1.4269997446070422  and R2= 0.40283628413238615
Exp. validation set RMSE= 1.96264350978536  and R2= 0.1933909923990288

GP and Var:

R2 and RMSE for dataset  0 :  [ 0.1615501   0.62945115  0.32910832  0.21326012  0.53135357  0.17658443
  0.57297149 -0.03724752  0.5578685   0.54168668  0.5061946   0.49424832
  0.34444683  0.54417386  0.33917092  0.3700254   0.39539464  0.69368146
  0.48337406  0.31562581] [1.69769595 1.25895385 1.68546936 1.71405419 1.39038358 2.07916612
 1.51005481 2.09573256 1.32920988 1.37431879 1.42451099 1.48683826
 1.54072045 1.34963846 1.40800825 1.56792138 1.6487457  1.10204589
 1.43120269 1.70232105]
Mean:  0.4081461376561052 1.5398496099391987
Std:  0.1775271803423536 0.2410841913533161
Min:  -0.03724751621258138 1.1020458883582138
Max:  0.693681462214685 2.095732557236884
Test set RMSE= 1.4085571219501651  and R2= 0.4181720921417793
Exp. validation set RMSE= 1.9215350240576787  and R2= 0.22682672826779116

GP and Cor:

R2 and RMSE for dataset  0 :  [0.16250124 0.5686104  0.3482697  0.27310483 0.56273912 0.14205313
 0.55660779 0.11126369 0.43399855 0.46554388 0.48712824 0.57968575
 0.35140223 0.39694883 0.30912413 0.37489667 0.36224101 0.57870265
 0.47412138 0.40091596] [1.69673274 1.35838199 1.6612256  1.64757372 1.34301941 2.12231503
 1.53871541 1.93990794 1.50392649 1.48409695 1.45175149 1.35544568
 1.53252515 1.55236884 1.43966233 1.56184764 1.69334704 1.2924311
 1.44396211 1.59271669]
Mean:  0.3969929580198415 1.5605976680803388
Std:  0.14138476123242064 0.19583019046484382
Min:  0.11126368535424136 1.2924311005363793
Max:  0.5796857506675335 2.1223150312025014
Test set RMSE= 1.6579347204877648  and R2= 0.1939156781821576
Exp. validation set RMSE= 1.7808846058450363  and R2= 0.33587201287412094

GP and Opt:

R2 and RMSE for dataset  0 :  [0.6689661  0.68628073 0.5686372  0.47810748 0.70267341 0.39197672
 0.72527432 0.48608212 0.55558336 0.54346406 0.731601   0.76726629
 0.59544369 0.70311391 0.25970189 0.30464372 0.50550766 0.76215102
 0.60856506 0.56852829] [1.06673757 1.15839767 1.35149986 1.39604527 1.10746205 1.78665112
 1.21119453 1.47516813 1.33264044 1.37165134 1.05021528 1.00861335
 1.21034555 1.08921285 1.49026653 1.6472763  1.49106805 0.9710994
 1.24578425 1.35167045]
Mean:  0.5806784008512152 1.2906499996680938
Std:  0.14203960933210535 0.21332165123039878
Min:  0.25970188643887104 0.9710994003537786
Max:  0.7672662858413611 1.7866511195278505
Test set RMSE= 1.304087666250623  and R2= 0.5012772313000012
Exp. validation set RMSE= 1.8290542665963756  and R2= 0.2994592478736501


'''    
    