#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:25:28 2020

@author: armi tiihonen
"""

from Functions_downselection_training_RF import define_groups_yvalue
from Downselection_xgb import XGB_train_test_newdata
from Downselection_gp import GP_train_test_newdata
from Main_downselection import RF_train_test_newdata, fetch_csv

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
    
    X_morgan_train = fetch_csv(path_x + 'x_morganfromdmpnn_train_seed3', index_col=None)
    X_morgan_train.index = y_refdata_train.index
    X_morgan_train.drop(columns=['smiles'], inplace=True)
    X_morgancount_train = fetch_csv(path_x + 'x_morgancountfromdmpnn_train_seed3', index_col=None)
    X_morgancount_train.index = y_refdata_train.index
    X_morgancount_train.drop(columns=['smiles'], inplace=True)
    X_rdkit_train = fetch_csv(path_x + 'x_rdkitfromdmpnn_train_seed3', index_col=None)
    X_rdkit_train.index = y_refdata_train.index
    X_rdkit_train.drop(columns=['smiles'], inplace=True)
    
    X_selfies_train = fetch_csv(path_x + 'x_selfies_train_seed3', index_col=None)
    X_smiles_train = fetch_csv(path_x + 'x_smiles_train_seed3', index_col=None)

    X_morgan_test = fetch_csv(path_x + 'x_morganfromdmpnn_test_seed3', index_col=None)
    X_morgan_test.drop(columns=['smiles'], inplace=True)
    X_morgancount_test = fetch_csv(path_x + 'x_morgancountfromdmpnn_test_seed3', index_col=None)
    X_morgancount_test.drop(columns=['smiles'], inplace=True)
    X_rdkit_test = fetch_csv(path_x + 'x_rdkitfromdmpnn_test_seed3', index_col=None)
    X_rdkit_test.drop(columns=['smiles'], inplace=True)
    X_selfies_test = fetch_csv(path_x + 'x_selfies_test_seed3', index_col=None)
    X_smiles_test = fetch_csv(path_x + 'x_smiles_test_seed3', index_col=None)

    X_morgan_newdata = fetch_csv(path_x + 'x_morganfromdmpnn_newdata', index_col=None)
    X_morgan_newdata.drop(columns=['smiles'], inplace=True)
    X_morgancount_newdata = fetch_csv(path_x + 'x_morgancountfromdmpnn_newdata', index_col=None)
    X_morgancount_newdata.drop(columns=['smiles'], inplace=True)
    X_rdkit_newdata = fetch_csv(path_x + 'x_rdkitfromdmpnn_newdata', index_col=None)
    X_rdkit_newdata.drop(columns=['smiles'], inplace=True)
    X_selfies_newdata = fetch_csv(path_x + 'x_selfies_newdata', index_col=None)
    X_smiles_newdata = fetch_csv(path_x + 'x_smiles_newdata', index_col=None)
    
    # Drop one mol with nans in morgan and rdkit datasets.    
    y_refdata_train = y_refdata_train.drop(X_morgan_train[(X_morgan_train=='Invalid SMILES').any(axis=1)].index)
    X_morgancount_train=X_morgancount_train.drop(X_morgan_train[(X_morgan_train=='Invalid SMILES').any(axis=1)].index)
    X_rdkit_train=X_rdkit_train.drop(X_morgan_train[(X_morgan_train=='Invalid SMILES').any(axis=1)].index)
    #X_smiles_train=X_smiles_test.drop(X_morgan_test[X_morgan_test.isnull().any(axis=1)].index)
    #X_selfies_train=X_selfies_test.drop(X_morgan_test[X_morgan_test.isnull().any(axis=1)].index)
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
                    ]
    
    all_refdata_y = [y_refdata_train, y_refdata_test, y_refdata_newdata, groups_refdata_train,
                     y_smiles_train, y_smiles_test, y_smiles_newdata, groups_smiles_train,
                     y_selfies_train, y_selfies_test, y_selfies_newdata, groups_selfies_train
                     ]
    
    return all_refdata_x, all_refdata_y

def fetch_downselected_data(path_x = './Data/Downselection_data_files/',
                  path_y = './Data/Downselection_data_files/'):
    
    X_init_train = fetch_csv(path_x + 'x_init_train_seed3', index_col=None)
    X_var_train = fetch_csv(path_x + 'x_var_train_seed3', index_col=None)
    X_cor_train = fetch_csv(path_x + 'x_cor_train_seed3', index_col=None)
    X_opt_train = fetch_csv(path_x + 'x_opt_train_seed3', index_col=None)

    X_init_test = fetch_csv(path_x + 'x_init_test_seed3', index_col=None)
    X_var_test = fetch_csv(path_x + 'x_var_test_seed3', index_col=None)
    X_cor_test = fetch_csv(path_x + 'x_cor_test_seed3', index_col=None)
    X_opt_test = fetch_csv(path_x + 'x_opt_test_seed3', index_col=None)

    X_init_newdata = fetch_csv(path_x + 'x_init_newdata', index_col=None)
    X_var_newdata = fetch_csv(path_x + 'x_var_newdata', index_col=None)
    X_cor_newdata = fetch_csv(path_x + 'x_cor_newdata', index_col=None)
    X_opt_newdata = fetch_csv(path_x + 'x_opt_newdata', index_col=None)
    
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
                    X_morgancount_train, X_morgancount_test, X_morgancount_newdata
                    ] = all_refdata_x
    
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
    cv_results_rf_morgan, test_rf_results_morgan, newdata_rf_results_morgan = RF_train_test_newdata(
            [X_morgan_train, X_morgan_test, X_morgan_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, ho_rf_morgan,
            saveas='./Results/rf_morgan_seed' + str(random_state),
            random_state=random_state)
    print('\nRF and Morgan count:\n')
    cv_results_rf_morgancount, test_results_rf_morgancount, newdata_results_rf_morgancount = RF_train_test_newdata(
            [X_morgancount_train, X_morgancount_test, X_morgancount_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, ho_rf_morgancount,
            saveas='./Results/rf_morgancount_seed'  + str(random_state),
            random_state=random_state)
    print('\nRF and Rdkit:\n')
    cv_results_rf_rdkit, test_results_rf_rdkit, newdata_results_rf_rdkit = RF_train_test_newdata(
            [X_rdkit_train, X_rdkit_test, X_rdkit_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, ho_rf_rdkit,
            saveas='./Results/rf_rdkit_seed'  + str(random_state),
            random_state=random_state)
    print('\nRF and Selfies:\n')
    cv_results_rf_selfies, test_results_rf_selfies, newdata_results_rf_selfies = RF_train_test_newdata(
            [X_selfies_train, X_selfies_test, X_selfies_newdata],
            [y_selfies_train, y_selfies_test, y_selfies_newdata], groups_selfies_train, ho_rf_selfies,
            saveas='./Results/rf_selfies_seed' + str(random_state),
            random_state=random_state)
    print('\nRF and Smiles:\n')
    cv_results_rf_smiles, test_results_rf_smiles, newdata_results_rf_smiles = RF_train_test_newdata(
            [X_smiles_train, X_smiles_test, X_smiles_newdata],
            [y_smiles_train, y_smiles_test, y_smiles_newdata], groups_smiles_train, ho_rf_smiles,
            saveas='./Results/rf_smiles_seed' + str(random_state),
            random_state=random_state)
    ###
    print('\nRF and Init:\n')
    cv_results_rf_init, test_results_rf_init, newdata_results_rf_init = RF_train_test_newdata(
            [X_init_train, X_init_test, X_init_newdata],
            [y_train, y_test, y_newdata], groups_train, ho_rf_init,
            saveas='./Results/rf_init_seed' + str(random_state),
            random_state=random_state)
    print('\nRF and Var:\n')
    cv_results_rf_var, test_results_rf_var, newdata_results_rf_var = RF_train_test_newdata(
            [X_var_train, X_var_test, X_var_newdata],
            [y_train, y_test, y_newdata], groups_train, ho_rf_var,
            saveas='./Results/rf_var_seed'  + str(random_state),
            random_state=random_state)
    print('\nRF and Cor:\n')
    cv_results_rf_cor, test_results_rf_cor, newdata_results_rf_cor = RF_train_test_newdata(
            [X_cor_train, X_cor_test, X_cor_newdata],
            [y_train, y_test, y_newdata], groups_train, ho_rf_cor,
            saveas='./Results/rf_cor_seed'  + str(random_state),
            random_state=random_state)
    print('\nRF and Opt:\n')
    cv_results_rf_opt, test_results_rf_opt, newdata_results_rf_opt = RF_train_test_newdata(
            [X_opt_train, X_opt_test, X_opt_newdata],
            [y_train, y_test, y_newdata], groups_train, ho_rf_opt,
            saveas='./Results/rf_opt_seed'  + str(random_state),
            random_state=random_state)
    ##########################################################################
    # Train XGB models.
    
    print('\nXGB and Morgan:\n')
    cv_results_xgb_morgan, test_xgb_results_morgan, newdata_xgb_results_morgan = XGB_train_test_newdata(
            [X_morgan_train, X_morgan_test, X_morgan_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, ho_xgb_morgan,
            saveas='./Results/xgb_morgan_seed' + str(random_state),
            random_state=random_state)
    print('\nXGB and Morgan count:\n')
    cv_results_xgb_morgancount, test_results_xgb_morgancount, newdata_results_xgb_morgancount = XGB_train_test_newdata(
            [X_morgancount_train, X_morgancount_test, X_morgancount_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, ho_xgb_morgancount,
            saveas='./Results/xgb_morgancount_seed'  + str(random_state),
            random_state=random_state)
    print('\nXGB and Rdkit:\n')
    cv_results_xgb_rdkit, test_results_xgb_rdkit, newdata_results_xgb_rdkit = XGB_train_test_newdata(
            [X_rdkit_train, X_rdkit_test, X_rdkit_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, ho_xgb_rdkit,
            saveas='./Results/xgb_rdkit_seed'  + str(random_state),
            random_state=random_state)
    print('\nXGB and Selfies:\n')
    cv_results_xgb_selfies, test_results_xgb_selfies, newdata_results_xgb_selfies = XGB_train_test_newdata(
            [X_selfies_train, X_selfies_test, X_selfies_newdata],
            [y_selfies_train, y_selfies_test, y_selfies_newdata], groups_selfies_train, ho_xgb_selfies,
            saveas='./Results/xgb_selfies_seed' + str(random_state),
            random_state=random_state)
    print('\nXGB and Smiles:\n')
    cv_results_xgb_smiles, test_results_xgb_smiles, newdata_results_xgb_smiles = XGB_train_test_newdata(
            [X_smiles_train, X_smiles_test, X_smiles_newdata],
            [y_smiles_train, y_smiles_test, y_smiles_newdata], groups_smiles_train, ho_xgb_smiles,
            saveas='./Results/xgb_smiles_seed' + str(random_state),
            random_state=random_state)
    ###
    print('\nXGB and Init:\n')
    cv_results_xgb_init, test_results_xgb_init, newdata_results_xgb_init = XGB_train_test_newdata(
            [X_init_train, X_init_test, X_init_newdata],
            [y_train, y_test, y_newdata], groups_train, ho_xgb_init,
            saveas='./Results/xgb_init_seed' + str(random_state),
            random_state=random_state)
    print('\nXGB and Var:\n')
    cv_results_xgb_var, test_results_xgb_var, newdata_results_xgb_var = XGB_train_test_newdata(
            [X_var_train, X_var_test, X_var_newdata],
            [y_train, y_test, y_newdata], groups_train, ho_xgb_var,
            saveas='./Results/xgb_var_seed'  + str(random_state),
            random_state=random_state)
    print('\nXGB and Cor:\n')
    cv_results_xgb_cor, test_results_xgb_cor, newdata_results_xgb_cor = XGB_train_test_newdata(
            [X_cor_train, X_cor_test, X_cor_newdata],
            [y_train, y_test, y_newdata], groups_train, ho_xgb_cor,
            saveas='./Results/xgb_cor_seed'  + str(random_state),
            random_state=random_state)
    print('\nXGB and Opt:\n')
    cv_results_xgb_opt, test_results_xgb_opt, newdata_results_xgb_opt = XGB_train_test_newdata(
            [X_opt_train, X_opt_test, X_opt_newdata],
            [y_train, y_test, y_newdata], groups_train, ho_xgb_opt,
            saveas='./Results/xgb_opt_seed'  + str(random_state),
            random_state=random_state)
    
    ##############################################################################
    # Train GP models.
    
    print('Running GP takes longer because kernel optimization is integrated into the implementation.\n')
    print('\nGP and Morgan:\n')
    cv_results_gp_morgan, test_gp_results_morgan, newdata_gp_results_morgan = GP_train_test_newdata(
            [X_morgan_train, X_morgan_test, X_morgan_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, None,
            saveas='./Results/gp_morgan_seed' + str(random_state),
            random_state=random_state)
    print('\nGP and Morgan count:\n')
    cv_results_gp_morgancount, test_results_gp_morgancount, newdata_results_gp_morgancount = GP_train_test_newdata(
            [X_morgancount_train, X_morgancount_test, X_morgancount_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, None,
            saveas='./Results/gp_morgancount_seed'  + str(random_state),
            random_state=random_state)
    print('\nGP and Rdkit:\n')
    cv_results_gp_rdkit, test_results_gp_rdkit, newdata_results_gp_rdkit = GP_train_test_newdata(
            [X_rdkit_train, X_rdkit_test, X_rdkit_newdata],
            [y_refdata_train, y_refdata_test, y_refdata_newdata], groups_refdata_train, None,
            saveas='./Results/gp_rdkit_seed'  + str(random_state),
            random_state=random_state)
    print('\nGP and Selfies:\n')
    cv_results_gp_selfies, test_results_gp_selfies, newdata_results_gp_selfies = GP_train_test_newdata(
            [X_selfies_train, X_selfies_test, X_selfies_newdata],
            [y_selfies_train, y_selfies_test, y_selfies_newdata], groups_selfies_train, None,
            saveas='./Results/gp_selfies_seed' + str(random_state),
            random_state=random_state)
    print('\nGP and Smiles:\n')
    cv_results_gp_smiles, test_results_gp_smiles, newdata_results_gp_smiles = GP_train_test_newdata(
            [X_smiles_train, X_smiles_test, X_smiles_newdata],
            [y_smiles_train, y_smiles_test, y_smiles_newdata], groups_smiles_train, None,
            saveas='./Results/gp_smiles_seed' + str(random_state),
            random_state=random_state)
    ###
    print('\nGP and Init:\n')
    cv_results_gp_init, test_results_gp_init, newdata_results_gp_init = GP_train_test_newdata(
            [X_init_train, X_init_test, X_init_newdata],
            [y_train, y_test, y_newdata], groups_train, None,
            saveas='./Results/gp_init_seed' + str(random_state),
            random_state=random_state)
    print('\nGP and Var:\n')
    cv_results_gp_var, test_results_gp_var, newdata_results_gp_var = GP_train_test_newdata(
            [X_var_train, X_var_test, X_var_newdata],
            [y_train, y_test, y_newdata], groups_train, None,
            saveas='./Results/gp_var_seed'  + str(random_state),
            random_state=random_state)
    print('\nGP and Cor:\n')
    cv_results_gp_cor, test_results_gp_cor, newdata_results_gp_cor = GP_train_test_newdata(
            [X_cor_train, X_cor_test, X_cor_newdata],
            [y_train, y_test, y_newdata], groups_train, None,
            saveas='./Results/gp_cor_seed'  + str(random_state),
            random_state=random_state)
    print('\nGP and Opt:\n')
    cv_results_gp_opt, test_results_gp_opt, newdata_results_gp_opt = GP_train_test_newdata(
            [X_opt_train, X_opt_test, X_opt_newdata],
            [y_train, y_test, y_newdata], groups_train, None,
            saveas='./Results/gp_opt_seed'  + str(random_state),
            random_state=random_state)
    
'''
Print-outs from the code:

RF and Morgan:

R2 and RMSE for dataset  0 :  [0.35324023 0.11157975 0.22234895 0.23706851 0.42851296 0.35721269
 0.61277975 0.37337339 0.35830066 0.40675167 0.53552869 0.70608994
 0.48038213 0.49947573 0.39539813 0.1683513  0.41374536 0.71358585
 0.26959521 0.46971269] [1.7692417  1.99164695 2.04447411 1.97662432 1.53246681 1.70000942
 1.40657508 1.62891745 1.55124869 1.61104359 1.39916904 1.13344979
 1.44098637 1.61559767 1.31033387 1.80149358 1.66599799 1.06195844
 1.93402984 1.68828146]
Mean:  0.4056516802928821 1.6131773084683871
Std:  0.1574390650898225 0.26405700034790947
Min:  0.11157975419664101 1.0619584367632287
Max:  0.7135858485301332 2.0444741077194157
Test set RMSE= 2.1405469634883425  and R2= -0.3436778014367048
New dataset RMSE= 1.7086764410532187  and R2= 0.3886359665371606

RF and Morgan count:

R2 and RMSE for dataset  0 :  [0.44364831 0.20899232 0.27746212 0.2471814  0.52566026 0.29673909
 0.63508369 0.31752652 0.42792871 0.46863736 0.64774576 0.77437857
 0.41122445 0.48511582 0.46172944 0.22987911 0.47006865 0.65758501
 0.31392153 0.54736014] [1.64093121 1.87928861 1.97069559 1.96348023 1.3961518  1.7781808
 1.36546486 1.69995549 1.46467311 1.52470048 1.21848127 0.99308229
 1.53388471 1.63860933 1.23636738 1.73357332 1.58394878 1.16114557
 1.87442582 1.55978821]
Mean:  0.44239341220673467 1.5608414439170968
Std:  0.1544131686957441 0.2664997080416677
Min:  0.20899232364280207 0.99308229114283
Max:  0.7743785666959231 1.9706955861222353
Test set RMSE= 1.9656871235737026  and R2= -0.13311609025907822
New dataset RMSE= 1.72582667832005  and R2= 0.3763016732629928

RF and Rdkit:

R2 and RMSE for dataset  0 :  [0.3056079  0.33285317 0.38583261 0.32318928 0.62464988 0.48953789
 0.67656577 0.27276633 0.40077319 0.48930945 0.65686048 0.62471621
 0.50536977 0.39966511 0.44209429 0.40415693 0.39506617 0.67361899
 0.26900347 0.48860039] [1.83323464 1.72589291 1.81690664 1.86172263 1.24195587 1.51495396
 1.2855142  1.75481625 1.4990332  1.49474786 1.2026136  1.28078125
 1.40591207 1.76936504 1.25871563 1.52485556 1.69233082 1.13363364
 1.93481311 1.65794237]
Mean:  0.4580118651257017 1.5444870622546434
Std:  0.13050261562711285 0.24428504380003574
Min:  0.2690034718465548 1.1336336377907865
Max:  0.6765657710780348 1.9348131050823247
Test set RMSE= 1.39030530158433  and R2= 0.4331528353039604
New dataset RMSE= 2.38331769949605  and R2= -0.1894442717086584

RF and Selfies:

R2 and RMSE for dataset  0 :  [-4.72807586e-01  1.28024753e-01 -1.16545121e-01 -3.15756713e-01
 -8.11274581e-02 -1.25119144e-01  6.97451403e-02 -3.24845772e-01
 -5.23383202e-02 -4.32725660e-04 -8.42065931e-02  2.47765320e-01
 -1.44021114e-01 -2.10786810e-02 -1.76027127e-01 -7.26447182e-02
  1.33478225e-01  2.84376108e-02 -5.51833550e-03  1.45252852e-01] [2.25006135 1.93125293 2.17436675 2.21664814 2.11178852 2.43040645
 2.22876997 2.36852144 2.05066881 2.0304869  2.11078456 1.81330866
 2.0353397  2.01998172 1.87831876 2.04592972 1.97381943 1.96267498
 1.99667801 1.90245104]
Mean:  -0.06198827531445179 2.0766128924354383
Std:  0.16991441698623452 0.15714476518984463
Min:  -0.472807585817804 1.813308662836293
Max:  0.24776532035782917 2.4304064484817225
Test set RMSE= 1.8647375697991115  and R2= -0.019720294492755697
New dataset RMSE= 2.074015777669788  and R2= 0.09924986814043768

RF and Smiles:

R2 and RMSE for dataset  0 :  [ 0.38308291 -0.09964291  0.18322415  0.15202925  0.27876978  0.17548889
  0.41321967  0.04021111  0.31370945  0.31326979  0.46000305  0.46446107
  0.24051927  0.38904605  0.31181041 -0.19620597  0.32329851  0.51853942
  0.11549196  0.37847501] [1.72794165 2.21579191 2.0952733  2.08387564 1.72157063 1.92537769
 1.73149668 2.01596282 1.60424112 1.73333339 1.50864269 1.52999701
 1.74211256 1.78494521 1.39798101 2.16055708 1.78990464 1.37686275
 2.12829711 1.82775732]
Mean:  0.2579400429553035 1.8050961114676045
Std:  0.1819854911922821 0.24515122391837485
Min:  -0.19620597362464975 1.3768627516574932
Max:  0.5185394191995317 2.215791911596066
Test set RMSE= 1.8016904443083193  and R2= 0.04806790114020276
New dataset RMSE= 1.8315244479943749  and R2= 0.2975657761690005

RF and Init:

R2 and RMSE for dataset  0 :  [0.3182764  0.62954513 0.39785816 0.26466231 0.29064215 0.16805094
 0.52915618 0.3001113  0.33664034 0.45540162 0.36574063 0.38300059
 0.32354466 0.32397885 0.16993702 0.25946456 0.3680365  0.68743985
 0.23366259 0.44481329] [1.5308249  1.25879419 1.59677624 1.65711397 1.71058502 2.08991209
 1.58563314 1.72150841 1.62814232 1.49811248 1.61443785 1.64224306
 1.56509048 1.64360697 1.57803325 1.69994819 1.68563554 1.11321703
 1.74310301 1.5332543 ]
Mean:  0.36249815311910116 1.6047986213136163
Std:  0.13193774773709718 0.18536653418976626
Min:  0.16805094256102304 1.1132170260401504
Max:  0.6874398525627456 2.089912091824589
Test set RMSE= 1.1965045888440924  and R2= 0.58016914043256
New dataset RMSE= 2.199879962074483  and R2= -0.013393677475660892

RF and Var:

R2 and RMSE for dataset  0 :  [0.38218409 0.63018068 0.3453451  0.31084226 0.2742482  0.27882303
 0.50457917 0.29126392 0.16316969 0.41755513 0.37870902 0.34348273
 0.33551604 0.39228836 0.05395678 0.2472965  0.31313895 0.61342098
 0.33218534 0.39854688] [1.45730648 1.25771394 1.66494875 1.60423609 1.73023876 1.94581066
 1.62649003 1.73235514 1.82867503 1.54929337 1.59784781 1.69401846
 1.55117978 1.55835576 1.68467494 1.71385758 1.75732515 1.23803288
 1.62720029 1.59586279]
Mean:  0.35033664294275735 1.6207711838430332
Std:  0.12907091976027885 0.1631536607390511
Min:  0.053956780446867136 1.2380328760579873
Max:  0.6301806804676722 1.9458106583239292
Test set RMSE= 1.1893163228864636  and R2= 0.5851984410896836
New dataset RMSE= 2.1192762027840764  and R2= 0.059507480510009314

RF and Cor:

R2 and RMSE for dataset  0 :  [0.44885599 0.60064301 0.343755   0.33467184 0.35753398 0.33458858
 0.49800581 0.41940865 0.20030442 0.36750817 0.43065155 0.43034348
 0.41704058 0.44000334 0.25613455 0.4095711  0.31583366 0.57750993
 0.45615759 0.51066803] [1.37642927 1.30697633 1.66696953 1.5762566  1.62793499 1.86906659
 1.6372448  1.56794181 1.78764048 1.61448385 1.52959678 1.57798037
 1.45291112 1.49592752 1.49385286 1.51791189 1.75387457 1.2942593
 1.46841758 1.43944957]
Mean:  0.4074594624433817 1.552756290960041
Std:  0.09604567037994179 0.14550350235860043
Min:  0.20030442486301459 1.2942592967413802
Max:  0.6006430108188192 1.8690665948059648
Test set RMSE= 1.2577357418209227  and R2= 0.5360999424475581
New dataset RMSE= 1.9705522009764809  and R2= 0.1868772527269683

RF and Opt:

R2 and RMSE for dataset  0 :  [0.52305829 0.68251867 0.58806293 0.41982748 0.53220404 0.51003086
 0.58241763 0.54428919 0.45004736 0.4792154  0.57592165 0.572026
 0.57734374 0.5491394  0.25092279 0.42054666 0.43684507 0.69855641
 0.57989423 0.5056082 ] [1.28042459 1.16532261 1.320718   1.4719313  1.38912142 1.60385099
 1.49325973 1.38911859 1.48245145 1.4649922  1.32011248 1.36774064
 1.23712487 1.3422672  1.49907692 1.50373738 1.59122511 1.09324138
 1.29060208 1.44687259]
Mean:  0.5239238000929644 1.3876595778189458
Std:  0.09667685332985525 0.13076542937689245
Min:  0.2509227921813655 1.0932413815147357
Max:  0.6985564139176359 1.603850991995379
Test set RMSE= 1.093782292173194  and R2= 0.6491613775156464
New dataset RMSE= 1.9426782806743375  and R2= 0.20971817859834951

XGB and Morgan:

R2 and RMSE for dataset  0 :  [ 0.21985117  0.02381697  0.09225192  0.21886746  0.27544442  0.25144668
  0.51860374  0.12157204  0.1912278   0.38225538  0.4004459   0.59224394
  0.16700813  0.30770334  0.19891845 -0.19978836  0.29791852  0.72593329
  0.08597442  0.40698811] [1.94314138 2.08770321 2.20887888 2.00006325 1.72553487 1.83454787
 1.56832204 1.92862474 1.7415211  1.64396858 1.58966249 1.33504306
 1.82447626 1.90005761 1.50829256 2.16378987 1.82316127 1.03881552
 2.16351813 1.78534   ]
Mean:  0.26393416574008177 1.7907231346088062
Std:  0.20242708582702204 0.28409894131035474
Min:  -0.19978835603793765 1.038815515826624
Max:  0.725933288654696 2.2088788811370392
Test set RMSE= 2.1559292089668745  and R2= -0.3630588721632062
New dataset RMSE= 2.0426940250355816  and R2= 0.1262506618979179

XGB and Morgan count:

R2 and RMSE for dataset  0 :  [ 0.22301611 -0.09742701  0.23168645  0.24869823  0.52038025  0.20319998
  0.63264011  0.2530513   0.17316575  0.3799842   0.403531    0.69868789
  0.19526602  0.29684211  0.46315044  0.133043    0.38075513  0.81624715
  0.10391416  0.52159558] [1.93919587 2.21355825 2.03216272 1.96150116 1.40390078 1.89274605
 1.37002899 1.77844339 1.76086014 1.64698789 1.58556729 1.14763387
 1.79326306 1.91490432 1.23473433 1.83933799 1.71223172 0.85060381
 2.14218104 1.60356596]
Mean:  0.33907139189782437 1.6911704315429088
Std:  0.2155976538226699 0.3376319065780989
Min:  -0.09742701269626042 0.8506038098341254
Max:  0.8162471514147736 2.2135582528404507
Test set RMSE= 2.0309870289254794  and R2= -0.2096505312796324
New dataset RMSE= 2.102784542900326  and R2= 0.07408786812322177

XGB and Rdkit:

R2 and RMSE for dataset  0 :  [0.29133243 0.06483268 0.28377029 0.27777567 0.4417801  0.53622114
 0.70850143 0.27062904 0.17824682 0.26159245 0.55845171 0.28430257
 0.2832593  0.44753227 0.19778449 0.052406   0.47679423 0.60444222
 0.33787804 0.36705476] [1.8519828  2.04337367 1.96207407 1.92316887 1.51457417 1.44401982
 1.22039973 1.757393   1.75544137 1.79736836 1.36420565 1.76872286
 1.69238383 1.69736057 1.5093597  1.92297621 1.5738654  1.2480019
 1.84140941 1.844473  ]
Mean:  0.34622938317125185 1.6866277201736506
Std:  0.167934949107036 0.23090879413653156
Min:  0.05240600205132495 1.2203997316700348
Max:  0.7085014302022667 2.0433736711196486
Test set RMSE= 1.896084536463071  and R2= -0.054292249095125644
New dataset RMSE= 3.0916768140770294  and R2= -1.0015590197109963

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
New dataset RMSE= 2.8888070088413302  and R2= -0.7475003879581577

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
New dataset RMSE= 2.1271444309600263  and R2= 0.0525109911194509

XGB and Init:

R2 and RMSE for dataset  0 :  [ 0.27183543  0.58154211  0.07897737  0.24880151  0.28498835 -0.01808532
  0.37391035  0.27603054  0.00488431  0.35801323  0.29778918  0.56942526
  0.10850721  0.12191058  0.3740546   0.17102532  0.3038131   0.31763374
  0.19041396  0.39681417] [1.58210799 1.33786705 1.97483042 1.6748901  1.71738845 2.31191434
 1.82844767 1.75087355 1.9941353  1.626557   1.6987194  1.37189015
 1.79671338 1.87321318 1.37034163 1.79859505 1.76921496 1.644833
 1.7916144  1.59815988]
Mean:  0.2656142497618269 1.725615345518687
Std:  0.15661894937009138 0.22201625848352954
Min:  -0.018085320298312002 1.3378670478307844
Max:  0.5815421106448464 2.311914343475151
Test set RMSE= 1.5272307060046773  and R2= 0.3160018682218344
New dataset RMSE= 1.6029589633772359  and R2= 0.46194703215683885

XGB and Var:

R2 and RMSE for dataset  0 :  [ 0.24134226  0.46044944  0.13411055  0.18066168  0.17860309  0.08738952
  0.4792623   0.11754138 -0.33349481 -0.12996165  0.13513386  0.41157575
  0.22200392  0.2772839  -0.58866206  0.15696596  0.16325922  0.41574489
  0.13988815  0.21637358] [1.61489504 1.51915872 1.91481086 1.74920454 1.8407233  2.18888214
 1.6675305  1.93304442 2.30841293 2.15793371 1.88522129 1.60376356
 1.6784507  1.69942381 2.18311388 1.81378296 1.93960434 1.52199875
 1.84667515 1.82158396]
Mean:  0.14827354553883662 1.8444107282013746
Std:  0.24931998620075652 0.22149875838384755
Min:  -0.5886620640051075 1.5191587192447882
Max:  0.4792622966143745 2.308412933931831
Test set RMSE= 1.5838679589159732  and R2= 0.2643291169264953
New dataset RMSE= 1.8817733061959063  and R2= 0.2584937264117001

XGB and Cor:

R2 and RMSE for dataset  0 :  [ 0.57676706  0.6284889   0.01995555  0.10623145  0.31853559  0.13817725
  0.57894451  0.25870081  0.19840754 -0.36013642  0.53343738  0.4872574
  0.63366298  0.33953113 -0.07098882  0.13343198  0.01253893  0.28866441
  0.35576831  0.50598183] [1.20617711 1.26058744 2.03712438 1.82692828 1.67661584 2.12710354
 1.49945674 1.77170504 1.78975938 2.36754085 1.38465907 1.49707914
 1.15175548 1.62459088 1.79247495 1.83892531 2.10706373 1.67938515
 1.59821087 1.44632576]
Mean:  0.2841678884514791 1.6841734472539447
Std:  0.2592046085547278 0.3129899517213432
Min:  -0.3601364200688273 1.1517554846225941
Max:  0.6336629792149882 2.367540854957527
Test set RMSE= 1.6250941260423688  and R2= 0.22553345498609667
New dataset RMSE= 1.7829204712788491  and R2= 0.33435271393865607

XGB and Opt:

R2 and RMSE for dataset  0 :  [0.12703836 0.55984146 0.39755716 0.18474977 0.23237657 0.51053359
 0.51355391 0.28713182 0.08026015 0.04418864 0.49405038 0.50133697
 0.11646214 0.5038602  0.1869661  0.20284183 0.08452607 0.73392218
 0.52041627 0.52363049] [1.73228336 1.37211858 1.59717529 1.74483526 1.77945122 1.60302796
 1.61169045 1.73739783 1.91712458 1.98468844 1.44192118 1.47638168
 1.78867924 1.4080558  1.56176237 1.76374182 2.0288067  1.02711098
 1.37893965 1.42025601]
Mean:  0.3402622035482893 1.6187724199418656
Std:  0.2006452960751286 0.2363667539273782
Min:  0.04418864002325196 1.0271109838946801
Max:  0.7339221821195103 2.028806702610671
Test set RMSE= 1.4732337090502798  and R2= 0.36351391158884316
New dataset RMSE= 2.2723296100875547  and R2= -0.08124192729552271
Running GP takes longer because kernel optimization is integrated into the implementation.


GP and Morgan:

R2 and RMSE for dataset  0 :  [0.32209622 0.01764185 0.21413084 0.04668775 0.51170018 0.42808418
 0.55467941 0.5134696  0.43813658 0.21306174 0.57609016 0.64887175
 0.4620137  0.5438412  0.4825351  0.3210298  0.49232704 0.56096549
 0.10340974 0.35716259] [1.81133884 2.09429598 2.05524858 2.20952516 1.4165476  1.60355454
 1.50841304 1.43532302 1.45154671 1.85549316 1.33668011 1.2388763
 1.46623453 1.54233488 1.21223738 1.62775155 1.55032711 1.31479976
 2.14278388 1.85883057]
Mean:  0.3903967448353161 1.636607134980124
Std:  0.1804204040095942 0.3012443138708364
Min:  0.017641846715703347 1.2122373835459623
Max:  0.6488717520124296 2.2095251594166303
Test set RMSE= 2.5225768796068837  and R2= -0.8660979805065108
Exp. validation set RMSE= 1.8314796575227885  and R2= 0.29760013222192405

GP and Morgan count:

R2 and RMSE for dataset  0 :  [0.38024481 0.1216862  0.23860005 0.05526003 0.56388488 0.4254514
 0.5057151  0.51617942 0.49451133 0.34054608 0.49688585 0.65232684
 0.3981237  0.3419244  0.4805188  0.43528715 0.35871118 0.61843278
 0.13712545 0.35654135] [1.73191175 1.98028632 2.02299895 2.19956859 1.33871602 1.60724122
 1.58917806 1.43132029 1.37680155 1.69856142 1.45620999 1.23276599
 1.55085593 1.8525014  1.21459682 1.48448768 1.74244132 1.22573281
 2.10210882 1.85972854]
Mean:  0.39589783978217324 1.6349006722699873
Std:  0.1567839595185058 0.291205795031308
Min:  0.05526002612875003 1.2145968234164437
Max:  0.652326836746002 2.1995685885116703
Test set RMSE= 2.230301690292084  and R2= -0.45872305856883466
Exp. validation set RMSE= 1.9300889457295587  and R2= 0.21992767699596338

GP and Rdkit:

R2 and RMSE for dataset  0 :  [ 0.05805827  0.14696882  0.06460371  0.06836723  0.29625744  0.01228506
  0.19149164  0.16396924  0.25903959  0.01694618  0.27208536  0.11938081
  0.23204546  0.06366363  0.05855426  0.34014513  0.00762227  0.0610783
 -0.00116787  0.21722708] [2.13514656 1.95157653 2.2422656  2.18425703 1.7005711  2.10733456
 2.03247937 1.88150678 1.66691378 2.0738516  1.75158399 1.96195541
 1.7518043  2.20971669 1.63510208 1.60467458 2.16755353 1.92275953
 2.26430443 2.05119586]
Mean:  0.13243108028571354 1.96482766586156
Std:  0.10445258261821903 0.21046094114853034
Min:  -0.0011678714687854974 1.6046745841413599
Max:  0.34014513175851147 2.2643044250452724
Test set RMSE= 1.8459383592658412  and R2= 0.0007365319034998707
Exp. validation set RMSE= 2.222755434639922  and R2= -0.034578822174547996

GP and Selfies:

R2 and RMSE for dataset  0 :  [-0.09483574 -0.00462827 -0.02177605 -0.10347526 -0.03709864 -0.04815327
 -0.04393798 -0.09124932 -0.02694282 -0.02623985 -0.00213878 -0.02463137
 -0.00815833 -0.01305073 -0.02083333 -0.00506757 -0.02394691 -0.02332189
 -0.04895208 -0.03897601] [1.93997367 2.07295452 2.08004398 2.02997301 2.06834037 2.34580576
 2.36103065 2.14959505 2.02577387 2.05650933 2.02932601 2.11630805
 1.91066372 2.01202527 1.75       1.98043408 2.14563774 2.01427689
 2.03934576 2.09747802]
Mean:  -0.0353707104167121 2.0612747872320902
Std:  0.0290533695799071 0.13035156322072083
Min:  -0.10347526265463514 1.7500000000000007
Max:  -0.0021387832699619747 2.361030646800647
Test set RMSE= 1.8466185312615795  and R2= 3.892441924335799e-13
Exp. validation set RMSE= 2.224217538662914  and R2= -0.03594033891235027

GP and Smiles:

R2 and RMSE for dataset  0 :  [-1.80613665e-01 -1.14110820e-01  2.00905441e-02  1.17746297e-01
 -1.74485545e-03 -5.76237484e-02 -2.38570370e-02 -1.99400402e-02
 -1.56087409e-02 -1.90623604e-01 -8.67955935e-06 -1.52925756e-02
 -1.67276560e-02  5.75788533e-03 -3.06080401e-02 -2.07971671e-01
 -2.57467033e-02 -1.33789208e-03 -2.00663401e-02 -4.48903187e-03] [2.39039446 2.23032077 2.29499746 2.1255832  2.02892712 2.1806365
 2.28719688 2.07817434 1.95154625 2.28231862 2.05301737 2.10664165
 2.01567334 2.27701938 1.7107793  2.17115654 2.20369452 1.98564044
 2.2855755  2.32360249]
Mean:  -0.03913881871277073 2.1491448060682727
Std:  0.07614727158075606 0.15804418248938956
Min:  -0.20797167133395278 1.7107792965110655
Max:  0.11774629680741433 2.390394458650726
Test set RMSE= 1.846634552800752  and R2= -1.7352374672263693e-05
Exp. validation set RMSE= 2.25770639482441  and R2= -0.06737038502701154

GP and Init:

R2 and RMSE for dataset  0 :  [0.14925405 0.66245184 0.3375792  0.15912747 0.54367339 0.14117016
 0.59310867 0.05717029 0.55028603 0.52129818 0.57624708 0.53702941
 0.26552682 0.43235519 0.31586686 0.38421405 0.39747306 0.74599691
 0.47868053 0.3920857 ] [1.71009918 1.20158625 1.67479493 1.77204214 1.37198662 2.12340685
 1.4740204  1.99807271 1.34055929 1.40455512 1.31960587 1.42256382
 1.63082686 1.50610819 1.43261978 1.55016401 1.64590936 1.00353503
 1.4376892  1.60441176]
Mean:  0.4120297439621933 1.531227868877688
Std:  0.18341271114834834 0.250185366077544
Min:  0.05717028577269312 1.003535029762228
Max:  0.7459969119943564 2.1234068483600392
Test set RMSE= 1.426999743066604  and R2= 0.40283628542165595
Exp. validation set RMSE= 1.9626435098914479  and R2= 0.19339099231182877

GP and Var:

R2 and RMSE for dataset  0 :  [ 0.15989805  0.62021014  0.31928209  0.20599431  0.51242134  0.16584024
  0.55761298 -0.04979625  0.55128631  0.53522184  0.49567251  0.48112297
  0.33884824  0.53586953  0.3236282   0.36413479  0.3913743   0.6758003
  0.47125775  0.30071242] [1.69936766 1.27455553 1.69776763 1.72195092 1.41818968 2.09268697
 1.53697026 2.10837164 1.33906758 1.38397774 1.43960785 1.50600802
 1.54728553 1.36187696 1.42447022 1.57523479 1.6542183  1.13375534
 1.44788826 1.72076897]
Mean:  0.39781960275448275 1.5542009921428217
Std:  0.17637026498225394 0.23819542780366054
Min:  -0.04979624656492443 1.133755336699495
Max:  0.6758002976801835 2.108371642179896
Test set RMSE= 1.404208044638264  and R2= 0.42175946257278096
Exp. validation set RMSE= 1.9135447194547328  and R2= 0.2332435201952926

GP and Cor:

R2 and RMSE for dataset  0 :  [0.15075117 0.56340709 0.31359895 0.23364371 0.57333386 0.13869374
 0.51906515 0.08899356 0.48327404 0.46551625 0.49740181 0.60045368
 0.31061854 0.46109877 0.32535925 0.35009298 0.36334885 0.50748154
 0.46929386 0.41998789] [1.70859382 1.36654965 1.70484001 1.69170381 1.3266491  2.12646605
 1.60253454 1.96406285 1.4369709  1.48413532 1.43713759 1.32153488
 1.57997306 1.46748078 1.42264621 1.59253276 1.69187566 1.39741145
 1.45057468 1.56715946]
Mean:  0.391770734679131 1.5670416292967433
Std:  0.14673370967000665 0.20184589957681598
Min:  0.08899356012437865 1.321534879138967
Max:  0.6004536766613976 2.1264660464060947
Test set RMSE= 1.740053868191873  and R2= 0.1120857876218474
Exp. validation set RMSE= 1.7817119886520254  and R2= 0.33525477386832725

GP and Opt:

R2 and RMSE for dataset  0 :  [0.66046411 0.72308087 0.55148567 0.42445314 0.54228513 0.40368439
 0.65617527 0.49899633 0.38506735 0.56452541 0.69038465 0.653268
 0.48436286 0.73893025 0.22585194 0.17580951 0.48367974 0.74596988
 0.66390558 0.50629143] [1.0803493  1.0883374  1.37810662 1.46605177 1.37407199 1.76936626
 1.35497984 1.4565155  1.56758684 1.33963861 1.12797411 1.2310963
 1.36644445 1.0214006  1.52395675 1.79339751 1.52362209 1.00358843
 1.15436644 1.44587249]
Mean:  0.5389335752269167 1.3533361650675808
Std:  0.15674554198089743 0.22210774059337055
Min:  0.1758095087447682 1.003588432503655
Max:  0.7459698779163382 1.7933975053956341
Test set RMSE= 1.2312866058859173  and R2= 0.5554056581131781
Exp. validation set RMSE= 1.7198734868700638  and R2= 0.3805971122602948

'''    
    