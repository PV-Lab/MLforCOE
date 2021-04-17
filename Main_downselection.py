#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:25:28 2020

This code reads molecular descriptor Excel, formats and curates data,
performs statistical stages of descriptor downselection, and applies RFE
descriptor downselection results, and creates and saves train and test
datasets in a format that is suitable for using with ML model training.

@author: armi tiihonen
"""

from Functions_downselection_training_RF import plot_heatmap, plot_RF_test, RF_feature_analysis, save_to_csv_pickle, save_to_pickle, fetch_pickle, fetch_pickled_HO, read_molecule_excel, compare_features_barplot, analyze_RF_for_multiple_seeds, clean_mics, var_filtering, cor_filtering, pick_xy_from_corrmatrix, define_groups_yvalue, dropHighErrorSamples, pick_xy_from_columnlist, logmic, predict_plot_RF, scale, inverseScale, corrMatrix
import numpy as np
import pandas as pd
#import sys
#sys.path.append('/home/armi/Dropbox (MIT)/Armiwork2019/2020/Bazan team/Codes/From Antti')
from set_figure_defaults import FigureDefaults
#sys.path = ['/home/armi/anaconda3/envs/coe-rf/lib/python3.8/site-packages']
from sklearn.model_selection import train_test_split
from Functions_training_GP import analyze_GP_for_multiple_seeds, GP_feature_analysis, predict_plot_GP
import os
import pickle

def RF_train_test_newdata(X_list, y_list, groups_train, ho_params,
                      saveas='Plot_result', save_cv = False, 
                      save_cv_path = './Data/Downselection_data_files/CV_splits/', 
                      save_cv_fingerprint = 'opt', n_folds=20, random_state=3): 
    
    [X_train, X_test, X_newdata] = X_list
    [y_train, y_test, y_newdata] = y_list
    
    ###############################################################################
    # 20 stratified, ortherwise random cross-validation repetitions to train set
    # to estimate accuracy.
    ho_params_cv = [ho_params]
    R2_all, RMSE_all, top_features_all, features_all, X_tests4, y_tests4, X_trains4, y_trains4, regressors4 = analyze_RF_for_multiple_seeds(
            [X_train], [y_train], ho_params = ho_params_cv, bar_plot = False,
            save_pickle = False, groups = groups_train, plotting=False,
            n_seeds=n_folds, test_proportion=0.2)
    cv_results = [R2_all, RMSE_all, top_features_all, features_all, X_tests4,
                  y_tests4, X_trains4, y_trains4, regressors4]

    if save_cv is True:
        save_cv_splits_to_csvs(X_train, y_train, X_tests4[0], y_tests4[0], X_trains4[0], y_trains4[0], path=save_cv_path, fingerprint=save_cv_fingerprint)
    ###############################################################################
    #If test_proportion = None, the function assumes to get X and y as lists of 2
    # datasets so that the first one is train and the second one is test.
    listX = [X_train, X_test]
    listy = [y_train, y_test]
    n_estimators = ho_params['n_estimators']
    max_depth = ho_params['max_depth']
    min_samples_split = ho_params['min_samples_split']
    min_samples_leaf = ho_params['min_samples_leaf']
    max_features = ho_params['max_features']
    bootstrap = ho_params['bootstrap']
    max_samples = ho_params['max_samples']
    
    feature_weights1, top_feature_weights1, regressor1, R21, RMSE1, scaler_test1, X_test1, y_test1, y_pred1, X_train1, y_train1 = RF_feature_analysis(
        listX, listy, groups = None, groups_only_for_plotting = True, 
                            test_indices = None, test_proportion = None, top_n = 21, 
                            n_estimators = n_estimators, max_depth = max_depth, 
                            min_samples_split = min_samples_split,
                            min_samples_leaf = min_samples_leaf, 
                            max_features = max_features, bootstrap = bootstrap,
                            i='', random_state = random_state, sample_weighing = False, 
                            plotting = True, saveas = saveas+'_test', title = None, max_samples=max_samples)
    test_results = [feature_weights1, top_feature_weights1, regressor1, R21, RMSE1, scaler_test1, X_test1, y_test1, y_pred1, X_train1, y_train1]
    print('Test set RMSE=', RMSE1, ' and R2=', R21)
    
    R2_newdata, RMSE_newdata, y_pred_newdata = predict_plot_RF(regressor1, X_newdata, y_newdata, plotting=True, title=None, groups = None, saveas = saveas+'_newdata')
    newdata_results = [R2_newdata, RMSE_newdata, y_pred_newdata]
    print('New dataset RMSE=', RMSE_newdata, ' and R2=', R2_newdata)
    
    ###############################################################################

    return cv_results, test_results, newdata_results

def divide_train_test_newdata_X_y(dataset_train, dataset_test, dataset_newdata, corrMatrixVar, corrMatrixCor, save = True, random_state = None, y_column=3):
    
    # Divide into X and y.
    X_init_newdata, y_init_newdata = pick_xy_from_columnlist(dataset_newdata, dataset_newdata.columns[y_column::])
    X_var_newdata, y_var_newdata = pick_xy_from_corrmatrix(dataset_newdata, corrMatrixVar)
    X_cor_newdata, y_cor_newdata = pick_xy_from_corrmatrix(dataset_newdata, corrMatrixCor)
    groups_newdata = define_groups_yvalue(y_init_newdata)
    
    X_init_train, y_init_train = pick_xy_from_columnlist(dataset_train, dataset_train.columns[y_column::])
    groups_train = define_groups_yvalue(y_init_train)
    
    X_init_test, y_init_test = pick_xy_from_columnlist(dataset_test, dataset_test.columns[y_column::])
    groups_test = define_groups_yvalue(y_init_test)
    
    X_var_train, y_var_train = pick_xy_from_corrmatrix(dataset_train, corrMatrixVar)
    X_var_test, y_var_test = pick_xy_from_corrmatrix(dataset_test, corrMatrixVar)
    
    X_cor_train, y_cor_train = pick_xy_from_corrmatrix(dataset_train, corrMatrixCor)
    X_cor_test, y_cor_test = pick_xy_from_corrmatrix(dataset_test, corrMatrixCor)
    
    data_train = [X_init_train, y_init_train,
                  X_var_train, y_var_train,
                  X_cor_train, y_cor_train, groups_train]
    data_test = [X_init_test, y_init_test, X_var_test, y_var_test,
                  X_cor_test, y_cor_test]
    data_newdata = [X_init_newdata, y_init_newdata, X_var_newdata, y_var_newdata,
                  X_cor_newdata, y_cor_newdata]
    
    if save == True:
        
        X_savedata = [X_init_newdata, X_var_newdata, X_cor_newdata,
                  X_init_train, X_var_train, X_cor_train,
                  X_init_test, X_var_test, X_cor_test]
        corresponding_x_paths = ['./Data/Downselection_data_files/x_init_newdata',
                        './Data/Downselection_data_files/x_var_newdata',
                        './Data/Downselection_data_files/x_cor_newdata',
                        './Data/Downselection_data_files/x_init_train',
                        './Data/Downselection_data_files/x_var_train',
                        './Data/Downselection_data_files/x_cor_train',
                        './Data/Downselection_data_files/x_init_test',
                        './Data/Downselection_data_files/x_var_test',
                        './Data/Downselection_data_files/x_cor_test'
                        ]
        y_savedata = [y_init_newdata, y_var_newdata, y_cor_newdata,
                  y_init_train, y_var_train, y_cor_train,
                  y_init_test, y_cor_test, y_cor_test]
        corresponding_dataset = [dataset_newdata, dataset_newdata, dataset_newdata, 
                                 dataset_train, dataset_train, dataset_train,
                                 dataset_test, dataset_test, dataset_test]
        corresponding_y_paths = ['./Data/Downselection_data_files/y_init_newdata',
                        './Data/Downselection_data_files/y_var_newdata',
                        './Data/Downselection_data_files/y_cor_newdata',
                        './Data/Downselection_data_files/y_init_train',
                        './Data/Downselection_data_files/y_var_train',
                        './Data/Downselection_data_files/y_cor_train',
                        './Data/Downselection_data_files/y_init_test',
                        './Data/Downselection_data_files/y_var_test',
                        './Data/Downselection_data_files/y_cor_test'
                        ]
        y_savedatasmiles = []
        
        for i in range(len(y_savedata)):
            
            # We want to save y with smiles (for compatibility with chemprop).
            smiles = corresponding_dataset[i].loc[y_savedata[i].index, ['smiles']].copy()
            y_savedatasmiles.append(pd.concat([smiles, y_savedata[i]], axis=1, verify_integrity=True))
            
            if ((random_state is not None) and (i>2)):
                
                save_to_csv_pickle(y_savedatasmiles[i], corresponding_y_paths[i] + '_seed'+ str(random_state))                
                save_to_csv_pickle(X_savedata[i], corresponding_x_paths[i] + '_seed'+ str(random_state), index=False) # chemprop does not know how to treat index
            
            else:
                
                save_to_csv_pickle(y_savedatasmiles[i], corresponding_y_paths[i])                
                save_to_csv_pickle(X_savedata[i], corresponding_x_paths[i], index=False)
                
        if random_state is not None:
            save_to_csv_pickle(dataset_test, './Data/Downselection_data_files/dataset_test_seed'+ str(random_state))
            save_to_csv_pickle(groups_test, './Data/Downselection_data_files/groups_test_seed'+ str(random_state))
            save_to_csv_pickle(dataset_train, './Data/Downselection_data_files/dataset_train_seed'+ str(random_state))
            save_to_csv_pickle(groups_train, './Data/Downselection_data_files/groups_train_seed'+ str(random_state))
            
        else:
            save_to_csv_pickle(dataset_test, './Data/Downselection_data_files/dataset_test')
            save_to_csv_pickle(groups_test, './Data/Downselection_data_files/groups_test')
            save_to_csv_pickle(dataset_train, './Data/Downselection_data_files/dataset_train')
            save_to_csv_pickle(groups_train, './Data/Downselection_data_files/groups_train')
    
        save_to_csv_pickle(groups_newdata, './Data/Downselection_data_files/groups_newdata')
        save_to_csv_pickle(dataset_newdata, './Data/Downselection_data_files/dataset_newdata')
        
    return data_train, data_test, data_newdata

def save_opt_data(Xlist, ylist, datasetlist):
    
    y_savedatasmiles = []
    
    for i in range(len(ylist)):
            
            # We want to save y with smiles (for compatibility with chemprop).
            smiles = datasetlist[i].loc[ylist[i].index, ['smiles']].copy()
            y_savedatasmiles.append(pd.concat([smiles, ylist[i]], axis=1, verify_integrity=True))
            
    save_to_csv_pickle(Xlist[0], './Data/Downselection_data_files/x_opt_train_seed'+ str(random_state), index=False)
    save_to_csv_pickle(Xlist[1], './Data/Downselection_data_files/x_opt_test_seed'+ str(random_state), index=False)
    save_to_csv_pickle(Xlist[2], './Data/Downselection_data_files/x_opt_newdata', index=False)
    save_to_csv_pickle(y_savedatasmiles[0], './Data/Downselection_data_files/y_opt_train_seed'+ str(random_state))
    save_to_csv_pickle(y_savedatasmiles[1], './Data/Downselection_data_files/y_opt_test_seed'+ str(random_state))
    save_to_csv_pickle(y_savedatasmiles[2], './Data/Downselection_data_files/y_opt_newdata')
        
    

def fetch_csv(filename, index_col=0):
    """
    Fetches any variable saved into a picklefile with the given filename.
    
    Parameters:
        filename (str): filename of the pickle file
        
    Returns:
        variable (any pickle compatible type): variable that was saved into the picklefile.
    """
    variable = pd.read_csv(filename+'.csv', index_col=index_col)
    return variable


def save_cv_splits_to_csvs(X_train, y_train, X_cvtests, y_cvtests, X_cvtrains, y_cvtrains, path = './Data/Downselection_data_files/CV_splits/Seed3/', fingerprint = 'opt'):
    
    all_splits = []
    indices_to_drop = [14, 108, 109]
    if len(indices_to_drop)>0:
        print('WARNING: Dropping molecules with indices ' + str(indices_to_drop) + ' - if input Excel file is modified, please modify this part of the code. Reason: Chemprop scaffold balancing did not work for these molecules. \n')
    y_train = y_train.copy().drop(indices_to_drop) # Molecules 'COE-D6C-HAc' and 'COE-Y4IM2' cannot be read in chemprop, and the chemprop indexing code breaks with these. TO DO: IMPORTANT!!! Replace with an implementation that takes molecule names instead of dataframe indices. Now the code removes wrong molecules if the Excel file is removed.
    for i in range(len(X_cvtests)):
        save_to_csv_pickle(X_cvtests[i], path + 'x_' + fingerprint + '_test_' + str(i))
        save_to_csv_pickle(y_cvtests[i], path + 'y_' + fingerprint + '_test_' + str(i))
        save_to_csv_pickle(X_cvtrains[i], path + 'x_' + fingerprint + '_train_' + str(i))
        save_to_csv_pickle(y_cvtrains[i], path + 'y_' + fingerprint + '_train_' + str(i))
        
        for k in indices_to_drop:
            try:
                y_cvtrains[i] = y_cvtrains[i].drop(k)
            except KeyError:
                pass
            try:
                y_cvtests[i] = y_cvtests[i].drop(k)
            except KeyError:
                pass
        
        val_test_length = int(len(y_cvtests[i].index)/2)
        current_split = [np.array([list(y_train.index).index(j) for j in y_cvtrains[i].index]),
                         np.array([list(y_train.index).index(j) for j in y_cvtests[i].index[0:val_test_length]]),
                         np.array([list(y_train.index).index(j) for j in y_cvtests[i].index[val_test_length:(-1)]])
                         ]

        #print(current_split)
        all_splits.append(current_split)
        with open(os.path.join(path, 'stratified_split_indices_cv' + str(i) + '_train.pckl'), 'wb') as wf:
            pickle.dump(current_split, wf)
    # These are saved just for record-keeping.
    save_to_csv_pickle(X_train, path + 'x_' + fingerprint + '_train')
    save_to_csv_pickle(y_train, path + 'y_' + fingerprint + '_train')
    
    # These are compatible with chemprop.
    with open(os.path.join(path, 'stratified_split_indices_cv_train.pckl'), 'wb') as wf:
            pickle.dump(all_splits, wf)


###############################################################################
if __name__ == "__main__":
    
    ##########################################################################
    # INPUT VALUES
    
    source_excel_file = './Data/04142021 5k descriptors.xlsx'
    y_column=3 # Location of MIC data in source Excel file.
    variance_limit = 0.1 # Relative variance filtering limit value.
    test_proportion = 0.2 # Test dataset proportion of the full dataset.
    random_state = 3 # Random state for train test divisions and model trainings.
    plotCorrMatrix = True # Setting this code to False makes the code run faster.
    # If True, saves the train-test datafiles.
    save = True
    # If this is True, saves the repeated subsampling splits for train-test.
    save_cv = False
    # Figure formatting options.
    mystyle = FigureDefaults('nature_comp_mat_sc')
    
    
    # Implementatation for including new data into the pipeline either for
    # new predictions or as an additional, experimental validation set. Here
    # is an example.
    name_newdata = ['COE2-3C-C3propylOH', 'COE2-3C-C3TEEDA', 'COE-A7', 'COE-D62OH',
                #'COE-Tt6', 
                'COE-Tt66C', 'COE-Tt6IM11', 'COE-A6', 'COE-A6IM1', 'COE-A6IM2',
                'COE-A6PR1', 'COE-A8', 'Amethyl', 'Amidine', 'COE-D4EtIM2', #'COE-D62N22N2',
                'COE-Y62C', 'COE-Y6N24C']#, 'COE2-3F']
    
    # Hyperparameters. Have been optimized separately.
    ho_init = {'bootstrap': True, 'max_depth': 11, 'max_features': 'sqrt',
                'max_samples': 0.99, 'min_samples_leaf': 1,
                'min_samples_split': 5, 'n_estimators': 117}
    ho_var = {'bootstrap': True, 'max_depth': 11, 'max_features': 0.3,
               'max_samples': 0.99, 'min_samples_leaf': 1,
               'min_samples_split': 3, 'n_estimators': 545}
    ho_cor = {'bootstrap': True, 'max_depth': 13, 'max_features': 0.5,
               'max_samples': 0.99, 'min_samples_leaf': 2,
               'min_samples_split': 4, 'n_estimators': 140}
    ho_opt = {'bootstrap': True, 'max_depth': 18, 'max_features': 'log2',
               'max_samples': 0.99, 'min_samples_leaf': 1,
               'min_samples_split': 2, 'n_estimators': 236}
    
    # Opt fingerprint. RFE has been performed separately.
    opt_descr_names = ['MSD', 'MaxDD', 'TI2_L', 'MATS4v', 'MATS4i', 'P_VSA_MR_5', 'P_VSA_MR_6',
       'TDB06s', 'RDF040m', 'Mor20m', 'Mor25m', 'Mor31m', 'Mor10v', 'Mor20v',
       'Mor25v', 'Mor26s', 'R7u', 'H-046', 'H-047', 'SHED_AL', 'ALOGP']
    '''
    opt_descr_names = ['MSD', 'MaxDD', 'TI2_L', 'MATS4v', 'MATS4i', 'P_VSA_MR_5', 'P_VSA_MR_6',
       'TDB06s', 'RDF040m', 'Mor20m', 'Mor25m', 'Mor31m', 'Mor10v', 'Mor20v',
       'Mor25v', 'R7u', 'H-046', 'H-047', 'SHED_AL', 'ALOGP']
    opt_descr_names = ['MSD', 'MaxDD', 'TI2_L', 'MATS4v', 'MATS4i', 'P_VSA_MR_5', 'P_VSA_MR_6',
       'TDB06s', 'RDF040m', 'Mor20m', 'Mor25m', 'Mor31m', 'Mor10v', 'Mor20v',
       'Mor25v', 'Mor26s', 'R7u', 'H-046', 'H-047', 'SHED_AL', 'ALOGP']
    opt_descr_names = ['O%', 'MSD', 'MaxDD', 'CIC4', 'TI2_L', 'MATS2p', 'P_VSA_MR_5', 'P_VSA_LogP_2', 'Mor20m', 'Mor25m', 'Mor31m', 'Mor10v', 'Mor20v', 'Mor25v', 'Mor13s', 'Mor06m', 'R4i', 'R7u', 'H-047', 'ALOGP', 'H7u', 'TDB07s', 'TDB10s', 'F02[C-N]']
    opt_descr_names = ['MSD', 'MaxDD', 'TI2_L', 'MATS4v', 'MATS4i', 'P_VSA_MR_5', 'TDB06s',
       'RDF040m', 'Mor06m', 'Mor20m', 'Mor25m', 'Mor31m', 'Mor10v', 'Mor20v',
       'Mor25v', 'R7u', 'H-046', 'H-047', 'SHED_AL', 'ALOGP']
    '''
    ##########################################################################
    # CODE EXECUTION STARTS
    
    # Data curation.
    ##########################################################################
    dataset_full = read_molecule_excel(source_excel_file, column_class = None)
    dataset = dataset_full.copy()
    dataset = dataset[dataset.loc[:, 'No.'] != 140] # ALOGP for this molecule is nan and needs to be re-calculated.
    dataset = dataset.rename(columns={'No.':'no', 'NAME': 'name',
                                          'SMILES ':'smiles',
                                          'MIC VALUE (Y VALUE)':'log2mic'})
    
    
    dataset = clean_mics(dataset, y_column)
    
    dataset = logmic(dataset, y_column)

    # Drop outlier molecules using leave one out cross validation and RF (not
    # hyperparameter optimized). Principle: If the molecule cannot be learned
    # with data from all the other molecules, it is either very different from
    # the rest of the molecules, or has experienced an experimental data
    # collection error.
    corrMatrixFull = corrMatrix(dataset, y_column)
    X, y = pick_xy_from_corrmatrix(dataset, corrMatrixFull)
    groups = define_groups_yvalue(y) # The molecules are divided in high/intermediate/low MIC groups for stratifying train/test splits.
    
    # This step takes some minutes so we can fetch the results from a pickle instead.
    #X, y, dataset, groups = dropHighErrorSamples(y, X, dataset, groups = groups, rmse_lim=3.5)
    #save_to_csv_pickle(X, './Data/X_outliers_dropped')
    #save_to_csv_pickle(y, './Data/y_outliers_dropped')
    #save_to_csv_pickle(dataset, './Data/dataset_outliers_dropped')
    #save_to_csv_pickle(groups, './Data/groups_outliers_dropped')

    X=fetch_csv('./Data/X_outliers_dropped')
    y=fetch_csv('./Data/y_outliers_dropped')
    dataset = fetch_csv('./Data/dataset_outliers_dropped')
    groups = fetch_csv('./Data/groups_outliers_dropped')
    '''
    Results from the single-molecule test set filtering (lim=3.5 which is default):
    
    outliers = ['COE-A8C-HAc', 'COE-D63SO3', 'COE2-2pip', 'COE2-3C-C3dipropyl-H',
                'COE-S6', 'COE-PYRAZINE-3C-BUTYL', 'Amidine', 'Y62C', 'Y6N24C']
    dataset_outliers = dataset[dataset.name.isin(outliers)]
    dataset = dataset[~dataset.name.isin(outliers)]    
    X, y = pick_xy_from_corrmatrix(dataset, corrMatrixFull)
    groups = define_groups_yvalue(y)
    
    
    There are  9  molecules with RMSE> 3.5 . These will be dropped from the analysis.
        no                   name  log2mic
1      2.0            COE-A8C-HAc      9.0
14    15.0            COE-D6C-HAc      9.0
23    24.0             COE-D63SO3      9.0
42    43.0              COE2-2pip      7.0
102  110.0                 COE-S6     10.0
117  127.0  COE-PYRAZINE-3C-BUTYL      8.0
137  147.0                Amidine      0.0
141  151.0                   Y62C      7.0
142  152.0                 Y6N24C      6.0
    
    There are  9  molecules with RMSE> 3.5 . These will be dropped from the analysis.
            no                   name  log2mic
            1      2.0            COE-A8C-HAc      9.0
            23    24.0             COE-D63SO3      9.0
            42    43.0              COE2-2pip      7.0
            #61    63.0   COE2-3C-C3dipropyl-H      6.0
            102  110.0                 COE-S6     10.0
            117  127.0  COE-PYRAZINE-3C-BUTYL      8.0
            #137  147.0                Amidine      0.0
            #141  151.0                   Y62C      7.0
            #142  152.0                 Y6N24C      6.0
            #Additionally 128
            
       If taking newdata out:
           There are  9  molecules with RMSE> 3.5 . These will be dropped from the analysis.
        no                        name  log2mic
        1      2.0                 COE-A8C-HAc      9.0
        #14    15.0                 COE-D6C-HAc      9.0
        23    24.0                  COE-D63SO3      9.0
        42    43.0                   COE2-2pip      7.0
        #67    69.0  COE2-3C-C3Npropylimidazole      1.0
        #97    99.0                     COE2-4C      9.0
        102  110.0                      COE-S6     10.0
        117  127.0       COE-PYRAZINE-3C-BUTYL      8.0
        #118  128.0          COE2-3C-C3A-PROPYL      1.0
            
            Old:
                There are  11  molecules with RMSE>3.5. These will be dropped from the analysis.
           No.                   NAME  MIC VALUE (Y VALUE)
    1      2.0            COE-A8C-HAc                  9.0
    23    24.0             COE-D63SO3                  9.0
    42    43.0              COE2-2pip                  7.0
    #58    60.0        COE2-3C-C3DABCO                  8.0
    #97    99.0                COE2-4C                  9.0
    102  110.0                 COE-S6                 10.0
    #109  119.0              COE-Y6IM2                  1.0
    117  127.0  COE-PYRAZINE-3C-BUTYL                  8.0
    #118  128.0     COE2-3C-C3A-PROPYL                  1.0
    137  147.0                Amidine                  0.0
    142  152.0                 Y6N24C                  6.0
    '''

    # Forming of train, test.
    ##########################################################################
    
    # Divide into train, test, and new data sets. New data is separated first.
    dataset_newdata = dataset.loc[dataset.index.intersection(dataset[dataset.name.isin(name_newdata)].index)].copy()
    dataset = dataset[~dataset.name.isin(name_newdata)]

    # Train-test division is done using stratification according to MIC value. --> need to define groups.
    # X,y here are only used for defining groups, nothing else.
    X, y = pick_xy_from_corrmatrix(dataset, corrMatrixFull)
    if len(groups.shape) == 2:
        groups = groups.loc[y.index,:]
    else:
        groups = pd.DataFrame(groups[y.index])
    
    # The source Excel file has changed since model development and therefore
    # train test split does not produce the same results as used in the model.
    # To repeat the same result, let's pick the values by hand.
    
    # Seed 3:
    train_idx = [19, 35, 21, 70, 52, 88, 112, 58, 26, 46, 
                 85, 27, 114, 50, 86, 110, 55, 99, 16, 17, 
                 8, 94, 25, 89, 127, 45, 63, 82, 15, 91, 
                 62, 28, 75, 3, 77, 29, 119, 106, 98, 80, 
                 40, 124, 59, 97, 125, 113, 38, 20, 129, 47, 
                 92, 39, 111, 68, 81, 44, 53, 95, 109, 87, 
                 66, 118, 49, 107, 123, 67, 7, 24, 135, 60, 
                 13, 116, 22, 30, 65, 4, 41, 14, 
                 18, 108]
    test_idx = [54, 128, 126, 76, 96, 32, 84, 69, 6, 115, 
                5, 2, 56, 12, 90, 79, 74, 93, 31, 37]
    # Seed 5:
    #train_idx = [32, 80, 15, 95, 129, 21, 111, 107, 25, 3, 
    #             8, 13, 28, 112, 26, 69, 115, 37, 63, 35, 
    #             27, 54, 56, 90, 96, 53, 79, 24, 88, 67, 
    #             65, 62, 87, 74, 77, 84, 119, 66, 127, 17, 
    #             75, 61, 22, 135, 110, 98, 4, 39, 41, 5, 
    #             40, 30, 124, 47, 114, 2, 81, 93, 19, 14, 
    #             106, 45, 68, 99, 94, 89, 70, 113, 60, 125, 
    #             76, 59, 46, 92, 86, 29, 126] 
    #test_idx = [20, 31, 49, 12, 91, 82, 50, 6, 18, 108, 
    #             52, 7, 38, 116, 123, 85, 128, 16, 55, 44]
    
    dataset_train = dataset.loc[train_idx,:].copy()
    dataset_test = dataset.loc[test_idx,:].copy()
    
    # IF YOU WANT TO PERFORM THE PIPELINE FROM ZERO, ACTIVATE THE NEXT LINES OF CODE! 
    #dataset_train, dataset_test = train_test_split(dataset,
    #                                                   test_size=test_proportion,
    #                                                  random_state=random_state,
    #                                                   stratify=groups)
    
    
    # Descriptor downselections (statistical).
    ##########################################################################
    
    # Var. stage.
    corrMatrixInitial, corrMatrixVar = var_filtering(dataset_train, y_column, variance_limit, plotCorrMatrix=plotCorrMatrix)
    
    # Cor. stage and correlation matrices for stages Var., Cor. intermediate, and
    # Cor. final (with spearman correlation - more slow but resulted in better
    # results with this data). 
    corrMatrixVar, corrMatrixCorX, corrMatrixCor = cor_filtering(dataset_train,
                                                                 y_column,
                                                                 filterWithCorrMatrix = True,
                                                                 corrMatrixForFiltering = corrMatrixVar,
                                                                 plotCorrMatrix = plotCorrMatrix)

    # Forming of train, test.
    ##########################################################################
    
    # Divide into X and y of each downselection stage.
    [[X_init_train, y_init_train, X_var_train, y_var_train, X_cor_train, y_cor_train, groups_train],
     [X_init_test, y_init_test, X_var_test, y_var_test, X_cor_test, y_cor_test],
     [X_init_newdata, y_init_newdata, X_var_newdata, y_var_newdata, X_cor_newdata, y_cor_newdata]] = divide_train_test_newdata_X_y(
         dataset_train, dataset_test, dataset_newdata, corrMatrixVar,
         corrMatrixCor, save = save, random_state=random_state, y_column=y_column)
    
    ###############################################################################
    
    # Perform hyperparameter optimization for Init., Var., and Cor. stages at
    # this point if you run for new data.
    
    ###############################################################################
    
    mystyle = FigureDefaults('nature_comp_mat_tc')
    
    print('\nInit. fingerprint:\n')
    cv_results_init, test_results_init, newdata_results_init = RF_train_test_newdata(
            [X_init_train, X_init_test, X_init_newdata],
            [y_init_train, y_init_test, y_init_newdata], groups_train, ho_init,
            saveas='./Results/rf_init_seed' + str(random_state), save_cv = save_cv,
            save_cv_path = './Data/Downselection_data_files/CV_splits/Seed'+str(random_state) + '/', 
            save_cv_fingerprint = 'init', random_state=random_state)
    
    print('\nVar. fingerprint:\n')
    cv_results_var, test_results_var, newdata_results_var = RF_train_test_newdata(
            [X_var_train, X_var_test, X_var_newdata],
            [y_var_train, y_var_test, y_var_newdata], groups_train, ho_var,
            saveas='./Results/rf_var_seed'  + str(random_state), save_cv = save_cv,
            save_cv_path = './Data/Downselection_data_files/CV_splits/Seed'+str(random_state) + '/',
            save_cv_fingerprint='var', random_state=random_state)
    
    print('\nCor. fingerprint:\n')
    cv_results_cor, test_results_cor, newdata_results_cor = RF_train_test_newdata(
            [X_cor_train, X_cor_test, X_cor_newdata],
            [y_cor_train, y_cor_test, y_cor_newdata], groups_train, ho_cor,
            saveas='./Results/rf_cor_seed'  + str(random_state), save_cv = save_cv,
            save_cv_path = './Data/Downselection_data_files/CV_splits/Seed'+str(random_state) + '/',
            save_cv_fingerprint='cor', random_state=random_state)
    
    ###############################################################################
    
    # Perform RFE for Cor. stage at this point if you run for new data.
    
    ###############################################################################
    
    # Defining opt. fingerprint and plotting correlation matrix.
    ###############################################################################
    opt_cols = list(dataset_train.columns[0:(y_column+1)])
    opt_cols.extend(opt_descr_names)
    dataset_opt_train = dataset_train.loc[:, opt_cols]
    dataset_opt_test = dataset_test.loc[:, opt_cols]
    dataset_opt_newdata = dataset_newdata.loc[:, opt_cols]
    
    corrMatrixOpt = corrMatrix(dataset_opt_train, y_column)
    
    mystyle = FigureDefaults('nature_comp_mat_dc')
    mystyle.figure_size = 12
    plot_heatmap(corrMatrixOpt, 'Opt. fingerprint correlations', vmin=-1, vmax=1, ticklabels=True)
    
    X_opt_train, y_opt_train = pick_xy_from_corrmatrix(dataset_train, corrMatrixOpt)
    X_opt_test, y_opt_test = pick_xy_from_corrmatrix(dataset_test, corrMatrixOpt)
    X_opt_newdata, y_opt_newdata = pick_xy_from_corrmatrix(dataset_newdata, corrMatrixOpt)
    
    
    if save == True:
        
        save_opt_data([X_opt_train, X_opt_test, X_opt_newdata],
                  [y_opt_train, y_opt_test, y_opt_newdata],
                  [dataset_opt_train, dataset_opt_test, dataset_opt_newdata])
    
    ###############################################################################
    
    mystyle = FigureDefaults('nature_comp_mat_tc')
    
    print('\nOpt. fingerprint:\n')
    cv_results_opt, test_results_opt, newdata_results_opt = RF_train_test_newdata(
            [X_opt_train, X_opt_test, X_opt_newdata],
            [y_opt_train, y_opt_test, y_opt_newdata], groups_train, ho_opt,
            saveas='./Results/rf_opt_seed' + str(random_state), save_cv = save_cv, 
            save_cv_path = './Data/Downselection_data_files/CV_splits/Seed'+str(random_state) + '/',
            save_cv_fingerprint = 'opt', random_state=random_state)
    # Save the fully trained RF model to a pickle for later use.
    save_to_pickle(test_results_opt[2], './Results/RF_regressor_opt_seed' + str(random_state))
    
'''
Print-outs:
    
Initial dataset: 5258 descriptors
After dropping constant or almost constant descriptors: 1662 descriptors
After dropping constant or almost constant descriptors: 1662 descriptors
After filtering out highly correlated descriptors (limit 0.9: 329 descriptors
Correlation with Y higher than 0.05: 233 descriptors

Init. fingerprint:

R2 and RMSE for dataset  0 :  [0.3182764  0.62954513 0.39785816 0.26466231 0.29064215 0.16805094
 0.52915618 0.3001113  0.33664034 0.45540162 0.36574063 0.38300059
 0.32354466 0.32397885 0.16993702 0.25946456 0.3680365  0.68743985
 0.23366259 0.44481329] [1.5308249  1.25879419 1.59677624 1.65711397 1.71058502 2.08991209
 1.58563314 1.72150841 1.62814232 1.49811248 1.61443785 1.64224306
 1.56509048 1.64360697 1.57803325 1.69994819 1.68563554 1.11321703
 1.74310301 1.5332543 ]
Mean:  0.36249815311910105 1.6047986213136163
Std:  0.13193774773709716 0.1853665341897662
Min:  0.16805094256102304 1.1132170260401506
Max:  0.6874398525627454 2.089912091824589
Test set RMSE= 1.1965045888440924  and R2= 0.58016914043256
New dataset RMSE= 2.199879962074483  and R2= -0.013393677475660892

Var. fingerprint:

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
Max:  0.6301806804676722 1.9458106583239294
Test set RMSE= 1.1893163228864636  and R2= 0.5851984410896836
New dataset RMSE= 2.1192762027840764  and R2= 0.059507480510009314

Cor. fingerprint:

R2 and RMSE for dataset  0 :  [0.44885599 0.60064301 0.343755   0.33467184 0.35753398 0.33458858
 0.49800581 0.41940865 0.20030442 0.36750817 0.43065155 0.43034348
 0.41704058 0.44000334 0.25613455 0.4095711  0.31583366 0.57750993
 0.45615759 0.51066803] [1.37642927 1.30697633 1.66696953 1.5762566  1.62793499 1.86906659
 1.6372448  1.56794181 1.78764048 1.61448385 1.52959678 1.57798037
 1.45291112 1.49592752 1.49385286 1.51791189 1.75387457 1.2942593
 1.46841758 1.43944957]
Mean:  0.4074594624433817 1.552756290960041
Std:  0.0960456703799418 0.14550350235860046
Min:  0.2003044248630147 1.29425929674138
Max:  0.6006430108188192 1.8690665948059648
Test set RMSE= 1.2577357418209225  and R2= 0.5360999424475583
New dataset RMSE= 1.9705522009764813  and R2= 0.18687725272696787

Opt. fingerprint:

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

'''