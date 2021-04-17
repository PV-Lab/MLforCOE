#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 16:52:47 2021

@author: armi
"""


from Functions_downselection_training_RF import plot_heatmap, save_to_csv_pickle, save_to_pickle, fetch_pickle, pick_xy_from_corrmatrix, define_groups_yvalue, pick_xy_from_columnlist
from Functions_training_GP import GP_feature_analysis, analyze_GP_for_multiple_seeds, predict_plot_GP
from set_figure_defaults import FigureDefaults

from Main_downselection import fetch_csv, save_cv_splits_to_csvs

def GP_train_test_newdata(X_list, y_list, groups_train, ho_params,
                      saveas='Plot_result', save_cv = False, 
                      save_cv_path = './Data/Downselection data files/CV_splits/', 
                      save_cv_fingerprint = 'opt', n_folds=20, random_state=3): 
    
    [X_train, X_test, X_newdata] = X_list
    [y_train, y_test, y_newdata] = y_list
    
    ###############################################################################
    # 20 stratified, ortherwise random cross-validation repetitions to train set
    # to estimate accuracy.
    if ho_params is not None:
        ho_params_cv = [ho_params]
    else:
        ho_params_cv = None
    R2_all, RMSE_all, top_features_all, features_all, X_tests4, y_tests4, X_trains4, y_trains4, regressors4, scalers4 = analyze_GP_for_multiple_seeds(
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
    list_X = [X_train, X_test]
    list_y = [y_train, y_test]
    if ho_params is not None:
        print('HO params not implemented for GP!\n')
    else:
        feature_weights1, top_feature_weights1, regressor1, R21, RMSE1, scaler1, X_test1, y_test1, y_pred1, X_train1, y_train1, std_pred1 = GP_feature_analysis(
                        list_X, list_y, groups=None,
                        groups_only_for_plotting = False,
                        test_indices = None, test_proportion = None,
                        top_n = 21, random_state = random_state,
                        sample_weighing = False, plotting=True,
                        saveas = saveas, title = None)
    test_results = [feature_weights1, top_feature_weights1, regressor1, R21, RMSE1, scaler1, X_test1, y_test1, y_pred1, X_train1, y_train1, std_pred1]
    print('Test set RMSE=', RMSE1, ' and R2=', R21)
    
    R2_newdata, RMSE_newdata, y_pred_newdata, std_pred_newdata = predict_plot_GP(regressor1, X_newdata, y_newdata, scaler1, plotting=True, title=None, groups = None, saveas = saveas+'_newdata')
    val_results = [R2_newdata, RMSE_newdata, y_pred_newdata, std_pred_newdata]
    print('Exp. validation set RMSE=', RMSE_newdata, ' and R2=', R2_newdata)
    ###############################################################################

    return cv_results, test_results, val_results