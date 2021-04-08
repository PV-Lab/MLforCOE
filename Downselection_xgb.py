#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:25:28 2020

Performs downselection using XGB instead of RF.

@author: armi tiihonen
"""

from Functions_downselection_training_RF import plot_heatmap, save_to_csv_pickle, save_to_pickle, fetch_pickle, pick_xy_from_corrmatrix, define_groups_yvalue, pick_xy_from_columnlist
from Functions_training_XGB import XGB_feature_analysis, analyze_XGB_for_multiple_seeds, predict_plot_XGB
from set_figure_defaults import FigureDefaults

from Main_downselection import fetch_csv, save_cv_splits_to_csvs


def XGB_train_test_newdata(X_list, y_list, groups_train, ho_params,
                           saveas = 'XGB_result', save_cv = False, 
                           save_cv_path = './Results/CV_splits_XGB/Seed3/', 
                      save_cv_fingerprint = 'opt', n_folds=20,
                      random_state = 3):
    
    [X_train, X_test, X_val] = X_list
    [y_train, y_test, y_val] = y_list
    
    ###############################################################################
    # 20 stratified, ortherwise random cross-validation repetitions to train set
    # to estimate accuracy.
    if ho_params is not None:
        ho_params_cv = [ho_params]
    else:
        ho_params_cv = None
    R2_all, RMSE_all, top_features_all, features_all, X_tests4, y_tests4, X_trains4, y_trains4, regressors4 = analyze_XGB_for_multiple_seeds(
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
        n_estimators = ho_params['n_estimators']
        max_depth = ho_params['max_depth']
        gamma = ho_params['gamma']
        eta = ho_params['eta']
        feature_weights1, top_feature_weights1, regressor1, R21, RMSE1, scaler_test1, X_test1, y_test1, y_pred1, X_train1, y_train1 = XGB_feature_analysis(
                        list_X, list_y, groups=None,
                        groups_only_for_plotting = False,
                        test_indices = None, test_proportion = None,
                        top_n = 21, random_state = random_state,
                        sample_weighing = False, plotting=True,
                        saveas = saveas, title = None,
                        max_depth=max_depth, gamma = gamma,
                        n_estimators=n_estimators, eta = eta)
    else:
        feature_weights1, top_feature_weights1, regressor1, R21, RMSE1, scaler_test1, X_test1, y_test1, y_pred1, X_train1, y_train1 = XGB_feature_analysis(
                        list_X, list_y, groups=None,
                        groups_only_for_plotting = False,
                        test_indices = None, test_proportion = None,
                        top_n = 21, random_state = ransom_state,
                        sample_weighing = False, plotting=True,
                        saveas = saveas, title = None)
    test_results = [feature_weights1, top_feature_weights1, regressor1, R21, RMSE1, scaler_test1, X_test1, y_test1, y_pred1, X_train1, y_train1]
    print('Test set RMSE=', RMSE1, ' and R2=', R21)
    
    
    R2_val, RMSE_val, y_pred_val = predict_plot_XGB(regressor1, X_val, y_val, plotting=True, title=None, groups = None, saveas = 'Plot_result_new_dataset_val')
    val_results = [R2_val, RMSE_val, y_pred_val]
    print('Exp. validation set RMSE=', RMSE_val, ' and R2=', R2_val)
    ###############################################################################

    return cv_results, test_results, val_results


###############################################################################

if __name__ == "__main__":
    
    # New dataset.
    #dataset_n = read_molecule_excel('./10232020 5k descriptors.xlsx',
    #                                column_class = None)
    #dataset_n = dataset_n[dataset_n.loc[:, 'No.'] != 140] # Temporary fix, ALOGP for this is nan.
    
    y_column=3
    variance_limit = 0.1
    
    test_proportion = 0.2
    
    save = False
    ###############################################################################
    
    # Don't do the filtering of init, var, cor from the beginning.
    
    X_init_train = fetch_csv('./Results/x_init_train_seed5')
    y_init_train = fetch_csv('./Results/y_init_train_seed5')
    X_var_train = fetch_csv('./Results/x_var_train_seed5')
    y_var_train = fetch_csv('./Results/y_var_train_seed5')
    X_cor_train = fetch_csv('./Results/x_cor_train_seed5')
    y_cor_train = fetch_csv('./Results/y_cor_train_seed5')
    groups_train =  fetch_csv('./Results/groups_train_seed5')
    X_init_test = fetch_csv('./Results/x_init_test_seed5')
    y_init_test = fetch_csv('./Results/y_init_test_seed5')
    X_var_test = fetch_csv('./Results/x_var_test_seed5')
    y_var_test = fetch_csv('./Results/y_var_test_seed5')
    X_cor_test = fetch_csv('./Results/x_cor_test_seed5')
    y_cor_test = fetch_csv('./Results/y_cor_test_seed5')
    X_init_val = fetch_csv('./Results/x_init_val')
    y_init_val = fetch_csv('./Results/y_init_val')
    X_var_val = fetch_csv('./Results/x_var_val')
    y_var_val = fetch_csv('./Results/y_var_val')
    X_cor_val = fetch_csv('./Results/x_cor_val')
    y_cor_val =  fetch_csv('./Results/y_cor_val')
    dataset_cor_train = fetch_csv('./Results/dataset_train_seed5')
    dataset_cor_test = fetch_csv('./Results/dataset_test_seed5')
    dataset_cor_val = fetch_csv('./Results/dataset_val')
    ###############################################################################
    
    random_state = 5
    # Results from the final run with seed 5 2/12/2021 (Run ID 9239973):
    ho_init = {'eta': 0.16303971225231562, 'gamma': 0.10020848581862363, 'max_depth': 3, 'n_estimators': 227}
    ho_var = {'eta': 0.17153129326979943, 'gamma': 2.078168135370535, 'max_depth': 4, 'n_estimators': 379}
    ho_cor = {'eta': 0.3054092191428568, 'gamma': 1.9386571606670655, 'max_depth': 5, 'n_estimators': 357}
    
    mystyle = FigureDefaults('nature_comp_mat_tc')
    
    ###############################################################################
    
    cv_results_init, test_results_init, val_results_init = XGB_train_test_val(
            [X_init_train, X_init_test, X_init_val],
            [y_init_train, y_init_test, y_init_val], groups_train, ho_init,
            saveas='./Results/xgb_init_seed' + str(random_state))
    
    cv_results_var, test_results_var, val_results_var = XGB_train_test_val(
            [X_var_train, X_var_test, X_var_val],
            [y_var_train, y_var_test, y_var_val], groups_train, ho_var,
            saveas='./Results/xgb_var_seed'  + str(random_state))
    
    cv_results_cor, test_results_cor, val_results_cor = XGB_train_test_val(
            [X_cor_train, X_cor_test, X_cor_val],
            [y_cor_train, y_cor_test, y_cor_val], groups_train, ho_cor,
            saveas='./Results/xgb_cor_seed'  + str(random_state))
    if save == True:
        save_to_pickle([cv_results_init, test_results_init, val_results_init,
                   cv_results_var, test_results_var, val_results_var,
                   cv_results_cor, test_results_cor, val_results_cor],
                   './Results/XGBresults_init_var_cor')   
    ###############################################################################
    # Do RFE and HO here.
    
    opt_descriptors_from_zero = ['MSD', 'MaxDD', 'TI2_L', 'MATS4m', 'MATS2v', 'MATS5e', 'GGI8', 'VE1_RG',
           'RDF035m', 'Mor25m', 'Mor26m', 'Mor25v', 'Mor26v', 'Mor16s', 'H-047',
           'CATS2D_03_AL', 'SHED_AA']
    opt_descriptors_from_RF = ['O%', 'MSD', 'MaxDD', 'CIC3', 'TI2_L', 'MATS8m', 'MATS2v', 'MATS7e',
           'P_VSA_MR_5', 'VE1sign_G', 'VE1sign_G/D', 'Mor22u', 'Mor20m', 'Mor25m',
           'Mor26m', 'Mor31m', 'Mor10v', 'Mor20v', 'Mor25v', 'R4i', 'H-047',
           'CATS2D_03_AL', 'SHED_AA', 'MLOGP2', 'ALOGP'] # Seed 5 top, train RMSE 1.35 +- 0.23. Final!
    opt_descriptors_from_RF_and_XGB = ['MSD', 'MaxDD', 'TI2_L', 'MATS2v', 'Mor25m', 'Mor26m', 'Mor25v', 'H-047', 'CATS2D_03_AL', 'SHED_AA']
    ho_opt = {'eta': 0.47182303948199855, 'gamma': 3.969100460481385, 'max_depth': 7, 'n_estimators': 496}
    ho_opt_from_zero = {'eta': 0.47331631649261163, 'gamma': 1.8026534027919072, 'max_depth': 4, 'n_estimators': 360}
    ###############################################################################
    # From XGB pipeline:
    
    opt_descriptors = ['log2mic']
    opt_descriptors.extend(opt_descriptors_from_zero)
    X_opt_train, y_opt_train = pick_xy_from_columnlist(dataset_cor_train, opt_descriptors)
    X_opt_test, y_opt_test = pick_xy_from_columnlist(dataset_cor_test, opt_descriptors)
    X_opt_val, y_opt_val = pick_xy_from_columnlist(dataset_cor_val, opt_descriptors)
        
    cv_results, test_results, val_results = XGB_train_test_val(
            [X_opt_train, X_opt_test, X_opt_val], [y_opt_train, y_opt_test, y_opt_val],
            groups_train, ho_opt_from_zero, saveas = './Results/XGB_opt_test')
    if save == True:
        save_to_pickle([cv_results, test_results, val_results], './Results/xgbresults_opt_from_zero')
        # Save also as csv for HO.
        save_to_csv_pickle(X_opt_train, './Results/x_opt_train_xgb_from_zero_seed5')
        save_to_csv_pickle(y_opt_train, './Results/y_opt_train_xgb_from_zero_seed5')
        save_to_csv_pickle(X_opt_test, './Results/x_opt_test_xgb_from_zero_seed5')
        save_to_csv_pickle(y_opt_test, './Results/y_opt_test_xgb_from_zero_seed5')
        save_to_csv_pickle(X_opt_val, './Results/x_opt_val_xgb_from_zero_seed5')
        save_to_csv_pickle(y_opt_val, './Results/y_opt_val_xgb_from_zero_seed5')
    
    corrMatrixOpt = dataset_cor_train.loc[:,opt_descriptors].corr(method='spearman')#
    mystyle = FigureDefaults('nature_comp_mat_dc')
    #plot_heatmap(corrMatrixOpt, './Results/XGB optimized descriptor set', vmin=-1, vmax=1)
    ###############################################################################
    # With RF Opt features.
    
    opt_descriptors = ['log2mic']
    opt_descriptors.extend(opt_descriptors_from_RF)
    X_opt_train_rf, y_opt_train = pick_xy_from_columnlist(dataset_cor_train, opt_descriptors)
    X_opt_test_rf, y_opt_test = pick_xy_from_columnlist(dataset_cor_test, opt_descriptors)
    X_opt_val_rf, y_opt_val = pick_xy_from_columnlist(dataset_cor_val, opt_descriptors)
    
    mystyle = FigureDefaults('nature_comp_mat_tc')
    
    cv_results, test_results, val_results = XGB_train_test_val(
            [X_opt_train_rf, X_opt_test_rf, X_opt_val_rf], [y_opt_train, y_opt_test, y_opt_val],
            groups_train, ho_opt, saveas = './Results/XGB_opt_rf_test')
    if save == True:
        save_to_pickle([cv_results, test_results, val_results], './Results/xgbresults_opt_from_rf')
        # No need to save X,y anymore.
        
        
    '''
    ome/armi/anaconda3/lib/python3.7/site-packages/dask/config.py:161: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
      data = yaml.load(f.read()) or {}
    /home/armi/anaconda3/lib/python3.7/site-packages/distributed/config.py:20: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
      defaults = yaml.load(f)
    ['/home/armi/anaconda3/lib/libxgboost.so']
    R2 and RMSE for dataset  0 :  [ 0.06324641  0.48831275  0.12420949  0.42543462  0.61067766  0.3128875
      0.24939128  0.11867686 -0.27909091  0.29136594  0.22523551  0.04077217
      0.46915351 -0.09085007 -0.18990663 -0.11166795  0.26502348  0.21407741
      0.47453578  0.38629665] [1.77395225 1.42784939 1.65020033 1.44007706 1.4465848  1.60185384
     1.49179006 1.72366555 2.29700892 1.32684315 1.83367305 2.131043
     1.54490633 2.1172416  2.10245504 1.86269157 1.3544631  1.83348332
     1.49371025 1.77077399]
    Mean:  0.20438907361050057 1.711213330392205
    Std:  0.23801558235751683 0.2770845386231334
    Min:  -0.2790909062870708 1.3268431516297026
    Max:  0.6106776587006821 2.297008922149448
    /home/armi/anaconda3/lib/python3.7/site-packages/matplotlib/font_manager.py:1241: UserWarning: findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans.
      (prop.get_family(), self.defaultFamily[fontext]))
    /home/armi/anaconda3/lib/python3.7/site-packages/matplotlib/font_manager.py:1241: UserWarning: findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans.
      (prop.get_family(), self.defaultFamily[fontext]))
    Test set RMSE= 1.3992316815763617  and R2= 0.4220371073868596
    Exp. validation set RMSE= 2.1353183490890424  and R2= 0.142218644071756
    R2 and RMSE for dataset  0 :  [ 0.31946457  0.46924036  0.47051132  0.47496372  0.39619503  0.25905744
      0.29222812  0.43759182  0.14898148  0.08195141  0.07770593  0.07266369
      0.37796151  0.25020844 -0.37164211  0.1874831   0.30252079  0.27970232
      0.47499449  0.43150612] [1.51200982 1.45421643 1.28311296 1.37660918 1.80151373 1.66341738
     1.44859692 1.376928   1.87361986 1.51022414 2.00065293 2.09531807
     1.67234667 1.75532752 2.25730727 1.59246395 1.31945955 1.75526658
     1.49305814 1.70430292]
    Mean:  0.271664477367551 1.6472876010322768
    Std:  0.19933327768020354 0.2576134815460027
    Min:  -0.3716421078687726 1.2831129595488044
    Max:  0.4749944939911377 2.257307270101511
    Test set RMSE= 1.4940432888871393  and R2= 0.3410581995369152
    Exp. validation set RMSE= 2.3850002071661156  and R2= -0.07010940413130706
    R2 and RMSE for dataset  0 :  [-0.10708625  0.49831217  0.33303802  0.62397485  0.49646621  0.14330497
      0.36638214  0.50965754  0.12345855  0.18700671 -0.09874376  0.26407045
      0.52927691  0.23992231 -0.10569984  0.02477584 -0.11503947  0.37017287
      0.48076379  0.48531871] [1.92850145 1.41382901 1.44008156 1.16499604 1.64514258 1.7886365
     1.37061225 1.28568675 1.90150821 1.42118972 2.1836612  1.86659221
     1.45479041 1.76732686 2.02669735 1.74463923 1.66830544 1.64133699
     1.48483186 1.6216349 ]
    Mean:  0.2624666354841739 1.6410000259231072
    Std:  0.23868165343199718 0.2568639528372083
    Min:  -0.11503947020430272 1.1649960391379441
    Max:  0.6239748512672975 2.1836611983156877
    Test set RMSE= 1.2904072400709226  and R2= 0.5084425549144043
    Exp. validation set RMSE= 2.533659766711618  and R2= -0.20766902845063817
    R2 and RMSE for dataset  0 :  [ 0.33962082  0.49965964  0.49512768  0.64644294  0.67914926  0.40792607
     -0.11091281  0.38018174  0.01749765  0.63131269  0.18523098  0.45580712
      0.64460029  0.15340393  0.55570354  0.52995684  0.60167545  0.40544789
      0.57219431  0.65798668] [1.48945    1.41192905 1.25293151 1.12965482 1.31322989 1.48695194
     1.81485066 1.44549835 2.01316224 0.9570567  1.88041757 1.60512092
     1.26408481 1.86520259 1.28471472 1.21121789 0.99712358 1.59471119
     1.34777708 1.3219207 ]
    Mean:  0.4374006356698633 1.4343503089218699
    Std:  0.21708667355363093 0.283410022603662
    Min:  -0.11091280955773164 0.9570566990220408
    Max:  0.6791492575257327 2.0131622413904124
    Test set RMSE= 1.65646488064562  and R2= 0.18999973407754756
    Exp. validation set RMSE= 2.4217836062644977  and R2= -0.10337212625717562
    /home/armi/anaconda3/lib/python3.7/site-packages/matplotlib/font_manager.py:1241: UserWarning: findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans.
      (prop.get_family(), self.defaultFamily[fontext]))
    R2 and RMSE for dataset  0 :  [ 0.36337324  0.53349373  0.49581489  0.47139993  0.66870485  0.35662691
     -0.0872709   0.58972318  0.03611658  0.48875216 -0.00537143  0.27879845
      0.58283945  0.19381023  0.34389067  0.4180935   0.3429274   0.47589464
      0.5718361   0.5950198 ] [1.46241855 1.36335466 1.2520785  1.38127328 1.33443301 1.55003109
     1.79543542 1.17604491 1.99399576 1.12700104 2.08881663 1.84781996
     1.36952289 1.82014727 1.56119943 1.34765898 1.28066968 1.49725709
     1.34834122 1.43846992]
    Mean:  0.3857236684871999 1.501798464534287
    Std:  0.2064191449089255 0.26362287448599037
    Min:  -0.08727090161011697 1.1270010390628609
    Max:  0.6687048455889498 2.0888166345184738
    Test set RMSE= 1.47325760926892  and R2= 0.359265539994458
    Exp. validation set RMSE= 2.171454685036958  and R2= 0.11294023740550574
    '''