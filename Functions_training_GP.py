#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 18:32:05 2021

@author: armi
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import operator
import sklearn as sklearn

from Functions_downselection_training_RF import save_to_pickle, compare_features_barplot, define_scale, scale, inverseScale, plot_RF_test

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern, RationalQuadratic



def analyze_GP_for_multiple_seeds(list_X, list_y, ho_params = None, n_seeds = 20, save_pickle = False, bar_plot = True, groups = None, groups_only_for_plotting = False, test_proportion = 0.21, top_n = 20, plotting=True, saveas = None, title=True):
    n_datasets = len(list_X)
    
    # Let's repeat y stratification. At the same, let's create a dataset for
    # RF hyperparameter optimization.
    R2_all2 = np.zeros((n_seeds,n_datasets))
    RMSE_all2 = np.zeros((n_seeds,n_datasets))
    top_features_all2 = []
    features_all2 = []
    X_tests = []
    y_tests = []
    X_trains = []
    y_trains = []
    regressors = []
    scalers = []
    filenames = ['X_tests_imp', 'y_tests_imp', 'X_tests', 'y_tests',
                 'X_trains_imp', 'y_trains_imp', 'X_trains', 'y_trains']
    for j in range(n_datasets):
        if ho_params is not None:
            n_estimators = ho_params[j]['n_estimators']
            max_depth = ho_params[j]['max_depth']
            min_samples_split = ho_params[j]['min_samples_split']
            min_samples_leaf = ho_params[j]['min_samples_leaf']
            max_features = ho_params[j]['max_features']
            bootstrap = ho_params[j]['bootstrap']
        top_features_temp = []
        features_temp = []
        X_tests_temp = []
        y_tests_temp = []
        X_trains_temp = []
        y_trains_temp = []
        regressors_temp = []
        scalers_temp = []
        if title is not None:
                title_temp = True
        else:
            title_temp = None
        for i in range(n_seeds):
            if saveas is not None:
                saveas_temp = saveas+str(i)
            else:
                saveas_temp = saveas
            if ho_params is None:
                feature_weights, top_feature_weights, regressor, R2, RMSE, scaler, X_test, y_test, y_pred, X_train, y_train = GP_feature_analysis(
                        list_X[j], list_y[j], groups=groups,
                        groups_only_for_plotting = groups_only_for_plotting,
                        test_indices = None, test_proportion = test_proportion,
                        top_n = top_n, i='', random_state = i,
                        sample_weighing = False, plotting=plotting, saveas = saveas_temp, title = title_temp)
            else:
                feature_weights, top_feature_weights, regressor, R2, RMSE, scaler, X_test, y_test, y_pred, X_train, y_train = GP_feature_analysis(
                        list_X[j], list_y[j], groups=groups,
                        groups_only_for_plotting = groups_only_for_plotting,
                        test_indices = None, test_proportion = test_proportion,
                        top_n = top_n, i='', random_state = i,
                        sample_weighing = False, n_estimators=n_estimators,
                        max_depth=max_depth, min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf, 
                        max_features=max_features, bootstrap=bootstrap, plotting=plotting, saveas = saveas_temp, title = title_temp)
            R2_all2[i,j] = R2
            RMSE_all2[i,j] = RMSE
            top_features_temp.append(top_feature_weights.copy())
            features_temp.append(feature_weights.copy())
            X_tests_temp.append(X_test.copy())
            y_tests_temp.append(y_test.copy())
            X_trains_temp.append(X_train.copy())
            y_trains_temp.append(y_train.copy())
            regressors_temp.append(regressor)
            scalers_temp.append(scaler)
            
        top_features_all2.append(top_features_temp)
        features_all2.append(features_temp)
        X_tests.append(X_tests_temp)
        y_tests.append(y_tests_temp)
        X_trains.append(X_trains_temp)
        y_trains.append(y_trains_temp)
        regressors.append(regressors_temp)
        scalers.append(scalers_temp)
        print('R2 and RMSE for dataset ', j, ': ', R2_all2[:,j], RMSE_all2[:,j])
        print('Mean: ', np.mean(R2_all2[:,j]), np.mean(RMSE_all2[:,j]))
        print('Std: ', np.std(R2_all2[:,j]), np.std(RMSE_all2[:,j]))
        print('Min: ', np.min(R2_all2[:,j]), np.min(RMSE_all2[:,j]))
        print('Max: ', np.max(R2_all2[:,j]), np.max(RMSE_all2[:,j]))
        if save_pickle == True:
            # Pickles for HO:
            if j == 0:
                save_to_pickle(X_tests, filenames[2])
                save_to_pickle(y_tests, filenames[3])
                save_to_pickle(X_trains, filenames[6])
                save_to_pickle(y_trains, filenames[7])
            if j == 1:
                save_to_pickle(X_tests, filenames[0])
                save_to_pickle(y_tests, filenames[1])
                save_to_pickle(X_trains, filenames[4])
                save_to_pickle(y_trains, filenames[5])
    
    # Plot the results. Compare feature weights of two methods. E.g., here the top
    # 50 feature weights of FilteredImportant dataset are compared to the top 50
    # feature weights of the Filtered dataset.
    if (bar_plot == True) and (n_datasets>1):
        compare_features_barplot(top_features_all2[0][0], top_features_all2[1][0])
    
    return R2_all2, RMSE_all2, top_features_all2, features_all2, X_tests, y_tests, X_trains, y_trains, regressors, scalers

def GP_feature_analysis(X, y, groups = None, groups_only_for_plotting = False, 
                        test_indices = None, test_proportion = 0.1, top_n = 5, 
                        n_estimators = 100, max_depth = None, 
                        min_samples_split = 2, min_samples_leaf = 1, 
                        max_features = 'auto', bootstrap = True, i='', 
                        random_state = None, sample_weighing = True, 
                        plotting = True, saveas = None, title = True):
    """
    Splits 'X' and 'y' to train and test sets so that 'test_proportion' of
    samples is in the test set. Fits a
    (sklearn) random forest model to the data according to RF parameters
    ('n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
    'max_features', 'bootstrap'). Estimates feature importances and determines
    'top_n' most important features. A plot and printouts for describing the
    results.
    
    Parameters:
        X (df): X data (features in columns, samples in rows)
        y (df): y data (one column, samples in rows)
        test_proportion (float, optional): Proportion of the test size from the original data.
        top_n (float, optional): The number of features in output 'top_feature_weights'
        n_estimators (int, optional): Number of trees in the forest
        max_depth (int, optional): Maximum depth of the tree
        min_samples split (int, optional): minimum number of samples required to split an internal node (could also be a float, see sklearn documentation)
        min_samples_leaf (int, optional): The minimum number od samples to be at a leaf node (could also be a float, see sklearn documentation)
        max_features (str, float, string, or None, optional): The number of features to consider when looking for the best split (see the options in sklearn documentation, 'sqrt' means max number is sqrt(number of features))
        bootstrap (boolean, optional): False means the whole dataset is used for building each tree, True means bootstrap of samples is used
        TO DO: Add value range that works for 5K dataset
        i (int, optional): Optional numeric index for figure filename.
        random_state (int, optional): Seed for train test split.

    Returns:
        feature_weights (df): weights of all the features
        top_feature_weights (df): weights of the features with the most weight
        regressor (RandomForestRegressor) RF regressor
        R2 (float): R2 value of the prediction for the test set.
    """
    if test_proportion == 0:
        # Use the whole dataset for both training and "testing".
        X_train = X.copy()
        X_test = X.copy()
        y_train = y.copy()
        y_test = y.copy()
    elif test_proportion == None:
        # Assume X and y are lists with two datasets...
        # Use dataset 0 as train and dataset 1 as test.
        X_train = X[0].copy()
        X_test = X[1].copy()
        y_train = y[0].copy()
        y_test = y[1].copy()
    else:
        # Split into test and train sets, and scale with StandardScaler.
        if test_indices is None:
            if groups is not None:
                if groups_only_for_plotting == False:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion, random_state=random_state, stratify=groups)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion, random_state=random_state)
                #shufflesplit = sklearn.model_selection.ShuffleSplit(n_splits=1, test_size=test_proportion, random_state=random_state)
                #X_train, X_test, y_train, y_test = shufflesplit.split(X, y, groups=groups)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion, random_state=random_state)
        else:
            #X_test = X.copy() # Are these needed?
            #y_test = y.copy() # Are these needed?
            X_test = X[test_indices].copy()
            y_test = y[test_indices].copy()
            #X_train = X.copy()
            #y_train = y.copy()
            X_train = X[~test_indices].copy()
            y_train = y[~test_indices].copy()
            #print(y_test)
    if sample_weighing:
        #sample_weight = np.divide(1,y_train.iloc[:,0]+0.1)
        #sample_weight = np.abs(y_train.iloc[:,0]-8.5)
        #sample_weight = np.abs(y_train.iloc[:,0]-4.1)
        sample_weight = y_train.copy()
        sample_weight[y_train<=3] = 5
        sample_weight[y_train>=8] = 5
        sample_weight[(y_train>3)&(y_train<8)] = 1
        sample_weight = sample_weight.squeeze()
    else:
        sample_weight = None
    #print(sample_weight)
    #X_train_s, X_test_s, y_train_s, y_test_s, scaler_train, scaler_test = scale(X_train, X_test, y_train, y_test)
    
    # Uncomment this part if you want to upsample the data.
    # This works only with class data. For that, you need to modify splitAndScale function and input y.
    #smote = SMOTE()
    #print(y_train_s.shape)
    #plot_2d_space(X_train_s, y_train_s, 'Original PCA')
    #X_train_s, y_train_s = smote.fit_sample(X_train_s, y_train_s)
    #print(y_train_s.shape, X_train_s.shape)
    #plot_2d_space(X_train_s, y_train_s, 'SMOTE over-sampling')
    
    #y_smogn = y_train_s.copy().join(X_train_s).reset_index(drop=True)
    #print(y_smogn.columns.get_loc('log(MIC)'))
    #print(y_smogn)
    #data_smogn = smogn.smoter(data = y_smogn, y = 'log(MIC)',
    #                          samp_method = 'extreme', under_samp = True,
    #                          rel_xtrm_type='both', rel_thres = 0.9, rel_method = 'auto',
    #                          rel_coef = 0.8)#, rel_ctrl_pts_rg = [[2,1,0], [8,1,0], [128,0,0]]) 
    #print(data_smogn)
    #y_train_s = data_smogn.iloc[:,0]
    #X_train_s = data_smogn.iloc[:,1::]
    #plot_2d_space(X_train_s, y_train_s, 'Smogned PCA')
    X_train_scaled, y_train_scaled, scaler_train = define_scale(X_train, y_train)
    #X_train_scaled = X_train
    #X_test_scaled = X_test
    #y_train_scaled = y_train
    #y_test_scaled = y_test
    kernel = Matern()
    regressor = GaussianProcessRegressor(kernel=kernel, random_state=None, alpha=1e-3, n_restarts_optimizer = 50, normalize_y = True).fit(X_train_scaled, y_train_scaled)
    
    R2, RMSE, y_pred = predict_plot_GP(regressor, X_test, y_test, scaler_train,
                                       plotting=plotting, title=title, 
                                       groups = groups, saveas = saveas)
    feature_weight = np.zeros(X_train.columns.shape)#regressor.feature_importances_
    #print('Feature weights for RF with ' + str(X.shape[1]+1) + ' features: ', feature_weight)
    '''
    y_pred = regressor.predict(X_test)
    y_pred = pd.Series(data=y_pred, index=y_test.index)
    #y_pred = y_pred.round() # MIC are exponents of two.
    feature_weight = regressor.feature_importances_
    #print('Feature weights for RF with ' + str(X.shape[1]+1) + ' features: ', feature_weight)
    #regressor.score(X_test_s, y_test_s)
    # Transform back to the original units.
    #X_test, y_test, y_pred = inverseScale(X_test_s, y_test_s, y_pred_s, scaler_test)
    R2 = sklearn.metrics.r2_score(y_test, y_pred)
    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(mse)
    #y_pred = np.exp2(y_pred) # Exponential data didn't look good in the plot.
    #y_test = np.exp2(y_test)
    if plotting is True:
        if title is not None:
            title_temp = 'Results/log_MIC RF with ' + str(X_train.shape[1]) + ' features'+str(i)
        else:
            title_temp = None
        if groups is not None:
            plot_RF_test(y_test, y_pred,
                     title = title_temp,
                     groups=groups.loc[y_test.index], saveas = saveas)
        else:
            plot_RF_test(y_test, y_pred,
                     title = title_temp,
                     groups=None, saveas = saveas)
    '''
    # Sort the features by importance.
    features = np.array(list(X_train.columns))
    #print('Features set : ', features)
    assert len(features) == len(feature_weight)
    i = 0
    l_dict = []
    while i < len(feature_weight):
        l_dict.append({features[i]:feature_weight[i]})
        i += 1
    res = sorted(zip(features, feature_weight), key = operator.itemgetter(1), reverse = True) 
    # Let's take the top features from the original set.
    top_features = [i[0] for i in res[:top_n]]
    #print('Top ', top_n, ' of the given features: ', top_features)
    
    # Let's put features into two small dataframes.
    feature_weights = pd.DataFrame(feature_weight.reshape((1,len(feature_weight))),
                                   columns = features,
                                   index = [0])
    top_feature_weights = feature_weights.loc[:, top_features].copy()
    #pd.DataFrame((feature_weights.loc[0,top_features].values).reshape((1, len(top_features))), columns = top_features, index = [0])
    
    return feature_weights, top_feature_weights, regressor, R2, RMSE, scaler_train, X_test, y_test, y_pred, X_train, y_train

def predict_plot_GP(regressor, X_test, y_test, scaler, plotting=True, title=None, 
                    groups = None, saveas = ''):
    
    X_test_scaled, y_test_scaled = scale(X_test, y_test, scaler)
    
    y_pred_scaled_array = regressor.predict(X_test_scaled)
    
    y_pred_scaled = y_test.copy()
    y_pred_scaled.iloc[:,-1] = y_pred_scaled_array
        
    X_pred, y_pred = inverseScale(X_test_scaled, y_pred_scaled, scaler)
    
    R2 = sklearn.metrics.r2_score(y_test, y_pred)
    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(mse)
    
    if plotting is True:
        if title is not None:
            title_temp = 'Results/log_MIC GP with ' + str(X_test.shape[1]) + ' features'
        else:
            title_temp = None
        if groups is not None:
            plot_RF_test(y_test, y_pred,
                     title = title_temp,
                     groups=groups.loc[y_test.index], saveas = saveas)
        else:
            plot_RF_test(y_test, y_pred,
                     title = title_temp,
                     groups=None, saveas = saveas)
    
    return R2, RMSE, y_pred