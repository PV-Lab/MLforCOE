#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 18:09:20 2021

Hyperparameter optimization for Init., Var., Cor. fingerprints and RF model.
Running this takes long so it is better to run on a server.

@author: armi tiihonen
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from bayes_opt import BayesianOptimization

def fetch_pickle(filename):
    """
    Fetches any variable saved into a picklefile with the given filename.
    
    Parameters:
        filename (str): filename of the pickle file
        
    Returns:
        variable (any pickle compatible type): variable that was saved into the picklefile.
    """
    with open(filename, 'rb') as picklefile:
        variable = pickle.load(picklefile)
    return variable

def fetch_csv(filename):
    """
    Fetches any variable saved into a picklefile with the given filename.
    
    Parameters:
        filename (str): filename of the pickle file
        
    Returns:
        variable (any pickle compatible type): variable that was saved into the picklefile.
    """
    variable = pd.read_csv(filename+'.csv', index_col=0)
    return variable


def rf_cv(max_depth, min_samples_leaf, min_samples_split,n_estimators, max_features, max_samples):
    max_features_options = ['log2', 'sqrt', 0.3, 0.5, None]
    params = {'bootstrap': True, 'max_depth': int(max_depth),
              'max_features': max_features_options[int(max_features)], 'min_samples_leaf': int(min_samples_leaf),
              'min_samples_split': int(min_samples_split), 
              'n_estimators': int(n_estimators), 'max_samples': max_samples}
    regressor.set_params(**params)
    regressor.fit(X_feature,np.ravel(y))
    #print(sorted(sklearn.metrics.SCORERS.keys()))
    scores = cross_val_score(regressor, X_feature, np.ravel(y), cv=20, scoring='neg_mean_squared_error')
        #cval = cross_val_score(regressor, X_feature, np.ravel(y),
    #                       scoring='neg_log_loss', cv=20)
    print(np.mean(scores))
    return np.mean(scores)#regressor.score(X_feature, np.ravel(y))#-1.0 * cval.mean()

#     we want to minimize error, so adding negative sign here because the BO later maximizes things
#    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

def plot_test(y_test, y_pred, title = None, xlabel = 'Measured $Y = \log_2(MIC)$', ylabel = 'Predicted $Y = \log_2(MIC)$', legend = ['Ideal', 'Result'], groups = None):
    """
    Plots the results of predicting test set y values using the random forest
    model.
    3
    Parameters:
        y_test (df): Experimental test set y values.
        y_pred (df): Predicted test set y values.
        title (str, optional): Title of the plot
        xlabel (str, optional)
        ylabel (str, optional)
        legend (str (2,), optional)
    """
    
    fig, ax  = plt.subplots(1,1)
    fig.set_figheight(5)
    fig.set_figwidth(5)
    if groups is not None:
        groups_obj = pd.concat([y_test,y_pred], axis=1).groupby(np.array(groups))
        cmap=plt.get_cmap('tab10')
        for name, group in groups_obj:
            # Works only for groups with numeric names that are max cmap length:
            ax.plot(group.iloc[:,0], group.iloc[:,1], marker="o", linestyle="", label=int(name), color = cmap.colors[int(name)])
            ax.legend()
    else:
        ax.scatter(y_test,y_pred, color = 'red')
    ax_max = 10
    if np.max(y_test.values)>ax_max:
        ax_max = np.max(y_test).values
    ax_min = 0
    if np.min(y_test.values)<ax_min:
        ax_min = np.min(y_test.values)
    ax.plot([ax_min, ax_max], [ax_min, ax_max], '--', color='black')
    ax.set_aspect('equal', 'box')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.savefig(title+'.pdf')
    plt.savefig(title+'.svg')
    #plt.savefig(title+'.png')#, dpi=600)
    #plt.show()
    
def fetch_my_data(X_path, y_path, groups_path):
    # I remember your code starts from here. import X and y. 
    X_feature = fetch_csv(X_path)
    y = fetch_csv(y_path)#'y_for_regressor_imp_no_ho')
    groups = fetch_csv(groups_path)
    print(X_feature)
    print(y)
    print(groups)
    # XGB is not compatible with special characters.
    #X_feature.columns = X_feature.columns.str.replace('[', '').str.replace(']','').str.replace('<','')
    #X_feature.columns = X_feature.columns.str.replace('(', '').str.replace(')','')
    #print(X_feature,y,groups)
    
    # change input args accordingly
    #X_train, X_test, y_train, y_test = train_test_split(X_feature, y, test_size=0.2, random_state=0)
    
    # as before, load the data
    # X is n by d, y is 1 by d
    #dtrain = xgb.DMatrix(data = X_feature, label=y)#xgb.DMatrix(data = np.array(X_train), label=np.array(y_train))
    print('Data loaded.')
    
    return X_feature, y, groups#, dtrain
    
def ho_with_bo_rf(run_name):
    
    # Below we can also use BO to tune XGB's hyperparams. This is customized version. 
    
    # set param range, I find these 4 most important
    # Feel free to change. 
    param_range =  {'max_depth': (2,20),
                   'min_samples_leaf': (1,5),
                   'min_samples_split': (2,8),
                   'n_estimators': (50,800),
                   'max_features': (0,4),
                   'max_samples': (0.1,0.99)}

    
    rf_bo = BayesianOptimization(rf_cv, param_range, verbose=3)
    
    # change the initial, acquisition function and n_iters. 
    # Given large enough n_iter, we should reach a relatively accurate model
    rf_bo.maximize(n_iter=300, init_points=50, acq='ei', alpha=1e-3,
    n_restarts_optimizer=3)
    
    # returns best performing params dict
    params = rf_bo.max['params']
    print(params)
    
    params['max_depth']= int(params['max_depth'])
    params['n_estimators']= int(params['n_estimators'])
    params['min_samples_leaf']= int(params['min_samples_leaf'])
    params['min_samples_split']= int(params['min_samples_split'])
    max_features_options = ['log2', 'sqrt', 0.3, 0.5, None]
    params['max_features']= max_features_options[int(params['max_features'])]
    
    print('Optimized parameters (', run_name, '): ', params)
    
    
    # this is final model
    # xgb_reg_final = xgb.XGBRegressor(**params).fit(np.array(X_feature), y)
    
    
    # you could also try the grid version as seen in RF tuning code, 
    # you should find similar solutions if running long enough (even better since it is exhausted search)
    # but it takes quite long and not sure if improves accuracy that much
    # so I would suggest using the BO guided hyperparameter optimization
    
    return None


if __name__ == "__main__":

    # In the server, Run1 are to optimize datasets Init, Var, and Cor. Don't look
    # Opt results from Run 1. Run 3 are to HO Opt fingerprint.
    X_path = ['./Data/Downselection data files/x_init_train_seed3', 
              './Data/Downselection data files/x_var_train_seed3',
              './Data/Downselection data files/x_cor_train_seed3',
              ]
    y_path = ['./Data/Downselection data files/y_init_train_seed3', 
              './Data/Downselection data files/y_var_train_seed3',
              './Data/Downselection data files/y_cor_train_seed3',
              ]
    groups_path = ['./Data/Downselection data files/groups_train_seed3',
                   './Data/Downselection data files/groups_train_seed3',
                   './Data/Downselection data files/groups_train_seed3',
                   ]
    run_name = ['Init', 'Var', 'Cor']
    
    for i in range(len(X_path)):
        X_feature, y, groups = fetch_my_data(X_path[i], y_path[i], groups_path[i])
        regressor = RandomForestRegressor(n_jobs = -2, criterion='mse')
        ho_with_bo_rf(run_name[i])
        print('BO HO run ', i, ' ended!\n\n\n')
