#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:28:12 2020

@author: armi
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFECV, RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split#, StratifiedShuffleSplit
import pickle
import sklearn
from set_figure_defaults import FigureDefaults
from RFE_RF import ranking, plot_ranking, fetch_pickle, fetch_csv, save_to_csv_pickle, score_feature_ranking_deep_cv, score_feature_ranking_shallow_cv, rfe_ranking, rfecv_ranking
import xgboost as xgb
sns.set_palette('colorblind')


if __name__ == "__main__":

    # Choose if you want to run RFE (takes long):
    run_on_server = False
    # If the above is set to False, the code will only load the data and plot
    # figures.
    
    test_proportion = 0.2
    ho_params_cor = {'eta': 0.3054092191428568, 'gamma': 1.9386571606670655, 'max_depth': 5, 'n_estimators': 357}
    
    # Optimum number of molecular descriptors has been determined manually
    # from the RFE results. This is done by finding the minimum error region
    # by looking at the dataframe+graphs or by using idxmin. This variable
    # affects to how many descriptors are printed out when reporting results,
    # not to RFE itself.
    optimum_cutoff = 18
    ###########################################################################
    # Load data.
    X = fetch_csv('./Data/Downselection_data_files/x_cor_train_seed3', index_col=None)
    y = fetch_csv('./Data/Downselection_data_files/y_cor_train_seed3').iloc[:,[-1]] # Dropping smiles.
    groups = fetch_csv('./Data/Downselection_data_files/groups_train_seed3')
    
    # XGB doesn't like special characters.
    X.columns = X.columns.str.replace('[', '').str.replace(']','').str.replace('<','')
    X.columns = X.columns.str.replace('(', '').str.replace(')','')
    ###########################################################################
    if run_on_server == True:
        
        regressor = regressor = xgb.XGBRegressor(**ho_params_cor)
        regressor.set_params(**ho_params_cor)
       
        print('\n RFE for RF starts. \n')
        
        rfe_s_results, rfe_s_ranks = score_feature_ranking_shallow_cv(regressor,
                                                                      X, y, groups,
                                                                      n_seeds = 20,
                                                                      test_proportion = 0.2,
                                                                      filename = './Results/RFE_s_xgb_ranking')
        print('Shallow RFE results and ranks: \n')
        print(rfe_s_results)
        print(rfe_s_ranks)
        print('These are top-25 features:')
        rfe_s_chosen_features = rfe_s_ranks[0].iloc[0,(rfe_s_ranks[0].iloc[0,:]<26).values]
        print(rfe_s_chosen_features)
        save_to_csv_pickle(rfe_s_results, './Data/Downselection_data_files/rfe_xgb_results')
        save_to_csv_pickle(rfe_s_ranks[0], './Data/Downselection_data_files/rfe_xgb_ranks')
        save_to_csv_pickle(rfe_s_chosen_features, './Data/Downselection_data_files/rfe_xgb_top25_features')
    
    else:
        
        # The optimum cut-off is defined manually from the top features list.
        rfe_s_results = fetch_csv('./Data/Downselection_data_files/rfe_xgb_results')
        rfe_s_ranks = fetch_csv('./Data/Downselection_data_files/rfe_xgb_ranks')
        
        rmse = []
        for i in range(rfe_s_results.loc[:,'#features'].max()):
            rmse.append(np.mean(rfe_s_results[rfe_s_results.loc[:,'#features']==(i+1)].loc[:,'RMSE']))
        rmse_df = pd.DataFrame(data=rmse, columns=['mean RMSE'])
        
        
        
        chosen_features = rfe_s_ranks.loc[:,(rfe_s_ranks<(optimum_cutoff+1)).values[0]].columns
        print('Chosen features are: ', chosen_features)
    
        from set_figure_defaults import FigureDefaults
        sns.set_palette('gray')
        mystyle = FigureDefaults('nature_comp_mat_tc')
        plot_ranking(rfe_s_results, savefig='./Results/rfe_xgb_ranking')
