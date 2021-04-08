#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:28:12 2020

@author: armi tiihonen
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFECV, RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import sklearn

# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

def rfe_ranking(regressor, X, y, rel_ranks, int_ranks):
    colnames = X.columns
    regressor.fit(X,np.ravel(y))
    #rfe = RFECV(regressor, verbose=3, cv=20)
    rfe = RFE(regressor, n_features_to_select=1,#(X.shape[1]-2), 
    verbose=3)
    #stop the search when only the last feature is left
    rfe.fit(X,np.ravel(y))
    rel_ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
    int_ranks["RFE"] = dict(zip(colnames, rfe.ranking_))

def rfecv_ranking(regressor, X, y, rel_ranks, int_ranks):
    colnames = X.columns
    regressor.fit(X,y)
    # To do: implement stratified splitting into here using StratifiedShuffleSplit.
    #splitter = StratifiedShuffleSplit(n_splits=20, test_size=0.2, train_size=0.8)
    rfecv = RFECV(regressor, verbose=3, cv=50, n_jobs=-1, scoring='neg_root_mean_squared_error')
    rfecv.fit(X,np.ravel(y))
    rel_ranks["RFECV"] = ranking(list(map(float, rfecv.ranking_)), colnames, order=-1)
    int_ranks["RFECV"] = dict(zip(colnames, rfecv.ranking_))
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

def plot_ranking(ranking_scores, savefig=None, x_variable = '#features'):
    color = np.array(sns.color_palette())[0,:]
    #sns.set_style('ticks')
    sns.relplot(x=x_variable, y="RMSE", kind="line", ci="sd", data=ranking_scores,
        height=plt.rcParams['figure.figsize'][0],
        aspect=plt.rcParams['figure.figsize'][1]/plt.rcParams['figure.figsize'][0],
        color=color)
    plt.legend(['RF'])
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig+'_RMSE.svg')
        plt.savefig(savefig+'_RMSE.png')
        plt.savefig(savefig+'_RMSE.pdf')
    sns.relplot(x=x_variable, y="MSE", kind="line", ci="sd", data=ranking_scores,
        height=plt.rcParams['figure.figsize'][0],
        aspect=plt.rcParams['figure.figsize'][1]/plt.rcParams['figure.figsize'][0],
        color=color)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig+'_MSE.svg')
        plt.savefig(savefig+'_MSE.png')
        plt.savefig(savefig+'_MSE.pdf')
    sns.relplot(x=x_variable, y="R2", kind="line", ci="sd", data=ranking_scores,
        height=plt.rcParams['figure.figsize'][0],
        aspect=plt.rcParams['figure.figsize'][1]/plt.rcParams['figure.figsize'][0],
        color=color)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig+'_R2.svg')
        plt.savefig(savefig+'_R2.png')
        plt.savefig(savefig+'_R2.pdf')
    sns.relplot(x=x_variable, y="RMSE", kind="line", data=ranking_scores,
                estimator='var', ci=None,
        height=plt.rcParams['figure.figsize'][0],
        aspect=plt.rcParams['figure.figsize'][1]/plt.rcParams['figure.figsize'][0],
        color=color)
    plt.ylabel('$\sigma$(RMSE)')
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig+'_varRMSE.svg')
        plt.savefig(savefig+'_varRMSE.png')
        plt.savefig(savefig+'_varRMSE.pdf')
    sns.relplot(x=x_variable, y="R2", kind="line", data=ranking_scores, 
                estimator='var', ci=None,
        height=plt.rcParams['figure.figsize'][0],
        aspect=plt.rcParams['figure.figsize'][1]/plt.rcParams['figure.figsize'][0],
        color=color)
    plt.ylabel('$\sigma$(R$^2$)')
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig+'_varR2.svg')
        plt.savefig(savefig+'_varR2.png')
        plt.savefig(savefig+'_varR2.pdf')
                
def score_feature_ranking_deep_cv(regressor, X, y, groups, n_seeds = 20, test_proportion = 0.2):
    '''
    This function does a separate RFE for every cross validation split and
    averages the results.

    Parameters
    ----------
    regressor : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    groups : TYPE
        DESCRIPTION.
    rank : TYPE
        DESCRIPTION.
    n_seeds : TYPE, optional
        DESCRIPTION. The default is 20.
    test_proportion : TYPE, optional
        DESCRIPTION. The default is 0.2.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    results : TYPE
        DESCRIPTION.

    '''
    features = X.columns
    n_features = features.shape[0]
    results = pd.DataFrame(columns=['#features', 'seed', 'RMSE', 'R2', 'MSE', 'y_test', 'y_pred', 'features'])
    rank_rfe = []
    for i in range(n_seeds):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion, random_state=i, stratify=groups)
        rel_ranks_temp = {}
        int_ranks_temp = {}
        rfe_ranking(regressor, X_train, y_train, rel_ranks_temp, int_ranks_temp)
        rank_rfe.append(pd.DataFrame({k: [v] for k, v in int_ranks_temp["RFE"].items()}))
        for j in range(n_features):
            print(i,j)
            #mask = rank.values[0] <= (j+1)
            mask = rank_rfe[i].values[0] <= (j+1)
            #if not np.all(rank.columns == X.columns):
            if not np.all(rank_rfe[i].columns == X.columns):
                raise Exception("Something wrong with the data!")
            #X_train, X_test, y_train, y_test = train_test_split(X.loc[:,mask], y, test_size=test_proportion, random_state=i, stratify=groups)
            regressor.fit(X_train.loc[:,mask],np.ravel(y_train))
            y_pred = regressor.predict(X_test.loc[:,mask])
            R2 = sklearn.metrics.r2_score(y_test, y_pred)
            mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
            RMSE = np.sqrt(mse)
            results.loc[len(results)] = [j+1, i, RMSE, R2, mse, y_test, y_pred, X_train.columns]
    plot_ranking(results, savefig='RFE_d_RF_ranking')
    return results, rank_rfe            

def score_feature_ranking_shallow_cv(regressor, X, y, groups, n_seeds = 20, test_proportion = 0.2, filename = './Results/RFE_s_rf_ranking'):
    '''
    This function performs RFE for the whole input dataset and then estimates
    cross validation score.

    Parameters
    ----------
    regressor : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    groups : TYPE
        DESCRIPTION.
    rank : TYPE
        DESCRIPTION.
    n_seeds : TYPE, optional
        DESCRIPTION. The default is 20.
    test_proportion : TYPE, optional
        DESCRIPTION. The default is 0.2.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    results : TYPE
        DESCRIPTION.

    '''
    rel_ranks = {}
    int_ranks = {}
    rfe_ranking(regressor, X, y, rel_ranks, int_ranks)
    rank = pd.DataFrame({k: [v] for k, v in int_ranks["RFE"].items()})
    
    features = X.columns
    n_features = features.shape[0]
    results = pd.DataFrame(columns=['#features', 'seed', 'RMSE', 'R2', 'MSE', 'y_test', 'y_pred', 'features'])
    for i in range(n_seeds):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion, random_state=i, stratify=groups)
        for j in range(n_features):
            print(i,j)
            mask = rank.values[0] <= (j+1)
            if not np.all(rank.columns == X.columns):
                raise Exception("Something wrong with the data!")
            #X_train, X_test, y_train, y_test = train_test_split(X.loc[:,mask], y, test_size=test_proportion, random_state=i, stratify=groups)
            regressor.fit(X_train.loc[:,mask],np.ravel(y_train))
            y_pred = regressor.predict(X_test.loc[:,mask])
            R2 = sklearn.metrics.r2_score(y_test, y_pred)
            mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
            RMSE = np.sqrt(mse)
            results.loc[len(results)] = [j+1, i, RMSE, R2, mse, y_test, y_pred, X_train.columns]
    plot_ranking(results, savefig=filename)
    return results, [rank]            

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

def save_to_csv_pickle(dataset, filename, join_with = None):
    """
    Saves any dataset to a csv file and picklefile with the given filename.
    
    Parameters:
        dataset (any pickle and to_csv compatible type): dataset to be saved into file
        filename (str): filename used for both csv and pickle file
    """
    dataset.to_csv(filename + '.csv', index=True)
    picklefile = open(filename, 'wb')
    pickle.dump(dataset,picklefile)
    picklefile.close()
    if join_with is not None:
        (join_with.join(dataset)).to_csv(filename + '.csv', index=True)


if __name__ == "__main__":
    
    # Choose if you want to run RFE (takes long):
    run_on_server = False
    # If the above is set to False, the code will only load the data and plot
    # figures.
    
    test_proportion = 0.2
    ho_params_cor = {'bootstrap': True, 'max_depth': 13, 'max_features': 0.5,
                     'max_samples': 0.99, 'min_samples_leaf': 2,
                     'min_samples_split': 4, 'n_estimators': 140}
    
    # Optimum number of molecular descriptors has been determined manually
    # from the RFE results. This is done by finding the minimum error region
    # by looking at the dataframe+graphs or by using idxmin. This variable
    # affects to how many descriptors are printed out when reporting results,
    # not to RFE itself.
    optimum_cutoff = 9
    
    ###########################################################################
    # Load data.
    X = fetch_csv('./Data/Downselection_data_files/x_cor_train_seed3')
    y = fetch_csv('./Data/Downselection_data_files/y_cor_train_seed3')
    groups = fetch_csv('./Data/Downselection_data_files/groups_train_seed3')
    
    ###########################################################################
    if run_on_server == True:
        
        regressor = RandomForestRegressor(n_jobs = -2, criterion='mse')
        regressor.set_params(**ho_params_cor)
       
        print('\n RFE for RF starts. \n')
        
        rfe_s_results, rfe_s_ranks = score_feature_ranking_shallow_cv(regressor,
                                                                      X, y, groups,
                                                                      n_seeds = 20, test_proportion = 0.2)
        print('Shallow RFE results and ranks: \n')
        print(rfe_s_results)
        print(rfe_s_ranks)
        print('These are top-25 features:')
        rfe_s_chosen_features = rfe_s_ranks[0].iloc[0,(rfe_s_ranks[0].iloc[0,:]<26).values]
        print(rfe_s_chosen_features)
        save_to_csv_pickle(rfe_s_results, './Data/Downselection_data_files/rfe_rf_results')
        save_to_csv_pickle(rfe_s_ranks[0], './Data/Downselection_data_files/rfe_rf_ranks')
        save_to_csv_pickle(rfe_s_chosen_features, './Data/Downselection_data_files/rfe_rf_top25_features')
    
    else:
        
        # The optimum cut-off is defined manually from the top features list.
        rfe_s_results = fetch_csv('./Data/Downselection_data_files/rfe_rf_results')
        rfe_s_ranks = fetch_csv('./Data/Downselection_data_files/rfe_rf_ranks')
        
        rmse = []
        for i in range(rfe_s_results.loc[:,'#features'].max()):
            rmse.append(np.mean(rfe_s_results[rfe_s_results.loc[:,'#features']==(i+1)].loc[:,'RMSE']))
        rmse_df = pd.DataFrame(data=rmse, columns=['mean RMSE'])
        
        
        
        chosen_features = rfe_s_ranks.loc[:,(rfe_s_ranks<(optimum_cutoff+1)).values[0]].columns
        print('Chosen features are: ', chosen_features)
    
        from set_figure_defaults import FigureDefaults
        sns.set_palette('gray')
        mystyle = FigureDefaults('nature_comp_mat_tc')
        plot_ranking(rfe_s_results, savefig='./Results/rfe_rf_ranking')