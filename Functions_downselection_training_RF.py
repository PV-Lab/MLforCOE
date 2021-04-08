from set_figure_defaults import FigureDefaults
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import operator
import warnings
import pickle
import sklearn as sklearn


def plot_heatmap(corrMatrix, title = '', vmin=None, vmax=None, cmap=None, ticklabels=False):
    """
    Plots a correlation matrix as a labeled heatmap.
    
    Parameters:
        corrMatrix (df): Correlation matrix
        title (str, optional): Title of the plot
        annot (boolean, optional): Show the value of each matrix cell in the plot.
    """
    if cmap == None:
        cmap_heatmap = "icefire"
    else:
        cmap_heatmap = cmap
    sn.heatmap(corrMatrix, vmin=vmin, vmax=vmax, cmap=cmap_heatmap, square=True, xticklabels=ticklabels, yticklabels=ticklabels, rasterized=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./Results/'+title+'.png', dpi=300)
    plt.savefig('./Results/'+title+'.pdf')
    #plt.savefig('./Results/'+title+'.svg')
    plt.show()
    

def plot_RF_test(y_test, y_pred, title = None, xlabel = 'Measured $\log_2(MIC)$', ylabel = 'Predicted $\log_2(MIC)$', legend = ['Ideal', 'Result'], groups = None, saveas = None):
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
    sn.set_palette('colorblind')
    def_color = 'k'#np.array(sn.color_palette())[0,:]
    #fig, ax  = plt.subplots(1,1)
    ##fig.set_figheight(5)
    ##fig.set_figwidth(5)
    if groups is not None:
        groups_obj = pd.concat([y_test, y_pred], axis=1).groupby(groups)
        cmap=plt.get_cmap('tab10')
        for name, group in groups_obj:
            # Works only for groups with numeric names that are max cmap length:
            fig, ax  = plt.subplots(1,1)
            ax.plot(group.iloc[:,0], group.iloc[:,1], marker=".", linestyle="", label=int(name), color = cmap.colors[int(name)])
            #ax.legend()
    else:
        sn.scatterplot(x=y_test.values.ravel(),y=y_pred.values.ravel(), color=def_color)
        #ax.scatter(y_test,y_pred, color = 'red', marker='.')
    ax_max = 10
    if np.max(y_test.values)>ax_max:
        ax_max = np.max(y_test).values
    ax_min = 0
    if np.min(y_test.values)<ax_min:
        ax_min = np.min(y_test.values)
    plt.plot([ax_min, ax_max], [ax_min, ax_max], '--', color='black')
    #plt.gca().set_aspect('equal', 'box')
    if title is not None:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if (saveas is None) and (title is not None):
        plt.savefig(title+'.pdf')
        plt.savefig(title+'.svg')
        plt.savefig(title+'.png', dpi=300)
        #plt.show()
    elif (saveas is not None):
        plt.savefig(saveas+'.pdf')
        plt.savefig(saveas+'.svg')
        plt.savefig(saveas+'.png', dpi=300)
    plt.show()
    
def splitAndScale(X, y, test_size, random_state = None):
    """
    Splits the data into train and test sets. Scales the train and test sets
    using a StandardScaler (sklearn). The datasets are being scaled separately
    to avoid "leaking" information from train to test set.
    
    Parameters:
        X (df): X data to be split and scaled (features in columns, samples in rows)
        y (df): y data to be split and scaled (one column, samples in rows)
        test_size (float): Proportion of the test size from the original data.
    
    Returns:
        X_train_scaled (df): X data of the train set
        X_test_scaled (df): X data of the test set
        y_train_scaled (df): y data of the train set
        y_test_scaled (df): y data of the test set
        scaler_train (StandardScaler): StandardScaler that is needed for scaling the
        train set back to initial units.
        scaler_test (StandardScaler): StandardScaler that is needed for scaling the
        test set back to initial units.
        random_state (int, optional): Seed for train test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    # Scale.
    scaler_test = preprocessing.StandardScaler()
    scaler_train = preprocessing.StandardScaler()
    test_scaled = X_test.copy()
    test_scaled[y_test.columns[0]] = y_test.values
    train_scaled = X_train.copy()
    train_scaled[y_train.columns[0]] = y_train.values
    test_scaled = pd.DataFrame(scaler_test.fit_transform(test_scaled), columns=test_scaled.columns, index=test_scaled.index)
    train_scaled = pd.DataFrame(scaler_train.fit_transform(train_scaled), columns=train_scaled.columns, index=train_scaled.index)
    X_train_scaled = train_scaled.iloc[:,:-1]
    y_train_scaled = train_scaled.iloc[:,[-1]]#y_train#
    X_test_scaled = test_scaled.iloc[:,:-1]
    y_test_scaled = test_scaled.iloc[:,[-1]]#y_test#
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_train, scaler_test


    
def define_scale(X_train, y_train):
    
    scaler_train = preprocessing.StandardScaler()
    train_scaled = X_train.copy()
    train_scaled[y_train.columns[-1]] = y_train.values
    
    train_scaled = pd.DataFrame(scaler_train.fit_transform(train_scaled), columns=train_scaled.columns, index=train_scaled.index)
    
    X_train_scaled = train_scaled.iloc[:,:-1]
    y_train_scaled = train_scaled.iloc[:,[-1]]

    return X_train_scaled, y_train_scaled, scaler_train

def scale(X_data, y_data, scaler):
    
    data_scaled = X_data.copy()
    data_scaled[y_data.columns[-1]] = y_data.values
    
    data_scaled = pd.DataFrame(scaler.transform(data_scaled), columns=data_scaled.columns, index=data_scaled.index)
    
    X_data_scaled = data_scaled.iloc[:,:-1]
    y_data_scaled = data_scaled.iloc[:,[-1]]

    return X_data_scaled, y_data_scaled

def inverseScale(X_data, y_data, scaler):
    
    datasets_scaled = X_data.copy()
    datasets_scaled[y_data.columns[-1]] = y_data.values
    
    datasets_unscaled = pd.DataFrame(scaler.inverse_transform(datasets_scaled), columns=datasets_scaled.columns, index = datasets_scaled.index)
    
    X_data_unscaled = datasets_unscaled.iloc[:,:-1]
    y_data_unscaled = datasets_unscaled.iloc[:,[-1]]
    
    return X_data_unscaled, y_data_unscaled


def RF_feature_analysis(X, y, groups = None, groups_only_for_plotting = False, 
                        test_indices = None, test_proportion = 0.1, top_n = 5, 
                        n_estimators = 100, max_depth = None, 
                        min_samples_split = 2, min_samples_leaf = 1, 
                        max_features = 'auto', bootstrap = True, i='', 
                        random_state = None, sample_weighing = True, 
                        plotting = True, saveas = None, title = True, max_samples = None):
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
    
    # Fit and estimate feature importances.
    regressor = RandomForestRegressor(n_estimators = n_estimators, 
                                         max_depth = max_depth,
                                          min_samples_split = min_samples_split,
                                          min_samples_leaf = min_samples_leaf,
                                          max_features = max_features,
                                          bootstrap = bootstrap,
                                          n_jobs = -2, criterion='mse',
                                          max_samples = max_samples)
    
    #regressor = RandomForestRegressor(n_jobs = -2, criterion='mse')
    #print(X_train.shape, y_train.shape)
    regressor.fit(X_train,np.ravel(y_train),  sample_weight = sample_weight)
    R2, RMSE, y_pred = predict_plot_RF(regressor, X_test, y_test, 
                                       plotting=plotting, title=title, 
                                       groups = groups, saveas = saveas)
    feature_weight = regressor.feature_importances_
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
    
    scaler_test = None
    
    return feature_weights, top_feature_weights, regressor, R2, RMSE, scaler_test, X_test, y_test, y_pred, X_train, y_train
    
def predict_plot_RF(regressor, X_test, y_test, plotting=True, title=None, groups = None, saveas = '', ):
    y_pred = regressor.predict(X_test)
    if y_test is None:
        y_pred = pd.DataFrame(y_pred, index=X_test.index, columns=['log2mic'])
        R2 = None
        mse = None
        RMSE = None
    
    else:
        y_pred = pd.DataFrame(data=y_pred, index=y_test.index, columns=['log2mic'])
        R2 = sklearn.metrics.r2_score(y_test, y_pred)
        mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
        RMSE = np.sqrt(mse)
        
    #y_pred = np.exp2(y_pred) # Exponential data didn't look good in the plot.
    #y_test = np.exp2(y_test)

    if plotting is True:
        if title is not None:
            title_temp = 'Results/log_MIC RF with ' + str(X_test.shape[1]) + ' features'
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

def save_to_csv_pickle(dataset, filename, join_with = None, index=True):
    """
    Saves any dataset to a csv file and picklefile with the given filename.
    
    Parameters:
        dataset (any pickle and to_csv compatible type): dataset to be saved into file
        filename (str): filename used for both csv and pickle file
    """
    dataset.to_csv(filename + '.csv', index=index)
    picklefile = open(filename, 'wb')
    pickle.dump(dataset,picklefile)
    picklefile.close()
    if join_with is not None:
        (join_with.join(dataset)).to_csv(filename + '.csv', index=index)

def save_to_pickle(dataset, filename):
    """
    Saves any dataset to a csv file and picklefile with the given filename.
    
    Parameters:
        dataset (any pickle and to_csv compatible type): dataset to be saved into file
        filename (str): filename used for both csv and pickle file
    """
    picklefile = open(filename, 'wb')
    pickle.dump(dataset,picklefile)
    picklefile.close()

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

def fetch_pickled_HO(filename):
    """
    Fetches random forest regression hyperparamaters saved into a picklefile
    and returns each hyperparameter.
    
    Parameters:
        filename (str): Filename of the pickle file. An example of the variable
                        that is expected to be stored in the pickle file:
                            pickled_variable = {'bootstrap': True,\n",
                            'max_depth': 18,\n",
                            'max_features': 'sqrt',\n",
                            'min_samples_leaf': 1,\n",
                            'min_samples_split': 2,\n",
                            'n_estimators': 300}
    
    Returns:
        n_estimators (int, optional): Number of trees in the forest
        max_depth (int, optional): Maximum depth of the tree
        min_samples split (int, optional): minimum number of samples required
            to split an internal node (could also be a float, see sklearn
            documentation)
        min_samples_leaf (int, optional): The minimum number od samples to be
        at a leaf node (could also be a float, see sklearn documentation)
        max_features (str, float, string, or None, optional): The number of
        features to consider when looking for the best split (see the options
        in sklearn documentation, 'sqrt' means max number is sqrt(number of
        features))
        bootstrap (boolean, optional): False means the whole dataset is used
        for building each tree, True means bootstrapping of samples is used
    """
    ho = fetch_pickle(filename)
    bootstrap = ho['bootstrap']
    max_depth = ho['max_depth']
    max_features = ho['max_features']
    min_samples_leaf = ho['min_samples_leaf']
    min_samples_split = ho['min_samples_split']
    n_estimators = ho['n_estimators']
    
    return n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap

def read_molecule_excel(filename, sheet_smiles_y_id = 'SMILES',
                        column_smiles = 'SMILES ',
                        column_y = 'MIC VALUE (Y VALUE)',
                        column_id = 'No.',
                        column_class = 'Class',
                        column_name = 'NAME',
                        sheet_features = ['1k','2k','3k','4k','5k','300'],
                        start_column_features = 2):
    """
    Reads molecule ID, output to be optimized, and features from the given
    sheets of the given Excel file, and outputs them as a single DataFrame.
    
    Parameters:
        filename (str): Filename of the dataset Excel file.
        sheet_smiles_y_idx (str,optional): To do
        column_smiles (str,optional): To do
        column_y (str,optional): To do
        column_id (str,optional): To do
        sheet_features ([str],optional): To do
        start_column_features (int,optional): To do
    
    Returns:
        dataset_original (df): Dataframe with molecules on each row, and
        columns in this order: [Idx, y value, feature0, feature1, ...]
    """
    datasets = pd.read_excel(filename,
                                     sheet_name = [sheet_smiles_y_id].extend(
                                         sheet_features),
                                     na_values='na', convert_float = False)
    if column_class is not None:
        dataset_original = (datasets[sheet_smiles_y_id]).loc[:, [column_id, column_name, column_class, column_smiles, column_y]]
    else:
        dataset_original = (datasets[sheet_smiles_y_id]).loc[:, [column_id, column_name, column_smiles, column_y]]
    for i in range(len(sheet_features)):
        dataset_original = pd.concat([dataset_original,
                                    datasets[sheet_features[i]
                                             ].iloc[:, start_column_features::]],
                                    axis=1)
    
    return dataset_original


# Old version, might still be in use somewhere. Doesn't have regressors as output.
'''
def analyze_RF_for_multiple_seeds(list_X, list_y, ho_params = None, n_seeds = 20, save_pickle = False, bar_plot = True, groups = None, groups_only_for_plotting = False, test_proportion = 0.21, top_n = 20, plotting=True):
    n_datasets = len(list_X)
    
    # Let's repeat y stratification. At the same, let's create a dataset for
    # RF hyperparameter optimization.
    R2_all2 = np.zeros((n_seeds,n_datasets))
    RMSE_all2 = np.zeros((n_seeds,n_datasets))
    top_features_all2 = [[None]*n_seeds]*n_datasets
    features_all2 = [[None]*n_seeds]*n_datasets
    X_tests = [[None]*n_seeds]*n_datasets
    y_tests = [[None]*n_seeds]*n_datasets
    X_trains = [[None]*n_seeds]*n_datasets
    y_trains = [[None]*n_seeds]*n_datasets
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

        for i in range(n_seeds):
            if ho_params is None:
                feature_weights, top_feature_weights, regressor, R2, RMSE, scaler_test, X_test, y_test, y_pred, X_train, y_train = RF_feature_analysis(
                        list_X[j], list_y[j], groups=groups,
                        groups_only_for_plotting = groups_only_for_plotting,
                        test_indices = None, test_proportion = test_proportion,
                        top_n = top_n, i='', random_state = i,
                        sample_weighing = False, plotting=plotting)
            else:
                feature_weights, top_feature_weights, regressor, R2, RMSE, scaler_test, X_test, y_test, y_pred, X_train, y_train = RF_feature_analysis(
                        list_X[j], list_y[j], groups=groups,
                        groups_only_for_plotting = groups_only_for_plotting,
                        test_indices = None, test_proportion = test_proportion,
                        top_n = top_n, i='', random_state = i,
                        sample_weighing = False, n_estimators=n_estimators,
                        max_depth=max_depth, min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf, 
                        max_features=max_features, bootstrap=bootstrap, plotting=plotting)
            R2_all2[i,j] = R2
            RMSE_all2[i,j] = RMSE
            top_features_all2[j][i] = top_feature_weights.copy()
            features_all2[j][i] = feature_weights.copy()
            X_tests[j][i] = X_test.copy()
            y_tests[j][i] = y_test.copy()
            X_trains[j][i] = X_train.copy()
            y_trains[j][i] = y_train.copy()
            #if (i == 0) and (j==0):
            #    top_feature_weights2 = top_feature_weights
            #if (i == 0) and (j==1):
            #    top_feature_weights_imp2 = top_feature_weights_imp
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
    
    return R2_all2, RMSE_all2, top_features_all2, features_all2, X_tests, y_tests, X_trains, y_trains
'''
def analyze_RF_for_multiple_seeds(list_X, list_y, ho_params = None, n_seeds = 20, save_pickle = False, bar_plot = True, groups = None, groups_only_for_plotting = False, test_proportion = 0.21, top_n = 20, plotting=True, saveas = None, title=True):
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
            max_samples = ho_params[j]['max_samples']
        top_features_temp = []
        features_temp = []
        X_tests_temp = []
        y_tests_temp = []
        X_trains_temp = []
        y_trains_temp = []
        regressors_temp = []
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
                feature_weights, top_feature_weights, regressor, R2, RMSE, scaler_test, X_test, y_test, y_pred, X_train, y_train = RF_feature_analysis(
                        list_X[j], list_y[j], groups=groups,
                        groups_only_for_plotting = groups_only_for_plotting,
                        test_indices = None, test_proportion = test_proportion,
                        top_n = top_n, i='', random_state = i,
                        sample_weighing = False, plotting=plotting, saveas = saveas_temp, title = title_temp)
            else:
                feature_weights, top_feature_weights, regressor, R2, RMSE, scaler_test, X_test, y_test, y_pred, X_train, y_train = RF_feature_analysis(
                        list_X[j], list_y[j], groups=groups,
                        groups_only_for_plotting = groups_only_for_plotting,
                        test_indices = None, test_proportion = test_proportion,
                        top_n = top_n, i='', random_state = i,
                        sample_weighing = False, n_estimators=n_estimators,
                        max_depth=max_depth, min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf, 
                        max_features=max_features, bootstrap=bootstrap, plotting=plotting, saveas = saveas_temp, title = title_temp, max_samples = max_samples)
            R2_all2[i,j] = R2
            RMSE_all2[i,j] = RMSE
            top_features_temp.append(top_feature_weights.copy())
            features_temp.append(feature_weights.copy())
            X_tests_temp.append(X_test.copy())
            y_tests_temp.append(y_test.copy())
            X_trains_temp.append(X_train.copy())
            y_trains_temp.append(y_train.copy())
            regressors_temp.append(regressor)
            
        top_features_all2.append(top_features_temp)
        features_all2.append(features_temp)
        X_tests.append(X_tests_temp)
        y_tests.append(y_tests_temp)
        X_trains.append(X_trains_temp)
        y_trains.append(y_trains_temp)
        regressors.append(regressors_temp)
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
    
    return R2_all2, RMSE_all2, top_features_all2, features_all2, X_tests, y_tests, X_trains, y_trains, regressors

def compare_features_barplot(feature_weights1, feature_weights2, filename_fig = None, title=None):
    features_to_append = feature_weights2.copy()
    rf_features_for_plots = feature_weights1.copy()
    rf_features_for_plots = rf_features_for_plots.append(features_to_append, sort=False, ignore_index = True)
    rf_features_for_plots=pd.melt(rf_features_for_plots.reset_index(), value_vars=rf_features_for_plots.columns,
                  id_vars = 'index')
    plt.figure()
    sn.barplot(x='value', y='variable', hue='index', data = rf_features_for_plots)
    if title is not None:
        plt.title(title)
    plt.show()
    if filename_fig is not None:
        plt.savefig(filename_fig+'.png')
        plt.savefig(filename_fig+'.pdf')
        plt.savefig(filename_fig+'.svg')
    return None

# The following functions are meant for functionalizing the feature selection code. Not used in this file.
def clean_mics(dataset, y_column):
    # Replace e.g. '>128' with 128*2 in y data (in column 2).
    idx = dataset[dataset.iloc[:,y_column].str.find('>')==0].index
    y_column_label = dataset.columns[y_column]
    dataset.loc[idx,y_column_label] = dataset.loc[idx,y_column_label].str[1::]
    dataset.loc[:,y_column_label] = np.double(dataset.loc[:,y_column_label])
    # Approximate "MIC>X" values with the next highest available MIC value (2*X).
    dataset.loc[idx, y_column_label] = dataset.loc[idx, y_column_label]*2
    # Drop rows with y data nan, and columns with any nan.
    dataset = dataset.dropna(axis=0, how='all', subset=[y_column_label])
    dataset = dataset.dropna(axis=1, how='any')
    if (y_column_label != 'MIC VALUE (Y VALUE)') and (y_column_label != 'log2mic'):
        warnings.warn('Dataset is not as expected. Check that everything is ok.')
    return dataset

def logmic(dataset, y_column):
    # First, take log from Y feature.
    dataset.iloc[:,y_column] = np.log2(dataset.iloc[:,y_column])
    return dataset

def corrMatrix(dataset, y_column, corrMethod='spearman'):
    corrMatrix = dataset.iloc[:,y_column::].corr(method=corrMethod)
    return corrMatrix

def var_filtering(dataset, y_column, variance_limit=0.1, plotCorrMatrix = True, corrMethod = 'spearman'):
    corrMatrixInitial = dataset.iloc[:,y_column::].corr(method=corrMethod)
    if plotCorrMatrix == True:
        plot_heatmap(corrMatrixInitial, 'Initial dataset: ' 
                 + str(corrMatrixInitial.shape[0]-1) + ' descriptors')
    print('Initial dataset: ' + str(corrMatrixInitial.shape[0]-1) + ' descriptors')
    # Drop constant features (note: this goes through also the No., SMILES, and y
    # value columns but it shouldn't be a problem because they are not constants)
    # Not needed anymore after variance filtering is implemented.
    # dataset = dataset.drop(columns=dataset.columns[(dataset == dataset.iloc[0,:]).all()])
    # Drop almost constant features (do not check No, SMILES, y value columns).
    idx_boolean = [False]*(y_column)
    idx_boolean.append(True)
    idx_boolean.extend(((np.var(dataset.iloc[:,(y_column+1)::])/np.mean(dataset.iloc[:,(y_column+1)::]))>variance_limit).values) #Numpy booleans here instead of python booleans, is it ok?
    
    corrMatrixVar = dataset.iloc[:,idx_boolean].corr(method=corrMethod)
    if plotCorrMatrix == True:
        plot_heatmap(corrMatrixVar, 'After dropping constant or almost constant descriptors: ' 
                 + str(corrMatrixVar.shape[0]-1) + ' descriptors')
    print('After dropping constant or almost constant descriptors: ' 
                 + str(corrMatrixVar.shape[0]-1) + ' descriptors')
    return corrMatrixInitial, corrMatrixVar

def cor_filtering(dataset, y_column, filterWithCorrMatrix = False, corrMatrixForFiltering = None, plotCorrMatrix = True, corrMethod = 'spearman', corr_limit1 = 0.9, corr_limit2 = 0.05):
    # Full correlation matrix with corrMatrixForFiltering taken into account.
    if filterWithCorrMatrix == False:
        corrMatrix = dataset.iloc[:,y_column::].corr(method=corrMethod)#'pearson')#
    else:
        corrMatrix = (dataset.loc[:,corrMatrixForFiltering.columns]).corr(method=corrMethod)#
    if plotCorrMatrix == True:
        plot_heatmap(corrMatrix, 'After dropping constant or almost constant descriptors: ' 
                 + str(corrMatrix.shape[0]-1) + ' descriptors')
    print('After dropping constant or almost constant descriptors: ' 
                 + str(corrMatrix.shape[0]-1) + ' descriptors')
    '''
    # See which features correlate with Y more than others.
    corrMatrixImportant = corrMatrix.loc[:,(np.abs(corrMatrix.iloc[0,:])>0.01).values]
    plot_heatmap(corrMatrixImportant)
    # --> Still a lot of correlating features.
    '''
    
    # Next, we want to drop features correlating too much with each other.
    # Mask upper triangle to drop only the other one of each two correlated features.
    corr_limit = corr_limit1 # Final value: 0.95
    tri_corrMatrix = pd.DataFrame(np.triu(corrMatrix,1), index = corrMatrix.index,
                                  columns = corrMatrix.columns)
    # List column names of highly correlated features.
    to_drop = [c for c in tri_corrMatrix.columns if any(np.abs(tri_corrMatrix[c]) > corr_limit)]
    # And drop them.
    corrMatrixCorX = corrMatrix.drop(columns = to_drop, index = to_drop)
    if plotCorrMatrix == True:
        plot_heatmap(corrMatrixCorX, 'After filtering out highly correlated descriptors (limit ' +
                str(corr_limit) + ': ' + str(corrMatrixCorX.shape[0]-1) + ' descriptors')
    print('After filtering out highly correlated descriptors (limit ' +
                str(corr_limit) + ': ' + str(corrMatrixCorX.shape[0]-1) + ' descriptors')
    # See again which of the remaining features correlate with Y.
    corr_limit = corr_limit2 # Final values: 0.025
    corrMatrixCor = corrMatrixCorX.loc[(np.abs(
            corrMatrixCorX.iloc[0,:])>corr_limit).values,(np.abs(
                    corrMatrixCorX.iloc[0,:])>corr_limit).values]
    if plotCorrMatrix == True:
        plot_heatmap(corrMatrixCor, 'Correlation with Y higher than ' + 
                 str(corr_limit) + ': ' + str(corrMatrixCor.shape[0]-1) +
                 ' descriptors')#, True)
    print('Correlation with Y higher than ' + 
                 str(corr_limit) + ': ' + str(corrMatrixCor.shape[0]-1) +
                 ' descriptors')
    # --> results in top75 features.
    return corrMatrix, corrMatrixCorX, corrMatrixCor

def pick_xy_from_columnlist(dataset, columnlist):
    y = pd.DataFrame(dataset.loc[:,columnlist[0]])
    X = dataset.loc[:,columnlist[1::]]
    return X, y

def pick_xy_from_corrmatrix(dataset, corrMatrix):
    X,y = pick_xy_from_columnlist(dataset, corrMatrix.columns)
    return X, y

def define_groups_yvalue(y):
    # RF with y value stratification.
    groups_yvalue = y.copy()
    
    groups_yvalue[y<3] = 1
    groups_yvalue[y>6] = 3
    groups_yvalue[(y>=3)&(y<=6)] = 2
    groups_yvalue = groups_yvalue.squeeze()
    return groups_yvalue

def dropHighErrorSamples(y, X, dataset, groups = None, rmse_lim = 3.5):
    # 1 sample at a time as a test set for 10 seeds. This will be utilized for
    # dropping the moleculest with the largest test set error.
    R2_all1 = np.zeros((y.shape[0],10))
    RMSE_all1 = np.zeros((y.shape[0],10))
    top_feature_weights_all1 = [[None]*10]*y.shape[0]
    for i in range(10):
        for j in range(y.shape[0]):
            test_indices = y.index == y.index[j]
            feature_weights_1, top_feature_weights_all1[j][i], regressor1, R21, RMSE1, scaler_test1, X_test1, y_test1, y_pred1, X_train1, y_train1 = RF_feature_analysis(
                    X, y, groups=None, test_indices = test_indices, test_proportion = 0.2, top_n = 15, i='', random_state = i, sample_weighing = False, plotting = False)
            print(R21, RMSE1)
            print(top_feature_weights_all1[j][i].columns)
            # R2 should not be used for 1 sample. To do: remove
            R2_all1[j,i] = R21
            RMSE_all1[j,i] = RMSE1
    print('R2 and RMSE with single-molecule test sets: ', R2_all1, RMSE_all1)
    print('Mean: ', np.mean(R2_all1), np.mean(RMSE_all1))
    print('Std: ', np.std(R2_all1), np.std(RMSE_all1))
    print('Min: ', np.min(R2_all1), np.min(RMSE_all1))
    print('Max: ', np.max(R2_all1), np.max(RMSE_all1))
    single_mol_rmse = np.mean(RMSE_all1, axis=1)
    print('There are ', np.sum(single_mol_rmse>rmse_lim), ' molecules with RMSE>', rmse_lim, '. These will be dropped from the analysis.')
    print(dataset.loc[single_mol_rmse>=rmse_lim, ['no', 'name', 'log2mic']])#, 'Class']])

    X = X[single_mol_rmse<rmse_lim]
    y = y[single_mol_rmse<rmse_lim]
    dataset_new = dataset[single_mol_rmse<rmse_lim]
    if groups is not None:
        groups = groups[single_mol_rmse<rmse_lim]
    else:
        groups = None
    
    return X, y, dataset_new, groups

if __name__ == "__main__":

    #plt.rcParams.update({'font.size': 12})
    #plt.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif'})
    mystyle = FigureDefaults('nature_comp_mat_dc')
    
    ###############################################################################
    # BLOCK 0: INPUT VARIABLES
    ###############################################################################
    # Dataset
    #dataset_original = pd.read_excel(r'./03132020 5K descriptors of 101 COE.xlsx',
    #                                 na_values='na', convert_float = False)
    filename = '07032020 updates 5k descriptors classes.xlsx'
    y_column = 4 # The code assumes features start after y data  column.
    dataset_original = read_molecule_excel(filename, column_class='Class')#'Simplified Class')
    
    seed = 8
    test_proportion = 0.1
    # Pickle files that contain round 1 optimized hyperparameters for random forest
    # regression (will be needed in block 2 of the code).
    pickle_ho_incorr_features = 'HO_result_5K_incorrelated_features'
    pickle_ho_incorr_features_imp = 'HO_result_5K_incorrelated_important_features'
    # Pickle files that contain round 2 optimized hyperparameters for random forest
    # regression (will be needed in block 3 of the code).
    pickle_ho_incorr_features2 = 'HO_result_5K_incorrelated_features_ho1'
    pickle_ho_incorr_features_imp2 = 'HO_result_5K_incorrelated_important_features_ho1'
    
    ###############################################################################
    # BLOCK 1: DATA FILTERING
    ###############################################################################
    # Filtering data utilizing correlation matrices. Removing constant and almost
    # constant values. Scaling to 0 mean and unit variance. Y data is treated as
    # log2(Y).
    '''    
    plot_heatmap(dataset_original.iloc[:,y_column::].corr(), title = 'Starting point: ' + 
                 str(dataset_original.shape[1]-y_column-1) + ' features')
    '''
    dataset = dataset_original.copy()
    
    # Replace e.g. '>128' with 128*2 in y data (in column 2).
    idx = dataset[dataset.iloc[:,y_column].str.find('>')==0].index
    dataset.iloc[idx,y_column] = dataset.iloc[idx,y_column].str[1::]
    dataset.iloc[:,y_column] = np.double(dataset.iloc[:,y_column])*2
    # Drop rows with y data nan, and columns with any nan.
    dataset = dataset.dropna(axis=0, how='all', subset=[dataset.columns[y_column]])
    dataset = dataset.dropna(axis=1, how='any')
    if dataset.columns[y_column] != 'MIC VALUE (Y VALUE)':
        warnings.warn('Dataset is not as expected. Check that everything is ok.')
    
    # Initial correlation matrix.
    # --> A lot of ones there. --> needs filtering.
    # Also different scales in the dataset --> needs scaling.
    corrMatrixInitial = dataset.iloc[:,y_column::].corr()
    '''plot_heatmap(corrMatrixInitial, title = 'After dropping NaNs: ' +
                 str(corrMatrixInitial.shape[0]-1) + ' features')
    '''
    # First, take log from Y feature.
    dataset.iloc[:,y_column] = np.log2(dataset.iloc[:,y_column])
    # Drop constant features (note: this goes through also the No., SMILES, and y
    # value columns but it shouldn't be a problem because they are not constants)
    dataset = dataset.drop(columns=dataset.columns[(dataset == dataset.iloc[0,:]).all()])
    # Drop almost constant features (do not check No, SMILES, y value columns).
    idx_boolean = [True]*(y_column+1)
    idx_boolean.extend(((np.var(dataset.iloc[:,(y_column+1)::])/np.mean(dataset.iloc[:,(y_column+1)::]))>0.1).values)
    dataset = dataset.iloc[:,idx_boolean]
    
    # Spearman might be affected by certain scaling operations, showing
    # correlations where it doesn't exist. RF is not affected by scaling.
    # So let's not use it for now.
    '''
    # Scale the whole dataset. (It doesn't actually seem to affect correlation
    # matrix. TO DO: Check and remove if true.)
    dataset_scaled = dataset.copy()
    # Remove the mean and scale to unit variance.
    scaler = preprocessing.StandardScaler() #Other tested options: PowerTransformer()#MinMaxScaler()
    # Scale.
    dataset_scaled.iloc[:,(y_column+1)::] = pd.DataFrame(scaler.fit_transform(
            dataset_scaled.iloc[:,(y_column+1)::]), columns=dataset_scaled.iloc[:,(y_column+1)::].columns,
            index=dataset_scaled.iloc[:,(y_column+1)::].index)
    
    # Full correlation matrix
    corrMatrix = dataset_scaled.iloc[:,y_column::].corr(method='spearman')#'pearson')#
    plot_heatmap(corrMatrix, 'After dropping constant or almost constant features: ' 
                 + str(corrMatrix.shape[0]-1) + ' features')
    '''
    # Full correlation matrix
    corrMatrix = dataset.iloc[:,y_column::].corr(method='spearman')#'pearson')#
    '''plot_heatmap(corrMatrix, 'After dropping constant or almost constant features: ' 
                 + str(corrMatrix.shape[0]-1) + ' features')
    '''
    '''
    # See which features correlate with Y more than others.
    corrMatrixImportant = corrMatrix.loc[:,(np.abs(corrMatrix.iloc[0,:])>0.01).values]
    plot_heatmap(corrMatrixImportant)
    # --> Still a lot of correlating features.
    '''
    
    # Next, we want to drop features correlating too much with each other.
    # Mask upper triangle to drop only the other one of each two correlated features.
    corr_limit = 0.9 # Final value: 0.95
    tri_corrMatrix = pd.DataFrame(np.triu(corrMatrix,1), index = corrMatrix.index,
                                  columns = corrMatrix.columns)
    # List column names of highly correlated features.
    to_drop = [c for c in tri_corrMatrix.columns if any(np.abs(tri_corrMatrix[c]) > corr_limit)]
    # And drop them.
    corrMatrixFiltered = corrMatrix.drop(columns = to_drop, index = to_drop)
    '''plot_heatmap(corrMatrixFiltered, 'After filtering out highly correlated features (limit ' +
                str(corr_limit) + ': ' + str(corrMatrixFiltered.shape[0]-1) + ' features')
    '''
    # See again which of the remaining features correlate with Y.
    corr_limit = 0.05 # Final values: 0.025
    corrMatrixFilteredImportant = corrMatrixFiltered.loc[(np.abs(
            corrMatrixFiltered.iloc[0,:])>corr_limit).values,(np.abs(
                    corrMatrixFiltered.iloc[0,:])>corr_limit).values]
    '''plot_heatmap(corrMatrixFilteredImportant, 'Correlation with Y higher than ' + 
                 str(corr_limit) + ': ' + str(corrMatrixFilteredImportant.shape[0]-1) +
                 ' features')#, True)
    # --> results in top75 features.
    '''
    ###############################################################################
    # BLOCK 2: RF WITHOUT HO 
    ###############################################################################
    # Let's do Random Forest for purpose of selecting most important features.
    ###############################################################################
    
    # Default RF for the FilteredImportant features (top 75):
    # Data
    # We are not using dataset_scaled because scaling needs to be done separately
    # for train and test sets.
    y_imp = pd.DataFrame(dataset.loc[:,corrMatrixFilteredImportant.columns[0]])
    X_imp = dataset.loc[:,corrMatrixFilteredImportant.columns[1::]]
    
    y = pd.DataFrame(dataset.loc[:,corrMatrixFiltered.columns[0]])
    X = dataset.loc[:,corrMatrixFiltered.columns[1::]]
    
    groups = dataset.loc[:,'Class']
    
    test_proportion = 0.21
    
    for i in range(2):
        feature_weights_imp, top_feature_weights_imp, regressor_imp, R2_imp, RMSE_imp, scaler_test_imp, X_test_imp, y_test_imp, y_pred_imp, X_train_imp, y_train_imp = RF_feature_analysis(
            X_imp, y_imp, groups=groups, groups_only_for_plotting = True, test_proportion = test_proportion, top_n = 20, i='', random_state = i, sample_weighing = False)
        print('Default RF, seed ', i, ', X_imp: ', R2_imp)
        feature_weights, top_feature_weights, regressor, R2, RMSE, scaler_test, X_test, y_test, y_pred, X_train, y_train = RF_feature_analysis(
            X, y, groups=groups, groups_only_for_plotting = True, test_proportion = test_proportion, top_n = 20, i='', random_state = i, sample_weighing = False)
        print('Default RF, seed ', i, ', X: ', R2, RMSE)
        if i == 0:
            top_feature_weights_00 = top_feature_weights
            top_feature_weights_imp_00 = top_feature_weights_imp
            
    # RF with y value stratification.
    groups_yvalue = y_imp.copy()
    groups_yvalue = groups_yvalue.squeeze()
    groups_yvalue.loc[groups_yvalue<3] = 1
    groups_yvalue.loc[groups_yvalue>7] = 2
    groups_yvalue.loc[(groups_yvalue>2)] = 3
    
    R2_all0, RMSE_all0, top_features_all0, features_all0, X_tests0, y_tests0, X_trains0, y_trains0, regressors0 = analyze_RF_for_multiple_seeds(
            [X, X_imp], [y, y_imp], bar_plot = False, save_pickle = False, groups = groups_yvalue, plotting=False)
    #plot_RF_test(y_tests0[0][0], y_preds0[0][0], groups = groups, title='Y value stratification, seed 0')
    compare_features_barplot(top_features_all0[1][0], top_feature_weights_imp_00,
                             title='1:y strat R2='+str(np.mean(R2_all0[:,1]))+
                             ', 2:default R2 '+str(R2_imp)) # To do fix R2_imp
    compare_features_barplot(top_features_all0[0][3], top_feature_weights_imp_00,
                             title='1:y strat R2='+str(np.mean(R2_all0[:,1]))+
                             ', 2:default R2 '+str(R2_imp)) # To do fix R2_imp
    '''
    R2_imp_all0 = np.zeros((20,1))
    RMSE_imp_all0 = np.zeros((20,1))
    for i in range(20):
        for j in range(1):
            feature_weights_imp, top_feature_weights_imp, regressor_imp, R2_imp, RMSE_imp, scaler_test_imp, X_test_imp, y_test_imp, y_pred_imp, X_train_imp, y_train_imp = RF_feature_analysis(
                    X_imp2, y_imp2, groups=groups_yvalue, groups_only_for_plotting = False, test_indices = None, test_proportion = test_proportion, top_n = 20, i='', random_state = i, sample_weighing = False, plotting=False)
            print(R2_imp, RMSE_imp)
            R2_imp_all0[i,j] = R2_imp
            RMSE_imp_all0[i,j] = RMSE_imp
            plot_RF_test(y_test_imp, y_pred_imp, groups = groups, title='Y value stratification, seed '+str(i))
            if i == 0:
                top_feature_weights_imp0 = top_feature_weights_imp
    print('R2 and RMSE with y stratification: ', R2_imp_all0, RMSE_imp_all0)
    print('Mean: ', np.mean(R2_imp_all0), np.mean(RMSE_imp_all0))
    print('Std: ', np.std(R2_imp_all0), np.std(RMSE_imp_all0))
    print('Min: ', np.min(R2_imp_all0), np.min(RMSE_imp_all0))
    print('Max: ', np.max(R2_imp_all0), np.max(RMSE_imp_all0))
    
    # Plot the feature weights with and without y stratification for X_imp.
    rf_features_for_plots = top_feature_weights_imp0.copy()
    rf_features_for_plots = rf_features_for_plots.append(top_feature_weights_imp_00, sort=False, ignore_index = True)
    rf_features_for_plots=pd.melt(rf_features_for_plots.reset_index(), value_vars=rf_features_for_plots.columns,
                  id_vars = 'index')
    plt.figure()
    sn.barplot(x='value', y='variable', hue='index', data = rf_features_for_plots)
    #plt.title(title)
    plt.show()
    plt.savefig('Barplot1.png')
    '''
    # 1 sample at a time as a test set for 10 seeds. This will be utilized for
    # dropping the moleculest with the largest test set error.
    
    R2_all1 = np.zeros((y_imp.shape[0],10))
    RMSE_all1 = np.zeros((y_imp.shape[0],10))
    top_feature_weights_all1 = [[None]*10]*y_imp.shape[0]
    for i in range(10):
        for j in range(y_imp.shape[0]):
            test_indices = y_imp.index == y_imp.index[j]
            feature_weights_1, top_feature_weights_all1[j][i], regressor1, R21, RMSE1, scaler_test1, X_test1, y_test1, y_pred1, X_train1, y_train1 = RF_feature_analysis(
                    X_imp, y_imp, groups=None, test_indices = test_indices, test_proportion = test_proportion, top_n = 15, i='', random_state = i, sample_weighing = False, plotting = False)
            print(R21, RMSE1)
            print(top_feature_weights_all1[j][i].columns)
            R2_all1[j,i] = R21
            RMSE_all1[j,i] = RMSE1
    save_to_pickle([R2_all1, RMSE_all1, top_feature_weights_all1], 'Single_mol_filtering_results')
    
    #[R2_all1, RMSE_all1, top_feature_weights_all1] = fetch_pickle('Single_mol_filtering_results')
    print('R2 and RMSE with single-molecule test sets: ', R2_all1, RMSE_all1)
    print('Mean: ', np.mean(R2_all1), np.mean(RMSE_all1))
    print('Std: ', np.std(R2_all1), np.std(RMSE_all1))
    print('Min: ', np.min(R2_all1), np.min(RMSE_all1))
    print('Max: ', np.max(R2_all1), np.max(RMSE_all1))
    single_mol_rmse = np.mean(RMSE_all1, axis=1)
    print('There are ', np.sum(single_mol_rmse>3.5), ' molecules with RMSE>3.5. These will be dropped from the analysis.')
    print(dataset.loc[single_mol_rmse>=3.5, ['No.', 'NAME', 'MIC VALUE (Y VALUE)', 'Class']])
    
    dataset_filtered = dataset.copy()[single_mol_rmse<3.5]
    X_imp = X_imp[single_mol_rmse<3.5]
    y_imp = y_imp[single_mol_rmse<3.5]
    groups_yvalue = groups_yvalue[single_mol_rmse<3.5]
    X = X[single_mol_rmse<3.5]
    y = y[single_mol_rmse<3.5]
    test_proportion = 0.21 # To keep test set size constant by he aboslute number of molecules.
    
    save_to_csv_pickle(X_imp, 'X_imp')
    save_to_csv_pickle(y_imp, 'y_imp', join_with = dataset.loc[y_imp.index, ['No.', 'SMILES ']])
    save_to_csv_pickle(X, 'X')
    save_to_csv_pickle(y, 'y', join_with = dataset.loc[y.index, ['No.', 'SMILES ']])
    save_to_csv_pickle(groups_yvalue, 'groups_yvalue')
    
    # Let's repeat y stratification. At the same, let's create a dataset for
    # RF hyperparameter optimization.
    R2_all2, RMSE_all2, top_features_all2, features_all2, X_tests, y_tests, X_trains, y_trains, regressors2 = analyze_RF_for_multiple_seeds(
            [X, X_imp], [y, y_imp], bar_plot = False, save_pickle = True, groups = groups_yvalue, plotting=False)
    compare_features_barplot(top_features_all0[1][0], top_features_all2[1][0],
                             title='0: y strat R2='+str(np.mean(R2_all0[:,1]))+
                             ', 1: y strat and 6 mols dropped '+str(np.mean(R2_all2[:,1])))
    
    ###############################################################################
    # BLOCK 3: RF AND HO ROUND 1 FOR STATISTICALLY DOWNSELECTED FEATURES
    ###############################################################################
    # Run RF_tuning_for_Armi.ipynb for step 0 and step 1 (or check the the
    # corresponding pickle files already exist) before running further.
    ho_params_opt1 = {'bootstrap': True, 'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
    ho_params = [ho_params_opt1, ho_params_opt1]
    test_proportion = 0.2
    R2_all3, RMSE_all3, top_features_all3, features_all3, X_tests3, y_tests3, X_trains3, y_trains3, regressors2 = analyze_RF_for_multiple_seeds(
            [X, X_imp], [y, y_imp], ho_params = ho_params, bar_plot = False,
            save_pickle = False, groups = groups_yvalue, plotting=False,
            n_seeds=20, test_proportion = test_proportion)
    compare_features_barplot(top_features_all3[0][0], top_features_all3[1][0],
                             title='0: HO opt X R2='+str(np.mean(R2_all3[0,:]))+
                             ', 1: HO opt X_imp R2='+str(np.mean(R2_all2[1,:])))
    compare_features_barplot(top_features_all3[0][0], top_features_all2[0][0],
                             title='0: HO opt X seed 0 R2='+str(np.mean(R2_all3[0,:]))+
                             ', 1: no HO opt X seed 0 R2='+str(np.mean(R2_all2[0,:])))
    compare_features_barplot(top_features_all3[0][3], top_features_all2[0][3],
                             title='0: HO opt X seed 3 R2='+str(np.mean(R2_all3[3,:]))+
                             ', 1: no HO opt X seed 3 R2='+str(np.mean(R2_all2[3,:])))
    
    # Final graph about HO round 1:
    compare_features_barplot(top_features_all3[1][3], top_features_all2[1][3],
                             title='0: HO opt, RMSE={:.2f}, R2={:.2f}, 1: no HO opt, RMSE={:.2f}, R2={:.2f}'.format(
                                     np.mean(RMSE_all3[3,1]), np.mean(R2_all3[3,1]),
                                     np.mean(RMSE_all2[3,1]), np.mean(R2_all2[3,1])),
                             filename_fig='Top_features_HO1')
    
    # Analysis:
    print('Mean R2 for X and X_imp before HO: ', np.mean(R2_all2, axis=0),
          ',\nMean R2 for X and X_imp after HO: ', np.mean(R2_all3, axis=0),
          ',\nMean RMSE for X and X_imp before HO: ', np.mean(RMSE_all2, axis=0),
          ',\nMean RMSE for X and X_imp after HO: ', np.mean(RMSE_all3, axis=0))
    print('X_imp is better than X for all the cases above.')
    print('All the mean values are better without the HO.')
    print('Std R2 for X and X_imp before HO: ', np.std(R2_all2, axis=0),
          ',\nStd R2 for X and X_imp after HO: ', np.std(R2_all3, axis=0),
          ',\nStd RMSE for X and X_imp before HO: ', np.std(RMSE_all2, axis=0),
          ',\nStd RMSE for X and X_imp after HO: ', np.std(RMSE_all3, axis=0))
    print('All the std values are better with the HO.')
    print('The choices: X_imp, with HO.')
    print('All RMSE with this choice:\n', RMSE_all3[:,1],
          '\nAll R2 with this choice:\n', R2_all3[:,1])
    print('Best RMSE with seed ', np.where(RMSE_all3[:,1] == np.min(RMSE_all3[:,1]))[0][0],
            '\nBest R2 with seed ', np.where(R2_all3[:,1] == np.max(R2_all3[:,1]))[0][0])
    print('Best combination with seed 3.')
    '''
    # For X_imp:
    print('All RMSE with this choice:\n', RMSE_all3[:,0],
          '\nAll R2 with this choice:\n', R2_all3[:,0])
    print('Best RMSE with seed ', np.where(RMSE_all3[:,0] == np.min(RMSE_all3[:,0]))[0][0],
            '\nBest R2 with seed ', np.where(R2_all3[:,0] == np.max(R2_all3[:,0]))[0][0])
    print('Best combination with seed 3.')
    '''
    chosen_seed = 3
    
    X_train = X_trains3[1][chosen_seed].copy()
    y_train = y_trains3[1][chosen_seed].copy()
    X_test = X_tests3[1][chosen_seed].copy()
    y_test = y_tests3[1][chosen_seed].copy()
    groups_train = groups_yvalue[y_train.index].copy()
    groups_test = groups_yvalue[y_test.index].copy()
    
    cols_to_pick = list(dataset_filtered.columns[0:(y_column+1)]) + list(X_train.columns)
    dataset_filtered = dataset_filtered.loc[:,cols_to_pick]
    save_to_pickle(dataset_filtered, 'July_dataset_filtered')
    save_to_csv_pickle(X_train, 'X_train')
    save_to_csv_pickle(y_train, 'y_train', join_with = dataset.loc[y_train.index, ['No.', 'SMILES ']])
    save_to_csv_pickle(X_test, 'X_test')
    save_to_csv_pickle(y_test, 'y_test', join_with = dataset.loc[y_test.index, ['No.', 'SMILES ']])
    save_to_csv_pickle(groups_train, 'groups_train')
    save_to_csv_pickle(groups_test, 'groups_test')
    save_to_pickle(chosen_seed, 'chosen_seed')
    save_to_pickle(features_all3, 'features_all3')
    # TO DO: I should actually do also the HO for the correct seed. Now it has been
    # done for seeds 0 and 1, I think...
    
    mystyle = FigureDefaults('nature_comp_mat_tc')
    # Finally, let's estimate accuracies using the train set only. All the X_imp features.
    R2_all4, RMSE_all4, top_features_all4, features_all4, X_tests4, y_tests4, X_trains4, y_trains4, regressors4 = analyze_RF_for_multiple_seeds(
        [X_train], [y_train], ho_params = ho_params, bar_plot = False,
        save_pickle = False, groups = groups_train, plotting=False, n_seeds=20, test_proportion=0.2)
    R2_all4, RMSE_all4, top_features_all4, features_all4, X_tests4, y_tests4, X_trains4, y_trains4, regressors4 = analyze_RF_for_multiple_seeds(
        [X_imp], [y_imp], ho_params = ho_params, bar_plot = False,
        save_pickle = False, groups = groups_yvalue, plotting=True, n_seeds=4, test_proportion=0.2)

    # Here are the features from which only nans and strings have been removed.
    X_cons = dataset_original.loc[:,corrMatrixInitial.columns[1::]].loc[y_train.index]    
    X_cons_all = dataset_original.loc[:,corrMatrixInitial.columns[1::]].loc[y_imp.index]    
    R2_all5, RMSE_all5, top_features_all5, features_all5, X_tests5, y_tests5, X_trains5, y_trains5, regressors5 = analyze_RF_for_multiple_seeds(
        [X_cons], [y_train], ho_params = None, bar_plot = True,
        save_pickle = False, groups = groups_train, plotting=False, n_seeds=20, test_proportion=0.2)
    R2_all5, RMSE_all5, top_features_all5, features_all5, X_tests5, y_tests5, X_trains5, y_trains5, regeressors5 = analyze_RF_for_multiple_seeds(
        [X_cons_all], [y_imp], ho_params = None, bar_plot = True,
        save_pickle = False, groups = groups_yvalue, plotting=True, n_seeds=4, test_proportion=0.2)
    ## Here are the features after doing the variance filtering.
    X_cons = dataset.loc[:,corrMatrix.columns[1::]].loc[y_train.index]    
    X_cons_all = dataset.loc[:,corrMatrix.columns[1::]].loc[y_imp.index]    
    R2_all5, RMSE_all5, top_features_all5, features_all5, X_tests5, y_tests5, X_trains5, y_trains5, regressors5 = analyze_RF_for_multiple_seeds(
        [X_cons], [y_train], ho_params = None, bar_plot = False,
        save_pickle = False, groups = groups_train, plotting=False, n_seeds=20, test_proportion=0.2)
    R2_all5, RMSE_all5, top_features_all5, features_all5, X_tests5, y_tests5, X_trains5, y_trains5, regeressors5 = analyze_RF_for_multiple_seeds(
        [X_cons_all], [y_imp], ho_params = None, bar_plot = False,
        save_pickle = False, groups = groups_yvalue, plotting=True, n_seeds=4, test_proportion=0.2)
    # For curiosity, this is withiut y stratification.
    R2_all6, RMSE_all6, top_features_all6, features_all6, X_tests6, y_tests6, X_trains6, y_trains6, regressors6 = analyze_RF_for_multiple_seeds(
        [X_cons], [y_train], ho_params = None, bar_plot = False,
        save_pickle = False, groups = None, plotting=False, n_seeds=20, test_proportion=0.2)
