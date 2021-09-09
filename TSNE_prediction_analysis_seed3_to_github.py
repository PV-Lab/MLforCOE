#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:19:08 2021

Analyse prediction errors of new datapoints.

@author: armi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib

sns.set_palette('copper')

# This is used only for reading classes of COE molecules.
df_classes = pd.read_excel('07032020 updates 5k descriptors classes.xlsx')

features_dir = ['./Newdata/x_opt_train_seed3.csv',
                './Newdata/x_opt_test_seed3.csv',
                './Newdata/x_opt_newdata.csv',
                './Newdata/x_opt_outliers.csv']
opt_features = ['MSD', 'MaxDD', 'TI2_L', 'MATS4v', 'MATS4i', 'P_VSA_MR_5', 'P_VSA_MR_6',
       'TDB06s', 'RDF040m', 'Mor20m', 'Mor25m', 'Mor31m', 'Mor10v', 'Mor20v',
       'Mor25v', 'Mor26s', 'R7u', 'H-046', 'H-047', 'SHED_AL', 'ALOGP'] # Used for Taylor's test data.
dat_dir = ['./Newdata/preds_opt_train_seed3.csv',
           './Newdata/preds_opt_test_seed3.csv',
           './Newdata/preds_opt_newdata.csv',
           './Newdata/preds_opt_outliers.csv']
dataset_dir = ['./Newdata/dataset_opt_train.csv',
               './Newdata/dataset_opt_test.csv',
               './Newdata/dataset_opt_newdata.csv',
               './Newdata/dataset_opt_outliers.csv']
dataset_names = ['Train', 'Test', 'New data', 'Outliers']

from set_figure_defaults import FigureDefaults
mystyle = FigureDefaults('nature_comp_mat_dc_fixed_height')


df_ydata = []
datasets=[]
X_all = []
for i in range(len(features_dir)):
    df_ydata.append(pd.read_csv(dat_dir[i], index_col=0))
    df_ydata[i].loc[:, 'Dataset'] = dataset_names[i]
    datasets.append(pd.read_csv(dataset_dir[i], index_col=0))
    all_nos = []
    all_classes = []
    all_prederrors = []
    all_predrelerrors = []
    
    for j in range(df_ydata[i].shape[0]):
        smiles = df_ydata[i].iloc[j,df_ydata[i].columns.get_loc('smiles')]
        nos = int(datasets[i][datasets[i].loc[:,'smiles'] == smiles].loc[:,'no'].values[0])
        if df_classes.Class[df_classes.loc[:,'SMILES '].isin([smiles])].shape[0]>0:
            molclass = str(df_classes.Class[df_classes.loc[:,'SMILES '].isin([smiles])].values[0])
        else:
            molclass = ''
        all_nos.append(nos)
        all_classes.append(molclass)
        # If newdata (== Data 3), set to prediction error, if test or train, set to 0.
        #if i > 0:
        prederror = df_ydata[i].iloc[j,df_ydata[i].columns.get_loc('error')]
        all_prederrors.append(np.sqrt(prederror**2))
        
        predrelerror = df_ydata[i].iloc[j,df_ydata[i].columns.get_loc('relerror')]
        all_predrelerrors.append(np.sqrt(predrelerror**2))
        
    df_ydata[i].loc[:,'no'] = all_nos
    df_ydata[i].loc[:,'class'] = all_classes
    df_ydata[i].loc[:, 'prederror'] = all_prederrors
    df_ydata[i].loc[:, 'predrelerror'] = all_predrelerrors
    X_all.append(pd.read_csv(features_dir[i]).loc[:,opt_features])
X = pd.concat(X_all, ignore_index=True).values

# Normal TSNE option.
X_transformed_tsne = TSNE(n_components=2, metric='euclidean', perplexity=40, learning_rate=10, random_state=0, n_iter=10000).fit_transform(X)

y_df = pd.concat(df_ydata)

def similarity_plot(df, y_name, label_texts = None, saveas = None,
                    X_name = ['X1', 'X2'], hue_norm = None):
    
    ax = sns.scatterplot(data = df, x=X_name[0], y=X_name[1],hue=y_name,
                         style='Dataset', palette='copper_r',
                         hue_norm = hue_norm)
    if label_texts is not None:
        for j in range(df.shape[0]):
            ax.text(df.loc[df.index[j],X_name[0]], df.loc[df.index[j],X_name[1]], str(df.loc[df.index[j],label_texts]), fontsize= 'xx-small')
    plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.) # Padding is added in order to improve the proportions of the final figure combined from multiple tsne plots 
    ax.tick_params(axis='both', bottom=False, top=False,
                                     left = False, right=False,
                                     labelbottom = False, labeltop=False,
                                     labelleft = False, labelright=False)
    plt.tight_layout()
    
    if saveas is not None:
        plt.savefig(saveas+".svg")
        plt.savefig(saveas+".png")
        plt.savefig(saveas+".pdf")
    plt.show()
    
def similarity_plot_all_options(X_transformed, y_df, y_options, label_text_options, saveas = None):
    
    # Build df for plotting.
    df = y_df.copy()
    df.loc[:,['X1', 'X2']] = X_transformed
    
    if 'predrelerror' in y_options: # We want to rescale coloring.
        hue_norm = (0, 3)
    elif 'prederror' in y_options: # We want to rescale coloring.
        hue_norm = (0, 3)
    else:
        hue_norm = None
    
    # Replace datasheet content names with final parameter names.
    old_name = ['log2mic', 'prederror', 'predrelerror', 'pred_log2mic']
    new_name = ['Measured\n$\log_2(MIC)$', 'RMSE', 'Relative\nRMSE', 'Predicted\n$\log_2(MIC)$']
    for i in range(len(old_name)):
        if old_name[i] in df.columns:
            df = df.rename(columns={old_name[i]: new_name[i]})
        if old_name[i] in y_options:
            y_options = [option.replace(old_name[i], new_name[i]) for option in y_options]
        if old_name[i] in label_text_options:
            label_text_options = [option.replace(old_name[i], new_name[i]) for option in label_text_options]
    
    
    for i in range(len(y_options)):
        
        for j in range(len(label_text_options)):
            if saveas is not None:
                similarity_plot(df, y_options[i],
                                label_texts = label_text_options[j],
                                saveas='./Similarity_plot_options/'+saveas+y_options[i]+label_text_options[j],
                                hue_norm = hue_norm)
            else:
                similarity_plot(df, y_options[i],
                                label_texts = label_text_options[j],
                                hue_norm = hue_norm)
        similarity_plot(df, y_options[i], label_texts = None,
                        saveas='./Similarity_plot_options/'+saveas+y_options[i],
                        hue_norm = hue_norm)


# MIC value
similarity_plot_all_options(X_transformed_tsne, y_df, ['log2mic', 'Dataset'],
                            ['no'#, 'class' # Class has been classified by molecule type - tsne should map each class appr. to its own blob - if not, you will have to tune tsne settings
                            ], saveas='RF_opt_T-SNE_')
# Prediction error
similarity_plot_all_options(X_transformed_tsne, y_df, ['prederror'
                            ], ['no'
                            ], saveas='RF_opt_T-SNE_')
# Relative prediction error
similarity_plot_all_options(X_transformed_tsne, y_df, ['predrelerror'
                            ], ['no'
                            ], saveas='RF_opt_T-SNE_')
# Predicted MIC value
similarity_plot_all_options(X_transformed_tsne, y_df, ['pred_log2mic'
                            ], ['no'
                            ], saveas='RF_opt_T-SNE_')

########################################
# Same again, but with PCA.
########################################

pca = PCA(n_components=2, random_state=0)
X_transformed_pca = pca.fit_transform(X)

# MIC value
similarity_plot_all_options(X_transformed_pca, y_df, ['log2mic', 'Dataset'],
                            ['no'
                            ], saveas='RF_opt_PCA_')
# Prediction error
similarity_plot_all_options(X_transformed_pca, y_df, ['prederror'#, 'Dataset'
                            ], ['no'
                            ], saveas='RF_opt_PCA_')
                                            
                                            

# PCA with only train data
pca_train = PCA(n_components=2, random_state=0)
X_transformed_pca_train = pca_train.fit_transform(X_all[0])
# MIC value
similarity_plot_all_options(X_transformed_pca_train, df_ydata[0], ['log2mic'#, 'Dataset'
                            ], ['no'#, 'class'
                            ], saveas='RF_opt_train_PCA_')
# Prediction error
similarity_plot_all_options(X_transformed_pca_train, df_ydata[0], ['prederror'#, 'Dataset'
                            ], ['no'#, 'class'
                            ], saveas='RF_opt_train_PCA_')

