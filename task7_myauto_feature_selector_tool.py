import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


#Pearson_cor
def cor_selector(X, y,num_feats):
    cor_list=[]

    for i in feature_name:
        cor= np.corrcoef(X[i],y)[0,1]
        cor_list.append(cor)
    cor_list=[0 if np.isnan(i) else i for i in cor_list]

    cor_feature=X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature

#Chi-square
def chi_squared_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector=SelectKBest(chi2,k=num_feats)
    chi_selector.fit(X_norm,y)
    chi_support=chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    

    # Your code ends here
    return chi_support, chi_feature

def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm=MinMaxScaler().fit_transform(X)
    rfe_selector=RFE(estimator=LogisticRegression(),n_features_to_select=num_feats,step=10,verbose=5)
    rfe_selector.fit(X_norm,y)
    rfe_support=rfe_selector.get_support()
    rfe_feature=X.loc[:,rfe_support].columns.tolist()
    
    # Your code ends here
    return rfe_support, rfe_feature
    

def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm=MinMaxScaler().fit_transform(X)
    embedded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2",),max_features=num_feats)
    embedded_lr_selector.fit(X_norm,y)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature=X.loc[:,embedded_lr_support].columns.tolist()
    return embedded_lr_support, embedded_lr_feature

def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    embedded_rf_selector=SelectFromModel(RandomForestClassifier(n_estimators=100),max_features=num_feats)
    embedded_rf_selector.fit(X,y)
    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:,embedded_rf_support].columns.tolist()
    return embedded_rf_support, embedded_rf_feature

def embedded_lgbm_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    lgbmc = LGBMClassifier(n_estimators=500,
                      learning_rate=0.05,
                      num_leaves=32,
                      colsample_bytree=0.2,
                      reg_alpha=3,
                      reg_lambda=1,
                      min_split_gain=0.01,
                      min_child_weight=40
)
    embedded_lgbm_selector = SelectFromModel(lgbmc,
                                            max_features=num_feats
    )
    embedded_lgbm_selector = embedded_lgbm_selector.fit(X, y)
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()
    
    # Your code ends here
    return embedded_lgbm_support, embedded_lgbm_feature


def preprocess_dataset(dataset_path):
    # Your code starts here (Multiple lines)
    df = pd.read_csv(dataset_path)
    numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
    catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']
    df = df[numcols+catcols]
    traindf = pd.concat([df[numcols], pd.get_dummies(df[catcols])],axis=1)
    features = traindf.columns

    traindf = traindf.dropna()
    traindf = pd.DataFrame(traindf,columns=features)
    y = traindf['Overall']>=87
    X = traindf.copy()
    del X['Overall']

    num_feats = X.shape[1]
    
    # Your code ends here
    return X, y, num_feats

def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support,cor_feature = cor_selector(X, y,num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    # if 'lgbm' in methods:
    #     embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
    
    feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embedded_lr_support,
                                    'Random Forest':embedded_rf_support})
    # count the selected times for each feature
    feature_selection_df['Total'] =feature_selection_df.drop('Feature',axis=1).sum(axis=1)
    # display the top 100
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    feature_selection_df.head(num_feats)
    
    features_list= feature_selection_df[feature_selection_df['Total']==5]
    best_features= features_list['Feature'].tolist()
    best_features
    
    
    #### Your Code ends here
    return best_features

best_features = autoFeatureSelector(dataset_path="fifa19.csv", methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf'])
best_features
