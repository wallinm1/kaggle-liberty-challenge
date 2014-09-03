# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RandomizedLasso, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

def make_dummies(df, variables):
    for variable in variables:
        dummies = pd.get_dummies(df[variable], prefix = variable)
        df = pd.concat([df, dummies], axis = 1)
        df = df.drop(variable, 1)
    return df

def main():
    print "read train"
    df_train = pd.read_csv('data/train.csv')
    print "read test"
    df_test = pd.read_csv('data/test.csv')
    sample = pd.read_csv('data/sampleSubmission.csv')
    
    cats = ['var1', 'var2', 'var3', 'var4', 'var5', 
            'var6', 'var7', 'var8', 'var9', 'dummy']
            
    print "convert mixed columns to strings"
    df_train.loc[:, cats] = df_train[cats].applymap(str)
    df_test.loc[:, cats] = df_test[cats].applymap(str)
    
    print "one-hot encoding"
    df_train = make_dummies(df_train, cats)
    df_test = make_dummies(df_test, cats)

    print "fill missing values"
    df_train = df_train.fillna(df_train.mean())
    df_test = df_test.fillna(df_test.mean())
    
    print "set binary labels"
    df_train['target_class'] = (df_train.target>0).astype(int)
    
    classes = df_train.target_class.values
    loss = df_train.target.values
    df_train = df_train.drop(['target', 'id', 'target_class'], axis = 1)
    df_test = df_test.drop(['id'], axis = 1)

    build_features = True #flag, determines whether features will be trained or read from file
    
    if build_features:
        print "univariate feature selectors"
        selector_clf = SelectKBest(score_func = f_classif, k = 'all')
        selector_reg = SelectKBest(score_func = f_regression, k = 'all')
        selector_clf.fit(df_train.values, classes)
        selector_reg.fit(df_train.values, loss)
        pvalues_clf = selector_clf.pvalues_
        pvalues_reg = selector_reg.pvalues_
        pvalues_clf[np.isnan(pvalues_clf)] = 1
        pvalues_reg[np.isnan(pvalues_reg)] = 1
        
        #put feature vectors into dictionary
        feats = {}
        feats['univ_sub01'] = (pvalues_clf<0.1)&(pvalues_reg<0.1) 
        feats['univ_sub005'] = (pvalues_clf<0.05)&(pvalues_reg<0.05)
        feats['univ_reg_sub005'] = (pvalues_reg<0.05)
        feats['univ_clf_sub005'] = (pvalues_clf<0.05)
        
        print "randomized lasso feature selector"
        sel_lasso = RandomizedLasso(random_state = 42, n_jobs = 4).fit(df_train.values, loss)
        #put rand_lasso feats into feature dict
        feats['rand_lasso'] = sel_lasso.get_support()
        
        print "l1-based feature selectors"
        X_sp = sparse.coo_matrix(df_train.values)
        sel_svc = LinearSVC(C=0.1, penalty = "l1", dual = False, random_state = 42).fit(X_sp, classes)
        feats['LinearSVC'] = np.ravel(sel_svc.coef_>0)
        sel_log = LogisticRegression(C=0.01, random_state = 42).fit(X_sp, classes)
        feats['LogReg'] = np.ravel(sel_log.coef_>0)
        
        feat_sums = np.zeros(len(feats['rand_lasso']))
        for key in feats:
            feat_sums+=feats[key].astype(int)
        feats['ensemble'] = feat_sums>=5 #take features which get 5 or more votes
        joblib.dump(feats, 'features/feats.pkl', compress = 3)
    
    else:
        feats = joblib.load('features/feats.pkl')
    
    xtrain = df_train.values
    xtest = df_test.values
    
    print "fitting gb-regressor"
    reg_gbr = GradientBoostingRegressor(n_estimators = 3000, learning_rate = 0.001, max_depth =5, random_state = 42, verbose = 100, min_samples_leaf=5)
    reg_gbr.fit(xtrain[:, feats['ensemble']], loss)
    gbr_preds = reg_gbr.predict(xtest[:, feats['ensemble']])
    sample['target'] = gbr_preds
    sample.to_csv('submissions/gbm_sub.csv', index = False)
    reg_lin = LinearRegression()
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    print "fitting linear regressor"
    reg_lin.fit(xtrain[:, feats['rand_lasso']], loss)
    lin_preds = reg_lin.predict(xtest[:, feats['rand_lasso']])
    gbr_order = gbr_preds.argsort().argsort() #maps smallest value to 0, second-smallest to 1 etc.
    lin_order = lin_preds.argsort().argsort()
    #averaging
    mean_order = np.vstack((gbr_order, lin_order)).mean(0)    
    sample['target'] = mean_order
    sample.to_csv('submissions/mean_sub.csv', index = False)
    
if __name__ == "__main__":
    main()