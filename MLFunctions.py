# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:54:59 2017

@author: Programming
"""


def splitPredictorTarget(dataset, target_name):
    
    X = dataset.loc[:, dataset.columns != target_name]
    Y = dataset[['Survived']].dropna()
    
    return X, Y;


from sklearn import model_selection


def trainDataset(X, Y, models, val_size, seed):

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=val_size, random_state=seed)

    # evaluate each model in turn
    results = []
    result_means = []
    resultStd = []
    names = []
    
    print('\nCross Validation with verification set\n')
    
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train.values.ravel(), cv=kfold, scoring='accuracy')
        results.append(cv_results)
        result_means.append(cv_results.mean())
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
    print('\n')
    
    #inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    
    return results, result_means;
