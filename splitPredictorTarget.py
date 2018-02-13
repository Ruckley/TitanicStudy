# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:54:59 2017

@author: Programming
"""


def splitPredictorTarget(dataset, target_name):
    
    X = dataset.loc[:, dataset.columns != target_name]
    Y = dataset[['Survived']].dropna()
    
    return X, Y;