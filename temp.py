# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:20:19 2017

@author: Programming
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv('C:/Users/Programming/Desktop/train.csv')


dataset['Title'][pd.isnull(dataset['Age'])].value_counts()

# Replace Name predictor with Titles
dataset['Name'] = dataset['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
dataset=dataset.rename(columns = {'two':'new_name'})
dataset = dataset.rename(columns = {'Name':'Title'})

age_group_ds = pd.DataFrame(columns = ['Age Group','Survived'])

age_bins = [0,5,10,15,20,30,40,50,60,70,80,90,100]
age_group_names=['(0,5]', '(5,10]', '(10,15]', '(15,20]', '(20,30]','(30,40]', '(40,50]', '(50,60]', '(60,70]', '(70,80]', '(80,90]', '(90,100]']


dataset['Age Group'] = pd.cut(dataset['Age'], age_bins, labels = age_group_names)
# sort so that data will be in numerical order for plotting
dataset.sort_values(by = 'Age')




# function to find the percentage of people who survived against a given variable and plot
def findPercentageSurvived(variable, dataset, plot, sort_by_survived):
    survived_percentages = []

    for title in dataset[variable].unique():
        survival_split = dataset.Survived[dataset[variable] == title].value_counts()

        if survival_split.size == 1:
            survived_percentages.append(survival_split.keys()[0]*100)
        else:
            survived_percentages.append((survival_split[1]/(survival_split[0]+survival_split[1]))*100)

    data = np.array(survived_percentages).T
    variable_survived = pd.DataFrame(data = data, index = dataset[variable].unique(), columns = ['% Survived'])
    
    if sort_by_survived == True:
        variable_survived = variable_survived.sort_values(by='% Survived')
    
    if plot == True:
        variable_survived.plot(kind='bar', title = '% Survived by Passenger ' + variable,legend=False)
    return variable_survived
    
findPercentageSurvived('Age Group', dataset, True, True);
