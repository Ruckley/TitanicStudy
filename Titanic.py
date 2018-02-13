#%%
# Load libraries
import pandas
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
from statsmodels.stats.outliers_influence import variance_inflation_factor


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#%%
dataset = pandas.read_csv('C:/Users/Programming/Desktop/train.csv')

#%%
########################## Pre-Processing #####################################

#%%
#### General pre-Processing

#Passenger name  gives no information so drop
dataset.drop(['Name'], axis = 1, inplace = True)

#### Deal with missing Data

#Calculate percentage of missing data for each column
print('\nOriginal Dataset\n')
print((dataset.isnull().sum()/len(dataset))*100)

#Cabin number  is not present for the majority of passengers so drop
dataset.drop(['Cabin'], axis = 1, inplace = True)

# A toiny num ber of Embarked data is missing so I will just remove these rows
dataset = dataset[pandas.notnull(dataset['Embarked'])]

#Ticket may present information but is non-numeric and difficult to transform so I will drop it for now
dataset.drop(['Ticket'], axis = 1, inplace = True)

#Age is missing some values. As most are there and we know age will be an important factor in survival, I will replace the NaNs with the average age
#Imputer from scikit-learn will do this for me
dataset[['Age']] = Imputer().fit_transform(dataset[['Age']].values)

#Sex is non numeric, as there are only 2 classes we can swap male and female for 0 and 1
dataset[['Sex']] = dataset[['Sex']].replace('male', 0)
dataset[['Sex']] = dataset[['Sex']].replace('female', 1)

# We can see from the PassengerId scatter plot that passengerId gives no information so we can remove it
dataset.drop(['PassengerId'], axis = 1, inplace = True)

print('\nDataset with NaNs removed\n')
print((dataset.isnull().sum()/len(dataset))*100)

age_ranges = []
i = 0;
dataset = dataset.sort_values(by='Age')

age_group_ds = pandas.DataFrame(columns = ['Age Group','Survived'])

group = 0

for age in dataset['Age'].values:
    group = ciel(age/5)



#%%

#### Testing Parameters

validation_size = 0.20
seed = 7
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#%%
bl_ds = dataset.copy();
bl_ds.drop(['Embarked', 'Pclass'], axis = 1, inplace = True)

[X,Y] = splitPredictorTarget(bl_ds, 'Survived')
[bl_results, bl_result_means] = trainDataset(X, Y, models, validation_size, seed)



#%%

#### Full Integer Encoding

ie_ds = dataset.copy();
le = preprocessing.LabelEncoder()
ie_ds['Embarked'] = le.fit_transform(dataset['Embarked'])


[X,Y] = splitPredictorTarget(ie_ds, 'Survived')
[full_ie_results, full_ie_result_means] = trainDataset(X, Y, models,  validation_size, seed)

#%%

# Confirm that title has a correlation with survival


survived_percentages = []

for title in dataset['Title'].unique():
    survival_split = dataset.Survived[dataset['Title'] == title].value_counts()
    
    if survival_split.size == 1:
        survived_percentages.append(survival_split.keys()[0]*100)
    else:
        survived_percentages.append((survival_split[1]/(survival_split[0]+survival_split[1]))*100)

data = np.array(survived_percentages).T
title_survived = pandas.DataFrame(data = data, index = dataset['Title'].unique(), columns = ['% Survived']).sort_values(by='% Survived')
title_survived.plot(kind='bar', legend=False)
 
#%%df.set_value('C', 'x', 10)

#%%

#### One-Hot-Encoding

# use one hot encoding for embarked and Pclass
ohe_ds = dataset.copy()

one_hot = pandas.get_dummies(dataset['Embarked'])
ohe_ds.drop(['Embarked'], axis = 1, inplace = True)
ohe_ds = ohe_ds.join(one_hot)

one_hot = pandas.get_dummies(dataset['Pclass'])
ohe_ds.drop(['Pclass'], axis = 1, inplace = True)
ohe_ds = ohe_ds.join(one_hot)

[X,Y] = splitPredictorTarget(ohe_ds, 'Survived')
[full_ohe_results, full_ohe_result_means] = trainDataset(X, Y, models, validation_size, seed)

#%%

### One-Hot Encoding removing column

ohe_drop_ds = ohe_ds.copy()
ohe_drop_ds.drop(['S', 3], axis = 1, inplace = True)

[X,Y] = splitPredictorTarget(ohe_drop_ds, 'Survived')
[full_ohe_drop_results, full_ohe_drop_result_means] = trainDataset(X, Y, models, validation_size, seed)

#%%

### Graph validation results

results = [bl_result_means, full_ie_result_means,full_ohe_result_means, full_ohe_drop_result_means]
model_names, mode_objects = map(list, zip(*models))


accuracy_comparison_encoding = pandas.DataFrame(data=results, columns=model_names, index = ['baseline', 'Integer Encoded', 'One Hot Encoded', 'One Hot Encoded, dropped column'])
accuracy_comparison_algorithm = accuracy_comparison_encoding.T

accuracy_plot = accuracy_comparison_algorithm.plot(kind='bar', title = "Accuracy Comparison", legend = True, ylim = [0.65,0.85])


#%%
######################## Testing #############################################

#test_dataset = pandas.read_csv('C:/Users/Programming/Desktop/test.csv')

#test_dataset.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis = 1, inplace = True)
#test_dataset[['Age']] = Imputer().fit_transform(test_dataset[['Age']].values)
#test_dataset[['Sex']] = test_dataset[['Sex']].replace('male', 0)
#test_dataset[['Sex']] = test_dataset[['Sex']].replace('female', 1)
#one_hot = pandas.get_dummies(test_dataset['Embarked'])
#test_dataset.drop(['Embarked'], axis = 1, inplace = True)
#test_dataset = test_dataset.join(one_hot)
#one_hot = pandas.get_dummies(test_dataset['Pclass'])
#test_dataset.drop(['Pclass'], axis = 1, inplace = True)
#test_dataset = test_dataset.join(one_hot)

#print(test_dataset.head(10))

#print('\n CART PREDICTION\n')
#cart = DecisionTreeClassifier()
#cart.fit(X_train, Y_train)
# =============================================================================
# predictions = cart.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
# 
# print('\n NB PREDICTION\n')
# 
# nb = DecisionTreeClassifier()
# nb.fit(X_train, Y_train)
# predictions = nb.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
# =============================================================================

#predictions = knn.predict(X_validation)
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))


    