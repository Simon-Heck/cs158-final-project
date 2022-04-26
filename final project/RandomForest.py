from random import Random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.datasets import make_classification



_clf = RandomForestClassifier(max_depth=2, random_state=0)

def main():
    RandomForestClassifier.

    setParameters()
    data = loadDataset()
    preProcessData(data)

    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

    X, y = 
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    
    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    
    
    
def setParameters(clf: RandomForestClassifier):
    clf.set_params()
    pass

def preProcessData(data):
    scaler = preprocessing.StandardScaler().fit(data)

def hyperTune(): #?
    pass

def loadModel():
    pass

def loadDataset(file_name):
    return pd.read_csv(file_name) 

if __name__ == "main":
    main()