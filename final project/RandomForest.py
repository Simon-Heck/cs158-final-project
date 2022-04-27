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
import random
from sklearn.utils import shuffle

def main():
    # print("hi")
    # RandomForestClassifier.

    # setParameters()
    # data = loadDataset()
    # preProcessData(data)

    df = pd.read_csv("C:\\Users\\simon\Desktop\\ComputerScience\\cs158 final project\\data\\hearts.csv")
    #shuffle
    # df = shuffle(df)
    
    df = df.sample(frac=1).reset_index(drop=True)
    y = df['output'].to_numpy()
    X = df.drop(columns=['output']).to_numpy()
    

    # print(X.shape)
    # print(y.shape)
    # exit()

    # X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=True)
    # print(X.shape, y.shape)
    # X, y = 
    # n_estimators=50
    #max_depth=10
    #random_state=0

    #estimators 
    for k in range(20,40):
        # n_estimators = i
        #max depth
        for j in range(1,10):
            # max_depth = j
            clf = RandomForestClassifier(n_estimators=k, max_depth=j, random_state=0, bootstrap=False)

            #train
            clf.fit(X[50:], y[50:])

            trues = 0
            for i in range(50):
                # R = random.randint(0,1)
                if(clf.predict(X[[i]]) == [1] and [1]==y[i]):
                    trues +=1
                elif(clf.predict(X[[i]]) == [0] and [0]==y[i]):
                    trues += 1
                # if([R] == [1] and [1]==y[i]):
                #     trues +=1
                # elif([R] == [0] and [0]==y[i]):
                #     trues += 1
            # print(trues)
            print(k,j)
            
            print(f'Accuracy: {trues/50}')

    # prediction = clf.predict(X[[50]])
    # print(y[50])
    # print(prediction)
    # evaluate the model
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # n_scores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # # report performance
    # print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    
    
    
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

if __name__ == "__main__":
    main()