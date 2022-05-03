from random import Random
from aiohttp import TraceRequestExceptionParams
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from setuptools import setup
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
import pickle
import sys
from data_cleaning import *

def main():
    lblCol = "Label"
    if sys.argv[1] == "tune":
        df = clean_data(f"C:\\Users\\simon\Desktop\\ComputerScience\\cs158 final project\\ModelData\\{sys.argv[2]}DEV.csv",lblCol,sys.argv[3])
        devX = df.drop(columns=[lblCol]).to_numpy()
        devY = df[lblCol].to_numpy()
        bestParams = hyperParamTune(devX,devY)
        tunedNEstimators = bestParams[1]
        tunedMaxDepth = bestParams[2]
        clf = RandomForestClassifier(n_estimators=tunedNEstimators, max_depth=tunedMaxDepth, random_state=0, bootstrap=True)
        joblib.dump(clf, "RFModelPreTrain")
        
    elif sys.argv[1] == "train":
        df = clean_data(f"C:\\Users\\simon\Desktop\\ComputerScience\\cs158 final project\\ModelData\\{sys.argv[2]}TRAIN.csv",lblCol,sys.argv[3])
        trainX = df.drop(columns=[lblCol]).to_numpy()
        trainY = df[lblCol].to_numpy()
        clf = joblib.load("RFModelPreTrain")
        clf.fit(trainX, trainY)
        devAccuraccy = clf.score(trainX, trainY)
        print(f'Development Accuracy: {devAccuraccy}')
        clf = joblib.dump(clf, "RFTrained")

    elif sys.argv[1] == "test":
        df = clean_data(f"C:\\Users\\simon\Desktop\\ComputerScience\\cs158 final project\\ModelData\\{sys.argv[2]}TEST.csv",lblCol,sys.argv[3])
        clf = joblib.load("RFTrained")
       
        testX = df.drop(columns=[lblCol]).to_numpy()
        testY = df[lblCol].to_numpy()
        # print(clf.)
        testAccuraccy = clf.score(testX, testY)
        print(f'Test Accuracy: {testAccuraccy}')
    elif sys.argv[1] == "compare":
        dfTrain = clean_data(f"C:\\Users\\simon\Desktop\\ComputerScience\\cs158 final project\\ModelData\\{sys.argv[2]}TRAIN.csv",lblCol,sys.argv[3])
        dftest = clean_data(f"C:\\Users\\simon\Desktop\\ComputerScience\\cs158 final project\\ModelData\\{sys.argv[2]}TEST.csv",lblCol,sys.argv[3])
        trainX = dfTrain.drop(columns=[lblCol]).to_numpy()
        trainY = dfTrain[lblCol].to_numpy()
        testX = dftest.drop(columns=[lblCol]).to_numpy()
        testY = dftest[lblCol].to_numpy()
        for depth in range(1,21):
            clf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=0, bootstrap=True)
            clf.fit(trainX, trainY)
            print(f"{depth},{clf.score(trainX, trainY)}")





    # preProcessData(X,y)
    
    # #  print(f'Accuracy: {bestParams[0]} || n_estimators: {bestParams[1]} || max_depth: {bestParams[2]}')
    # tunedNEstimators = bestParams[1]
    # tunedMaxDepth = bestParams[2]
    # clf = RandomForestClassifier(n_estimators=tunedNEstimators, max_depth=tunedMaxDepth, random_state=0, bootstrap=True)
    # # clf.fit(trainX,trainY)
    

    # modelAccuraccy = test(testX,testY,joblib.load("RFModel"))
    # # initialSetup(trainX,trainY)


def test(X, Y, clf):
    correct = 0
    for i in range(len(Y)):
        if(clf.predict(X[[i]]) == Y[i]):
            correct +=1
    return correct/len(Y)

def classify(x, clf):
    return clf.predict(x)
        
# def preProcessData(X, y):
#     scaler = preprocessing.StandardScaler().fit(X,y)

def hyperParamTune(X, Y): #?
    # [accuraccy, n_estimators, max_depth]
    bestParams = [0,0,0]
    for k in range(20,40):
        # n_estimators = i
        #max depth
        for j in range(1,10):
            # max_depth = j
            # for r in range(0,10):
            clf = RandomForestClassifier(n_estimators=k, max_depth=j, random_state=0, bootstrap=True)
            #train
            clf.fit(X, Y)

            trues = 0
            for i in range(len(Y)):
                # R = random.randint(0,1)
                if(clf.predict(X[[i]]) == [1] and [1]==Y[i]):
                    trues +=1
                elif(clf.predict(X[[i]]) == [0] and [0]==Y[i]):
                    trues += 1

            accuracy = trues/len(Y)
            if accuracy > bestParams[0]:
                bestParams[1] = k
                bestParams[2] = j
                bestParams[0] = accuracy
    return bestParams
                

if __name__ == "__main__":
    main()