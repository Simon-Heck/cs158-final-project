from sklearn.ensemble import RandomForestClassifier 
import sys
from data_cleaning import *
from sklearn.model_selection import cross_val_score
__author__ = "Simon Heck, Stefanos Stoikos, Nicholas Marsano"

def main():
    # Ex cmd line input: python DecisionTree.py compare credit 1
    # column in panda dataframe name which contains the labels
    lblCol = "Label"
    job = sys.argv[1]
    dataset = sys.argv[2]
    #0 for min max, 1 for mean scaling
    scalingOption = sys.argv[3]
    if job == "compare":
        dfData = clean_data(f"C:\\Users\\simon\Desktop\\ComputerScience\\cs158 final project\\ModelData\\{dataset}DATA.csv",lblCol,scalingOption)
        X = dfData.drop(columns=[lblCol]).to_numpy()
        Y = dfData[lblCol].to_numpy()
        for depth in range(1,21):
            clf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=0, bootstrap=True)
            # cross_val_score handles training of model for each fold
            print(f"{depth},{cross_val_score(clf,X, Y,cv=10).mean()}")

if __name__ == "__main__":
    main()