
import pandas as pd
import os  
import sys


def main():
    # os.makedirs("C:\\Users\\simon\Desktop\\ComputerScience\\cs158 final project\\ModelData", exist_ok=True) 
    df = pd.read_csv(f"C:\\Users\\simon\Desktop\\ComputerScience\\cs158 final project\\ModelData\\{sys.argv[1]}DATA.csv")
    part_85 = df.sample(frac=.85)
    part_15 = df.drop(part_85.index)
    os.makedirs("C:\\Users\\simon\Desktop\\ComputerScience\\cs158 final project\\ModelData", exist_ok=True) 
    part_15.to_csv(f"C:\\Users\\simon\Desktop\\ComputerScience\\cs158 final project\\ModelData\\{sys.argv[1]}TEST.csv",index=False)
    part_85.to_csv(f"C:\\Users\\simon\Desktop\\ComputerScience\\cs158 final project\\ModelData\\{sys.argv[1]}TRAIN.csv",index=False)
    

if __name__ == "__main__":
    main()