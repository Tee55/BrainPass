import pandas as pd
import os
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import json

def main():
    df = pd.read_csv(os.path.join("dataset", "eeg-data.csv"), sep=",")
    
    labelList = list(dict.fromkeys(df["label"]))
    print(labelList)
    
    train_X = df[df.label == 'colorRound1-1']
    train_y = df[df.label == 'relax']
    
    
    
            

if __name__ == '__main__':
    main()
    