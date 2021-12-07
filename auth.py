import pandas as pd
import os
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

def main():
    df = pd.read_csv(os.path.join("dataset", "eeg-data.csv"), sep=",")
    
    labelList = list(dict.fromkeys(df["label"]))
    print(labelList)
    
    train_X =  []
    train_y = []
    
    for index, row in df.iterrows():
        if row["label"] != "unlabeled":
            #generator = TimeseriesGenerator(data=row["raw_values"], targets=train_y, length=200, batch_size=1)
            print(row["indra_time"], row["createdAt"], row["updatedAt"])
            
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    print(train_X.shape)
    print(train_y.shape)
    
    
            

if __name__ == '__main__':
    main()
    