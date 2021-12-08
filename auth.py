import pandas as pd
import os
import numpy as np
from datetime import datetime
from scipy.fft import fft
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import json
from keras.preprocessing.sequence import TimeseriesGenerator

class EEGutil:
    
    def __init__(self):
        self.sampling_rate = 512
        self.num_people = 31
        self.batch_size = 1
        self.num_channels = 3
        
        
    def create_model(self, n_timesteps, n_features):
        model = Sequential()
        model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
        model.add(Dense(self.num_people, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def checkDir(self):
        if not os.path.exists("dataset/"):
            os.mkdir("dataset/")
            
    def stimulus_duration(self, stimulus_name):
        df = pd.read_csv(os.path.join("dataset", "stimulus-times.csv"), sep=",")
        stimulus_list = list(df["event name"])
        print(stimulus_list)
        start_time = df.iloc[stimulus_list.index(stimulus_name)]["time"]
        end_time = df.iloc[stimulus_list.index(stimulus_name) + 1]["time"]
        FMT = '%Y-%m-%d %H:%M:%S.%f+00'
        interval = datetime.strptime(end_time, FMT) - datetime.strptime(start_time, FMT)
        interval = int(interval.total_seconds())
        return interval

    def bandpower(self, x, fmin, fmax):
        Pxx = fft(x)
        ind_min = fmin * 3
        ind_max = fmax * 3
        Pxx_abs = np.abs(Pxx)
        Pxx_pow = np.square(Pxx_abs)
        Pxx_sum = sum(Pxx_pow[ind_min: ind_max])
        return Pxx_sum
            
    def segmentation(self, x, time_length, time_shift):
        n = time_length * self.sampling_rate  # group size
        m = time_shift * self.sampling_rate  # overlap size
        return [x[i:i+n] for i in range(0, len(x), n-m)]
    
    def compute_features(self, x):
        epoches = self.segmentation(x, 3, 1)
        features = np.empty((len(epoches), 6))
        for i, epoch in enumerate(epoches):
            delta = self.bandpower(epoch, 1, 4)
            theta = self.bandpower(epoch, 4, 8)
            alpha = self.bandpower(epoch, 8, 13)
            beta = self.bandpower(epoch, 13, 30)
            gamma = self.bandpower(epoch, 30, 45)
            total_power = self.bandpower(epoch, 1, 45)

            features[i][0] = delta
            features[i][1] = theta
            features[i][2] = alpha
            features[i][3] = beta
            features[i][4] = gamma
            features[i][5] = total_power
                
        return features
    def main(self):
        df = pd.read_csv(os.path.join("dataset", "eeg-data.csv"), sep=",")
        df.raw_values = df.raw_values.map(json.loads)
        print(df.columns)
        df = df[df.label == 'colorRound1-1']
        
        self.duration = self.stimulus_duration('colorRound1-1')
        self.time_shift = 1
        print('Duration: %d' % self.duration)
        
        X_train = []
        y_train = []
        y_hats = np.arange(self.num_people)
        y_hats = np.unique(y_hats)
        for subject in range(0, self.num_people):
            data = df[df['id'] == subject+1].raw_values.tolist()
            raw_data = []
            for series in data:
                raw_data.extend(series)
            
            if len(raw_data) >= self.sampling_rate*self.duration:
                raw_data = raw_data[0:self.sampling_rate*self.duration]
                features = self.compute_features(raw_data)
                X_train.append(features)
                y_train.append(y_hats[subject])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        print(X_train.shape)
        print(y_train.shape)
        model = self.create_model(n_timesteps=X_train.shape[1], n_features=X_train.shape[2])
        model.fit(X_train, y_train, epochs=10, batch_size=self.batch_size, verbose=1)
    
    
if __name__ == '__main__':
    eegutil = EEGutil()
    eegutil.main()
    