import pandas as pd
import os
import numpy as np
from datetime import datetime
from scipy.fft import fft
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import json
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2

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
        
        # Six features
        features = np.empty((6, ))
        
        delta_avg = []
        theta_avg = []
        alpha_avg = []
        beta_avg = []
        gamma_avg = []
        total_power_avg = []
        for i, epoch in enumerate(epoches):
            delta = self.bandpower(epoch, 1, 4)
            theta = self.bandpower(epoch, 4, 8)
            alpha = self.bandpower(epoch, 8, 13)
            beta = self.bandpower(epoch, 13, 30)
            gamma = self.bandpower(epoch, 30, 45)
            total_power = self.bandpower(epoch, 1, 45)
            
            delta_avg.append(delta)
            theta_avg.append(theta)
            alpha_avg.append(alpha)
            beta_avg.append(beta)
            gamma_avg.append(gamma)
            total_power_avg.append(total_power)
            
        delta_avg = np.mean(np.array(delta_avg))
        theta_avg = np.mean(np.array(theta_avg))
        alpha_avg = np.mean(np.array(alpha_avg))
        beta_avg = np.mean(np.array(beta_avg))
        gamma_avg = np.mean(np.array(gamma_avg))
        total_power_avg = np.mean(np.array(total_power_avg))

        features[0] = delta
        features[1] = theta
        features[2] = alpha
        features[3] = beta
        features[4] = gamma
        features[5] = total_power
                
        return features
    def main(self):
        df = pd.read_csv(os.path.join("dataset", "eeg-data.csv"), sep=",")
        df.raw_values = df.raw_values.map(json.loads)
        print(df.columns)
        df = df[df.label == 'colorRound1-1']
        
        self.duration = self.stimulus_duration('colorRound1-1')
        self.time_shift = 1
        print('Duration: %d' % self.duration)
        
        features_all = []
        y_train = []
        integer_encoded = np.arange(0, self.num_people)
        # binary encode
        integer_encoded = np.reshape(integer_encoded, (-1, 1))
        onehot_encoder = OneHotEncoder(sparse=False)
        y_hats = onehot_encoder.fit_transform(integer_encoded)
        for subject in range(0, self.num_people):
            data = df[df['id'] == subject+1].raw_values.tolist()
            raw_data = []
            for series in data:
                raw_data.extend(series)
            
            if len(raw_data) >= self.sampling_rate*self.duration:
                raw_data = raw_data[0:self.sampling_rate*self.duration]
                features = self.compute_features(raw_data)
                features_all.append(features)
                y_train.append(y_hats[subject])
                
        features_all = np.array(features_all)
        scaler = MinMaxScaler(feature_range=(0, 1))
        features_all = scaler.fit_transform(features_all)
        y_train = np.array(y_train)
        print(features_all.shape)
        print(y_train.shape)
        
        # Fisher score
        fisher_score = chi2(features_all, y_train)    
        print('Score: ', fisher_score)
    
if __name__ == '__main__':
    eegutil = EEGutil()
    eegutil.main()
    