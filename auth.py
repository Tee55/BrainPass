import pandas as pd
import os
import numpy as np
from datetime import datetime
from scipy.fft import fft
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import chi2, SelectKBest
from progress.bar import Bar
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier

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
    
    def compute_features(self, x, num_features=5):
        epoches = self.segmentation(x, 3, 1)
        
        # Six features
        features = np.empty((num_features, ))
        
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

        features[0] = delta/total_power
        features[1] = theta/total_power
        features[2] = alpha/total_power
        features[3] = beta/total_power
        features[4] = gamma/total_power
                
        return features
    
    def fisher(self, x, y):
        
        features_all = []
        for raw_data in x:
            features = self.compute_features(raw_data)
            features_all.append(features)
        
        features_all = np.array(features_all)
        
        # Fisher score
        fisher_score,_ = chi2(features_all, y)    
        idx = np.argsort(fisher_score)
        idx = idx[::-1]
        return idx

    def cal_acc(self, predictions, y_val):
        score = 0
        for pred, y in zip(predictions, y_val):
            if pred == y:
                score += 1
        acc = score/len(predictions)
        return acc
        
    def main(self):
        df = pd.read_csv(os.path.join("dataset", "eeg-data.csv"), sep=",")
        df.raw_values = df.raw_values.map(json.loads)
        print(df.columns)
        df = df[df.label == 'colorRound1-1']
        
        self.duration = self.stimulus_duration('colorRound1-1')
        self.time_shift = 1
        print('Duration: %d' % self.duration)
        
        x = []
        y = []
        y_hats = np.arange(0, self.num_people)
        for subject in range(0, self.num_people):
            data = df[df['id'] == subject+1].raw_values.tolist()
            raw_data = []
            for series in data:
                raw_data.extend(series)
            
            if len(raw_data) >= self.sampling_rate*self.duration:
                raw_data = raw_data[0:self.sampling_rate*self.duration]
                x.append(raw_data)
                y.append(y_hats[subject])
                
        x = np.array(x)   
        y = np.array(y)
        
        idx = self.fisher(x, y)
        print("Fisher Features Index (High -> Low): {}".format(idx))
        for combi_number in range(1, len(idx)+1):
            combi_idx = combinations(idx, combi_number)
            combi_idx = list(combi_idx)
            
            for combi in combi_idx:
                features_all = []
                for raw_data in x:
                    features = self.compute_features(raw_data)
                    features = [features[index] for index in combi]
                    features_all.append(features)
                        
                data_x = np.array(features_all)
                data_y = y
                
                # Leave one out
                loo = LeaveOneOut()
                count = 0
                for train_indices, test_indices in loo.split(data_x):
                    train_X, val_X = data_x[train_indices], data_x[test_indices]
                    train_y, val_y = data_y[train_indices], data_y[test_indices]
                    
                    classifier = KNeighborsClassifier(n_neighbors=5)
                    classifier.fit(train_X, train_y)
                    preds = classifier.predict(val_X)
                    if preds[0] == val_y[0]:
                        count += 1
                print("Accuracy: {}".format(count))
    
if __name__ == '__main__':
    eegutil = EEGutil()
    eegutil.main()
    