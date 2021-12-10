import pandas as pd
import os
import numpy as np
from scipy.fft import fft
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import chi2
from progress.bar import Bar
from sklearn.model_selection import LeaveOneOut
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import butter, lfilter

class EEGutil:
    
    def __init__(self):
        self.batch_size = 1
        self.DATASET_DIR = "dataset/"
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder()
        self.duration = 3
        self.time_shift = 1
        self.n_epochs = 10
        self.sampling_rate = 128
        
        
    def create_model(self, n_timesteps, n_features):
        model = Sequential()
        model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
        model.add(Dense(self.n_epochs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def checkDir(self):
        if not os.path.exists("dataset/"):
            os.mkdir("dataset/")

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
        epoches = self.segmentation(x, self.duration, self.time_shift)
        
        delta_avg = []
        theta_avg = []
        alpha_avg = []
        beta_avg = []
        gamma_avg = []
        total_power_avg = []
        for epoch in epoches:
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

        features = np.empty((num_features, ))
        features[0] = delta_avg/total_power_avg
        features[1] = theta_avg/total_power_avg
        features[2] = alpha_avg/total_power_avg
        features[3] = beta_avg/total_power_avg
        features[4] = gamma_avg/total_power_avg
                
        return features
    
    def fisher(self, x, y):
        
        features_all = []
        for i, raw_data in enumerate(x):
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
    
    def load_dataset(self):
        
        x = []
        y = []
        
        for id_dir in os.listdir(self.DATASET_DIR):
            id_x = np.empty((self.n_epochs, self.sampling_rate * self.duration))
            for i in range(0, self.n_epochs):
                data_file = os.listdir(os.path.join(self.DATASET_DIR, id_dir))[i]
                general_info_df = pd.read_csv(os.path.join(self.DATASET_DIR, id_dir, data_file), delimiter=",", header=None, nrows=1, skipinitialspace=True)
                general_info = {}
                for col in general_info_df.columns:
                    text = general_info_df.iloc[0][col]
                    key, value = text.split(':', 1)
                    value = value.strip()
                    general_info[key] = value

                headers = general_info["labels"]
                header_list = headers.split(" ")
                
                df = pd.read_csv(os.path.join(self.DATASET_DIR, id_dir, data_file), delimiter=",", names=header_list, skiprows=1, skipinitialspace=True)
                df = df.drop(df.index[self.sampling_rate * self.duration:])
                id_x[i] = df["O1"]
            x.append(id_x)
            y.append(id_dir)
            
        x = np.array(x)
        y = self.label_encoder.fit_transform(y)
        
        print(x.shape, y.shape)
                
        return x, y
    
    def preprocessing(self, x):  
        minmax_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()
        for i, epoches in enumerate(x):
            epoches = minmax_scaler.fit_transform(epoches)
            epoches = standard_scaler.fit_transform(epoches)
            x[i] = epoches
        return x
    
    def filter(self, x, idx):
        
        if idx[0] == 0:
            low_fs = 1
            high_fs = 4
        elif idx[0] == 1:
            low_fs = 4
            high_fs = 8
        elif idx[0] == 2:
            low_fs = 8
            high_fs = 13
        elif idx[0] == 3:
            low_fs = 13
            high_fs = 30
        elif idx[0] == 4:
            low_fs = 30
            high_fs = 45
        
        for i, epoches in enumerate(x):
            b, a = butter(5, [low_fs, high_fs], btype='band', fs=self.sampling_rate)
            epoches = lfilter(b, a, epoches)
            x[i] = epoches
        return x
        
    def main(self):
        x, y = self.load_dataset()
        x = self.preprocessing(x)
        idx = self.fisher(x, y)
        print("Fisher Features Index (High -> Low): {}".format(idx))
        
        x = self.filter(x, idx)
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
                print("Combination: {}, Accuracy: {}".format(combi_number, count))
    
if __name__ == '__main__':
    eegutil = EEGutil()
    eegutil.main()
    