import pandas as pd
import os
import numpy as np
from scipy.fft import fft
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class EEGutil:
    
    def __init__(self):
        self.batch_size = 1
        self.DATASET_DIR = "dataset/"
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder()
        self.duration = 3
        self.time_shift = 1
        self.n_samples = 10
        self.sampling_rate = 128

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
    
    def extract_features(self, x):
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
            
            delta_avg.append(delta)
            theta_avg.append(theta)
            alpha_avg.append(alpha)
            beta_avg.append(beta)
            gamma_avg.append(gamma)
            
        delta_avg = np.mean(np.array(delta_avg))
        theta_avg = np.mean(np.array(theta_avg))
        alpha_avg = np.mean(np.array(alpha_avg))
        beta_avg = np.mean(np.array(beta_avg))
        gamma_avg = np.mean(np.array(gamma_avg))
        
        features = [delta_avg, theta_avg, alpha_avg, beta_avg, gamma_avg]
                
        return features

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
            for i in range(0, self.n_samples):
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
                x.append(df["O1"])
                y.append(id_dir)
            
        x = np.array(x)
        y = self.label_encoder.fit_transform(y)
                
        return x, y
    
    def preprocessing(self, x):
        
        new_x = []
        for series in x:
            features = self.extract_features(series)
            new_x.append(features)
            
        x = np.array(new_x)
        
        # Normalization
        x = preprocessing.normalize(x)
        
        df = pd.DataFrame(x)
        print(df)
             
        return x
        
    def main(self):
        x, y = self.load_dataset()
        x = self.preprocessing(x)
        
        print(x.shape, y.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        
        predictions = knn.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print("KNN Accuracy: {}".format(acc))
        
        predictions = lda.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print("LDA Accuracy: {}".format(acc))
    
if __name__ == '__main__':
    eegutil = EEGutil()
    eegutil.main()
    