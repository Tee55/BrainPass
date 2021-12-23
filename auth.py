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
from itertools import combinations, count
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, Flatten, MaxPooling1D, BatchNormalization, ReLU, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import shutil
import json
from argparse import ArgumentParser


class EEGutil:

    def __init__(self):
        self.batch_size = 1
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder()
        self.delta = [1, 4]
        self.theta = [4, 8]
        self.alpha = [8, 13]
        self.beta = [13, 30]
        self.gamma = [30, 45]

    def bandpower(self, x, f_range):
        Pxx = fft(x)
        ind_min = f_range[0] * 3
        ind_max = f_range[1] * 3
        Pxx_abs = np.abs(Pxx)
        Pxx_pow = np.square(Pxx_abs)
        Pxx_sum = sum(Pxx_pow[ind_min: ind_max])
        return Pxx_sum

    def segmentation(self, x, time_length, time_shift):
        n = time_length * self.sampling_rate  # group size
        m = time_shift * self.sampling_rate  # overlap size
        return [x[i:i+n] for i in range(0, len(x), n-m)]

    def get_bandpower(self, x):
        epoches = self.segmentation(x, self.duration, self.time_shift)

        delta = theta = alpha = beta = gamma = []
        for epoch in epoches:
            delta_val = self.bandpower(epoch, self.delta)
            theta_val = self.bandpower(epoch, self.theta)
            alpha_val = self.bandpower(epoch, self.alpha)
            beta_val = self.bandpower(epoch, self.beta)
            gamma_val = self.bandpower(epoch, self.gamma)

            delta.append(delta_val)
            theta.append(theta_val)
            alpha.append(alpha_val)
            beta.append(beta_val)
            gamma.append(gamma_val)
        
        delta = np.mean(np.array(delta))
        theta = np.mean(np.array(theta))
        alpha = np.mean(np.array(alpha))
        beta = np.mean(np.array(beta))
        gamma = np.mean(np.array(gamma))

        return delta, theta, alpha, beta, gamma

    def cal_acc(self, predictions, y_val):
        score = 0
        for pred, y in zip(predictions, y_val):
            if pred == y:
                score += 1
        acc = score/len(predictions)
        return acc

    def load_private_dataset(self):
        
        # Variable
        self.sampling_rate = 128
        self.num_people = 33
        self.duration = 3
        self.time_shift = 1
        
        x = []
        y = []

        for id_dir in os.listdir("private_dataset/"):
            
            # 10 file for each person
            for i in range(0, 10):
                data_file = os.listdir(
                    os.path.join("private_dataset/", id_dir))[i]
                general_info_df = pd.read_csv(os.path.join(
                    "private_dataset/", id_dir, data_file), delimiter=",", header=None, nrows=1, skipinitialspace=True)
                general_info = {}
                for col in general_info_df.columns:
                    text = general_info_df.iloc[0][col]
                    key, value = text.split(':', 1)
                    value = value.strip()
                    general_info[key] = value

                headers = general_info["labels"]
                header_list = headers.split(" ")

                df = pd.read_csv(os.path.join("private_dataset/", id_dir, data_file),
                                 delimiter=",", names=header_list, skiprows=1, skipinitialspace=True)
                df = df.drop(df.index[self.sampling_rate * self.duration:])
                x.append(df["O1"])
                y.append(id_dir)

        x = np.array(x)
        y = self.label_encoder.fit_transform(y)
        y = np.reshape(y, (-1, 1))
        return x, y
    
    def load_public_dataset(self):
        
        # Variable
        self.sampling_rate = 512
        self.num_people = 30
        self.duration = 3
        self.time_shift = 1
        
        df = pd.read_csv(os.path.join("public_dataset/", "eeg-data.csv"), sep=",")
        df.raw_values = df.raw_values.map(json.loads)
        
        x = []
        y = []
        for i in range(0, self.num_people):
            raw_list = df[df['id'] == i+1][df.label == "colorRound1-1"].raw_values.tolist()
            raw_data = []
            for raw in raw_list:
                raw_data.extend(raw)
                
            if len(raw_data) >= self.sampling_rate*self.duration:
                raw_data = raw_data[0:self.sampling_rate*self.duration]
            x.append(raw_data)
            y.append(i)
            
        x = np.array(x)
        y = self.label_encoder.fit_transform(y)
        y = np.reshape(y, (-1, 1))
        return x, y

    def preprocessing(self, x, combi=None):
        features = []
        for series in x:
            delta, theta, alpha, beta, gamma = self.get_bandpower(series)
            feature = [delta, theta, alpha, beta, gamma]
            if combi != None:
                feature = [feature[index] for index in combi]
            features.append(feature)
        x = np.array(features)
        return x

    def fisher(self, x, y):

        x = self.preprocessing(x)

        # Fisher score
        fisher_score, _ = chi2(x, y)
        idx = np.argsort(fisher_score)
        idx = idx[::-1]
        return idx

    def create_model(self, input_shape):
        input_layer = Input(input_shape)
        conv1 = Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = ReLU()(conv1)

        conv2 = Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = ReLU()(conv2)

        conv3 = Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = ReLU()(conv3)

        gap = GlobalAveragePooling1D()(conv3)

        output_layer = Dense(self.num_people, activation="softmax")(gap)

        return Model(inputs=input_layer, outputs=output_layer)

    def feature_selection(self, x, y):
        idx = self.fisher(x, y)
        print("Fisher Features Index (High -> Low): {}".format(idx))

        # Feature selection
        all_acc = []
        all_combi = []

        for combi_number in range(1, len(idx)+1):
            combi_idx = combinations(idx, combi_number)
            combi_idx = list(combi_idx)

            for combi in combi_idx:

                data_x = self.preprocessing(x, combi)

                X_train, X_test, y_train, y_test = train_test_split(
                    data_x, y, test_size=0.33)

                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(X_train, y_train.ravel())

                predictions = knn.predict(X_test)
                acc = accuracy_score(y_test, predictions)

                all_acc.append(acc)
                all_combi.append(combi)

        all_acc = np.array(all_acc)
        print("Highest Accuracy({}) with Combination {}".format(np.max(all_acc), all_combi[np.argmax(all_acc)]))
        return all_combi[np.argmax(all_acc)]

    def filter(self, x, f_range):
        b, a = butter(5, f_range, btype='band',
                      fs=self.sampling_rate)
        x = lfilter(b, a, x)
        return x

    def cleanTensorflowLogs(self):
        LOG_DIR = ['logs/']
        for logdir in LOG_DIR:
            subfolders = [os.path.join(logdir, f) for f in os.listdir(
                logdir) if not f.endswith(('.csv', '.gitignore'))]
            for subfolder in subfolders:
                if os.path.isdir(subfolder):
                    shutil.rmtree(subfolder)
                else:
                    os.remove(subfolder)
                    
    def features_extractions(self, x, combi):
        
        new_x = []
        for series in x:
            delta_series = self.filter(series, self.delta)
            theta_series = self.filter(series, self.theta)
            alpha_series = self.filter(series, self.alpha)
            beta_series = self.filter(series, self.beta)
            gamma_series = self.filter(series, self.gamma)
            
            features = [delta_series, theta_series, alpha_series, beta_series, gamma_series]
            features = [features[index] for index in combi]
            
            timeseries = []
            for i in range(len(delta_series)):
                each_features = []
                for feature in features:
                    each_features.append(feature[i])
                timeseries.append(each_features)
            new_x.append(timeseries)
        
        new_x = np.array(new_x)
        return new_x

    def run(self, dataset_type="public"):

        self.cleanTensorflowLogs()

        if dataset_type == "public":
            x, y = self.load_public_dataset()
        else:
            x, y = self.load_private_dataset()
            
        print("Original Shape: {}, {}".format(x.shape, y.shape))
        
        combi = self.feature_selection(x, y)
        x = self.features_extractions(x, combi)

        print("Train shape: {}, {}".format(x.shape, y.shape))

        tensorboard = TensorBoard(log_dir='logs', update_freq='batch')
        callbacks = [tensorboard]

        model = self.create_model(input_shape=x.shape[1:])
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )
        model.fit(x, y, batch_size=self.batch_size,
                  epochs=350, callbacks=callbacks)
        model.save_weights(os.path.join("model", "model.h5"))

        test_loss, test_acc = model.evaluate(x, y)

        print("Test accuracy", test_acc)
        print("Test loss", test_loss)


if __name__ == '__main__':
    eegutil = EEGutil()
    parser = ArgumentParser(description='EEG Authentication')
    parser.add_argument(
        '-d', '-dataset', '--dataset', default="public",
        help='Choose dataset type (Public, Private)'
    )
    args = parser.parse_args()
    eegutil.run(dataset_type=args.dataset)