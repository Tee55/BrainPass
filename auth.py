from numpy.lib.function_base import append
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
from itertools import combinations
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, Flatten, MaxPooling1D, BatchNormalization, ReLU, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import shutil


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

        delta = theta = alpha = beta = gamma = total_power = []
        for epoch in epoches:
            delta_val = self.bandpower(epoch, 1, 4)
            theta_val = self.bandpower(epoch, 4, 8)
            alpha_val = self.bandpower(epoch, 8, 13)
            beta_val = self.bandpower(epoch, 13, 30)
            gamma_val = self.bandpower(epoch, 30, 45)

            delta.append(delta_val)
            theta.append(theta_val)
            alpha.append(alpha_val)
            beta.append(beta_val)
            gamma.append(gamma_val)

        return delta, theta, alpha, beta, gamma

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
        channels = ["O1", "O2"]

        for id_dir in os.listdir(self.DATASET_DIR):
            for i in range(0, self.n_samples):
                data_file = os.listdir(
                    os.path.join(self.DATASET_DIR, id_dir))[i]
                general_info_df = pd.read_csv(os.path.join(
                    self.DATASET_DIR, id_dir, data_file), delimiter=",", header=None, nrows=1, skipinitialspace=True)
                general_info = {}
                for col in general_info_df.columns:
                    text = general_info_df.iloc[0][col]
                    key, value = text.split(':', 1)
                    value = value.strip()
                    general_info[key] = value

                headers = general_info["labels"]
                header_list = headers.split(" ")

                df = pd.read_csv(os.path.join(self.DATASET_DIR, id_dir, data_file),
                                 delimiter=",", names=header_list, skiprows=1, skipinitialspace=True)
                df = df.drop(df.index[self.sampling_rate * self.duration:])
                x.append([df[channel] for channel in channels])
                y.append(id_dir)
        print(headers)
        self.num_people = len(os.listdir(self.DATASET_DIR))

        x = np.array(x)
        y = self.label_encoder.fit_transform(y)
        y = np.reshape(y, (-1, 1))
        return x, y

    def preprocessing(self, x, combi=None):
        new_x = []
        for sample in x:
            delta_avg = theta_avg = alpha_avg = beta_avg = gamma_avg = []
            for channel in sample:
                delta, theta, alpha, beta, gamma = self.extract_features(
                    channel)
                delta_avg.append(delta)
                theta_avg.append(theta)
                alpha_avg.append(alpha)
                beta_avg.append(beta)
                gamma_avg.append(gamma)

            delta_avg = np.array(delta_avg)
            theta_avg = np.array(theta_avg)
            alpha_avg = np.array(alpha_avg)
            beta_avg = np.array(beta_avg)
            gamma_avg = np.array(gamma_avg)

            delta_avg = np.mean(delta_avg)
            theta_avg = np.mean(theta_avg)
            alpha_avg = np.mean(alpha_avg)
            beta_avg = np.mean(beta_avg)
            gamma_avg = np.mean(gamma_avg)

            features = np.array([delta_avg, theta_avg,
                                alpha_avg, beta_avg, gamma_avg])

            if combi != None:
                features = np.array([features[index] for index in combi])

            new_x.append(features)

        x = np.array(new_x)

        df = pd.DataFrame(x)
        # print(df)

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
                knn.fit(X_train, y_train)

                print("=====Combination {}=====".format(combi))

                predictions = knn.predict(X_test)
                acc = accuracy_score(y_test, predictions)
                print("KNN Accuracy: {}".format(acc))

                all_acc.append(acc)
                all_combi.append(combi)

        all_acc = np.array(all_acc)

        print("Highest Accuracy: {}".format(np.max(all_acc)))
        print("With Combination: {}".format(all_combi[np.argmax(all_acc)]))

        return all_combi[np.argmax(all_acc)]

    def filter(self, x, low_fs, high_fs):
        b, a = butter(5, [low_fs, high_fs], btype='band',
                      fs=self.sampling_rate)
        channel = lfilter(b, a, x)
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

    def main(self):

        self.cleanTensorflowLogs()

        x, y = self.load_dataset()
        print("Original Shape: {}, {}".format(x.shape, y.shape))

        new_x = []
        for sample in x:
            data = np.empty((sample.shape[1], sample.shape[0]))
            for i, channel in enumerate(sample):
                for j, each_point in enumerate(channel):
                    data[j][i] = each_point

            data = preprocessing.normalize(data)

            new_x.append(data)

        x = np.array(new_x)

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

        eval = model.evaluate(x)
        print(eval)


if __name__ == '__main__':
    eegutil = EEGutil()
    eegutil.main()