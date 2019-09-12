

import pickle
import numpy as np

import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.optimizers import  SGD
import keras.backend as K

np.random.seed(0)

n_rows = 5
n_cols = 5


class my_model:
    def __init__(self):
        self.batch_size = 8
        self.num_epochs = 100
        self.train_frac = 0.7
        self.val_frac = 0.9
        self.past_pv = 20
        self.optimizer = SGD(lr=10 ** (-3))
        self.make_data()
        self.make_model()
        self.train()
        self.plot_losses()
        self.save_test()

    def make_data(self):
        self.parking = pickle.load(open(r'C:\Users\deept\PycharmProjects\UC_project\parking_grid_sequence.pkl', 'rb'))
        self.dc_map = (sum(self.parking) > 0)
        self.weather = pickle.load(open(r'C:\Users\deept\PycharmProjects\UC_project\weather_sequences.pkl','rb'))
        self.weather_vars = [i for i in self.weather]
        self.n_weather = len(self.weather_vars)
        self.max_wv = dict()
        self.min_wv = dict()
        for wv in self.weather_vars:
            self.max_wv[wv] = np.max(self.weather[wv])
            self.min_wv[wv] = np.min(self.weather[wv])
        self.poi = pickle.load(open(r'C:\Users\deept\PycharmProjects\UC_project\poi_pv_mat.pkl', 'rb'))
        self.poi = self.poi / np.max(self.poi)
        x_data_parking = []
        x_data_weather = []
        x_data_poi = []
        y_data = []
        for k in range(20, int(self.train_frac * len(self.parking))):
            padded_parking_maps = [np.pad(self.parking[k - j], 1, mode='constant') for j in range(1, 21)]
            for i in range(n_rows):
                for j in range(n_cols):
                    if self.dc_map[i, j]:
                        x_data_parking.append(np.array(
                            np.concatenate([np.ndarray.flatten(pcm[i:i + 3, j:j + 3]) for pcm in padded_parking_maps])))
                        x_data_weather.append(
                            [(self.weather[wv][k] - self.min_wv[wv]) / self.max_wv[wv] for wv in self.weather_vars])
                        x_data_poi.append([self.poi[i, j]])
                        y_data.append([self.parking[k][i, j]])
        self.x_data_parking = np.array(x_data_parking)
        self.x_data_weather = np.array(x_data_weather)
        self.x_data_poi = np.array(x_data_poi)
        self.y_data = np.array(y_data)

    def make_model(self):
        inp_parking = Input(shape=(180,))
        hid1 = Dense(5, activation='tanh')(inp_parking)
        inp_weather = Input(shape=(self.n_weather,))
        hid2 = Dense(5, activation='tanh')(inp_weather)
        inp_poi = Input(shape=(1,))
        hid = Concatenate()([hid1, hid2, inp_poi])
        hid = Dense(1, activation='relu')(hid)
        self.model = Model([inp_parking, inp_weather, inp_poi], hid)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)

    def make_test(self):
        self.x1 = []
        self.x2 = []
        self.x3 = []
        self.y = []
        for i in range(n_rows):
            for j in range(n_cols):
                if self.dc_map[i,j]:
                    for k in range(int(self.train_frac*len(self.parking)), int(self.val_frac*len(self.parking))):
                        padded_parking_maps = [np.pad(self.parking[k-z],1,mode='constant') for z in range(1,21)]
                        x_data_parking = np.array( np.concatenate( [np.ndarray.flatten(pcm[i:i+3,j:j+3]) for pcm in padded_parking_maps]  ) )
                        x_data_weather = np.array([(self.weather[wv][k]-self.min_wv[wv])/self.max_wv[wv] for wv in self.weather_vars])
                        x_data_poi = np.array([self.poi[i,j]])
                        self.x1.append(x_data_parking)
                        self.x2.append(x_data_weather)
                        self.x3.append(x_data_poi)
                        self.y.append([self.parking[k][i,j]])
        self.x1 = np.array(self.x1)
        self.x2 = np.array(self.x2)
        self.x3 = np.array(self.x3)
        self.y = np.array(self.y)

    def test(self):
        self.test_loss_record.append(self.model.test_on_batch([self.x1, self.x2, self.x3], self.y))

    def my_loss(y_true, y_pred):
        return K.sum((y_true - y_pred) ** 2)

    def train(self):
        self.make_test()
        self.train_loss_record = []
        self.test_loss_record = []
        num_data = self.x_data_parking.shape[0]
        mbs = list(range(0, num_data - self.batch_size, self.batch_size))
        self.test()
        self.train_loss_record.append(
            self.model.test_on_batch([self.x_data_parking, self.x_data_weather, self.x_data_poi], self.y_data))
        for it in range(self.num_epochs):
            train_loss = 0
            np.random.shuffle(mbs)
            for b in mbs:
                # train on a batch
                train_inp_parking = self.x_data_parking[b:b + self.batch_size]
                train_inp_weather = self.x_data_weather[b:b + self.batch_size]
                train_inp_poi = self.x_data_poi[b:b + self.batch_size]
                train_true_op = self.y_data[b:b + self.batch_size]
                batch_train_loss = self.model.train_on_batch([train_inp_parking, train_inp_weather, train_inp_poi],
                                                             train_true_op)
                train_loss += batch_train_loss
                print('epoch', it, '| b', b, '| train_loss', train_loss / (b + 1), self.test_loss_record[-1])
            self.train_loss_record.append(train_loss / len(mbs))
            if it % 10 == 0:
                self.model.save('model.h5')
                self.plot_losses()
            self.test()
        self.model.save('model.h5')

    def plot_losses(self):
        plt.plot(self.train_loss_record)
        plt.plot(self.test_loss_record)
        plt.legend(['train_loss', 'test_loss'])
        plt.xlabel('iteration*50')
        plt.savefig('losses_pv.png')
        pickle.dump([self.train_loss_record, self.test_loss_record], open('losses_pv.pkl', 'wb'))

    def save_test(self):
        a = 1


my_model()