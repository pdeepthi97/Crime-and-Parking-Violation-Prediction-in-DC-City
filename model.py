import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam, SGD


class my_model:
    def __init__(self):
        self.batch_size = 8
        self.num_iter = 100000
        self.train_frac = 0.6
        self.optimizer = SGD(lr=0.000001)
        self.make_data()
        self.make_model()
        self.train()
        self.plot_losses()
        self.save_test()

    def make_data(self):
        parking = pickle.load(open(r'C:\Users\deept\PycharmProjects\UC_project\parking_violation.pkl','rb'))
        temp = pickle.load(open(r'C:\Users\deept\PycharmProjects\UC_project\weather_sequences.pkl','rb'))['temp']
        data = []
        for i in range(50, len(parking)):
            data.append(parking[i - 50:i] + temp[i - 4:i + 1] + [parking[i]])
        self.inp_dim = len(data[0]) - 1
        data = np.array(data)
        bp = int(self.train_frac * data.shape[0])
        self.train_data = data[:bp, :]
        self.test_data = data[bp:, :]

    def make_model(self):
        inp = Input(shape=(self.inp_dim,))
        hid = Dense(128, activation='tanh')(inp)
        hid = Dense(128, activation='tanh')(hid)
        hid = Dropout(0.5)(hid)
        op = Dense(1, activation='linear')(hid)
        self.model = Model(inp, op)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)

    def train(self):
        self.train_loss_record = []
        self.test_loss_record = []
        for it in range(self.num_iter):
            # train on a batch
            idx = np.random.randint(0, self.train_data.shape[0], self.batch_size)
            train_inp = self.train_data[idx, :-1]
            train_true_op = self.train_data[idx, -1]
            train_loss = self.model.train_on_batch(train_inp, train_true_op) / self.batch_size
            # test on a batch
            test_idx = np.random.randint(0, self.test_data.shape[0], self.batch_size)
            test_inp = self.test_data[test_idx, :-1]
            test_true_op = self.test_data[test_idx, -1]
            test_loss = self.model.test_on_batch(test_inp, test_true_op) / self.batch_size
            if it % 548 == 0:
                self.train_loss_record.append(train_loss)
                self.test_loss_record.append(test_loss)
            print('iteration', it, '| train_loss', train_loss, '| test loss', test_loss)

    def save_test(self):
        test_true_op = self.test_data[:, -1]
        test_pred_op = self.model.predict(self.test_data[:, :-1])
        pickle.dump([test_true_op, test_pred_op], open('prediction.pkl', 'wb'))
        plt.figure()
        plt.plot(test_true_op)
        plt.plot(test_pred_op)
        plt.legend(['actual', 'predicted'])
        plt.title('Results on validation sequence')
        plt.savefig('prediction.png')

    def plot_losses(self):
        plt.plot(self.train_loss_record)
        plt.plot(self.test_loss_record)
        plt.legend(['train_loss', 'test_loss'])
        plt.xlabel('iteration*500')
        plt.savefig('losses.png')


my_model()

