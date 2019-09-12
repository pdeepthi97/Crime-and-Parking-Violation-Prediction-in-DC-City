import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor as MLP
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam, SGD

class my_model:  
    def __init__(self):
        self.batch_size = 8
        self.num_epoch = 50
        self.num_iter = self.num_epoch*548
        self.train_frac = 0.7
        self.val_frac = 0.9
        self.optimizer = SGD(lr=0.000001)
        self.make_data()
        self.make_model()
        self.train()
        self.plot_losses()
        self.save_test()

    def make_data(self):
        crime = pickle.load(open('../../data/crime_sequences.pkl','rb'))
        self.weather = pickle.load(open('../../data/weather_sequences.pkl','rb'))
        self.weather_vars = [i for i in self.weather]
        self.n_weather = len(self.weather_vars)
        self.max_wv = dict()
        self.min_wv = dict()
        for wv in self.weather_vars:
            self.max_wv[wv] = np.max(self.weather[wv])
            self.min_wv[wv] = np.min(self.weather[wv])
        data = []
        for i in range(20, len(crime)):
            data.append( crime[i-20:i] + [(self.weather[wv][i]-self.min_wv[wv])/self.max_wv[wv] for wv in self.weather_vars] + [crime[i]] )
            if len(data[i-20])!=36:
                print (len(data[i-20]))
        print (len(data))
        self.inp_dim = len(data[0])-1
        data = np.array(data)
        bp = int(self.train_frac*data.shape[0])
        bp2 = int(self.val_frac*data.shape[0]) 
        self.train_data = data[:bp,:]
        self.val_data = data[bp:bp2,:]
        self.test_data = data[bp2:,:]
    
    def make_model(self):
        inp = Input(shape=(self.inp_dim,))
        hid = Dense(128, activation='tanh')(inp)
#        hid = Dense(128, activation='tanh')(inp)
        op = Dense(1, activation='relu')(hid)
        self.model = Model(inp, op)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
    
    def train(self):
        self.train_loss_record = []
        self.test_loss_record = []
        train_loss = 0
        test_loss = 0
        for it in range(self.num_iter):
            #train on a batch
            idx = np.random.randint(0, self.train_data.shape[0], self.batch_size)
            train_inp = self.train_data[idx, :-1]
            train_true_op = self.train_data[idx, -1]
            train_loss += self.model.train_on_batch(train_inp, train_true_op) 
            #test on a batch
            test_idx = np.random.randint(0, self.test_data.shape[0], self.batch_size)
            test_inp = self.val_data[test_idx, :-1]
            test_true_op = self.val_data[test_idx, -1] 
            test_loss += self.model.test_on_batch(test_inp, test_true_op) 
            if it%548==0:
                self.train_loss_record.append(train_loss/548)
                self.test_loss_record.append(test_loss/548)
                train_loss = 0
                test_loss = 0
            print ('iteration',it,'| train_loss',train_loss, '| test loss', test_loss)
            
    def save_test(self):
        test_true_op = self.test_data[:,-1]
        test_pred_op = self.model.predict(self.test_data[:,:-1])
        pickle.dump([test_true_op, test_pred_op], open('prediction.pkl', 'wb'))
        plt.figure()
        plt.plot(test_true_op)
        plt.plot(test_pred_op)
        plt.xlabel('time')
        plt.ylabel('crime count')
        plt.legend(['actual','predicted'])
        plt.savefig('prediction.png')
            
    def plot_losses(self):
        plt.plot(list(range(1, len(self.train_loss_record))), self.train_loss_record[1:])
        plt.plot(list(range(1, len(self.train_loss_record))),self.test_loss_record[1:])
        plt.legend(['training loss','validation loss'])
        plt.xlabel('epoch')
        plt.savefig('losses.png')

my_model()


#crime = pickle.load(open('../../data/crime_sequences.pkl','rb'))
#temp = pickle.load(open('../../data/weather_sequences.pkl','rb'))['temp']
#data = []
#for i in range(50, len(crime)):
#    data.append( crime[i-50:i] + temp[i-4:i+1] + [crime[i]] )
#data = np.array(data)
#bp = int(self.train_frac*self.inp_dim)
#self.train_data = data[:bp,:]
#self.test_data = data[bp:,:]
#reg = MLP(hidden_layer_sizes=(300), max_iter=200)
#reg.fit(data[:,:-1], data[:,-1])
#
#err = reg.predict(data[:,:-1]) - data[:,-1]
#print (min(err), max(err), sum(abs(err))/len(err))
#plt.plot(data[:,-1])
#plt.plot(err)