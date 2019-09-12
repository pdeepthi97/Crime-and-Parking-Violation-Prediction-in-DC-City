#not a cnn but an ann

import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor as MLP
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv2D, Concatenate
from keras.optimizers import Adam, SGD
import keras.backend as K
np.random.seed(0)

n_rows = 5
n_cols = 5
class my_model:  
    def __init__(self):
        self.batch_size = 8
        self.num_epochs = 100
        self.train_frac = 0.9
        self.past_crime = 20
        self.optimizer = SGD(lr=10**(-3))
        self.make_data()
        self.make_model()
        self.train()
        self.plot_losses()
        self.save_test()

    def make_data(self):
        crime = pickle.load(open('../../data/crime_mat_sequence.pkl','rb'))
        self.dc_map = (sum(crime)>0)
        weather = pickle.load(open('../../data/weather_sequences.pkl','rb'))
        weather_vars = [i for i in weather]
        self.n_weather = len(weather_vars)
        max_wv = dict()
        for wv in weather_vars:
            max_wv[wv] = np.max(weather[wv])
        poi = pickle.load(open('../../data/poi_mat.pkl','rb'))
        poi = poi/np.max(poi)
        x_data_crime = []
        x_data_weather = []
        x_data_poi = []
        y_data = []        
        for k in range(20, int(self.train_frac*len(crime))):
            padded_crime_maps = [np.pad(crime[k-j],1,mode='constant') for j in range(1,21)]
            for i in range(n_rows):
                for j in range(n_cols):
                    if self.dc_map[i,j]:
                        x_data_crime.append(np.array( np.concatenate( [np.ndarray.flatten(pcm[i:i+3,j:j+3]) for pcm in padded_crime_maps]  ) ))
                        x_data_weather.append([weather[wv][k]/max_wv[wv] for wv in weather_vars])
                        x_data_poi.append([poi[i,j]])
                        y_data.append([crime[k][i,j]])
        self.x_data_crime = np.array(x_data_crime)
        self.x_data_weather = np.array(x_data_weather)
        self.x_data_poi = np.array(x_data_poi)
        self.y_data = np.array(y_data)
        
    def make_model(self):
        inp_crime = Input(shape=(180,))
        hid1 = Dense(5, activation='tanh')(inp_crime)
        inp_weather = Input(shape=(self.n_weather,))
        hid2 = Dense(5, activation='tanh')(inp_weather)
        inp_poi = Input(shape=(1,))
        hid = Concatenate()([hid1,hid2,inp_poi])
        hid = Dense(1,activation='relu')(hid)
        self.model = Model([inp_crime,inp_weather,inp_poi],hid)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
    
    def train(self):
        self.train_loss_record = []
        self.test_loss_record = []
        num_data = self.x_data_crime.shape[0]
        mbs = list(range(0,num_data-self.batch_size, self.batch_size))
        for it in range(self.num_epochs):
            train_loss = 0
            np.random.shuffle(mbs)
            for b in mbs:
                #train on a batch
                train_inp_crime = self.x_data_crime[b:b+self.batch_size]
                train_inp_weather = self.x_data_weather[b:b+self.batch_size]
                train_inp_poi = self.x_data_poi[b:b+self.batch_size]
                train_true_op = self.y_data[b:b+self.batch_size]
                batch_train_loss = self.model.train_on_batch([train_inp_crime, train_inp_weather, train_inp_poi], train_true_op) 
                train_loss += batch_train_loss
                print ('epoch',it,'| b', b, '| train_loss',train_loss/(b+1))
            self.train_loss_record.append(train_loss/int(num_data/self.batch_size))
            
            
    def plot_losses(self):
        plt.plot(self.train_loss_record)
        plt.plot(self.test_loss_record)
        plt.legend(['train_loss','test_loss'])
        plt.xlabel('iteration*50')
        plt.savefig('losses.png')
        pickle.dump([self.train_loss_record, self.test_loss_record], open('losses.pkl','wb'))
        
    def save_test(self):
        a=1
#        crime = pickle.load(open('../../data/crime_mat_sequence.pkl','rb'))
#        self.dc_map = (sum(crime)>0)
#        temp = pickle.load(open('../../data/weather_sequences.pkl','rb'))['temp']
#        poi = pickle.load(open('../../data/poi_mat.pkl','rb'))
#        padded_crime_maps = [np.pad(c,1,mode='constant') for c in crime]
#        for i in range(n_rows):
#            for j in range(n_cols):
#                    if self.dc_map[i,j]:
#                        for i in range(int(self.train_frac*len(crime)), len(crime)):
#                            
#                        
#                        
#                        x_data_crime.append(np.array( np.concatenate( [np.ndarray.flatten(pcm[i:i+3,j:j+3]) for pcm in padded_crime_maps]  ) ))
#                        x_data_weather.append([temp[i]])
#                        x_data_poi.append([poi[i,j]])
#                        y_data.append([crime[i][i,j]])        

my_model()