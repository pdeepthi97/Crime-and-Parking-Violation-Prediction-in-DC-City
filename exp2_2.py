#using custom loss to not include cells not in DC
import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor as MLP
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Conv2D
from keras.optimizers import Adam, SGD
import keras.backend as K
np.random.seed(0)

n_rows = 5
n_cols = 5
class my_model:  
    def __init__(self):
        self.batch_size = 8
        self.num_iter = 10000
        self.train_frac = 0.7
        self.past_crime = 20
        self.lr =0.00000001
        self.make_data()
        self.train()
        self.plot_losses()
        self.save_test()

    def make_data(self):
        crime = pickle.load(open('../../data/crime_mat_sequence.pkl','rb'))
        temp = pickle.load(open('../../data/weather_sequences.pkl','rb'))['temp']
        poi = pickle.load(open('../../data/poi_mat.pkl','rb'))
        x_data = []
        y_data = []
        for i in range(50, len(crime)):
            x_data.append( crime[i-self.past_crime:i] + [poi] + [np.full((n_rows,n_cols),temp[i])] )
            y_data.append([crime[i]])
        self.n_inp_maps = len(x_data[0])
        bp = int(self.train_frac*len(x_data))
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        self.x_train = x_data[:bp,:,:,:]
        self.y_train = y_data[:bp,:,:,:]
        self.x_test = x_data[bp:,:,:,:]
        self.y_test = y_data[bp:,:,:,:]
        #extract cells overlapping with DC
        self.dc_map = (sum(crime)>0).astype(int)
        self.n_cells = np.sum(self.dc_map)
        
    def make_model(self, lr, first_make=True):
        inp = Input(shape=(self.n_inp_maps,n_rows,n_cols))
        hid = Conv2D(1, kernel_size=(3,3), padding='same', data_format='channels_first', activation='tanh')(inp)
        hid = Conv2D(1, kernel_size=(3,3), padding='same', data_format='channels_first', activation='relu')(hid)
        self.model = Model(inp, hid)
        def my_loss(y_true, y_pred):
            squared_error = K.abs(y_true-y_pred)
            filter_squared_error = self.dc_map * squared_error
            return K.sum(filter_squared_error)/self.n_cells
        if not(first_make):
            self.model.load_weights('model_weights.h5')
        self.model.compile(loss=my_loss, optimizer=SGD(lr=lr))
    
    def save_model(self):
        self.model.save_weights('model_weights.h5')

    def train_batch(self,it):
        #train on a batch
        idx = np.random.randint(0, self.x_train.shape[0], self.batch_size)
        train_inp = self.x_train[idx,:,:,:]
        train_true_op = self.y_train[idx,:,:,:]
        train_loss = self.model.train_on_batch(train_inp, train_true_op) / self.batch_size
        #test on a batch
        test_idx = np.random.randint(0, self.x_test.shape[0], self.batch_size)
        test_inp = self.x_test[test_idx,:,:,:]
        test_true_op = self.y_test[test_idx,:,:,:] 
        test_loss = self.model.test_on_batch(test_inp, test_true_op) /self.batch_size
        if it%50==0:
            self.train_loss_record.append(train_loss)
            self.test_loss_record.append(test_loss)
        print ('iteration',it,'| train_loss',train_loss, '| test loss', test_loss)        
    
    def train(self):
        self.train_loss_record = []
        self.test_loss_record = []
        self.make_model(self.lr)
        for it in range(5000):
            self.train_batch(it)
        self.save_model()
        self.make_model(0.9, False)
        for it in range(5000):
            self.train_batch(it)
            
    def save_test(self):
        for i in range(1):
            test_true_op = self.y_test[i:i+1,:,:,:]
            test_pred_op = np.round(self.model.predict(self.x_test[0:1,:,:,:])).astype(int)
            print (test_true_op)
            print (test_pred_op)
            print ( np.sum(np.abs(test_true_op-test_pred_op)*self.dc_map)/self.n_cells )
        
#        pickle.dump([test_true_op, test_pred_op], open('prediction.pkl', 'wb'))
#        plt.figure()
#        plt.plot(test_true_op)
#        plt.plot(test_pred_op)
#        plt.legend(['actual','predicted'])
#        plt.title('Results on validation sequence')
#        plt.savefig('prediction.png')
            
    def plot_losses(self):
        plt.plot(self.train_loss_record[5:])
        plt.plot(self.test_loss_record[5:])
        plt.legend(['train_loss','test_loss'])
        plt.xlabel('iteration*50')
        plt.savefig('losses.png')
        pickle.dump([self.train_loss_record, self.test_loss_record], open('losses.pkl','wb'))

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