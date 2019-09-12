from keras.models import load_model
import pickle
import numpy as np

n_rows = 5
n_cols = 5
model = load_model('model9.h5')
train_frac = 0.9
crime = pickle.load(open('../../data/crime_mat_sequence.pkl','rb'))
dc_map = (sum(crime)>0)
weather = pickle.load(open('../../data/weather_sequences.pkl','rb'))
weather_vars = [i for i in weather]
n_weather = len(weather_vars)
max_wv = dict()
min_wv = dict()
for wv in weather_vars:
    max_wv[wv] = np.max(weather[wv])
    min_wv[wv] = np.min(weather[wv])
poi = pickle.load(open('../../data/poi_mat.pkl','rb'))
poi = poi/np.max(poi)
seqs = [[[] for i in range(n_cols)] for i in range(n_rows)]
for i in range(n_rows):
    for j in range(n_cols):
        if dc_map[i,j]:
            for k in range(int(train_frac*len(crime)), len(crime)):
                padded_crime_maps = [np.pad(crime[k-z],1,mode='constant') for z in range(1,21)]
                x_data_crime = np.array( [np.concatenate( [np.ndarray.flatten(pcm[i:i+3,j:j+3]) for pcm in padded_crime_maps]  ) ])
                x_data_weather = np.array([[(weather[wv][k]-min_wv[wv])/max_wv[wv] for wv in weather_vars]])
                x_data_poi = np.array([poi[i,j]])
                seqs[i][j].append(model.predict([x_data_crime, x_data_weather, x_data_poi])[0][0])
                
pickle.dump(seqs, open('seqs1.pkl','wb'))
    