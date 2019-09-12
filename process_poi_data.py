import pickle
import numpy as np

n_rows = 5
n_cols = 5
crime_data = pickle.load(open('../../data/crime.pkl','rb'))
lats = [i[-2] for i in crime_data]
longs = [i[-1] for i in crime_data]
min_lat, max_lat = min(lats), max(lats)
min_long, max_long = min(longs), max(longs)
lat_spacing = (max_lat-min_lat)/n_rows
long_spacing = (max_long-min_long)/n_cols

data = np.zeros((n_rows,n_cols))
data_path = '../../data/poi opendatadc/Points_of_Interest.csv'
with open(data_path) as f:
    f.readline()
    lines = f.readlines()
for l in lines:
    line = l.split(',')
    lat = float(line[1])
    long = float(line[0])
    for i in range(n_rows):
        if min_lat+i*lat_spacing<=lat<=min_lat+(i+1)*lat_spacing:
            lat_slot = n_rows-i-1    
    for i in range(n_cols):
        if min_long+i*long_spacing<=long<=min_long+(i+1)*long_spacing:
            long_slot = i
    data[lat_slot,long_slot] += 1
pickle.dump(data, open('../../data/poi_mat.pkl','wb'))
    