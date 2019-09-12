import pickle
import numpy as np

n_rows = 5
n_cols = 5
parking_data = pickle.load(open(r'C:\Users\deept\PycharmProjects\UC_project\parking_violation.pkl', 'rb'))
longs = [i[-2] for i in parking_data]
lats = [i[-1] for i in parking_data]
min_lat, max_lat = (min(lats)), (max(lats))
min_long, max_long = min(longs), max(longs)
print(min_lat)
print(max_lat)
lat_spacing = (float(max_lat) - float(min_lat)) / n_rows
long_spacing = (float(max_long) - float(min_long)) / n_cols
print(lat_spacing)
print(long_spacing)
data = np.zeros((n_rows, n_cols))
data_path = (r'C:\Users\deept\PycharmProjects\UC_project\Points_of_Interest.csv')
with open(data_path) as f:
    f.readline()
    lines = f.readlines()
for l in lines:
    line = l.split(',')
    lat = float(line[1])
    long = float(line[0])
    print(lat)
    print(long)
    for i in range(n_rows):
        if min_lat + i * lat_spacing <= lat <= min_lat + (i + 1) * lat_spacing:
            lat_slot = n_rows - i - 1
    for i in range(n_cols):
        if min_long + i * long_spacing <= long <= min_long + (i + 1) * long_spacing:
            long_slot = i
    data[lat_slot, long_slot] += 1
pickle.dump(data, open(r'C:\Users\deept\PycharmProjects\UC_project\poi_pv_mat.pkl', 'wb'))
