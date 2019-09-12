import pickle
from datetime import date, timedelta
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

def get_date_to_index():
    global date_to_index
    date_to_index = dict()
    start_date = date(2015, 7, 1)
    delta = timedelta(days=1)
    date_to_index[(start_date.year, start_date.month, start_date.day)] = 0
    prev = start_date
    for i in range(1, 1096):
        prev = prev + delta
        date_to_index[(prev.year, prev.month, prev.day)] = i
        
def get_crime_sequences():
    global crime_sequences
    num_sub_slots = 1096*4
    crime_sequences = [np.zeros((n_rows, n_cols)) for i in range(num_sub_slots)]
    for record in crime_data:
        year = record[0]
        month = record[1]
        day = record[2]
        hour = record[3]
        lat = record[4]
        for i in range(n_rows):
            if min_lat+i*lat_spacing<=lat<=min_lat+(i+1)*lat_spacing:
                lat_slot = n_rows-i-1
        long = record[5]
        for i in range(n_cols):
            if min_long+i*long_spacing<=long<=min_long+(i+1)*long_spacing:
                long_slot = i
        sub_slot = date_to_index[(year, month, day)]*4 + int(hour/6)
        crime_sequences[sub_slot][lat_slot, long_slot] += 1

get_date_to_index()
get_crime_sequences()
pickle.dump(crime_sequences, open('../../data/crime_mat_sequence.pkl','wb'))
