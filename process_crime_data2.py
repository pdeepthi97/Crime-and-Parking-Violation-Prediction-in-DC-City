import pickle
from datetime import date, timedelta

crime_data = pickle.load(open('../../data/crime.pkl','rb'))

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
    crime_sequences = [0 for i in range(num_sub_slots)]
    for record in crime_data:
        year = record[0]
        month = record[1]
        day = record[2]
        hour = record[3]
        sub_slot = date_to_index[(year, month, day)]*4 + int(hour/6)
        crime_sequences[sub_slot] += 1

get_date_to_index()
get_crime_sequences()
pickle.dump(crime_sequences, open('../../data/crime_int_sequence.pkl','wb'))
