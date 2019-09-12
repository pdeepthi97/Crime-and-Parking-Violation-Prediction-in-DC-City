from datetime import date, timedelta
import pickle
import matplotlib.pyplot as plt

weather_data = pickle.load(open('../../data/weather.pkl','rb')) 

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
       

def get_weather_variables():
    global weather_sequences, missing
    missing=[]
    num_sub_slots = 1096*4 #divide each day into 4 parts
    weather_sequences = dict()
    weather_sequences_unmerged = dict()
    all_weather_variables = list(weather_data.values())[0].keys()
    for variable in all_weather_variables:
        weather_sequences_unmerged[variable] = [[] for i in range(num_sub_slots)]
        weather_sequences[variable] = [0 for i in range(num_sub_slots)]
    for record in weather_data:
        year = record[0]
        month = record[1]
        day = record[2]
        hour = record[3]
        slot = date_to_index[(year, month, day)]*4 
        if 0<=hour<=6:
            for variable in weather_data[record]:
                weather_sequences_unmerged[variable][slot].append(weather_data[record][variable])
        if 6<=hour<=12:
            for variable in weather_data[record]:
                weather_sequences_unmerged[variable][slot+1].append(weather_data[record][variable])     
        if 12<=hour<=18:
            for variable in weather_data[record]:
                weather_sequences_unmerged[variable][slot+2].append(weather_data[record][variable])
        if 18<=hour<=23:
            for variable in weather_data[record]:
                weather_sequences_unmerged[variable][slot+3].append(weather_data[record][variable])
        if hour==0: #put in last sub slot in previous day's slot
            for variable in weather_data[record]:
                weather_sequences_unmerged[variable][slot-1].append(weather_data[record][variable])
    for variable in all_weather_variables:
        for i in range(num_sub_slots):
            unmerged_sequence = weather_sequences_unmerged[variable][i]
            if not unmerged_sequence: #if it's empty
                if weather_sequences_unmerged[variable][i-1] and weather_sequences_unmerged[variable][i+1]:
                    weather_sequences[variable][i] = (sum(weather_sequences_unmerged[variable][i-1])/len(weather_sequences_unmerged[variable][i-1]) + sum(weather_sequences_unmerged[variable][i+1])/len(weather_sequences_unmerged[variable][i+1])) / 2 
                    missing.append(0)
                else:
                    weather_sequences[variable][i] = -1000
                    missing.append(1)
            else:
                weather_sequences[variable][i] = sum(unmerged_sequence)/len(unmerged_sequence)
                missing.append(0)

    
get_date_to_index()
get_weather_variables()
pickle.dump(weather_sequences, open('../../data/weather_sequences.pkl','wb'))
