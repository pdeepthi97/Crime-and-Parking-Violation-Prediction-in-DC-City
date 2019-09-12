import pickle
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt

crime_sequence = pickle.load(open('../../data/crime_sequences.pkl','rb'))
plt.figure()
plt.plot(crime_sequence)
plt.xlabel('time')
plt.ylabel('crime count for whole city')
crime_sequence = pd.Series(crime_sequence)
plt.figure()
autocorrelation_plot(crime_sequence)
plt.title('crime sequence autocorrelation')

weather_sequences = pickle.load(open('../../data/weather_sequences.pkl','rb'))
for var in weather_sequences:
    temp_sequence = pd.Series(weather_sequences[var])
    plt.figure()
    plt.scatter(temp_sequence, crime_sequence, s = 1)
    plt.xlabel(var)
    plt.ylabel('crime count')


