
import pandas as pd
import pickle
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt

parking_violation = pickle.load(open(r'C:\Users\deept\PycharmProjects\UC_project\parking_violation.pkl','rb'))
parking_violation = pd.Series(parking_violation)
plt.figure()
autocorrelation_plot(parking_violation)
plt.title('Parking Violation autocorrelation')
weather_sequences = pickle.load(open(r'C:\Users\deept\PycharmProjects\UC_project\weather_sequences.pkl','rb'))
for var in weather_sequences:
    temp_sequence = pd.Series(weather_sequences[var])
    plt.figure()
    plt.scatter(temp_sequence, parking_violation, s = 1)
    plt.xlabel(var)
    plt.ylabel('Parking violation count')
plt.show()