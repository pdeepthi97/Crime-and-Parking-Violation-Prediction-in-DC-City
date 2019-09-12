
import pickle
import csv

data = dict()
with open('combinedfile.csv', 'r') as source: # Create list of rows
    reader = csv.reader(source)
    next(reader, None)
    rd = list(reader)
    for r in rd:
      offense = r[3]
      t_year, t_month, temp1 = r[9].split('-')
      t_day,temp2 = temp1.split(' ')
      t_hour,t_min,t_sec,temp3= temp2.split(':')
      data[(t_year, t_month, t_day, t_hour, float(r[7]),float(r[8]))] = offense
pickle.dump(data, open(r'C:\Users\deept\PycharmProjects\UC_project\parking_violation.pkl','wb'))