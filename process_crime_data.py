import pytz, datetime
import pickle
import os

local = pytz.timezone ('US/Eastern')
dst_start_end_days = {2007:(11,4), 2008:(9,2), 2009:(8,1), 2010:(14,7), 2011:(13,6), 2012:(11,4),
                 2013:(10,3), 2014:(9,2), 2015:(8,1), 2016:(13,6), 2017:(12,5),
                 2018:(11,4)}

data = dict()
start_dtime = datetime.datetime(2015,7,1)
end_dtime = datetime.datetime(2018, 6, 30)

def convert_to_utc(dtime): 
    naive_dtime = datetime.datetime.strptime (dtime[:-5], "%Y-%m-%dT%H:%M:%S")
    if naive_dtime<start_dtime or naive_dtime>end_dtime: #not in the time period of interest
        return None
    # handle daylight savings
    is_dst = False
    year = naive_dtime.year
    month = naive_dtime.month
    day = naive_dtime.day
    hour = naive_dtime.hour
    if 3<month<11:
        is_dst = True
    elif month==3:
        if day>dst_start_end_days[year][0]:
            is_dst = True
        elif day==dst_start_end_days[year][0] and hour>=2:
            is_dst = True
    elif month == 11:
        if day<dst_start_end_days[year][1]:
            is_dst = True
        elif day==dst_start_end_days[year][1] and hour<=2:
            is_dst = True
    local_dtime = local.localize(naive_dtime, is_dst=is_dst)
    utc_dtime = local_dtime.astimezone(pytz.utc) 
    return utc_dtime

data_path = '../../data/crime opendatadc/'
file_paths = os.listdir(data_path)
for file_path in file_paths:
    print (file_path)
    with open(data_path+file_path) as f:
        f.readline() #read headers
        lines = f.readlines()
        for line in lines:
            line_split = line.split(',')
            offense = line_split[6]
            lat = float(line_split[18])
            long = float(line_split[19])
            start_time = line_split[21]
            if start_time=='':
                report_time = line_split[3]
                dtime = convert_to_utc(report_time)
            else:
                dtime = convert_to_utc(start_time)
            if dtime!=None:
                data[(dtime.year, dtime.month, dtime.day, dtime.hour, lat, long)] = offense
                
pickle.dump(data, open('../../data/crime.pkl','wb'))               
