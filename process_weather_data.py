import pickle

data = dict()
weather_data_path = "../../data/weather/weather.csv"
with open(weather_data_path) as f:
    f.readline() #read headers
    lines = f.readlines()
    for line in lines:
        line_split = line.split(',')
        temp = (float(line_split[7]) + float(line_split[8]))/2 - 273.15 #in celcius
        pres = float(line_split[9])
        humd = float(line_split[12])
        wind_speed = float(line_split[13])
        cloud_percent = float(line_split[23])
        weather_main = line_split[25]
        clouds = int(weather_main=='Clouds')
        drizzle = int(weather_main=='Drizzle')
        fog = int(weather_main=='Fog')
        haze = int(weather_main=='Haze')
        mist = int(weather_main=='Mist')
        rain = int(weather_main=='Rain')
        smoke = int(weather_main=='Smoke')
        snow = int(weather_main=='Snow')
        squall = int(weather_main=='Squall')
        tstorm = int(weather_main=='Thunderstorm')
        date = line_split[1]
        year = int(date[:4])
        month = int(date[5:7])
        if (2015<year<2018) or (year==2015 and month>6) or (year==2018 and month<7):
            day = int(date[8:10])
            hour = int(date[11:13])
            attr_dict = {'temp':temp, 'pres':pres, 'humd':humd, 'wind_speed':wind_speed, 
                         'cloud_percent':cloud_percent, 'clouds':clouds, 'drizzle':drizzle,
                         'fog':fog, 'haze':haze, 'mist':mist, 'rain':rain, 'smoke':smoke, 
                         'snow':snow, 'squall':squall, 'tstorm':tstorm}
            data[(year, month, day, hour)] = attr_dict
        
pickle.dump(data, open('../../data/weather.pkl','wb'))
