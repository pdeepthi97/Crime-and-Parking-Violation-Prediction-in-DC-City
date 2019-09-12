import csv
import pytz, datetime
import re
local = pytz.timezone ("US/Eastern")
with open('Parking_Violations_Issued_in_April_2017.csv', 'r') as source: # Create list of rows
    reader = csv.reader(source)
    next(reader, None)
    rd = list(reader)

with open('PV_April_2017.csv', 'w', newline='') as result:
    wtr = csv.writer(result)
    wtr.writerow(('Object_ID', 'Violation_Code', 'Violation_Description', 'RP_Plate_State', 'Address_ID', 'Street_seg_ID', 'lat','long','Ticket_Date_time',))
    for r in rd:
        t_date, temp = r[18].split('T')
        t_time, temp2 = temp.split('.')
        t_date_time = t_date + t_time
        naive = datetime.datetime.strptime(t_date_time, "%Y-%m-%d%H:%M:%S")
        local_dt = local.localize(naive, is_dst=None)
        utc_dt = local_dt.astimezone(pytz.utc)
        wtr.writerow((r[2], r[9], r[10], r[12], r[14], r[15], r[0], r[1],utc_dt))