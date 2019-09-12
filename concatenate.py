import pandas as pd
import glob, os

os.chdir(r"C:\Users\deept\PycharmProjects\UC_project")
results = pd.DataFrame([])

for counter, file in enumerate(glob.glob("PV*")):
    with open(file, 'r') as source:  # Create list of rows
       namedf = pd.read_csv(file)
       results = results.append(namedf)

results.to_csv(r'C:\Users\deept\PycharmProjects\UC_project\combinedfile.csv','wb')