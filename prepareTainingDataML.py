import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import yaml

sourceCodeDir = os.path.dirname(os.path.abspath(__file__)) #get the directory of the source code
parameterFile = os.path.join(sourceCodeDir, "parameters.yaml") #build the absolute file path of the expected parameter file

#Load parameters
with open(parameterFile, "r") as file: #load parameters from file, contains either custom values or the copied default values
    allParams = yaml.safe_load(file)

databaseParameters = allParams["databaseParameters"] #get the DB parameters
machineLearningParameters = allParams["machineLearningParameters"] #get the ML parameters

#parameters
sequenceLength = machineLearningParameters["SQEUENCE_LENGTH"] #get sequence length for each time window
minPointsRequired = machineLearningParameters["MIN_NUM_POINTS_SEQUENCE"] #get the minimum number of points per time window

features = ["altitude", "groundSpeed", "climbRate", "track", "turnRate"] #features used for the model

#connect to the databse
conn = sqlite3.connect(databaseParameters["DATABASE_PATH"])

#load all relevant columns
query = """
SELECT flightId, time, lat, lon, altitude, groundSpeed, climbRate, track, turnRate, state
FROM track_points
WHERE altitude IS NOT NULL AND groundSpeed IS NOT NULL AND climbRate IS NOT NULL
ORDER BY flightId, time;
"""

data = pd.read_sql_query(query, conn) #read relavant columns

#function to convert HH:MM:SS time into seconds past midnight
def time_to_seconds(t):
    h, m, s = map(float, t.split(':'))
    return int(h * 3600 + m * 60 + s)

data["timeSec"] = data["time"].apply(time_to_seconds) #convert the time from HH:MM:SS to seconds past midnight

#reduce data to relevant columns
data = data[["flightId", "timeSec"] + features + ["state"]]

#data processing
X_out = []
y_out = []

#group data after each flight
for flightId, flight in tqdm(data.groupby("flightId"), desc="Processing flights"):
    flight = flight.sort_values("timeSec").reset_index(drop=True) #sort the data from this flight after the time column
    timeArray = flight["timeSec"].values #get an array containing the time column

    for startIdx in range(len(flight)): #itterate over the time array and find time windows
        startTime = timeArray[startIdx] #get start time
        endTime = startTime + sequenceLength #compute end time

        window = flight[(flight["timeSec"] >= startTime) & (flight["timeSec"] < endTime)] #get points inside this time window
        n = len(window) #get number of points in time window

        if n >= minPointsRequired: 
            data = window[features].values #get the data when enough points are available

            if n < sequenceLength:
                #pad the data with the last data point, if not enough points are available
                pad = np.tile(data[-1], (sequenceLength - n, 1))
                data = np.vstack([data, pad])
            elif n > sequenceLength:
                #downsample data
                idxs = np.linspace(0, n - 1, sequenceLength).astype(int)
                data = data[idxs]

            label = 1 if window.iloc[-1]["state"] == "landing" else 0 #label the data point as 1 if the state is landing

            X_out.append(data) #append data set
            y_out.append(label) #append result array

#return data
X = np.array(X_out) #convert to numpy array
y = np.array(y_out) #convert to numpy array

print(f"Finished sequences: {X.shape}, labels: {y.shape}")

#get directory of the database
outputDir = os.path.dirname(os.path.abspath(databaseParameters["DATABASE_PATH"]))

#save the output in the same directory as the database
np.save(os.path.join(outputDir, "X.npy"), X)
np.save(os.path.join(outputDir, "y.npy"), y)

print(f"Data saved in:\n{os.path.join(outputDir, 'X.npy')}\n{os.path.join(outputDir, 'y.npy')}")