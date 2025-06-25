from ognClient import OgnClient
from dataPlotter import plotAltSpeedAndStates
import pickle
import time

client = OgnClient()

# Beispiel für das Laden eines pkl-Files
file1 = 'DD9B60_2025-04-27.pkl'
file2 = 'DDA286_2025-04-27.pkl'
file3 = 'combinedData.pkl'
with open(file3, 'rb') as f:
    flightData = pickle.load(f)  # erwartet z.B. ein Dict mit aircraftId als Keys

for data in flightData:
    print(data)
    time.sleep(0.1)
    client.processMessageDict(data)
    client.removeOldTracks()
    client.airborneDataWriteDetection()

aircraftId, aircraftData = next(iter(client.aircraftTracks.items()))
track = aircraftData["track"]

#plotAltSpeedAndStates(track)
