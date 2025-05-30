from ognClient import OgnClient
from dataPlotter import plotAltSpeedAndStates
import pickle

client = OgnClient()

# Beispiel für das Laden eines pkl-Files
file1 = 'DD9B60_2025-04-27.pkl'
file2 = 'DDA286_2025-04-27.pkl'
with open(file2, 'rb') as f:
    flightData = pickle.load(f)  # erwartet z.B. ein Dict mit aircraftId als Keys

for data in flightData:
    client.processMessageDict(data)
    client.removeOldTracks()

aircraftId, aircraftData = next(iter(client.aircraftTracks.items()))
track = aircraftData["track"]

#plotAltSpeedAndStates(track)
