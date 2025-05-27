from ognClient import OgnClient
from createTimeSeriesPlot import plotFlightProfileFromTrack
import pickle

client = OgnClient()

# Beispiel für das Laden eines pkl-Files
with open('DD9B60_2025-04-27.pkl', 'rb') as f:
    flightData = pickle.load(f)  # erwartet z.B. ein Dict mit aircraftId als Keys

for data in flightData:
    client.processMessageDict(data)   

plotFlightProfileFromTrack(client.aircraftTracks['DD9B60']['track'])