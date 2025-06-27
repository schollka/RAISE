from ognClient import OgnClient
from dataPlotter import plotAltSpeedAndStates
import pickle
import time


try:
    client = OgnClient()
    client.startServer()

    # Beispiel für das Laden eines pkl-Files
    file1 = 'DD9B60_2025-04-27.pkl'
    file2 = 'DDA286_2025-04-27.pkl'
    file3 = 'combinedData.pkl'
    with open(file2, 'rb') as f:
        flightData = pickle.load(f)  # erwartet z.B. ein Dict mit aircraftId als Keys

    for data in flightData:
        #print(data)
        #time.sleep(0.1)
        client.processMessageDict(data)
        client.removeOldTracks()
        client.monitorSignalReception()
        client.airborneDataWriteDetection()
except KeyboardInterrupt:
    print("KeyboardInterrupt")

finally:
    client.shutdown()


#aircraftId, aircraftData = next(iter(client.aircraftTracks.items()))
#track = aircraftData["track"]

#plotAltSpeedAndStates(track)

#Strg + K  dann  Strg + C
#Strg + K  dann  Strg + U



# import os
# baseDirectory = r"C:\Users\Kai\OneDrive - bwedu\Uni\SS 25\09_EffizienzProgrammieren\02_Database\01_DecodeLogs"

# # Step 1: Collect all matching .pkl files
# pklFiles = []
# for root, dirs, files in os.walk(baseDirectory):
#     for fileName in files:
#         if fileName.endswith('.pkl') and fileName != 'combinedData.pkl':
#             filePath = os.path.join(root, fileName)
#             pklFiles.append(filePath)

# # Step 2: Iterate with index

# totalFiles = len(pklFiles)
# for i, filePath in enumerate(pklFiles, start=1):
#     try:
#         with open(filePath, 'rb') as f:
#             flightData = pickle.load(f)
#             print(f'Loaded file {i} out of {totalFiles}: {filePath}')
#             client = OgnClient()

#             for data in flightData:
#                 client.processMessageDict(data)
#                 client.removeOldTracks()
#                 client.monitorSignalReception()
#                 client.airborneDataWriteDetection()

#             client.shutdown()
            

#     except Exception as e:
#         print(f'Failed to load file {i} out of {totalFiles}: {filePath} – {e}')

