from ognClient import OgnClient
import pickle
import os

BASE_DIR = r"C:\Users\Kai\OneDrive - bwedu\Uni\SS 25\09_EffizienzProgrammieren\02_Database\01_DecodeLogs"

# Alle .pkl-Dateien rekursiv sammeln
pklFiles = []
for root, dirs, files in os.walk(BASE_DIR):
    for fileName in files:
        if fileName.endswith('.pkl'):
            filePath = os.path.join(root, fileName)
            pklFiles.append(filePath)

# Verarbeiten mit Fortschrittsanzeige
totalFiles = len(pklFiles)
for i, filePath in enumerate(pklFiles, start=1):
    print(f"[{i}/{totalFiles}] Verarbeite: {filePath}")
    try:
        with open(filePath, 'rb') as file:
            flightData = pickle.load(file)
            client = OgnClient()
            for data in flightData:
                client.processMessageDict(data)
                client.removeOldTracks()
                client.airborneDataWriteDetection()
    except Exception as e:
        print(f"Fehler beim Laden von {filePath}: {e}")