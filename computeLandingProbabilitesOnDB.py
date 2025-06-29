import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from datetime import timedelta
import yaml
import os

#load config
sourceCodeDir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(sourceCodeDir, "parameters.yaml"), "r") as f:
    allParams = yaml.safe_load(f)

dbPath = allParams["databaseParameters"]["DATABASE_PATH"]
modelPath = allParams["machineLearningParameters"]["MODEL_PATH"]
print(modelPath)

sequenceLength = allParams["machineLearningParameters"]["SEQUENCE_LENGTH"]
sequenceTimeWindow = allParams["machineLearningParameters"]["SEQUENCE_TIME_WINDOW"]
minPointsRequired = allParams["machineLearningParameters"]["MIN_NUM_POINTS_SEQUENCE"]
features = allParams["machineLearningParameters"]["FEATURES"]

print("Loading model...")
model = tf.keras.models.load_model(modelPath)

print("Connecting to database...")
conn = sqlite3.connect(dbPath)

print("Loading arrival track points from database...")
query = """
SELECT flightId, time, lat, lon, altitude, groundSpeed, climbRate, track, turnRate, state
FROM track_points
WHERE category = 'arrival'
  AND altitude IS NOT NULL
  AND groundSpeed IS NOT NULL
  AND climbRate IS NOT NULL
ORDER BY flightId, time;
"""
df_all = pd.read_sql_query(query, conn)
conn.close()

if df_all.empty:
    print("No data found.")
    exit()

results = []

#convert time to datetime for correct time delta calculation
df_all["timestamp"] = pd.to_datetime(df_all["time"], format="%H:%M:%S.%f", errors="coerce")
df_all = df_all.dropna(subset=["timestamp"])  # remove malformed timestamps

print("Running inference flight by flight...")
for flightId, df in tqdm(df_all.groupby("flightId"), desc="Flights"):
    df = df.sort_values("timestamp").reset_index(drop=True)

    for i in tqdm(range(len(df)), desc=f"Flight {flightId}", leave=False):

        currentTime = df["timestamp"].iloc[i]
        windowStart = currentTime - timedelta(seconds=sequenceTimeWindow)
        window = df[(df["timestamp"] >= windowStart) & (df["timestamp"] <= currentTime)].copy()


        if len(window) < minPointsRequired:
            continue

        startTime = window["timestamp"].iloc[0]
        relTime = (window["timestamp"] - startTime).dt.total_seconds().values.reshape(-1, 1)
        featureMatrix = window[features].values
        sequence = np.hstack([relTime, featureMatrix])
        n = sequence.shape[0]

        if n < sequenceLength:
            pad = np.tile(sequence[-1], (sequenceLength - n, 1))
            sequence = np.vstack([sequence, pad])
        elif n > sequenceLength:
            idxs = np.linspace(0, n - 1, sequenceLength).astype(int)
            sequence = sequence[idxs]

        sequence = np.expand_dims(sequence, axis=0)  # shape: (1, sequenceLength, 6)      
        prob = model.predict(sequence, verbose=0)[0][0]

        lat = df["lat"].iloc[i]
        lon = df["lon"].iloc[i]
        heading = df["track"].iloc[i]

        results.append({
            "flightId": flightId,
            "timestamp": currentTime.time().strftime("%H:%M:%S"),
            "lat": lat,
            "lon": lon,
            "probability": prob,
            "heading": heading,
        })

#store as data frame
resultsDf = pd.DataFrame(results)
modelDir = os.path.dirname(modelPath)
csvPath = os.path.join(modelDir, "modelLandingProbabilities.csv")
resultsDf.to_csv(csvPath, index=False)
print(f"Results stored in {csvPath}")
