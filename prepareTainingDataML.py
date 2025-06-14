import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm

# === Parameter ===
dbPath = f"C:/Users/Kai/OneDrive - bwedu/Uni/SS 25/09_EffizienzProgrammieren/02_Database/02_Database/flightData_Labeled_replaced.db"
sequenceLength = 60        # Sekunden
minPointsRequired = 50     # Mindestanzahl, um ein Fenster zu verwenden
features = ["altitude", "groundSpeed", "climbRate", "track", "turnRate"]

# === Verbindung zur Datenbank ===
conn = sqlite3.connect(dbPath)

# Lade relevante Spalten
query = """
SELECT flightId, time, lat, lon, altitude, groundSpeed, climbRate, track, turnRate, state
FROM track_points
WHERE altitude IS NOT NULL AND groundSpeed IS NOT NULL AND climbRate IS NOT NULL
ORDER BY flightId, time;
"""
df = pd.read_sql_query(query, conn)

# Zeit in Sekunden umrechnen
def time_to_seconds(t):
    h, m, s = map(float, t.split(':'))
    return int(h * 3600 + m * 60 + s)

df["timeSec"] = df["time"].apply(time_to_seconds)

# Reduziere auf relevante Spalten
df = df[["flightId", "timeSec"] + features + ["state"]]

# === Verarbeitung ===
X_out = []
y_out = []

# Gruppiere nach Flug
for flightId, flight in tqdm(df.groupby("flightId"), desc="Verarbeite Flüge"):
    flight = flight.sort_values("timeSec").reset_index(drop=True)
    timeArray = flight["timeSec"].values

    for startIdx in range(len(flight)):
        startTime = timeArray[startIdx]
        endTime = startTime + sequenceLength

        window = flight[(flight["timeSec"] >= startTime) & (flight["timeSec"] < endTime)]
        n = len(window)

        if n >= minPointsRequired:
            data = window[features].values

            if n < sequenceLength:
                # Padding mit letztem Wert
                pad = np.tile(data[-1], (sequenceLength - n, 1))
                data = np.vstack([data, pad])
            elif n > sequenceLength:
                # Gleichmäßiges Sampling auf 60 Punkte
                idxs = np.linspace(0, n - 1, sequenceLength).astype(int)
                data = data[idxs]

            label = 1 if window.iloc[-1]["state"] == "landing" else 0

            X_out.append(data)
            y_out.append(label)

# === Speicherung ===
X = np.array(X_out)
y = np.array(y_out)

print(f"Fertige Sequenzen: {X.shape}, Labels: {y.shape}")
np.save("X.npy", X)
np.save("y.npy", y)
print("Daten gespeichert als 'X.npy' und 'y.npy'")
