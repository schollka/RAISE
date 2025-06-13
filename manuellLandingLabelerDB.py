
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Button
from matplotlib.patches import Circle
import os
import shutil
import yaml
import math

sourceCodeDir = os.path.dirname(os.path.abspath(__file__)) #get the directory of the source code
parameterFile = os.path.join(sourceCodeDir, "parameters.yaml") #build the absolute file path of the expected parameter file
defaultParameters = os.path.join(sourceCodeDir, "defaultParameters.yaml") #build the absolute file path of the default parameter file

# Check if parameterFile exists and copy default if nonexistent
if not os.path.exists(parameterFile):
    shutil.copy(defaultParameters, parameterFile) #copy default parameters
#Load parameters
with open(parameterFile, "r") as file: #load parameters from file, contains either custom values or the copied default values
    allParams = yaml.safe_load(file)
databaseParameters = allParams["databaseParameters"]
airportParameters = allParams["airportParameters"]
stateEstimationParameters = allParams["stateEstimationParameters"]

kmPerDegreeLat = 111  # Näherung
kmPerDegreeLon = 111 * abs(math.cos(math.radians(airportParameters["AIRPORT_LATITUDE"])))

# Verbindung zur SQLite-Datenbank
conn = sqlite3.connect(databaseParameters["DATABASE_PATH"])
cursor = conn.cursor()

# Lade alle Flüge mit category = 'arrival'
arrival_flight_ids = pd.read_sql_query(
    "SELECT DISTINCT flightId FROM track_points WHERE category = 'arrival';", conn
)

# Liste zur Speicherung aller Änderungen für Undo
undo_log = []

# Interaktive Auswahl und Datenbank-Update
for i, flight_id in enumerate(arrival_flight_ids["flightId"]):
    print(f"Flug {i+1}/{len(arrival_flight_ids)}: flightId = {flight_id}")

    # Lade alle Trackpunkte dieses Flugs
    track_data = pd.read_sql_query(
        f"""
        SELECT id, time, altitude, lat, lon, state
        FROM track_points
        WHERE flightId = {flight_id}
        ORDER BY id
        """, conn
    )

    if track_data.empty:
        continue

    # Zeit umwandeln
    track_data["time"] = pd.to_timedelta(track_data["time"])
    lat = track_data["lat"].values
    lon = track_data["lon"].values
    states = track_data["state"].values
    ids = track_data["id"].values

    selected_index = {"value": None}
    changed_entries = []

    def onclick(event: MouseEvent):
        if event.inaxes != ax:
            return
        dist = (lat - event.ydata)**2 + (lon - event.xdata)**2
        idx = dist.argmin()
        selected_index["value"] = idx

        ax.clear()
        ax.plot(lon, lat, label="Flugroute")
        ax.scatter(lon[idx:], lat[idx:], color="orange", label="Landing", s=20)
        ax.scatter(lon[idx], lat[idx], color="red", label="Start Landeanflug", zorder=5)
        ax.set_title(f"flightId {flight_id} – Klick auf Startpunkt des Landeanflugs")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()
        fig.canvas.draw()

    def on_done(event):
        plt.close()

    def on_undo(event):
        if undo_log:
            last_change = undo_log.pop()
            for entry in last_change:
                cursor.execute(
                    "UPDATE track_points SET state = ? WHERE id = ?;",
                    (entry["old_state"], entry["id"])
                )
            conn.commit()
            print(f"→ Änderungen für flightId {last_change[0]['flightId']} rückgängig gemacht.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lon, lat, label="Flugroute")
    ax.plot(airportParameters["AIRPORT_LONGITUDE"], airportParameters["AIRPORT_LATITUDE"], marker='x', color='black', markersize=12, mew=3, label='Airport')
    radiusDegLat = stateEstimationParameters["ON_GROUND_POSITION_RADIUS"] / 1000 / kmPerDegreeLat
    radiusDegLon = stateEstimationParameters["ON_GROUND_POSITION_RADIUS"] / 1000 / kmPerDegreeLon
    circle = Circle(
        (airportParameters["AIRPORT_LONGITUDE"], airportParameters["AIRPORT_LATITUDE"]),
        radius=radiusDegLon,
        edgecolor='black',
        facecolor='none',
        linewidth=1.5,
        linestyle='--',
        label=f'{int(stateEstimationParameters["ON_GROUND_POSITION_RADIUS"])} m Radius'
    )
    ax.add_patch(circle)
    ax.set_title(f"flightId {flight_id} – Klick auf Startpunkt des Landeanflugs")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()

    button_done_ax = plt.axes([0.75, 0.01, 0.1, 0.05])
    button_done = Button(button_done_ax, "Fertig")
    button_done.on_clicked(on_done)

    button_undo_ax = plt.axes([0.6, 0.01, 0.1, 0.05])
    button_undo = Button(button_undo_ax, "Undo")
    button_undo.on_clicked(on_undo)

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if selected_index["value"] is not None:
        start_idx = selected_index["value"]

        for idx in range(start_idx, len(track_data)):
            if states[idx] == "airborne":
                track_point_id = int(ids[idx])
                old_state = states[idx]
                cursor.execute(
                    "UPDATE track_points SET state = 'landing' WHERE id = ?;",
                    (track_point_id,)
                )
                changed_entries.append({
                    "id": track_point_id,
                    "old_state": old_state,
                    "flightId": int(flight_id)
                })

        conn.commit()
        if changed_entries:
            undo_log.append(changed_entries)
            print(f"→ Aktualisiert: {len(changed_entries)} Punkte auf 'landing' gesetzt (Undo möglich).")

# Verbindung schließen
conn.close()
