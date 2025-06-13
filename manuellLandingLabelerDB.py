
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Button
import os
import shutil
import yaml
import math

import geopandas as gpd
import contextily as cx
from shapely.geometry import Point

# Maximize plot window (only works with some backends)
plt.switch_backend("tkagg")
mng = plt.get_current_fig_manager()
try:
    mng.window.state("zoomed")
except AttributeError:
    try:
        mng.frame.Maximize(True)
    except AttributeError:
        pass  # Not supported in this environment

sourceCodeDir = os.path.dirname(os.path.abspath(__file__))
parameterFile = os.path.join(sourceCodeDir, "parameters.yaml")
defaultParameters = os.path.join(sourceCodeDir, "defaultParameters.yaml")

if not os.path.exists(parameterFile):
    shutil.copy(defaultParameters, parameterFile)

with open(parameterFile, "r") as file:
    allParams = yaml.safe_load(file)

databaseParameters = allParams["databaseParameters"]
airportParameters = allParams["airportParameters"]
stateEstimationParameters = allParams["stateEstimationParameters"]

kmPerDegreeLat = 111
kmPerDegreeLon = 111 * abs(math.cos(math.radians(airportParameters["AIRPORT_LATITUDE"])))

conn = sqlite3.connect(databaseParameters["DATABASE_PATH"])
cursor = conn.cursor()

arrival_flight_ids = pd.read_sql_query(
    "SELECT DISTINCT flightId FROM track_points WHERE category = 'arrival';", conn
)

for i, flight_id in enumerate(arrival_flight_ids["flightId"]):
    print(f"Flight {i+1}/{len(arrival_flight_ids)}: flightId = {flight_id}")

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

    track_data["time"] = pd.to_timedelta(track_data["time"])
    lat = track_data["lat"].values
    lon = track_data["lon"].values
    states = track_data["state"].values
    ids = track_data["id"].values

    selected_index = {"value": None}
    skip_flight = {"value": False}

    # GeoDataFrame erstellen
    gdf = gpd.GeoDataFrame(track_data, geometry=gpd.points_from_xy(track_data["lon"], track_data["lat"]), crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)

    def onclick(event: MouseEvent):
        if event.inaxes != ax:
            return
        x_click, y_click = event.xdata, event.ydata
        distances = ((gdf.geometry.x - x_click)**2 + (gdf.geometry.y - y_click)**2)
        idx = distances.argmin()
        selected_index["value"] = idx

        ax.clear()
        gdf.plot(ax=ax, label="Flight path", alpha=0.6, markersize=10)
        gdf.iloc[idx:].plot(ax=ax, color="orange", edgecolor="orange", label="Landing", markersize=10)
        gdf.iloc[[idx]].plot(ax=ax, color="red", edgecolor="red", label="Landing start", markersize=30)
        airport_geom = Point(airportParameters["AIRPORT_LONGITUDE"], airportParameters["AIRPORT_LATITUDE"])
        airport_gdf = gpd.GeoDataFrame(geometry=[airport_geom], crs="EPSG:4326").to_crs(epsg=3857)
        airport_gdf.plot(ax=ax, color="black", marker="x", markersize=100, label="Airport")
        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
        ax.set_title(f"flightId {flight_id} – Click start of landing")
        ax.legend()
        fig.canvas.draw()

    def on_done(event):
        plt.close()

    def on_skip(event):
        skip_flight["value"] = True
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        mng = plt.get_current_fig_manager()
        mng.window.state("zoomed")
    except Exception:
        pass

    gdf.plot(ax=ax, label="Flight path", alpha=0.6, markersize=10)
    airport_geom = Point(airportParameters["AIRPORT_LONGITUDE"], airportParameters["AIRPORT_LATITUDE"])
    airport_gdf = gpd.GeoDataFrame(geometry=[airport_geom], crs="EPSG:4326").to_crs(epsg=3857)
    airport_gdf.plot(ax=ax, color="black", marker="x", markersize=100, label="Airport")
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
    ax.set_title(f"flightId {flight_id} – Click start of landing")
    ax.legend()

    button_done_ax = plt.axes([0.75, 0.01, 0.1, 0.05])
    button_done = Button(button_done_ax, "Done")
    button_done.on_clicked(on_done)

    button_skip_ax = plt.axes([0.6, 0.01, 0.1, 0.05])
    button_skip = Button(button_skip_ax, "Skip flight")
    button_skip.on_clicked(on_skip)

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if skip_flight["value"]:
        print("→ Flight skipped.")
        continue

    if selected_index["value"] is not None:
        start_idx = selected_index["value"]
        changed_count = 0
        for idx in range(start_idx, len(track_data)):
            if states[idx] == "airborne":
                track_point_id = int(ids[idx])
                cursor.execute(
                    "UPDATE track_points SET state = 'landing' WHERE id = ?;",
                    (track_point_id,)
                )
                changed_count += 1

        conn.commit()
        print(f"→ Updated: {changed_count} points set to 'landing'.")

conn.close()
