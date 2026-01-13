
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import shutil
import yaml
import math
import sys

import geopandas as gpd
import contextily as cx
from shapely.geometry import Point

#read arguments
if len(sys.argv) > 1:
    filter_categories = [arg.strip() for arg in sys.argv[1:]]
    print(f"Filtering for categories: {filter_categories}")
else:
    filter_categories = None
    print("No category filter specified – showing all flights.")

sourceCodeDir = os.path.dirname(os.path.abspath(__file__)) #get the directory of the source code
parameterFile = os.path.join(sourceCodeDir, "parameters.yaml") #build the absolute file path of the expected parameter file

#Load parameters
with open(parameterFile, "r") as file: #load parameters from file, contains either custom values or the copied default values
    allParams = yaml.safe_load(file)

databaseParameters = allParams["databaseParameters"] #get the DB parameters
airportParameters = allParams["airportParameters"] #get airport parameters

#geometry parameters
kmPerDegreeLat = 111
kmPerDegreeLon = 111 * abs(math.cos(math.radians(airportParameters["AIRPORT_LATITUDE"])))

#connect to the database from the in the parameter file specified directory
conn = sqlite3.connect(databaseParameters["DATABASE_PATH"])

#load relevent flights
if filter_categories:
    placeholder = ",".join("?" for _ in filter_categories)
    query = f"SELECT DISTINCT flightId FROM track_points WHERE category IN ({placeholder})"
    arrival_flight_ids = pd.read_sql_query(query, conn, params=filter_categories)
else:
    arrival_flight_ids = pd.read_sql_query("SELECT DISTINCT flightId FROM track_points", conn)

#define colors
state_colors = {
    "airborne": "blue",
    "onGround": "green",
    "transitionAirGrnd": "purple",
    "landing": "orange"
}

#get data from database
for i, flight_id in enumerate(arrival_flight_ids["flightId"]):
    print(f"Showing flight {i+1}/{len(arrival_flight_ids)}: flightId = {flight_id}")

    track_data = pd.read_sql_query(
        f"""
        SELECT id, time, altitude, lat, lon, state, category
        FROM track_points
        WHERE flightId = ?
        ORDER BY id
        """, conn, params=(flight_id,)
    )

    if track_data.empty:
        continue
    
    #plot data
    gdf = gpd.GeoDataFrame(track_data, geometry=gpd.points_from_xy(track_data["lon"], track_data["lat"]), crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)

    def on_next(event):
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        mng = plt.get_current_fig_manager()
        mng.frame.Maximize(True)
    except Exception:
        pass

    for state, color in state_colors.items():
        mask = gdf["state"] == state
        if mask.any():
            gdf[mask].plot(ax=ax, label=state, color=color, edgecolor=color, markersize=10)

    airport_geom = Point(airportParameters["AIRPORT_LONGITUDE"], airportParameters["AIRPORT_LATITUDE"])
    airport_gdf = gpd.GeoDataFrame(geometry=[airport_geom], crs="EPSG:4326").to_crs(epsg=3857)
    airport_gdf.plot(ax=ax, color="black", marker="x", markersize=100, label="Airport")

    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
    ax.set_title(f"flightId {flight_id} – Category: {track_data['category'].iloc[0]}")
    ax.legend()

    button_next_ax = plt.axes([0.8, 0.01, 0.12, 0.05])
    button_next = Button(button_next_ax, "Next flight")
    button_next.on_clicked(on_next)

    plt.show()

conn.close()
