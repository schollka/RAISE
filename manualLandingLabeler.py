'''
              __|__
       --------(_)--------       
              RAISE         
 Runway Approach Identification for Silent Entries
------------------------------------------------------
    Tracking • Prediction • Silent Entry Detection     
------------------------------------------------------
------------------------------------------------------
    Tracking • Prediction • Silent Entry Detection     
------------------------------------------------------

data labeling and visualization script
'''

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Button
import os
import yaml
import math

import geopandas as gpd
import contextily as cx
from shapely.geometry import Point

sourceCodeDir = os.path.dirname(os.path.abspath(__file__)) #get the directory of the source code
parameterFile = os.path.join(sourceCodeDir, "parameters.yaml") #build the absolute file path of the expected parameter file

#Load parameters
print("Loading parameters.")
with open(parameterFile, "r") as file:
    allParams = yaml.safe_load(file)

databaseParameters = allParams["databaseParameters"]
airportParameters = allParams["airportParameters"]
print("Parameters loaded.")

#geometry parameters
kmPerDegreeLat = 111
kmPerDegreeLon = 111 * abs(math.cos(math.radians(airportParameters["AIRPORT_LATITUDE"])))

#connect to the database from the in the parameter file specified directory
conn = sqlite3.connect(databaseParameters["DATABASE_PATH"])
cursor = conn.cursor()

#get all ID's for the flights with category arrival
print("Loading data from database.")
arrival_flight_ids = pd.read_sql_query(
    "SELECT DISTINCT flightId FROM track_points WHERE category = 'arrival';", conn
)
print("Data loaded.")

#define colors for each state
state_colors = {
    "airborne": "blue",
    "onGround": "green",
    "transitionAirGrnd": "purple",
    "landing": "orange"
}

#Go trough the database with the previosly found ID's
for i, flight_id in enumerate(arrival_flight_ids["flightId"]):
    print(f"Flight {i+1}/{len(arrival_flight_ids)}: flightId = {flight_id}")

    #get the track data from DB
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

    #extract positon and state data
    track_data["time"] = pd.to_timedelta(track_data["time"])
    lat = track_data["lat"].values
    lon = track_data["lon"].values
    states = track_data["state"].values
    ids = track_data["id"].values

    selected_index = {"value": None}
    skip_flight = {"value": False}
    remove_flight = {"value": False}

    def plot_track():
        ax.clear()
        updated_track_data = pd.read_sql_query(
            f"""
            SELECT id, time, altitude, lat, lon, state
            FROM track_points
            WHERE flightId = {flight_id}
            ORDER BY id
            """, conn
        )
        gdf_updated = gpd.GeoDataFrame(updated_track_data,
            geometry=gpd.points_from_xy(updated_track_data["lon"], updated_track_data["lat"]),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        for state, color in state_colors.items():
            mask = gdf_updated["state"] == state
            if mask.any():
                gdf_updated[mask].plot(ax=ax, label=state, color=color, edgecolor=color, markersize=10)

        airport_geom = Point(airportParameters["AIRPORT_LONGITUDE"], airportParameters["AIRPORT_LATITUDE"])
        airport_gdf = gpd.GeoDataFrame(geometry=[airport_geom], crs="EPSG:4326").to_crs(epsg=3857)
        airport_gdf.plot(ax=ax, color="black", marker="x", markersize=100, label="Airport")

        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"Database-flightId {flight_id} – click on the point that marks the beginning of the approach")
        ax.legend()
        fig.canvas.draw()

    gdf = gpd.GeoDataFrame(track_data, geometry=gpd.points_from_xy(track_data["lon"], track_data["lat"]), crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)

    #mouse click action
    def onclick(event: MouseEvent):
        if event.inaxes != ax:
            return
        x_click, y_click = event.xdata, event.ydata
        distances = ((gdf.geometry.x - x_click)**2 + (gdf.geometry.y - y_click)**2)
        idx = distances.argmin()
        selected_index["value"] = idx

        ax.clear()
        for state, color in state_colors.items():
            mask = gdf["state"] == state
            if mask.any():
                gdf[mask].plot(ax=ax, label=state, color=color, edgecolor=color, markersize=10)


        #colour landing phase
        gdf.iloc[idx:].plot(ax=ax, color="orange", edgecolor="orange", label="Landing (marked)", markersize=10)
        gdf.iloc[[idx]].plot(ax=ax, color="red", edgecolor="red", label="Landing start", markersize=30)

        airport_geom = Point(airportParameters["AIRPORT_LONGITUDE"], airportParameters["AIRPORT_LATITUDE"])
        airport_gdf = gpd.GeoDataFrame(geometry=[airport_geom], crs="EPSG:4326").to_crs(epsg=3857)
        airport_gdf.plot(ax=ax, color="black", marker="x", markersize=100, label="Airport")

        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"Database-flightId {flight_id} – click on the point that marks the beginning of the approach")
        ax.legend()
        fig.canvas.draw()

    def on_done(event):
        plt.close()

    def on_skip(event):
        skip_flight["value"] = True
        plt.close()

    def on_remove(event):
        remove_flight["value"] = True
        plt.close()

    def on_reset(event):
        cursor.execute(
            "UPDATE track_points SET state = 'airborne' WHERE flightId = ? AND state = 'landing';",
            (flight_id,)
        )
        conn.commit()
        print("→ Reset: all 'landing' points set to 'airborne'.")
        plot_track()

    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        mng = plt.get_current_fig_manager()
        mng.frame.Maximize(True)
    except Exception:
        pass

    #colour points based on their state
    for state, color in state_colors.items():
        mask = gdf["state"] == state
        if mask.any():
            gdf[mask].plot(ax=ax, label=state, color=color, edgecolor=color, markersize=10)

    #set an X at the airport reference point
    airport_geom = Point(airportParameters["AIRPORT_LONGITUDE"], airportParameters["AIRPORT_LATITUDE"])
    airport_gdf = gpd.GeoDataFrame(geometry=[airport_geom], crs="EPSG:4326").to_crs(epsg=3857)
    airport_gdf.plot(ax=ax, color="black", marker="x", markersize=100, label="Airport")

    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"Database-flightId {flight_id} – click on the point that marks the beginning of the approach")
    ax.legend()

    button_done_ax = plt.axes([0.75, 0.01, 0.1, 0.05])
    button_done = Button(button_done_ax, "Done")
    button_done.on_clicked(on_done)

    button_skip_ax = plt.axes([0.6, 0.01, 0.1, 0.05])
    button_skip = Button(button_skip_ax, "Skip flight")
    button_skip.on_clicked(on_skip)

    button_remove_ax = plt.axes([0.45, 0.01, 0.1, 0.05])
    button_remove = Button(button_remove_ax, "Remove flight")
    button_remove.on_clicked(on_remove)

    button_reset_ax = plt.axes([0.3, 0.01, 0.1, 0.05])
    button_reset = Button(button_reset_ax, "Reset landing")
    button_reset.on_clicked(on_reset)

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if skip_flight["value"]:
        print("→ Flight skipped.")
        continue

    if remove_flight["value"]:
        cursor.execute("DELETE FROM track_points WHERE flightId = ?;", (flight_id,))
        conn.commit()
        print("→ Flight removed from database.")
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

# Count remaining flights per category
conn = sqlite3.connect(databaseParameters["DATABASE_PATH"])
cursor = conn.cursor()
for category in ["arrival", "departure", "inFlight"]:
    result = cursor.execute(
        "SELECT COUNT(DISTINCT flightId) FROM track_points WHERE category = ?;", (category,)
    )
    count = result.fetchone()[0]
    print(f"Remaining flights with category '{category}': {count}")
conn.close()
