import argparse
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
import numpy as np

# Argumente parsen
parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for plotting")
parser.add_argument("--separateByRunway", nargs='*', type=float, help="Optional: list of runway directions in degrees (e.g. 10 280)")
args = parser.parse_args()
threshold = args.threshold
runways = args.separateByRunway

# CSV laden
df = pd.read_csv("modelLandingProbabilities.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%H:%M:%S", errors="coerce")
df = df.dropna(subset=["timestamp"])
df["seconds"] = df["timestamp"].dt.hour * 3600 + df["timestamp"].dt.minute * 60 + df["timestamp"].dt.second
df = df.sort_values(by=["flightId", "seconds"])

segments = []

for flightId, group in df.groupby("flightId"):
    group = group.sort_values("seconds")
    aboveThreshold = group[group["probability"] > threshold]

    if not aboveThreshold.empty:
        firstHitTime = aboveThreshold.iloc[0]["seconds"]
        segment = group[group["seconds"] >= firstHitTime - 10].copy()
    else:
        endTime = group["seconds"].max()
        segment = group[group["seconds"] >= endTime - 10].copy()

    # Zuweisung zu Runway (falls aktiviert)
    if runways:
        headingWindow = group[group["seconds"] >= group["seconds"].max() - 30]
        if headingWindow.empty or "heading" not in headingWindow:
            continue
        avgHeading = headingWindow["heading"].mean() % 360
        print(f"⟶ Flight {flightId}: avgHeading = {avgHeading:.1f}°")

        assigned = False
        for runwayLabel in runways:
            headingDeg = (runwayLabel * 10) % 360
            diff = min(abs(avgHeading - headingDeg), 360 - abs(avgHeading - headingDeg))
            if diff <= 30:
                segment["assignedRunway"] = f"{int(runwayLabel):02d}"
                segments.append(segment)
                assigned = True
                break
        if not assigned:
            segment["assignedRunway"] = "unassigned"
            segments.append(segment)


    else:
        segment["assignedRunway"] = "combined"
        segments.append(segment)

if not segments:
    print("Keine Daten gefunden.")
    exit()

# Alle Segmente zusammenfügen
filteredDf = pd.concat(segments)

# Geometrie erzeugen
geometry = [Point(xy) for xy in zip(filteredDf["lon"], filteredDf["lat"])]
gdf = gpd.GeoDataFrame(filteredDf, geometry=geometry, crs="EPSG:4326")
gdf = gdf.to_crs(epsg=3857)

# Farben & Transparenz
colors = gdf["probability"].apply(lambda p: "blue" if p < threshold else "red")
alpha = gdf["probability"].apply(lambda p: 0.25)

# Plot je Runway oder kombiniert
if runways:
    for runwayId, runwayGroup in gdf.groupby("assignedRunway"):
        fig, ax = plt.subplots(figsize=(10, 10))
        runwayGroup.plot(ax=ax, color=colors.loc[runwayGroup.index], alpha=alpha.loc[runwayGroup.index], markersize=50)
        ax.set_xlim(runwayGroup.geometry.x.min() - 500, runwayGroup.geometry.x.max() + 500)
        ax.set_ylim(runwayGroup.geometry.y.min() - 500, runwayGroup.geometry.y.max() + 500)
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        ax.set_title(f"Runway {runwayId}")
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
else:
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color=colors, alpha=alpha, markersize=50)
    ax.set_xlim(gdf.geometry.x.min() - 500, gdf.geometry.x.max() + 500)
    ax.set_ylim(gdf.geometry.y.min() - 500, gdf.geometry.y.max() + 500)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_title("Alle Flüge kombiniert")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
