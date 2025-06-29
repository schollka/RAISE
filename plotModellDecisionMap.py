import argparse
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point

# Argumente parsen
parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for plotting")
args = parser.parse_args()
threshold = args.threshold

# CSV laden
df = pd.read_csv("modelLandingProbabilities.csv")

# Zeitstempel parsen: "HH:MM:SS" → datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%H:%M:%S", errors="coerce")

# Ungültige Zeitangaben entfernen
df = df.dropna(subset=["timestamp"])

# Zeit in Sekunden ab Mitternacht umwandeln (für Subtraktion)
df["seconds"] = df["timestamp"].dt.hour * 3600 + df["timestamp"].dt.minute * 60 + df["timestamp"].dt.second

# Nach Flug und Zeit sortieren
df = df.sort_values(by=["flightId", "seconds"])

# Gefilterte Abschnitte sammeln
segments = []

for flightId, group in df.groupby("flightId"):
    aboveThreshold = group[group["probability"] > threshold]
    if not aboveThreshold.empty:
        firstHit = aboveThreshold.iloc[0]["seconds"]
        segment = group[group["seconds"] >= firstHit - 10]
        segments.append(segment)

# Falls keine Segmente gefunden wurden
if not segments:
    print("Keine Datenpunkte über dem Threshold gefunden.")
    exit()

filteredDf = pd.concat(segments)

# Geometrie erzeugen
geometry = [Point(xy) for xy in zip(filteredDf["lon"], filteredDf["lat"])]
gdf = gpd.GeoDataFrame(filteredDf, geometry=geometry, crs="EPSG:4326")
gdf = gdf.to_crs(epsg=3857)

# Farben & Transparenz
colors = gdf["probability"].apply(lambda p: "blue" if p < threshold else "red")
alpha = gdf["probability"].apply(lambda p: 0.25 if p < threshold else 0.25)

# Plot
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, color=colors, alpha=alpha, markersize=50)

# Zoom auf Punktwolke
ax.set_xlim(gdf.geometry.x.min() - 500, gdf.geometry.x.max() + 500)
ax.set_ylim(gdf.geometry.y.min() - 500, gdf.geometry.y.max() + 500)

# Basemap hinzufügen
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_axis_off()
plt.tight_layout()
plt.show()
