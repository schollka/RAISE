import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
import numpy as np

#colormap für probability
import matplotlib as mpl
#cmap = plt.cm.turbo                          #oder: plt.cm.viridis
cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

def plotPointsColoredByProbability(ax, gdfSlice, opacity=0.35, size=50):
    #kompakter Scatter statt GeoPandas-plot, damit Colorbar geht
    xs = gdfSlice.geometry.x.values
    ys = gdfSlice.geometry.y.values
    sc = ax.scatter(
        xs, ys,
        c=gdfSlice["probability"].values,
        cmap=cmap,
        norm=norm,
        alpha=opacity,
        s=size,
        linewidths=0
    )
    return sc


# Argumente parsen
parser = argparse.ArgumentParser()
parser.add_argument("csvPath", type=str, help="Path to the CSV file with landing probabilities")
parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for plotting")
parser.add_argument("--separateByRunway", nargs='*', type=float, help="Optional: list of runway directions in degrees (e.g. 10 280)")
args = parser.parse_args()
threshold = args.threshold
runways = args.separateByRunway
csvPath = args.csvPath

timeToFirstHit = 10
avgHeadingTime = 45
numberBins = 20
landingColor = "#7f0288"
otherColor = "#db9600"
opacity = 0.50

# CSV laden
df = pd.read_csv(csvPath)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%H:%M:%S", errors="coerce")
df = df.dropna(subset=["timestamp"])
df["seconds"] = df["timestamp"].dt.hour * 3600 + df["timestamp"].dt.minute * 60 + df["timestamp"].dt.second
df = df.sort_values(by=["flightId", "seconds"])

# Globales Histogramm über alle Wahrscheinlichkeiten
fig_hist_all, ax_hist_all = plt.subplots(figsize=(8, 4))
allProbs = df["probability"]
ax_hist_all.hist(
    allProbs,
    bins=numberBins,
    color="blue",
    edgecolor="black",
    weights=np.ones_like(allProbs) / len(allProbs)
)
ax_hist_all.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")
ax_hist_all.set_xlabel("Probability")
ax_hist_all.set_ylabel("Relative Frequency")
ax_hist_all.set_title("Global Probability Distribution (All Data Points)")
ax_hist_all.set_xticks(np.arange(0, 1.05, 0.05))
ax_hist_all.legend()
plt.tight_layout()
plt.show()


segments = []

for flightId, group in df.groupby("flightId"):
    group = group.sort_values("seconds")
    aboveThreshold = group[group["probability"] > threshold]

    if not aboveThreshold.empty:
        firstHitTime = aboveThreshold.iloc[0]["seconds"]
        segment = group[group["seconds"] >= firstHitTime - timeToFirstHit].copy()
    else:
        endTime = group["seconds"].max()
        segment = group[group["seconds"] >= endTime - timeToFirstHit].copy()

    # Zuweisung zu Runway (falls aktiviert)
    if runways:
        runways = [int(float(r)) for r in runways]
        headingWindow = group[group["seconds"] >= group["seconds"].max() - avgHeadingTime]
        if headingWindow.empty or "heading" not in headingWindow:
            continue
        avgHeading = headingWindow["heading"].mean() % 360
        print(f"Flight {flightId}: avgHeading = {avgHeading:.1f}°")

        assigned = False
        for runwayLabel in runways:
            headingDeg = (runwayLabel * 10) % 360
            diff = min(abs(avgHeading - headingDeg), 360 - abs(avgHeading - headingDeg))
            if diff <= 20:
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
colors = gdf["probability"].apply(lambda p: otherColor if p < threshold else landingColor)
alpha = gdf["probability"].apply(lambda p: opacity)

# Plot je Runway oder kombiniert
if runways:
    for runwayId, runwayGroup in gdf.groupby("assignedRunway"):
        # Kartenplot
        fig, ax = plt.subplots(figsize=(10, 10))
        runwayGroup.plot(ax=ax, color=colors.loc[runwayGroup.index], alpha=alpha.loc[runwayGroup.index], markersize=50)
        legendHandles = [
            mpatches.Patch(color=landingColor, label="landing"),
            mpatches.Patch(color=otherColor, label="other")
        ]
        ax.legend(handles=legendHandles, title="Classification", loc="upper right")
        ax.set_xlim(runwayGroup.geometry.x.min() - 500, runwayGroup.geometry.x.max() + 500)
        ax.set_ylim(runwayGroup.geometry.y.min() - 500, runwayGroup.geometry.y.max() + 500)
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        ax.set_title(f"Approach Paths for Runway {runwayId} with Classification Threshold {threshold}")
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

        #zusatzplot: runway-spezifisch mit kontinuierlicher Einfärbung
        fig_c, ax_c = plt.subplots(figsize=(10, 10))
        sc = plotPointsColoredByProbability(ax_c, runwayGroup, opacity=opacity, size=50)
        ax_c.set_xlim(runwayGroup.geometry.x.min() - 500, runwayGroup.geometry.x.max() + 500)
        ax_c.set_ylim(runwayGroup.geometry.y.min() - 500, runwayGroup.geometry.y.max() + 500)
        ctx.add_basemap(ax_c, source=ctx.providers.OpenStreetMap.Mapnik)
        ax_c.set_title(f"Approach Paths (Runway {runwayId}) – colored by probability")
        ax_c.set_axis_off()
        cbar = plt.colorbar(sc, ax=ax_c, fraction=0.035, pad=0.01)
        cbar.set_label("Probability")
        plt.tight_layout()
        plt.show()


        # Histogramm mit relativer Häufigkeit und mehr X-Ticks
        fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
        probs = runwayGroup["probability"]  # oder gdf["probability"] für den kombinierten Plot
        ax_hist.hist(probs, bins=numberBins, color="blue", edgecolor="black", weights=np.ones_like(probs) / len(probs))
        ax_hist.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")
        ax_hist.set_xlabel("Probability")
        ax_hist.set_ylabel("Relative Frequency")
        ax_hist.set_title(f"Probability Distribution for Runway {runwayId}")  # oder passend ersetzen
        ax_hist.set_xticks(np.arange(0, 1.05, 0.05))
        ax_hist.legend()
        plt.tight_layout()
        plt.show()


    #Dritter Plot: Alle Flüge kombiniert
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color=colors, alpha=alpha, markersize=50)
    # Legende hinzufügen
    legendHandles = [
        mpatches.Patch(color=landingColor, label="landing"),
        mpatches.Patch(color=otherColor, label="other")
    ]
    ax.legend(handles=legendHandles, title="Classification", loc="upper right")
    ax.set_xlim(gdf.geometry.x.min() - 500, gdf.geometry.x.max() + 500)
    ax.set_ylim(gdf.geometry.y.min() - 500, gdf.geometry.y.max() + 500)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_title(f"Approach Paths with Classification Threshold {threshold}")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    #zusatzplot: alle flüge kombiniert mit kontinuierlicher Einfärbung
    fig_c_all, ax_c_all = plt.subplots(figsize=(10, 10))
    sc = plotPointsColoredByProbability(ax_c_all, gdf, opacity=opacity, size=50)
    ax_c_all.set_xlim(gdf.geometry.x.min() - 500, gdf.geometry.x.max() + 500)
    ax_c_all.set_ylim(gdf.geometry.y.min() - 500, gdf.geometry.y.max() + 500)
    ctx.add_basemap(ax_c_all, source=ctx.providers.OpenStreetMap.Mapnik)
    ax_c_all.set_title(f"Approach Paths – colored by probability")
    ax_c_all.set_axis_off()
    cbar = plt.colorbar(sc, ax=ax_c_all, fraction=0.035, pad=0.01)
    cbar.set_label("Probability")
    plt.tight_layout()
    plt.show()


    # Histogramm für alle Flüge kombiniert
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    probs = gdf["probability"]
    ax_hist.hist(probs, bins=numberBins, color="blue", edgecolor="black", weights=np.ones_like(probs) / len(probs))
    ax_hist.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")
    ax_hist.set_xlabel("Probability")
    ax_hist.set_ylabel("Relative Frequency")
    ax_hist.set_title("Probability Distribution (All Flights)")
    ax_hist.legend()
    plt.tight_layout()
    plt.show()


else:
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color=colors, alpha=alpha, markersize=50)
    # Legende hinzufügen
    legendHandles = [
        mpatches.Patch(color=landingColor, label="landing"),
        mpatches.Patch(color=otherColor, label="other")
    ]
    ax.legend(handles=legendHandles, title="Classification", loc="upper right")
    ax.set_xlim(gdf.geometry.x.min() - 500, gdf.geometry.x.max() + 500)
    ax.set_ylim(gdf.geometry.y.min() - 500, gdf.geometry.y.max() + 500)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_title(f"Approach Paths with Classification Threshold {threshold}")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    #zusatzplot: kombiniert, kontinuierliche Einfärbung
    fig_c, ax_c = plt.subplots(figsize=(10, 10))
    sc = plotPointsColoredByProbability(ax_c, gdf, opacity=opacity, size=50)
    ax_c.set_xlim(gdf.geometry.x.min() - 500, gdf.geometry.x.max() + 500)
    ax_c.set_ylim(gdf.geometry.y.min() - 500, gdf.geometry.y.max() + 500)
    ctx.add_basemap(ax_c, source=ctx.providers.OpenStreetMap.Mapnik)
    ax_c.set_title(f"Approach Paths – colored by probability")
    ax_c.set_axis_off()
    cbar = plt.colorbar(sc, ax=ax_c, fraction=0.035, pad=0.01)
    cbar.set_label("Probability")
    plt.tight_layout()
    plt.show()


    # Histogramm für alle Flüge kombiniert
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    probs = gdf["probability"]
    ax_hist.hist(probs, bins=numberBins, color="blue", edgecolor="black", weights=np.ones_like(probs) / len(probs))
    ax_hist.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")
    ax_hist.set_xlabel("Probability")
    ax_hist.set_ylabel("Relative Frequency")
    ax_hist.set_title("Probability Distribution (All Flights)")
    ax_hist.legend()
    plt.tight_layout()
    plt.show()

