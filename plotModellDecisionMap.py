import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point

df = pd.read_csv("modelLandingProbabilities.csv")
geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
gdf = gdf.to_crs(epsg=3857)

colors = gdf["probability"].apply(lambda p: "lightgray" if p < 0.6 else "red")
alpha = gdf["probability"].apply(lambda p: 0.5 if p < 0.6 else 0.7)

fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, color=colors, alpha=alpha, markersize=50)

# Zoom auf Punktwolke
ax.set_xlim(gdf.geometry.x.min() - 500, gdf.geometry.x.max() + 500)
ax.set_ylim(gdf.geometry.y.min() - 500, gdf.geometry.y.max() + 500)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_axis_off()
plt.tight_layout()
plt.show()
