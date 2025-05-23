import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import itertools
import math

class FlightPlotter:
    def __init__(self, airportLat, airportLon, airportRadius_km):
        self.airportLat = airportLat
        self.airportLon = airportLon
        self.airportRadius_km = airportRadius_km
        self.fig, self.ax = plt.subplots()
        self.aircraftColors = {}
        self.colorCycle = itertools.cycle(plt.cm.get_cmap('tab10').colors)
        self.lines = {}

        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        self.ax.set_title("Live Flight Tracks")
        self.ax.grid(True)

    def updatePlot(self, aircraftTracks):
        self.ax.clear()
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        self.ax.set_title("Live Flight Tracks")
        self.ax.grid(True)
        self.ax.set_aspect('equal', adjustable='datalim')  # ⬅️ This line sets equal scaling


        # Flughafenmarker zeichnen
        self.ax.plot(self.airportLon, self.airportLat, marker='x', color='black', markersize=12, mew=3, label='Airport')

        # Kreis um den Flughafen (nur optisch – nicht exakt metrisch genau auf Kugeloberfläche)
        # Für kleine Radien ist eine grobe Umrechnung ausreichend
        kmPerDegreeLat = 111  # Näherung
        kmPerDegreeLon = 111 * abs(math.cos(math.radians(self.airportLat)))
        radiusDegLat = self.airportRadius_km / kmPerDegreeLat
        radiusDegLon = self.airportRadius_km / kmPerDegreeLon

        circle = Circle(
            (self.airportLon, self.airportLat),
            radius=radiusDegLon,
            edgecolor='black',
            facecolor='none',
            linewidth=1.5,
            linestyle='--',
            label=f'{self.airportRadius_km*1000:.0f} m Radius'
        )
        self.ax.add_patch(circle)

        for aircraftId, trackInfo in aircraftTracks.items():
            track = trackInfo["track"]
            if not track:
                continue

            lats = [p["lat"] for p in track]
            lons = [p["lon"] for p in track]

            if aircraftId not in self.aircraftColors:
                self.aircraftColors[aircraftId] = next(self.colorCycle)

            # Plot full track
            self.ax.plot(lons, lats, label=aircraftId, color=self.aircraftColors[aircraftId])

            # Plot last known position as black rectangle
            lastLat = lats[-1]
            lastLon = lons[-1]
            self.ax.plot(lastLon, lastLat, marker='s', color='black', markersize=5)


        self.ax.legend(loc='upper right')

    def show(self):
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
