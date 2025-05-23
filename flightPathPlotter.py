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
        self.verbose = 0

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
        self.ax.set_aspect('equal', adjustable='datalim')

        # Flughafenmarker zeichnen
        self.ax.plot(self.airportLon, self.airportLat, marker='x', color='black', markersize=12, mew=3, label='Airport')

        # Für kleine Radien ist eine grobe Umrechnung ausreichend
        kmPerDegreeLat = 111  # Näherung
        kmPerDegreeLon = 111 * abs(math.cos(math.radians(self.airportLat)))
        radiusDegLat = self.airportRadius_km / kmPerDegreeLat
        radiusDegLon = self.airportRadius_km / kmPerDegreeLon

        # Kreis zeichnen
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

        # Plotbereich auf ±2,5 km beschränken
        halfExtentKm = 2.5
        extentDegLat = halfExtentKm / kmPerDegreeLat
        extentDegLon = halfExtentKm / kmPerDegreeLon
        self.ax.set_xlim(self.airportLon - extentDegLon, self.airportLon + extentDegLon)
        self.ax.set_ylim(self.airportLat - extentDegLat, self.airportLat + extentDegLat)

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
            if trackInfo['state'] == "airborne":
                self.ax.plot(lastLon, lastLat, marker='s', color='black', markersize=5)
            elif trackInfo['state'] == "onGround":
                self.ax.plot(lastLon, lastLat, marker='^', color='blue', markersize=3)
            else:
                self.ax.plot(lastLon, lastLat, marker='o', color='green', markersize=3)

        self.ax.legend(loc='upper right')

    def show(self):
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
