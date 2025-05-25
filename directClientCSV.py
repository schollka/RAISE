import argparse
from flightPathPlotter import FlightPlotter
from ognClient import OgnClient
from readCsvForProcessing import load_messages  # siehe vorheriger Schritt
import matplotlib.pyplot as plt

verbose = 0

def runClientFromCsv(client, csvPath):
    try:
        messages = load_messages(csvPath)

        # Schritt 1: Alle Nachrichten einlesen und verarbeiten
        for data in messages:
            if client.REALTIME_MODE:
                client.time.setSystemTime()

            client.processMessageDict(data)

        # Schritt 3: Plot mit allen Flugspuren anzeigen
        flightPlot = FlightPlotter(
            airportLat=client.AIRPORT_LATITUDE,
            airportLon=client.AIRPORT_LONGITUDE,
            airportRadius_km=client.ON_GROUND_POSITION_RADIUS / 1000.0
        )
        flightPlot.show()

        if verbose >= 1:
            print(f"📊 Plot wird erstellt mit {len(client.aircraftTracks)} Flugspuren")

        flightPlot.updatePlot(client.aircraftTracks)
        flightPlot.fig.canvas.draw()
        flightPlot.fig.canvas.flush_events()

        print("📄 Alle CSV-Daten verarbeitet und geplottet.")
        plt.ioff()       
        plt.show()       

    except FileNotFoundError:
        print(f"❌ CSV-Datei nicht gefunden: {csvPath}")
    except KeyboardInterrupt:
        print("\n⛔ Verarbeitung manuell abgebrochen.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verarbeite OGN-CSV-Daten und visualisiere Flugpfade.")
    parser.add_argument("--csv", required=True, help="Pfad zur CSV-Datei")
    args = parser.parse_args()

    client = OgnClient()
    runClientFromCsv(client, args.csv)
