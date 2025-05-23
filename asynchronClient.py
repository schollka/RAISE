import asyncio
import json
import websockets
from flightPathPlotter import FlightPlotter
from ognClient import OgnClient
verbose = 0

async def runClientWebSocket(self, uri="ws://localhost:8765"):
    flightPlot = FlightPlotter(
        airportLat=self.AIRPORT_LATITUDE,
        airportLon=self.AIRPORT_LONGITUDE,
        airportRadius_km=self.ON_GROUND_POSITION_RADIUS / 1000.0
    )
    flightPlot.show()

    try:
        async with websockets.connect(uri) as websocket:
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)

                    if data.get("type") == "eof":
                        print("📶 Stream vom Emulator wurde beendet.")
                        break

                    if self.REALTIME_MODE:
                        self.time.setSystemTime()

                    self.processMessageDict(data)
                    self.removeOldTracks()

                    if verbose >= 2:
                        print("------------------------------------")
                        for aircraftId, trackInfo in self.aircraftTracks.items():
                            if trackInfo["track"]:
                                lastPosition = trackInfo["track"][-1]
                                print(f"✈ {aircraftId} | "
                                    f"State: {trackInfo['state']} | "
                                    f"StableState: {trackInfo['stableState']} | "
                                    f"PrevStableState: {trackInfo['prevStableState']} | "
                                    f"Pos: {lastPosition['lat']:.5f}, {lastPosition['lon']:.5f} | "
                                    f"Alt: {lastPosition['alt']}m | "
                                    f"Spd: {lastPosition['speed']:.1f}m/s | "
                                    f"Last Package: {(self.time.getSystemTime() - lastPosition['timestamp']).total_seconds():.0f}s | "
                                    f"OGNtime: {lastPosition['time']} | "
                                    f"SysTime: {lastPosition['timestamp']}")
                        print("------------------------------------")

                    flightPlot.updatePlot(self.aircraftTracks)
                    flightPlot.fig.canvas.draw()
                    flightPlot.fig.canvas.flush_events()

                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON-Fehler: {e}")
                except websockets.ConnectionClosed:
                    print("🔌 Verbindung wurde vom Server geschlossen.")
                    break

    except ConnectionRefusedError:
        print("❌ Verbindung zum Emulator nicht möglich. Stelle sicher, dass der Server läuft.")
    except KeyboardInterrupt:
        print("\n⛔ Client manuell beendet.")

# Main-Aufruf mit asynchroner Struktur
if __name__ == "__main__":
    client = OgnClient()
    asyncio.run(runClientWebSocket(client))
