from flightPathPlotter import FlightPlotter
from ognClient import OgnClient
import socket
import select

def runClient(self):
        flightPlot = FlightPlotter(
            airportLat=self.AIRPORT_LATITUDE,
            airportLon=self.AIRPORT_LONGITUDE,
            airportRadius_km=self.ON_GROUND_POSITION_RADIUS / 1000.0
        )
        flightPlot.show()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            self.connectToOgnServer(sock)
            buffer = ""
            
            try:
                while True:
                    ready, _, _ = select.select([sock], [], [], 0)

                    if ready:
                        data = sock.recv(4096)
                        if not data:
                            print("Connection closed by server.")
                            break

                        buffer += data.decode(errors='ignore')
                        processedCount = 0  #Counter for number of recieved messages

                        while '\n' in buffer and processedCount < self.MAXIMUM_MESSAGES_IN_BUFFER:
                            processedCount += 1
                            if self.REALTIME_MODE:
                                self.time.setSystemTime() #set system time

                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            if not line or not line[0].isdigit():
                                continue
                            
                            self.processMessageLine(line)
                            self.removeOldTracks()

                            # Terminal Output (optional, can be moved to a separate method)
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
                                        f"OGNtime: {(lastPosition['time'])} | "
                                        f"SysTime: {(lastPosition['timestamp'])}")
                            print("------------------------------------")

                        flightPlot.updatePlot(self.aircraftTracks)
                        flightPlot.fig.canvas.draw()
                        flightPlot.fig.canvas.flush_events()

                        if '\n' in buffer:
                            buffer = '' #delete remaining buffer contents

                    else:
                        #this executes, when no new messages are processed
                        self.systemLoop() #call defined maintanance functions
  

            except KeyboardInterrupt:
                print("\nClient terminated by user.")


if __name__ == "__main__":
    client = OgnClient()
    runClient(client)