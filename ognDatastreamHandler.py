import socket
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from statistics import mean
from math import radians, sin, cos, sqrt, atan2


class OgnClient:
    ognRegex = re.compile(
        r"^(?P<recvTime>\d+\.\d+)sec:(?P<freq>\d+\.\d+)MHz: "
        r"(?P<netCode>\d+):(?P<rfLevel>\d+):(?P<aircraft>[A-F0-9]+) (?P<time>\d+): "
        r"\[\s*(?P<lat>[+-]?\d+\.\d+),\s*(?P<lon>[+-]?\d+\.\d+)\]deg\s+"
        r"(?P<alt>\d+)m\s+(?P<vs>[+-]?\d+\.\d+)m/s\s+(?P<speed>\d+\.\d+)m/s\s+"
        r"(?P<track>\d+\.\d+)deg\s+(?P<turnRate>[+-]?\d+\.\d+)deg/s\s+"
        r"(?P<aircraftType>__\d)\s+(?P<acftDim>\d{2}x\d{2})m\s+"
        r"(?P<stealth>[OS])\s+:(?P<noTrack>[0-9a-f]{3})__"
        r"(?P<freqOffset>[+-]?\d+\.\d+)kHz\s+(?P<snr>\d+\.\d+)/(?P<rssi>\d+\.\d+)dB/(?P<errCount>\d+)\s+"
        r"(?P<eStatus>\d+)e\s+(?P<distance>\d+\.\d+)km\s+(?P<bearing>\d+\.\d+)deg\s+(?P<elevAngle>[+-]?\d+\.\d+)deg"
        r"(?:\s*(?P<relayed>\+))?\s*$"
    )

    # System parameters
    BUFFER_SECONDS = 300
    LANDING_LOOKBACK_SECONDS = 30
    AIRPORT_ALTITUDE = 266
    ALTITUDE_TOLERANCE = 15
    MAX_ON_GROUND_SPEED = 20 / 3.6
    ON_GROUND_DETECTION_TIME_WINDOW = 30
    AIRPORT_LATITUDE = 49.002222
    AIRPORT_LONGITUDE = 9.086389
    ON_GROUND_POSITION_RADIUS = 750

    def __init__(self, host="127.0.0.1", port=50001):
        self.host = host
        self.port = port

        # Initialize aircraft tracks dictionary
        self.aircraftTracks = defaultdict(lambda: {
            "track": deque(maxlen=1000),
            "state": "unknown",
            "stableState": "unknown",
            "prevStableState": "unknown",
            "lastStateChange": datetime.now(timezone.utc),
            "landedSaved": False,
            "hasBeenAirborne": False
        })

    def parseOgnLine(self, line):
        match = self.ognRegex.match(line)
        if not match:
            return None
        d = match.groupdict()

        try:
            d["recvTime"] = float(d["recvTime"])
            d["freq"] = float(d["freq"])
            d["time"] = int(d["time"])
            d["lat"] = float(d["lat"])
            d["lon"] = float(d["lon"])
            d["alt"] = int(d["alt"])
            d["vs"] = float(d["vs"])
            d["speed"] = float(d["speed"])
            d["track"] = float(d["track"])
            d["turnRate"] = float(d["turnRate"])
            d["snr"] = float(d["snr"])
            d["rssi"] = float(d["rssi"])
            d["errCount"] = int(d["errCount"])
            d["eStatus"] = int(d["eStatus"])
            d["distance"] = float(d["distance"])
            d["bearing"] = float(d["bearing"])
            d["elevAngle"] = float(d["elevAngle"])
            d["timestamp"] = datetime.now(timezone.utc)
            d["reducedDataConfidence"] = d.get("flagged") == "!"
            d["relayed"] = bool(d.get("relayed"))
        except Exception as e:
            print(f"OGN message parsing error: {e}")
            return None
        return d

    @staticmethod
    def haversineDistance(lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius in meters
        phi1 = radians(lat1)
        phi2 = radians(lat2)
        dPhi = radians(lat2 - lat1)
        dLambda = radians(lon2 - lon1)

        a = sin(dPhi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dLambda / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def detectOnGroundState(self, track):
        if not track:
            return "unknown"

        now = datetime.now(timezone.utc)
        windowStart = now - timedelta(seconds=self.ON_GROUND_DETECTION_TIME_WINDOW)
        recentPoints = [p for p in track if p["timestamp"] >= windowStart]

        if len(recentPoints) < 5:
            return "unknown"

        avgAlt = mean(p["alt"] for p in recentPoints)
        minAltThres = self.AIRPORT_ALTITUDE - self.ALTITUDE_TOLERANCE
        maxAltThres = self.AIRPORT_ALTITUDE + self.ALTITUDE_TOLERANCE

        avgSpeed = mean(p["speed"] for p in recentPoints)
        avgLat = mean(p["lat"] for p in recentPoints)
        avgLon = mean(p["lon"] for p in recentPoints)
        distanceToAirport = self.haversineDistance(avgLat, avgLon, self.AIRPORT_LATITUDE, self.AIRPORT_LONGITUDE)

        if minAltThres <= avgAlt <= maxAltThres and avgSpeed <= self.MAX_ON_GROUND_SPEED and distanceToAirport <= self.ON_GROUND_POSITION_RADIUS:
            return "onGround"
        elif avgSpeed > self.MAX_ON_GROUND_SPEED and (avgAlt > maxAltThres or avgAlt < minAltThres):
            return "airborne"
        else:
            return "unknown"

    def removeOldTracks(self):
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.BUFFER_SECONDS)
        for aircraftId, data in list(self.aircraftTracks.items()):
            track = data["track"]
            while track and track[0]["timestamp"] < cutoff:
                track.popleft()
            if not track:
                del self.aircraftTracks[aircraftId]

    def dumpDataToDatabase(self, aircraftId, track):
        print(f"\n[DB] Dumping {len(track)} points for {aircraftId} to database.")
        # TODO: Replace with actual DB logic

    def debounceState(self, aircraftId, newState):
        entry = self.aircraftTracks[aircraftId]
        now = datetime.now(timezone.utc)

        if newState != entry["stableState"]:
            timeInCurrentState = now - entry["lastStateChange"]
            if timeInCurrentState >= timedelta(seconds=10):
                entry["stableState"] = newState
                entry["lastStateChange"] = now
                return True
        else:
            entry["lastStateChange"] = now
        return False

    def processAircraftState(self, aircraftId):
        aircraft = self.aircraftTracks[aircraftId]
        currentState = aircraft["state"]

        stableChanged = self.debounceState(aircraftId, currentState)

        if stableChanged:
            prevState = aircraft["prevStableState"]
            newState = aircraft["stableState"]
            aircraft["prevStableState"] = newState

            if prevState == "airborne" and newState == "onGround":
                if aircraft["hasBeenAirborne"] and not aircraft["landedSaved"]:
                    self.dumpDataToDatabase(aircraftId, list(aircraft["track"]))
                    aircraft["landedSaved"] = True

            elif prevState == "onGround" and newState == "airborne":
                aircraft["hasBeenAirborne"] = True
                aircraft["landedSaved"] = False

    def runClient(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            print(f"Connecting to {self.host}:{self.port}...")
            sock.connect((self.host, self.port))
            print("Connected. Waiting for OGN data...\n")

            buffer = ""
            try:
                while True:
                    data = sock.recv(4096)
                    if not data:
                        print("Connection closed by server.")
                        break
                    buffer += data.decode(errors='ignore')
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if not line or not line[0].isdigit():
                            continue
                        parsed = self.parseOgnLine(line)
                        if parsed:
                            aircraftId = parsed["aircraft"]
                            self.aircraftTracks[aircraftId]["track"].append(parsed)

                            self.aircraftTracks[aircraftId]["state"] = self.detectOnGroundState(self.aircraftTracks[aircraftId]["track"])
                            self.processAircraftState(aircraftId)
                            self.removeOldTracks()

                        # Terminal Output (optional, can be moved to a separate method)
                        print("------------------------------------")
                        for aircraftID, trackInfo in self.aircraftTracks.items():
                            if trackInfo["track"]:
                                lastPosition = trackInfo["track"][-1]
                                print(f"✈ {aircraftID} | "
                                      f"State: {trackInfo['state']} | "
                                      f"StableState: {trackInfo['stableState']} | "
                                      f"Pos: {lastPosition['lat']:.5f}, {lastPosition['lon']:.5f} | "
                                      f"Alt: {lastPosition['alt']}m | "
                                      f"Spd: {lastPosition['speed']:.1f}m/s | ")
                        print("------------------------------------")

            except KeyboardInterrupt:
                print("\nClient terminated by user.")


if __name__ == "__main__":
    client = OgnClient()
    client.runClient()
