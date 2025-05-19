import socket
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from statistics import mean
from math import radians, sin, cos, sqrt, atan2

#System parameters
HOST = "127.0.0.1"  #localhost
PORT = 50001        #ogn-decode TCP port
BUFFER_SECONDS = 300  #5 minutes of data retention per aircraft in seconds

#Landing detection parameters
LANDING_LOOKBACK_SECONDS = 30 #Time window to consider data for landing detection [s]
AIRPORT_ALTITUDE = 266 #Airport altitude [m]
ALTITUDE_TOLERANCE = 15 #Tolerance for aicraft altitude when on ground [m]
MAX_ON_GOUND_SPEED = 20 / 3.6 #Maximum gound speed for an aicraft to be considered on ground [m/s]
ON_GROUND_DETECTION_TIME_WINDOW = 30 #The last X seconds of the track data will be used to determine the onGound or airborne state [s]
AIRPORT_LATITUDE = 49.002222 #Latitude of the airport reference position, use center of runway if possible [°]
AIRPORT_LONGITUDE = 9.086389 #Longitude of the airport reference position, use center of runway if possible [°]
ON_GROUND_POSITION_RADIUS = 750 #All aircrafts on the ground within this distance from the airport reference will be considered landed [m]

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
#regex for complex ogn message
# 0.585sec:868.174MHz: 1:2:DD9C20 142218: [ +48.95403,  +9.62327]deg  1401m  -3.2m/s  27.4m/s 204.5deg  +0.2deg/s __2 03x03m O :00f__-26.07kHz  4.0/15.0dB/2  0e    40.1km 097.3deg  +1.2deg + 

#Initilize the RAM storage for the aircraft data, maximum 1000 datasets per aircraft
aircraftTracks = defaultdict(lambda: {
    "track": deque(maxlen=1000),
    "state": "unknown",
    "stableState": "unknown",
    "stateChangeTime": None,
    "landedSaved": False,
    "hasBeenAirborne": False
})


def parseOgnLine(line):
    #get all the information from the OGN message
    match = ognRegex.match(line)
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

def removeOldTracks():
    #remove old data points that are no longer needed
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=BUFFER_SECONDS)
    for aircraftId, data in list(aircraftTracks.items()):
        track = data["track"]
        while track and track[0]["timestamp"] < cutoff:
            track.popleft()
        if not track:
            del aircraftTracks[aircraftId]

def haversineDistance(lat1, lon1, lat2, lon2):
    #Computes the distance between two points on earths surface

    R = 6371000 #Earth radius
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dPhi = radians(lat2 - lat1) #delta in latitude
    dLambda = radians(lon2 - lon1) #delta in longitude

    a = sin(dPhi / 2)**2 + cos(phi1) * cos(phi2) * sin(dLambda / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c #distance between both points in meters
    return distance

def detectOnGroundState(track):
    """
    Determines whether an aircraft is 'onGround' or 'airborne' based on the last 30 seconds of data.
    The average altitude must be below a set threshold and the average gound speed below a maximum to be considered on gound.
    """
    if not track:
        return "unknown" #no track data available

    now = datetime.now(timezone.utc) #get current time
    windowStart = now - timedelta(seconds=30) #calculate the start time of the time window

    #filter track to get only the data in the time window
    recentPoints = [p for p in track if p["timestamp"] >= windowStart]
    
    if len(recentPoints) < 5:
        return "unknown" #Not enough data to make a confident decision

    avgAlt = mean(p["alt"] for p in recentPoints) #calculate the average altitude
    minAltThres = AIRPORT_ALTITUDE - ALTITUDE_TOLERANCE #minimum altitude to be considered on ground at the airport
    maxAltThres = AIRPORT_ALTITUDE + ALTITUDE_TOLERANCE #maximum altitude to be considered on ground at the airport

    avgSpeed = mean(p["speed"] for p in recentPoints) #calculate average ground speed
    avgLat = mean(p["lat"] for p in recentPoints) #calculate average latitude
    avgLon = mean(p["lon"] for p in recentPoints) #calculate average longitude
    distanceToAirport = haversineDistance(avgLat, avgLon, AIRPORT_LATITUDE, AIRPORT_LONGITUDE)


    if minAltThres <= avgAlt <= maxAltThres and avgSpeed <= MAX_ON_GOUND_SPEED and distanceToAirport <= ON_GROUND_POSITION_RADIUS:
        return "onGround" #aircraft is on ground
    elif avgSpeed > MAX_ON_GOUND_SPEED and (avgAlt > maxAltThres or avgAlt < minAltThres):
        return "airborne" #aircraft is airborne
    else:
        return "unknown" #unknown state

def dumpDataToDatabase(aircraftId, track):
    print(f"\n[DB] Dumping {len(track)} points for {aircraftId} to database.")
    # TODO: Replace with actual DB logic

def debounceState(aircraftId, newState):
    entry = aircraftTracks[aircraftId]
    now = datetime.now(timezone.utc)
    
    if newState != entry["stableState"]:
        timeInCurrentState = now - entry["lastStateChange"]
        if timeInCurrentState >= timedelta(seconds=10):  # Debounce time: 10 seconds
            entry["stableState"] = newState
            entry["lastStateChange"] = now
            return True  # State changed
    elif newState == entry["stableState"]:
        entry["lastStateChange"] = now  # reset timer if stable state
    return False  # No effective change

def processAircraftState(aircraftId):
    aircraft = aircraftTracks[aircraftId]
    currentState = aircraft["state"]

    debounceState(aircraftId, currentState)  # Stabilen Zustand updaten
    stableState = aircraft["stableState"]

    # Prüfe Übergang (nur wenn stabiler Zustand sich geändert hat)
    if "prevStableState" not in aircraft:
        aircraft["prevStableState"] = stableState

    if stableState != aircraft["prevStableState"]:
        prevState = aircraft["prevStableState"]
        aircraft["prevStableState"] = stableState

        # Übergang airborne -> onGround
        if prevState == "airborne" and stableState == "onGround":
            if aircraft["hasBeenAirborne"] and not aircraft["landedSaved"]:
                dumpDataToDatabase(aircraftId)
                aircraft["landedSaved"] = True
        # Übergang onGround -> airborne
        elif prevState == "onGround" and stableState == "airborne":
            aircraft["hasBeenAirborne"] = True
            aircraft["landedSaved"] = False


def runClient(host, port):
    #client code
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        print(f"Connecting to {host}:{port}...")
        sock.connect((host, port))
        print("Connected. Waiting for OGN data...\n")

        buffer = ""
        try:
            while True:
                data = sock.recv(4096) #connect to TCp socket
                if not data:
                    print("Connection closed by server.")
                    break
                buffer += data.decode(errors='ignore')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line or not line[0].isdigit(): #continue if no new message was revied or if it dit not start with a number (all OGN messages start with a number)
                        continue
                    parsed = parseOgnLine(line) #get data from message with the parser
                    if parsed:
                        aircraftId = parsed["aircraft"]
                        aircraftTracks[aircraftId]["track"].append(parsed)
                        
                        # Zustandsbestimmung
                        aircraftTracks[aircraftId]["state"] = detectOnGroundState(aircraftTracks[aircraftId]["track"])

                        # Zustand verarbeiten und ggf. Daten speichern
                        processAircraftState(aircraftId)

                        # Alte Daten entfernen
                        removeOldTracks()

                        #Terminal Output
                        '''
                        print(f"✈ {aircraftId} | "
                              f"State: {aircraftTracks[aircraftId]["state"]} | "
                              f"StableState: {aircraftTracks[aircraftId]['stableState']} | "
                              f"{parsed['timestamp'].strftime('%H:%M:%S')} | "
                              f"Pos: {parsed['lat']:.5f}, {parsed['lon']:.5f} | "
                              f"Alt: {parsed['alt']}m | "
                              f"Spd: {parsed['speed']:.1f}m/s | "
                              f"V/S: {parsed['vs']:+.1f}m/s | "
                              f"Dist: {parsed['distance']:02.1f}km | "
                              f"Reduced Confidence: {parsed['reducedDataConfidence']} | "
                              f"Relayed: {parsed['relayed']}")   
                        '''

                        print("------------------------------------")
                        for aircraftID, trackInfo in aircraftTracks.items():
                            if trackInfo["track"]:
                                lastPosition = trackInfo["track"][-1]
                                print(f"✈ {aircraftId} | "
                                    f"State: {trackInfo["state"]} | "
                                    f"StableState: {trackInfo['stableState']} | "
                                    f"Pos: {lastPosition['lat']:.5f}, {lastPosition['lon']:.5f} | "
                                    f"Alt: {lastPosition['alt']}m | "
                                    f"Spd: {lastPosition['speed']:.1f}m/s | ")

                        print("------------------------------------")             
        except KeyboardInterrupt:
            print("\nClient terminated by user.")

if __name__ == "__main__":
    runClient(HOST, PORT)
