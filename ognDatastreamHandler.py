import socket
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

#System parameters
HOST = "127.0.0.1"  #localhost
PORT = 50001        #ogn-decode TCP port
BUFFER_SECONDS = 300  #5 minutes of data retention per aircraft in seconds

#Landing detection parameters
LANDING_LOOKBACK_SECONDS = 30 #Time window to consider data for landing detection [s]
AIRPORT_ALTITUDE = 266 #Airport altitude [m]
ALTITUDE_TOLERANCE = 15 #Tolerance for aicraft altitude when on ground [m]

ognRegex = re.compile(
    r"^(?P<recvTime>\d+\.\d+)sec:(?P<freq>\d{3}\.\d{3})MHz: "
    r"(?P<netCode>\d+):(?P<rfLevel>\d+):(?P<aircraft>[A-F0-9]+) (?P<time>\d+): "
    r"\[\s*(?P<lat>[+-]?\d+\.\d+),\s*(?P<lon>[+-]?\d+\.\d+)\]deg\s+"
    r"(?P<alt>\d+)m\s+(?P<vs>[+-]?\d+\.\d+)m/s\s+(?P<speed>\d+\.\d+)m/s\s+"
    r"(?P<track>\d+\.\d+)deg\s+(?P<turnRate>[+-]?\d+\.\d+)deg/s\s+"
    r"(?P<aircraftType>__\d)\s+(?P<acftDim>\d{2}x\d{2})m\s+"
    r"(?P<stealth>[OS])\s+:(?P<noTrack>[0-9a-f]{3})__"
    r"(?P<freqOffset>[+-]?\d+\.\d+)kHz\s+(?P<snr>\d+\.\d+)/(?P<rssi>\d+\.\d+)dB/(?P<errCount>\d+)\s+"
    r"(?P<eStatus>\d+)e\s+(?P<distance>\d+\.\d+)km\s+(?P<bearing>\d+\.\d+)deg\s+(?P<elevAngle>[+-]?\d+\.\d+)deg"
    r"(?:\s+(?P<flagged>!))?$"
)
#regex for complex ogn message
# 0.585sec:868.174MHz: 1:2:DD9C20 142218: [ +48.95403,  +9.62327]deg  1401m  -3.2m/s  27.4m/s 204.5deg  +0.2deg/s __2 03x03m O :00f__-26.07kHz  4.0/15.0dB/2  0e    40.1km 097.3deg  +1.2deg + 

#Initilize the RAM storage for the aircraft data, maximum 1000 datasets per aircraft
aircraftTracks = defaultdict(lambda: {
    "track": deque(maxlen=1000), #1000 datasets
    "state": "unknown" #default aircraft status
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
    except Exception as e:
        print(f"OGN message parsing error: {e}")
        return None
    return d

def removeOldTracks():
    #remove old data points that are no longer needed
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=BUFFER_SECONDS)
    for aircraftId, track in list(aircraftTracks.items()):
        while track and track[0]["timestamp"] < cutoff:
            track.popleft()
        if not track:
            del aircraftTracks[aircraftId]

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
                        aircraftId = parsed["aircraft"] #get aircraft ID
                        aircraftTracks[aircraftId].append(parsed) #append data to aircraft
                        removeOldTracks() #remove old data points

                        #Terminal Output
                        print(f"✈ {aircraftId} | "
                              f"{parsed['timestamp'].strftime('%H:%M:%S')} | "
                              f"Pos: {parsed['lat']:.5f}, {parsed['lon']:.5f} | "
                              f"Alt: {parsed['alt']}m | "
                              f"Spd: {parsed['speed']:.1f}m/s | "
                              f"V/S: {parsed['vs']:+.1f}m/s | "
                              f"Trk: {parsed['track']:03.1f}° | "
                              f"TrnRate: {parsed['turnRate']:03.1f}°/s | "
                              f"Dist: {parsed['distance']:02.1f}km | "
                              f"Bearing: {parsed['bearing']:03.1f}° | "
                              f"Reduced Confidence: {parsed['reducedDataConfidence']}")
        except KeyboardInterrupt:
            print("\nClient terminated by user.")

if __name__ == "__main__":
    runClient(HOST, PORT)
