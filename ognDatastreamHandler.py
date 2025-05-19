import socket
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta

HOST = "127.0.0.1"
PORT = 50001
BUFFER_SECONDS = 300  # 5 Minuten

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
)

# RAM Speicher für Flugzeuge
aircraftTracks = defaultdict(lambda: deque(maxlen=1000))

def parseOgnLine(line):
    match = ognRegex.match(line)
    if not match:
        return None
    d = match.groupdict()
    # Konvertiere passende Felder
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
        d["timestamp"] = datetime.utcnow()
    except Exception as e:
        print(f"⚠️ Fehler beim Parsen: {e}")
        return None
    return d

def removeOldTracks():
    now = datetime.utcnow()
    cutoff = now - timedelta(seconds=BUFFER_SECONDS)
    for aircraftId, track in list(aircraftTracks.items()):
        while track and track[0]["timestamp"] < cutoff:
            track.popleft()
        if not track:
            del aircraftTracks[aircraftId]

def runClient(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        print(f"🔌 Verbinde mit {host}:{port}...")
        sock.connect((host, port))
        print("✅ Verbunden. Warte auf OGN-Daten...")

        buffer = ""
        try:
            while True:
                data = sock.recv(4096)
                if not data:
                    print("Verbindung wurde vom Server geschlossen.")
                    break
                buffer += data.decode(errors='ignore')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line or not line[0].isdigit():
                        continue
                    parsed = parseOgnLine(line)
                    if parsed:
                        aircraftId = parsed["aircraft"]
                        aircraftTracks[aircraftId].append(parsed)
                        removeOldTracks()
        except KeyboardInterrupt:
            print("\n🛑 Client manuell beendet.")

if __name__ == "__main__":
    runClient(HOST, PORT)
