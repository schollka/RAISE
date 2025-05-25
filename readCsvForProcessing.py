import pandas as pd
from datetime import datetime
import re

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

def parse_time(ts):
    try:
        return datetime.fromisoformat(ts).strftime("%H%M%S")
    except:
        return None

def parse_raw_line(timestamp, raw):
    match = ognRegex.match(raw)
    if not match:
        return None
    d = match.groupdict()
    return {
        "timestamp": timestamp,
        "time": parse_time(timestamp),
        "aircraft": d["aircraft"],
        "lat": float(d["lat"]),
        "lon": float(d["lon"]),
        "altitude": int(d["alt"]),
        "climbRate": float(d["vs"]),
        "groundSpeed": float(d["speed"]),
        "track": float(d["track"]),
        "turnRate": float(d["turnRate"]),
        "frequency": float(d["freq"]),
        "ognTime": int(d["time"]),
        "recvTime": float(d["recvTime"]),
        "netCode": int(d["netCode"]),
        "rfLevel": int(d["rfLevel"]),
        "aircraftType": d["aircraftType"],
        "acftDim": d["acftDim"],
        "stealth": d["stealth"],
        "noTrack": d["noTrack"],
        "freqOffset": float(d["freqOffset"]),
        "snr": float(d["snr"]),
        "rssi": float(d["rssi"]),
        "errCount": int(d["errCount"]),
        "eStatus": int(d["eStatus"]),
        "distance": float(d["distance"]),
        "bearing": float(d["bearing"]),
        "elevAngle": float(d["elevAngle"]),
        "relayed": bool(d.get("relayed")),
        "raw": raw,
    }

def parse_structured_row(row):
    return {
        "timestamp": row["timestamp"],
        "time": parse_time(row["timestamp"]),
        "aircraft": row["id"],
        "lat": row["lat"],
        "lon": row["lon"],
        "altitude": row["altitude_m"],
        "climbRate": row["climb_rate_mps"],
        "groundSpeed": row["ground_speed_mps"],
        "track": row["track_deg"],
        "turnRate": row["turn_rate_degps"],
        "frequency": None,
        "ognTime": row.get("ogn_time", None),
        "recvTime": None,
        "netCode": None,
        "rfLevel": None,
        "aircraftType": None,
        "acftDim": None,
        "stealth": None,
        "noTrack": None,
        "freqOffset": None,
        "snr": None,
        "rssi": None,
        "errCount": None,
        "eStatus": None,
        "distance": None,
        "bearing": None,
        "elevAngle": None,
        "relayed": None,
        "raw": None,
    }

def load_messages(path):
    df = pd.read_csv(path, sep=None, engine="python")

    if "ogn_raw" in df.iloc[0].values.tolist():
        df.columns = ["timestamp", "ogn_raw"]
        return [
            parsed for _, row in df.iloc[1:].iterrows()
            if (parsed := parse_raw_line(row["timestamp"], row["ogn_raw"])) is not None
        ]
    elif "timestamp" in df.columns and "id" in df.columns:
        return [parse_structured_row(row) for _, row in df.iterrows()]
    else:
        raise ValueError("Unbekanntes Dateiformat")
