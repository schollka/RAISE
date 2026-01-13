'''
              __|__
       --------(_)--------       
              RAISE         
 Runway Approach Identification for Silent Entries
------------------------------------------------------
    Tracking • Prediction • Silent Entry Detection     
------------------------------------------------------

code for API webserver
'''

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List
import asyncio
import random
import string
from callsignDBLookUp import DDBLookup
from datetime import time, datetime

app = FastAPI(root_path="/api")

#enable CORS so that frontend JavaScript can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#obfuscation maps
idObfuscationMap: Dict[str, str] = {}
reverseObfuscationMap: Dict[str, str] = {}

def obfuscate_id(real_id: str) -> str:
    if real_id not in idObfuscationMap:
        rand = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        idObfuscationMap[real_id] = rand
        reverseObfuscationMap[rand] = real_id
    return idObfuscationMap[real_id]

def deobfuscate_id(obf_id: str) -> str:
    return reverseObfuscationMap.get(obf_id)

def is_callsign_translation_enabled():
    return globalConfig.get("LOOK_UP_ID_TO_CALLSIGN", False) 

#global variable to hold the aircraftTracks reference from RAISE main code
externalAircraftTracks = None

#a list to store all active WebSocket clients
websocketClients: List[WebSocket] = []

#REST endpoint: returns all aircrafts with latest position and state
@app.get("/aircrafts")
async def get_all_aircrafts():
    if externalAircraftTracks is None:
        return JSONResponse([])

    result = []
    for aircraftId, entry in externalAircraftTracks.items():
        if not entry["track"]:
            continue
        lastPoint = entry["track"][-1]  #get latest position

        #format lastTimeSeen to string (assume it's datetime or time object)
        lastSeen = lastPoint.get("time", None)
        if isinstance(lastSeen, (datetime, time)):
            lastSeenStr = lastSeen.isoformat()
        else:
            lastSeenStr = None

        result.append({
            "id": obfuscate_id(aircraftId),
            "lat": lastPoint.get("lat"),
            "lon": lastPoint.get("lon"),
            "heading": lastPoint.get("track",0),
            "alt": lastPoint.get("alt", 0),
            "speed": lastPoint.get("speed", 0),
            "flightState": entry.get("stableState", "unknown"),
            "receptionState": entry.get("receptionState", "normal"),
            "lastTimeSeen": lastSeenStr
        })
    return JSONResponse(result)

#REST endpoint: returns the full track (deque) of a specific aircraft
@app.get("/aircrafts/{aircraft_id}/track")
async def get_track(aircraft_id: str):
    real_id = deobfuscate_id(aircraft_id)
    if real_id is None or externalAircraftTracks is None or real_id not in externalAircraftTracks:
        return JSONResponse(status_code=404, content={"error": "Aircraft not found"})

    #prepare track with only lat/lon
    track = []
    for point in externalAircraftTracks[real_id]["track"]:
        if "lat" in point and "lon" in point:
            track.append({"lat": point["lat"], "lon": point["lon"]})

    return JSONResponse(track)

#REST endpoint: returns the callsign for an aircraft, or "XXXXX" if tracking not allowed
@app.get("/callsign/{aircraft_id}")
async def get_callsign(aircraft_id: str):
    real_id = deobfuscate_id(aircraft_id)
    if real_id is None:
        return {"aircraft": "?", "callsign": "XXXXX"}

    if is_callsign_translation_enabled():
        callsign = ddb.getCallsign(real_id)
    else:
        callsign = real_id
    return {"aircraft": real_id, "callsign": callsign}

#WebSocket endpoint: accepts and stores connection for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocketClients.append(websocket)
    try:
        while True:
            await asyncio.sleep(0.1)  #keep connection alive
    except Exception:
        websocketClients.remove(websocket)  #remove client on disconnect

#function to push a position update to all connected WebSocket clients
async def push_position_update(aircraftId: str):
    if externalAircraftTracks is None or not externalAircraftTracks[aircraftId]["track"]:
        return
    lastPoint = externalAircraftTracks[aircraftId]["track"][-1]

    #format lastTimeSeen to string (assume it's datetime or time object)
    lastSeen = lastPoint.get("timestamp", None)
    if isinstance(lastSeen, (datetime, time)):
        lastSeenStr = lastSeen.isoformat()
    else:
        lastSeenStr = None

    data = {
        "type": "positionUpdate",
        "aircraftId": obfuscate_id(aircraftId),
        "lat": lastPoint.get("lat"),
        "lon": lastPoint.get("lon"),
        "heading": lastPoint.get("track",0),
        "alt": lastPoint.get("alt", 0),
        "speed": lastPoint.get("speed", 0),
        "flightState": externalAircraftTracks[aircraftId].get("stableState", "unknown"),
        "receptionState": externalAircraftTracks[aircraftId].get("receptionState", "normal"),
        "lastTimeSeen": lastSeenStr
    }
    for ws in websocketClients:
        try:
            await ws.send_json(data)  #send update to each connected client
        except:
            pass  #ignore failed sends (e.g. broken connection)

#function to link RAISE aircraftTracks into this server module
def connect_aircraft_tracks(reference):
    global externalAircraftTracks
    externalAircraftTracks = reference

#set the config of the map
def set_map_config(airportParams):
    global mapConfig
    mapConfig = {
        "lat": airportParams.get("AIRPORT_LATITUDE", 48.749957),
        "lon": airportParams.get("AIRPORT_LONGITUDE", 9.105383),
        "zoom": airportParams.get("WEB_ZOOM_LEVEL", 12)
    }

#get map config
@app.get("/config")
async def get_config():
    return {
        "airport": {
            "lat": mapConfig["lat"],
            "lon": mapConfig["lon"]
        },
        "mapZoom": mapConfig["zoom"]
    }

#get global system config
def connect_config(configDict):
    global globalConfig, ddb
    globalConfig = configDict

    refreshSeconds = globalConfig.get("ID_DB_REFRESH_INTERVALL", 43200)  # fallback: 12h
    ddb = DDBLookup(refreshSeconds)
