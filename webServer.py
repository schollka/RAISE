from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List
import asyncio

#initialize the FastAPI app
app = FastAPI()

#enable CORS so that frontend JavaScript can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#global variable to hold the aircraftTracks reference from RAISE main code
externalAircraftTracks = None

#a list to store all active WebSocket clients
websocketClients: List[WebSocket] = []

#REST endpoint: returns all aircrafts with latest position and state
@app.get("/aircrafts")
async def get_all_aircrafts():
    if externalAircraftTracks is None:
        return JSONResponse([])  #return empty list if no data reference is set

    result = []
    for aircraftId, entry in externalAircraftTracks.items():
        if not entry["track"]:
            continue  #skip aircrafts with no track data
        lastPoint = entry["track"][-1]  #get latest position
        result.append({
            "id": aircraftId,
            "lat": lastPoint.get("lat"),
            "lon": lastPoint.get("lon"),
            "alt": lastPoint.get("alt"),
            "flightState": entry.get("flightState", "unknown"),
            "receptionState": entry.get("receptionState", "normal")
        })
    return JSONResponse(result)

#REST endpoint: returns the full track (deque) of a specific aircraft
@app.get("/aircrafts/{aircraft_id}/track")
async def get_track(aircraft_id: str):
    if externalAircraftTracks is None or aircraft_id not in externalAircraftTracks:
        return JSONResponse(status_code=404, content={"error": "Aircraft not found"})
    track = externalAircraftTracks[aircraft_id]["track"]
    return JSONResponse(list(track))  #convert deque to list for JSON serialization

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
    data = {
        "type": "positionUpdate",
        "aircraftId": aircraftId,
        "lat": lastPoint.get("lat"),
        "lon": lastPoint.get("lon"),
        "alt": lastPoint.get("alt"),
        "flightState": externalAircraftTracks[aircraftId].get("flightState", "unknown"),
        "receptionState": externalAircraftTracks[aircraftId].get("receptionState", "normal")
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
