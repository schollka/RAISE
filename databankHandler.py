# save_to_db.py

from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from collections import deque
from datetime import datetime
import json

Base = declarative_base()

class TrackPoint(Base):
    __tablename__ = 'track_points'

    id = Column(Integer, primary_key=True)
    flightId = Column(Integer)
    category = Column(String)

    # Zeit und Position
    timestamp = Column(DateTime)
    time = Column(Integer)
    recvTime = Column(Float)
    lat = Column(Float)
    lon = Column(Float)
    altitude = Column(Integer)
    distanceToAirport = Column(Float)

    # Bewegung
    climbRate = Column(Float)
    groundSpeed = Column(Float)
    track = Column(Float)
    turnRate = Column(Float)

    # Empfangsdaten
    freq = Column(Float)
    snr = Column(Float)
    rssi = Column(Float)
    errCount = Column(Integer)
    eStatus = Column(Integer)
    relayed = Column(Boolean)
    reducedDataConfidence = Column(Boolean)

    # Relativer Standort
    distance = Column(Float)
    bearing = Column(Float)
    elevAngle = Column(Float)

    # Zusatzinfos
    state = Column(String)
    aircraftStates = Column(Text)  # als JSON-String speichern
    flightEvents = Column(Text)    # als JSON-String speichern

# Hilfsfunktion zum Konvertieren von datetime-Objekten
def serializeDatetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def saveTrack(trackDeque: deque, dbPath: str, category: str = "default"):
    """
    Speichert alle Punkte in `trackDeque` als zusammengehörigen Flug mit gemeinsamer flightId
    und übergibt die angegebene Kategorie (z. B. 'departure', 'arrival', 'inFlight').
    """
    if not trackDeque:
        return  # nichts zu speichern

    engine = create_engine(f'sqlite:///{dbPath}')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # aktuelle maximale flightId holen
    lastFlightId = session.query(TrackPoint.flightId).order_by(TrackPoint.flightId.desc()).first()
    nextFlightId = (lastFlightId[0] + 1) if lastFlightId else 1

    for point in trackDeque:
        trackPoint = TrackPoint(
            flightId=nextFlightId,
            category=category,
            timestamp=point["timestamp"],
            time=point["time"],
            recvTime=point["recvTime"],
            lat=point["lat"],
            lon=point["lon"],
            altitude=point["alt"],
            distanceToAirport=point["distanceToAirport"],
            climbRate=point["vs"],
            groundSpeed=point["speed"],
            track=point["track"],
            turnRate=point["turnRate"],
            freq=point["freq"],
            snr=point["snr"],
            rssi=point["rssi"],
            errCount=point["errCount"],
            eStatus=point["eStatus"],
            relayed=point["relayed"],
            reducedDataConfidence=point["reducedDataConfidence"],
            distance=point["distance"],
            bearing=point["bearing"],
            elevAngle=point["elevAngle"],
            state=point["aircraftStates"].get("flightState", "unknown"),
            aircraftStates=json.dumps(point["aircraftStates"], default=serializeDatetime),
            flightEvents=json.dumps(point["flightEvents"], default=serializeDatetime)
        )
        session.add(trackPoint)

    session.commit()
    session.close()
