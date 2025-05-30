# save_to_db.py

from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from collections import deque
from datetime import datetime

Base = declarative_base()

class TrackPoint(Base):
    __tablename__ = 'track_points'

    id = Column(Integer, primary_key=True)
    flightId = Column(Integer)  # gemeinsame ID für zusammenhängenden Flug
    timestamp = Column(DateTime)
    lat = Column(Float)
    lon = Column(Float)
    altitude = Column(Integer)
    climbRate = Column(Float)
    groundSpeed = Column(Float)
    track = Column(Float)
    turnRate = Column(Float)
    state = Column(String)
    category = Column(String)  #category of stored data, departure, arrival, inFlight


def saveTrack(trackDeque: deque, dbPath: str, category: str):
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
        entry = TrackPoint(
            flightId=nextFlightId,
            timestamp=point['timestamp'],
            lat=point['lat'],
            lon=point['lon'],
            altitude=point['altitude'],
            climbRate=point['climbRate'],
            groundSpeed=point['groundSpeed'],
            track=point['track'],
            turnRate=point['turnRate'],
            state=point.get('state', 'unknown'),
            category=category
        )
        session.add(entry)

    session.commit()
    session.close()

