# save_to_db.py

from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from collections import deque

Base = declarative_base()

class TrackPoint(Base):
    __tablename__ = 'track_points'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    lat = Column(Float)
    lon = Column(Float)
    altitude = Column(Integer)
    climbRate = Column(Float)
    groundSpeed = Column(Float)
    track = Column(Float)
    turnRate = Column(Float)
    state = Column(String)

def saveTrackData(aircraftData: dict, dbPath: str):
    """
    Speichert die Trackdaten eines Flugzeugs anonymisiert in eine SQLite-Datenbank.
    :param aircraftData: Daten wie client.aircraftTracks['DD9B60']
    :param dbPath: Pfad zur Datenbankdatei (z.B. '/home/pi/flight_data.db')
    """
    engine = create_engine(f'sqlite:///{dbPath}')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    trackDeque = aircraftData.get('track', deque())
    for point in trackDeque:
        entry = TrackPoint(
            timestamp=point['timestamp'],
            lat=point['lat'],
            lon=point['lon'],
            altitude=point['altitude'],
            climbRate=point['climbRate'],
            groundSpeed=point['groundSpeed'],
            track=point['track'],
            turnRate=point['turnRate'],
            state=point.get('state', 'unknown')
        )
        session.add(entry)

    session.commit()
    session.close()
