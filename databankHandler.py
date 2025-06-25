'''
              __|__
       --------(_)--------       
              RAISE         
 Runway Approach Identification for Silent Entries
------------------------------------------------------
    Tracking • Prediction • Silent Entry Detection     
------------------------------------------------------

code for handling the database and write operations
'''

from sqlalchemy import create_engine, text, Column, Integer, Float, String, Boolean, DateTime, Time, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from collections import deque
from datetime import datetime
import json

Base = declarative_base()

class TrackPoint(Base):
    #data structure for storing the track data in the database
    __tablename__ = 'track_points'

    id = Column(Integer, primary_key=True)
    flightId = Column(Integer)
    aircraftId = Column(String)
    category = Column(String)

    #time and location
    timestamp = Column(DateTime) #system time stamp
    time = Column(Time) #time in the OGN message HHMMSS
    recvTime = Column(Float) #decode time
    lat = Column(Float) #latitude
    lon = Column(Float) #longitude
    altitude = Column(Integer) #GPS altitude
    distanceToAirport = Column(Float) #computed distance to reference point on airport

    #movement
    climbRate = Column(Float) #vertical speed
    groundSpeed = Column(Float) #GPS ground speed
    track = Column(Float) #heading
    turnRate = Column(Float) #turn rate

    #reciever information
    freq = Column(Float) #frequency
    snr = Column(Float) 
    rssi = Column(Float)
    errCount = Column(Integer)
    eStatus = Column(Integer)
    relayed = Column(Boolean) #flag if the message was relayed
    reducedDataConfidence = Column(Boolean) #flag if the data may be corrupted

    #relative position calculated by ogn-decode
    distance = Column(Float) #distance from reciever station
    bearing = Column(Float) #bearing
    elevAngle = Column(Float) #elevtion angle

    #additional information 
    state = Column(String) #current computed state
    aircraftStates = Column(Text)  #all aircraft state dictionary entires as JSON
    flightEvents = Column(Text)    #all flight event dictionary entries as JSON

def serializeDatetime(obj):
    #helper function to convert datetime obejcts into serialized data
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

class DatabaseService:
    '''
    main database class with function definitions for init, read and write operations
    '''

    def __init__(self, dbParameters):
        #init the database service with the corret path and system behavior
        self.parameters = dbParameters #get the parameters from the main program
        self.engine = create_engine(
                                    f'sqlite:///{self.parameters["DATABASE_PATH"]}',
                                    connect_args={"check_same_thread": False}
                                ) #create engine
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL;")) #enable write-ahed loggin mode for simulatanios writing and reading

        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def saveTrack(self, trackDeque: deque, aircraftId: str = None, category: str = "default"):
        #writing track data to the database
        if not trackDeque:
            return  #no data available

        session = self.Session() #get the current session

        lastFlightId = session.query(TrackPoint.flightId).order_by(TrackPoint.flightId.desc()).first() #get the last flight ID
        nextFlightId = (lastFlightId[0] + 1) if lastFlightId else 1 #count the flight ID one step up for the new dataset

        for point in trackDeque:
            #convert data into structure for each data point
            #privatize the aircraft ID if requested by the user
            if self.parameters["PRIVATE_AIRCRAFT_ID"]:
                thisAircraftId=None
            else:
                thisAircraftId=aircraftId
            #privatize the timestamp if requested by the user
            if self.parameters["PRIVATE_TIMESTAMP"]:
                thisTimeStamp=None
            else:
                thisTimeStamp=point["timestamp"] 
        
            trackPoint = TrackPoint(
                flightId=nextFlightId,
                aircraftId=thisAircraftId,
                category=category,
                timestamp=thisTimeStamp,
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
            session.add(trackPoint) #add the new data point to the data base

        session.commit() #commit the changes to the database
        session.close() #close session to the database

    def shutdown(self):
        if self.engine:
            self.engine.dispose()
