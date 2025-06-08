import socket
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone, time
from statistics import mean
from math import radians, sin, cos, sqrt, atan2
import select
from databankHandler import saveTrack
import yaml
import os
import shutil
import random

####################################################
################# OGN Client #######################
####################################################

class OgnClient:
    ####################################################
    ################# auxillary functions ##############
    ####################################################

    def safeFloat(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def safeInt(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
        
    def createPlaceHolderPendingState(self):
        #create a placeholder dictionary for the pending state dict
        return {
            "state": "unknown",
            "timestamp": self.time.getSystemTime()
        }

    def createPlaceHolderAircraftStates(self):
        #create a placeholder dictionary for the aircraft states dict
        return {
            "flightState": "unknown",
            "stableState": "unknown",
            "prevStableState": "unknown",
            "prevPrevStableState": "unknown",
            "pendingState": self.createPlaceHolderPendingState(),
        }
    
    def createPlaceHolderFlightEvent(self):
        #create a placeholder dictionary for the flight event dict
        return {
            "detectedTakeOff": False,
            "detectedTouchDown": False
        }
    
    def randomStorageFlag(self, probability):
        #returns True or False based on the random function and a given probability
        return random.random() < probability

    def __init__(self):
        '''
        Set up the OGN-Client and all its needed parameters
        '''

        ####################################################
        ################# load parameter file ##############
        ####################################################

        sourceCodeDir = os.path.dirname(os.path.abspath(__file__)) #get the directory of the source code
        parameterFile = os.path.join(sourceCodeDir, "parameters.yaml") #build the absolute file path of the expected parameter file
        defaultParameters = os.path.join(sourceCodeDir, "defaultParameters.yaml") #build the absolute file path of the default parameter file
        
        # Check if parameterFile exists and copy default if nonexistent
        if not os.path.exists(parameterFile):
            shutil.copy(defaultParameters, parameterFile) #copy default parameters
        #Load parameters
        with open(parameterFile, "r") as file: #load parameters from file, contains either custom values or the copied default values
            allParams = yaml.safe_load(file)

        #extract parameter dictionaries
        self.systemParameters = allParams["systemParameters"] #general system parameters
        self.airportParameters = allParams["airportParameters"] #airport parameters
        self.stateEstimationParameters = allParams["stateEstimationParameters"] #parameters used for state estimation
        self.signalReceptionParameters = allParams["signalReceptionParameters"] #signal reception state estimation parameters

        ####################################################
        ################# initialize system ################
        ####################################################

        self.host = self.systemParameters["HOST"] #define the host adress of the ogn-decode server
        self.port = self.systemParameters["PORT"] #define the port of the ogn-decode TCP server
        self.time = self.TimeManager() #initialize system time

        #initialize aircraft tracks dictionary
        self.aircraftTracks = defaultdict(lambda: {
            "track": deque(maxlen=self.systemParameters["DEQUE_LENGHT"]), #deque to store all data points

            #aircraft states
            "flightState": "unknown", #current calculated aircraft state
            "flightSubState": None, #substate for the "airborne" and "transitionAirGrnd" flightState
            "stableState": "unknown", #currently as stable determined aircraft state
            "pendingState": self.createPlaceHolderPendingState(), #pending states, currently in debounce
            "prevStableState": "unknown", #previos stable aircraft state
            "prevPrevStableState": "unknown", #previous previous stable aircraft state

            #flight events
            "detectedTakeOff": False, #boolean state if a take-off was detected with this data point
            "detectedTouchDown": False, #boolean state if a landing was detected with this data point
            "aircraftDepartedAirport": False, #boolen store information if aircraft has departed from this airport
            "departureTime": None, #time of departure
            "storeDeparture": False, #boolen to trigger the storage of the departure into the database
            "lastTimeDataWrittenToDB": None, #last timestamp at which data was written to the databse
            "airborneSince": None, #last time when the aircraft was stable airborne
            "inFlightDataStored": False, #boolen if in flight data was stored to the database

            #signal states
            "receptionState": "normal" #state of the signal reception
        })

    class TimeManager:
        '''
        The system relies on multiple time dependent calculations.
        The system time is abstracted behind this time manager to facilitate those calculations in two modes.
            Synchrone mode:
                If the system is operated as a real-time service, then the systems realtime is used
            Asynchrone mode:
                For development, testing and training the system is set to run in asynchrone mode.
                In this case the reference time is syntetically created based on the used dataset.
        '''
        def __init__(self):
            self.time = datetime.now(timezone.utc) #set time to realtime
        
        def setSystemTime(self):
            self.time = datetime.now(timezone.utc) #set time to realtime

        def setSystemTimeAsynchronousMode(self, asyntime: int, referenceDate: datetime = None):
            #this functions demodulates a timestamp in the format HHMMSS and creates a valid reference time from it
            hh = asyntime // 10000
            mm = (asyntime // 100) % 100
            ss = asyntime % 100

            #Specify the date
            if referenceDate is None:
                #the real current date is used, since all data is expected to be from the same date
                #in that case no relevant information is lost or added to the data by using the wrong date
                referenceDate = datetime.now(timezone.utc) 

            #Combine date and time into a valid structure
            asynSysTime = datetime.combine(referenceDate.date(), time(hh, mm, ss), tzinfo=timezone.utc)
            self.time = asynSysTime #set asynchrone system time

        def getSystemTime(self):
            return self.time #return the set time

    '''
    Regex expression for OGN message decode
    Example message: 0.936sec:868.370MHz: 1:2:DD9A70 142236: [ +49.00106,  +9.07859]deg   268m  +0.0m/s   0.0m/s 180.0deg  +0.0deg/s __1 04x04m O :01f__-30.02kHz 42.8/52.5dB/0  0e 0.1km 285.8deg -4.6deg + !
    Message blocks: 
        - recieving time of ogn-decode
        - frequency
        - network ID level
        - aircraft ID
        - GNSS time
        - position [Lat, Long]
        - GPS altitude
        - vertical speed
        - ground speed
        - heading
        - turn rate
        - aircraft type
        - aircraft dimension
        - stealth status
        - NoTrack hex-code
        - frequency offset
        - RSSI / SNR
        - error count
        - distance to reciever
        - bearing
        - elevation angle
        - + = relayed
        - ! = message maybe not valid
    '''
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

    def parseOgnLine(self, line):
        #decode recieved message into seperate data blocks

        match = self.ognRegex.match(line) #search for a match in the recieved message
        if not match:
            return None #if no match was found => probalby a system message, discard and move on
        d = match.groupdict() #create a dictionary based on the found match

        try:
            #try to demodulate the match into its data fields
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
            d["reducedDataConfidence"] = d.get("flagged") == "!"
            d["relayed"] = bool(d.get("relayed"))
            d["distanceToAirport"] = self.distanceToAirport(d["lat"], d["lon"])
            if not self.systemParameters["REALTIME_MODE"]: #switch for realtime operation or asynchrone operation
                self.time.setSystemTimeAsynchronousMode(asyntime=d["time"]) #create a timestamp based on the time in the recieved message
            d["timestamp"] = self.time.getSystemTime()
            d["aircraftStates"] = self.createPlaceHolderAircraftStates()
            d["flightEvents"] = self.createPlaceHolderFlightEvent()
        except Exception as e:
            print(f"OGN message parsing error: {e}")
            return None
        return d

    @staticmethod
    def haversineDistance(lat1, lon1, lat2, lon2):
        #Tool for the calculation of the distance between two points on earths sqherical surface
        R = 6371000  #earth radius in meters
        phi1 = radians(lat1) #convert to radians
        phi2 = radians(lat2) #convert to radians
        dPhi = radians(lat2 - lat1) #compute latitude delta
        dLambda = radians(lon2 - lon1) #compute longitude delta

        a = sin(dPhi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dLambda / 2) ** 2 #compute coefficient
        c = 2 * atan2(sqrt(a), sqrt(1 - a)) #compute coefficient
        distance = R * c #compute distance

        return distance
    
    def distanceToAirport(self, lat, lon):
        #compute the current distance from the aircraft to the airports reference position
        distanceToAirport = self.haversineDistance(lat, lon, self.airportParameters["AIRPORT_LATITUDE"], self.airportParameters["AIRPORT_LONGITUDE"])
        return distanceToAirport
    
    def connectToOgnServer(self, sock):
        #try to connect to to the ogn-decode TCP server and establish communication
        print(f"Connecting to {self.host}:{self.port}...")
        sock.connect((self.host, self.port))
        print("Connected. Waiting for OGN data...\n")

    def debounceFlightState(self, aircraftId, newState):
        '''
        The computed state of the aircraft can toggle between different states.
        In order to have a robust classification of the aircrafts state the "stableState" variable is used.
        The current state of the aircraft has to fullfill certian requirements
        to be considered as the current stable state of the aircraft.
        '''

        flgStableStateChanged = False #flag is set to True if the stable state of the aircraft is changed during debouncing
        now = self.time.getSystemTime() #get the current valid time
        aircraft = self.aircraftTracks[aircraftId] #get the data from the aircraft

        currentStableState = aircraft.get("stableState", "unknown") #get the currently set stable state of the aircraft
        prevStableState = aircraft.get("prevStableState", "unknown") #get the previosly set stable state of the aircraft
        prevPrevStableState = aircraft.get("prevPrevStableState", "unknown") #get the previosly previosly set stable state of the aircraft
        pendingStateState = aircraft.get("pendingState", {}).get("state", "unknown") #get the currently debouncing state
        pendingStateTimestamp = aircraft.get("pendingState", {}).get("timestamp", now) #get the time since the currently debouncing state first occured

        aircraft["state"] = newState #set the new state

        #debounce the new aircraft state
        if newState == currentStableState:
            #the currently stable state is the same as the new computed state => no change needed, no debouncing needed
            
            #prepare variables for the return dictionary
            newStableState = currentStableState
            newPrevStableState = prevStableState
            newPrevPrevStableState = prevPrevStableState
            newPendingState = {"state": pendingStateState,
                               "timestamp": pendingStateTimestamp}
        else:
            #the currently stable state is not the same as the new cumputed state => debouncing needed
            if newState == pendingStateState:
                #the new state already occured and is currently in the debouncing logic
                if now - pendingStateTimestamp >= timedelta(seconds=self.stateEstimationParameters["DEBOUNCE_TIME"]):
                    #the new state was present long enough to be considered stable => change all state variables
                    newStableState = newState #change the new stable state to the new and debounced state
                    newPrevStableState = currentStableState #set the previos stable state variable accordingly
                    newPrevPrevStableState = prevStableState #set the previos previos stable state to the previos stable state
                    newPendingState = {"state": pendingStateState,
                                       "timestamp": pendingStateTimestamp} #the information here will not be changed
                    
                    aircraft["stableState"] = newStableState #write to aircraft
                    aircraft["prevStableState"] = newPrevStableState #write to aircraft
                    aircraft["pendingState"] = newPendingState #write to aircraft
                    aircraft["prevPrevStableState"] = newPrevPrevStableState #write to aircraft
                    flgStableStateChanged = True #set flag

                    if newStableState == 'airborne':
                        aircraft["airborneSince"] = now

                    if newStableState == 'onGround':
                        aircraft["airborneSince"] = None

                else:
                    #the new state already occured but not long enough => wait for future data points
                    #prepare variables for the return dictionary
                    newStableState = currentStableState 
                    newPrevStableState = prevStableState
                    newPrevPrevStableState = prevPrevStableState
                    newPendingState = {"state": pendingStateState,
                                    "timestamp": pendingStateTimestamp}
            else:
                #the new state is currently not in the debouncing logic => start debouncing process
                #prepare variables for the return dictionary
                newStableState = currentStableState
                newPrevStableState = prevStableState
                newPrevPrevStableState = prevPrevStableState
                newPendingState = {"state": newState,
                                   "timestamp": now} #here the new state and its time is stored for debouncing with the next data point
                aircraft["pendingState"] = newPendingState

        #create return dictionary
        newStatesDict = self.createPlaceHolderAircraftStates()
        newStatesDict["flightState"] = newState
        newStatesDict["stableState"] = newStableState
        newStatesDict["prevStableState"] = newPrevStableState
        newStatesDict["prevPrevStableState"] = newPrevPrevStableState
        newStatesDict["pendingState"] = newPendingState

        return flgStableStateChanged, newStatesDict
    
    def detectFlightEvent(self, aircraftId, stateChanged = True):
        aircraft = self.aircraftTracks[aircraftId] #get the corresponding aircraft
        track = aircraft["track"] #get its data
        if not track:
            return #abort if no data is present    
        lastDataPoint = track[-1] #get the newest data point in the track data
        
        if stateChanged: #skip the computation if no change occured
            currentState = aircraft["stableState"] #get current state
            prevState = aircraft["prevStableState"] #get previos state
            prevPrevState = aircraft["prevPrevStableState"] #get previos previos state

            #take off detection
            if prevState == "onGround" and currentState == "airborne":
                #state transition path: onGround => airborne
                takeOff = True
            elif prevPrevState == "onGround" and prevState == "transitionAirGrnd" and currentState == "airborne":
                #state transition path: onGround => transitionAirGrnd => airborne
                takeOff = True
            else:
                takeOff = False

            #tpuch down detection
            if prevState == "airborne" and currentState == "onGround":
                #state transition path: airborne => onGround
                touchDown = True
            elif (prevPrevState in {"airborne", "landing", "transitionAirGrnd"} 
                  and prevState in {"landing", "transitionAirGrnd"}
                  and currentState == "onGround"):
                #state transition path: airborne/landing/transitionAirGrnd => landing/transitionAirGrnd => onGround
                touchDown = True
            else:
                touchDown = False 

            #store the information in a dictionary
            eventDict = self.createPlaceHolderFlightEvent() #get default dict
            eventDict["detectedTakeOff"] = takeOff
            eventDict["detectedTouchDown"] = touchDown

            #store the information in the aircraft overview
            aircraft["detectedTakeOff"] = takeOff
            aircraft["detectedTouchDown"] = touchDown
            lastDataPoint["flightEvent"] = eventDict

            #write data to database
            if touchDown:
                #store the last track points to the database
                self.writeDataToDatabase(aircraftId=aircraftId, track=track, category="arrival", duration=self.systemParameters["STORAGE_DURATION_ARRIVAL"])

            if takeOff:
                aircraft["aircraftDepartedAirport"] = True #set the flag, that the aircraft departed the airport
                aircraft["departureTime"] = self.time.getSystemTime() #set the departure time (time at which the take off was deteced)
                aircraft["storeDeparture"] = self.randomStorageFlag(self.systemParameters["PROBABILITY_OF_DEPATURE_STORAGE"]) #radnomly set the flag if this departure should be stored in the DB or not
                print(f"Departed: {aircraft["aircraftDepartedAirport"]} | {aircraft["departureTime"]} | StoreDep: {aircraft["storeDeparture"]}")      
        else:
            #no state change occured => no event can be detected
            #store the default information
            eventDict = self.createPlaceHolderFlightEvent()
            aircraft["detectedTakeOff"] = eventDict["detectedTakeOff"]
            aircraft["detectedTouchDown"] = eventDict["detectedTouchDown"]
            lastDataPoint["flightEvent"] = eventDict
   
    def stateMachine(self, aircraftId):
        '''
        The state machine classifies the aircrafts current state based on the newest data and a set of parameters
        The aircrafts states can be: 
            onGound: the aircraft is on the ground at the airport
            airborne: the aircraft is currently airborne
            transitionAirGrnd: the aircraft is in a transition state between onGound and airborne => take off or final approach
            unknown: the state could not be classified
        '''

        aircraft = self.aircraftTracks[aircraftId] #get the corresponding aircraft
        track = aircraft["track"] #get its data

        if not track:
            return #abort if no data is present
        
        lastDataPoint = track[-1] #get the newest data point in the track data

        altitude = lastDataPoint.get("alt", 0) #get the alitude value
        speed = lastDataPoint.get("speed", 0) #get the ground speed value
        distance = lastDataPoint.get("distanceToAirport", 0) #get the distance from the airport

        #determine limits for the state "onGround"
        minAlt = self.airportParameters["AIRPORT_ALTITUDE"] - self.stateEstimationParameters["ALTITUDE_TOLERANCE"]
        maxAlt = self.airportParameters["AIRPORT_ALTITUDE"] + self.stateEstimationParameters["ALTITUDE_TOLERANCE"]
        maxSpeed = self.stateEstimationParameters["MAX_ON_GROUND_SPEED"] / 3.6
        maxDist = self.stateEstimationParameters["ON_GROUND_POSITION_RADIUS"]

        #create flags, that correspond to the previosly set limits
        flgHeightGroundLevel = minAlt <= altitude <= maxAlt #aircrafts altitude can be considered on ground at the airport
        flgSpeedValidGound = speed <= maxSpeed #the aircrafts ground speed is so low that it can only be moving on the ground
        flgInsideAirportBoundaries = distance <= maxDist #its position is somewhere in the near vicinity of the airport

        if flgHeightGroundLevel and flgSpeedValidGound and flgInsideAirportBoundaries:
            flightState = "onGround" #if all three flags are true, then the aircraft is on the ground at the airport
        elif not flgHeightGroundLevel and not flgSpeedValidGound:
            flightState = "airborne" #if the altitude is outside the airports height and the velocity is higher then the maximum ground handling speed, then it is airbone
        elif flgHeightGroundLevel and flgInsideAirportBoundaries and not flgSpeedValidGound:
            flightState = "transitionAirGrnd" #if its speed is to high but it is in the airports vicinity and inside the height band, then it is landing or taking off
        else:
            #if none of those conditions is met, then further investigation is needed
            #for a better understanding of the aircrafts state, now averaged values are used for the classification

            now = self.time.getSystemTime() #get the current time
            windowStart = now - timedelta(seconds=self.stateEstimationParameters["STATE_DETECTION_TIME_WINDOW"]) #determine the start time of the time window
            recentPoints = [p for p in track if p["timestamp"] >= windowStart] #get all data points from the start time until now
            
            if len(recentPoints) >= self.stateEstimationParameters["MIN_NUMBER_DATA_POINTS_STATE_ESTIMATION"]: #ensure a minimum number of points is used
                #compute the average altidude, ground speed and distance
                avgAlt = mean(p["alt"] for p in recentPoints)
                avgSpeed = mean(p["speed"] for p in recentPoints)
                avgDist = mean(p["distanceToAirport"] for p in recentPoints)

                #compute the flags as before but with the average values
                flgAvrHeightGroundLevel = minAlt <= avgAlt <= maxAlt
                flgAvrSpeedValidGound = avgSpeed <= maxSpeed
                flgAvrInsideAirportBoundaries = avgDist <= maxDist

                #same classification logic as before
                if flgAvrHeightGroundLevel and flgAvrSpeedValidGound and flgAvrInsideAirportBoundaries:
                    flightState = "onGround"
                elif not flgAvrHeightGroundLevel and not flgAvrSpeedValidGound:
                    flightState = "airborne"
                elif flgAvrHeightGroundLevel and flgAvrInsideAirportBoundaries and not flgAvrSpeedValidGound:
                    flightState = "transitionAirGrnd"
                else:
                    flightState = "unknown" #if again no classification is met, then the aircrafts state is unknown
            else:
                flightState = "unknown" #if not enough points are present for a robust average, then the aircrafts state is unknown

        flgStableStateChanged, newStates = self.debounceFlightState(aircraftId, flightState) #debounce the computed state
        self.detectFlightEvent(aircraftId=aircraftId, stateChanged=flgStableStateChanged)

        #write all current states (state, stableState, ...) into the corresponding deque entry for storage
        lastDataPoint["aircraftStates"] = lastDataPoint.get("aircraftStates", self.createPlaceHolderAircraftStates())
        lastDataPoint["aircraftStates"] = newStates 

    def removeOldTracks(self):
        '''
        Remove all data points that are older then the maximum set time.
        The additional data is not necessary for the landing detection.
        Free up valuable RAM.
        '''
        now = self.time.getSystemTime() #get current time
        cutoff = now - timedelta(seconds=self.systemParameters["STORAGE_DURATION_SECONDS"]) #cutoff time, all older message will be deleted
        for aircraftId, data in list(self.aircraftTracks.items()):
            track = data["track"]
            while track and track[0]["timestamp"] < cutoff: #if data is to old
                track.popleft() #delete data
            if not track:
                del self.aircraftTracks[aircraftId] #delete aircraft entry when no data points are left

    def airborneDataWriteDetection(self):
        '''
        This function writes the departure data into the DB if the deparure was set to be stored in the DB.
        This function also decides if in-flight data is written to the DB or not and handles the process
        '''
        for aircraftId, data in list(self.aircraftTracks.items()):
            aircraft = self.aircraftTracks[aircraftId]
            now = self.time.getSystemTime() #get current time

            #write departure data
            if aircraft.get("storeDeparture"): #if the flag was set to store the departure data
                #check if the set amount of time since the departure has passed 
                if (now - aircraft['departureTime']).total_seconds() >= self.systemParameters['STORAGE_DURATION_AFT_DEPARTURE']:
                    aircraft['storeDeparture'] = False #set the flag to false to aviod storin the data multiple times
                    aircraft['lastTimeDataWrittenToDB'] = now #set the time of storage

                    #compute time window boundaries
                    lowerBound = aircraft['departureTime'] - timedelta(seconds=self.systemParameters['STORAGE_DURATION_PRE_DEPARTURE'])
                    upperBound = aircraft['departureTime'] + timedelta(seconds=self.systemParameters['STORAGE_DURATION_AFT_DEPARTURE'])
                    
                    track = data["track"]
                    recentPoints = [point for point in track if lowerBound <= point['timestamp'] <= upperBound] #extract the points inside the time window

                    if recentPoints:
                        #store data in database
                        saveTrack(recentPoints, dbPath=self.systemParameters['DATABASE_PATH'], category="departure")
                        print(f"\n[DB] Dumping {len(recentPoints)} points for {aircraftId} to database as category departure.")
        
            # write in-flight data
            if aircraft.get("stableState") != "airborne":
                return  #only continue when the aircraft is airborne
            
            if aircraft.get('inFlightDataStored'):
                return #continue if the in flight data was already stored

            airborneSince = aircraft.get("airborneSince") #get the time since the aircraft is airborne
            lastWritten = aircraft.get("lastTimeDataWrittenToDB") #get the time when the last data was dumped to the DB
            
            if not isinstance(airborneSince, datetime):
                return #check if a valid airborne time was recieved

            #get system parameters
            minAirborne = self.systemParameters["MINIMUM_TIME_AIRBORNE"]
            storeInterval = self.systemParameters["STORAGE_DURATION_IN_FLIGHT"]

            #time since the aircraft is airborne
            timeAirborne = (now - airborneSince).total_seconds()

            #do not continue if the aircraft is not airborne long enough
            if timeAirborne < (minAirborne + storeInterval):
                return

            #check if lastWritten was set, if so then how much time elapsed since then
            if isinstance(lastWritten, datetime):
                timeSinceLastWrite = (now - lastWritten).total_seconds()
                if timeSinceLastWrite < storeInterval:
                    return

            #if all conditions are met, then randomly decide to store the data
            if self.randomStorageFlag(probability=self.systemParameters['PROBABILITY_OF_IN_FLIGHT_STORAGE']):
                aircraft["lastTimeDataWrittenToDB"] = now
                aircraft["inFlightDataStored"] = True
                self.writeDataToDatabase(aircraftId=aircraftId, track=data["track"],category='inFlight', duration=storeInterval)
                    
    def writeDataToDatabase(self, aircraftId, track, category, duration):
        #Store the track data after the landing was detected into database
        now = self.time.getSystemTime()
        cutoff = now - timedelta(seconds=duration) #compute cutoff time

        recentPoints = [point for point in track if point['timestamp'] >= cutoff] #extract all points in the time window

        if recentPoints:
            #store data into the database
            saveTrack(recentPoints, dbPath=self.systemParameters['DATABASE_PATH'], category=category)
            print(f"\n[DB] Dumping {len(recentPoints)} points for {aircraftId} to database as category {category}.")

        if category == 'inFlight':
            from dataPlotter import plotAltSpeedAndStates
            plotAltSpeedAndStates(recentPoints)
        
    def processMessageLine(self, line):
        #used when the system runs in synchrone mode and recieves data from ogn-decode
        parsed = self.parseOgnLine(line)
        if not parsed:
            return

        aircraftId = parsed["aircraft"] #get the aircraft ID from message
        self.aircraftTracks[aircraftId]["track"].append(parsed) #append the recieved data
        self.stateMachine(aircraftId) #compute the state of the aircraft

    def printInfos(self):
        #print basic infos to the terminal
        print("------------------------------------")
        for aircraftId, trackInfo in self.aircraftTracks.items():
            if trackInfo["track"]:
                lastPosition = trackInfo["track"][-1]
                print(f"✈ {aircraftId} | "
                    f"State: {trackInfo['flightState']} | "
                    f"StableState: {trackInfo['stableState']} | "
                    f"PrevStableState: {trackInfo['prevStableState']} | "
                    f"Pos: {lastPosition['lat']:.5f}, {lastPosition['lon']:.5f} | "
                    f"Alt: {lastPosition['alt']}m | "
                    f"Spd: {lastPosition['speed']:.1f}m/s | "
                    f"Last Package: {(self.time.getSystemTime() - lastPosition['timestamp']).total_seconds():.0f}s | "
                    f"OGNtime: {(lastPosition['time'])} | "
                    f"SysTime: {(lastPosition['timestamp'])}")
        print("------------------------------------")
       
    def processMessageDict(self, data):
        #this is used, when the system operates in asynchrone mode and the data is already present in a dictionary
        try:
            #get and convert recieved data
            data["recvTime"] = self.safeFloat(data.get("recvTime"))
            data["freq"] = self.safeFloat(data.get("frequency"))
            data["time"] = self.safeInt(data.get("ognTime"))
            data["lat"] = self.safeFloat(data["lat"])
            data["lon"] = self.safeFloat(data["lon"])
            data["alt"] = self.safeInt(data["altitude"])
            data["vs"] = self.safeFloat(data["climbRate"])
            data["speed"] = self.safeFloat(data["groundSpeed"])
            data["track"] = self.safeFloat(data["track"])
            data["turnRate"] = self.safeFloat(data["turnRate"])
            data["snr"] = self.safeFloat(data.get("snr"))
            data["rssi"] = self.safeFloat(data.get("rssi"))
            data["errCount"] = self.safeInt(data.get("errCount"))
            data["eStatus"] = self.safeInt(data.get("eStatus"))
            data["distance"] = self.safeFloat(data.get("distance"))
            data["bearing"] = self.safeFloat(data.get("bearing"))
            data["elevAngle"] = self.safeFloat(data.get("elevAngle"))
            data["relayed"] = bool(data.get("relayed", False))
            data["reducedDataConfidence"] = data.get("flagged") == "!"
            data["distanceToAirport"] = self.distanceToAirport(data["lat"], data["lon"])

            if not self.systemParameters["REALTIME_MODE"]:
                self.time.setSystemTimeAsynchronousMode(asyntime=data["time"])
            data["timestamp"] = self.time.getSystemTime() #set time stamp
            data["aircraftStates"] = self.createPlaceHolderAircraftStates()
            data["flightEvents"] = self.createPlaceHolderFlightEvent()

        except Exception as e:
            print(f"Fehler beim Verarbeiten der OGN-Daten: {e}")
            return

        aircraftId = data["aircraft"] #get the aircraft ID from the recieved data
        self.aircraftTracks[aircraftId]["track"].append(data) #store data in deque
        self.stateMachine(aircraftId=aircraftId) #compute the state of the aircraft


    def monitorSignalReception(self):
        '''
        Finite state machine for the signal reception status.
        Classifies if messages are recieved normal or if the time since the last message is to long
        '''
        now = self.time.getSystemTime() #get current time
        for aircraftId, data in list(self.aircraftTracks.items()): #loop over aircrafts
            track = data["track"] #get track data
            if not track:
                continue #now data available

            lastTimestamp = track[-1]['timestamp'] #get last timestamp

            heartbeatCutoff = now - timedelta(seconds=self.signalReceptionParameters["AIRCRAFT_HEARBEAT_MISSING_TIME"]) #cutoff time for missing heartbeat
            lostCutoff = now - timedelta(seconds=self.signalReceptionParameters["AIRCRAFT_LOST_TIME"]) #cutoff time for lost aircraft

            if lastTimestamp < lostCutoff:
                newReceptionState = "aircraftLost" #cutoff time exceeded, aircraft lost
            elif lastTimestamp < heartbeatCutoff:
                newReceptionState = "heartbeatMissing" #cutoff time exceeded, aircraft not lost but missing
            else:
                newReceptionState = "normal" #normal

            if newReceptionState != data.get("receptionState", "normal"):
                prevState = data["receptionState"]
                data["receptionState"] = newReceptionState #write the new state
    
    def systemLoop(self):
        #Loop that executes while no new message is processed => maintanance
        if self.systemParameters["REALTIME_MODE"]:
            self.time.setSystemTime() #set system time
        self.removeOldTracks() #remove old track data
        self.monitorSignalReception() #monitor the signal reception from all aircrafts

    def runClient(self):
        '''
        run the client in a loop and constantly process the from ogn-decode recieved data.
        if no message is recieved then perform the needed realtime data maintanance
        '''
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            self.connectToOgnServer(sock) #connect to the server
            buffer = "" #buffer for the recieved messages
            
            try:
                while True:
                    ready, _, _ = select.select([sock], [], [], 0)

                    if ready:
                        data = sock.recv(4096) #get data from socket
                        if not data:
                            print("Connection closed by server.")
                            break

                        buffer += data.decode(errors='ignore') #store the recieved data in a buffer
                        processedCount = 0  #Counter for number of recieved messages

                        #process the messages in the buffer, a limit to how many messages can be processed is enforced
                        while '\n' in buffer and processedCount < self.systemParameters["MAXIMUM_MESSAGES_IN_INPUT_BUFFER"]:
                            processedCount += 1
                            if self.systemParameters["REALTIME_MODE"]:
                                self.time.setSystemTime() #set system time

                            line, buffer = buffer.split('\n', 1) #get a line from the buffer
                            line = line.strip()
                            if not line or not line[0].isdigit(): #if the revieved line is not an aircraft data package then continue
                                continue
                            
                            self.processMessageLine(line) #process the recieved data
                            self.removeOldTracks() #remove old data from the RAM

                        if '\n' in buffer:
                            buffer = '' #delete remaining buffer contents if too many messages were processed

                    else:
                        #this executes, when no new messages are processed
                        self.systemLoop() #call defined maintanance functions
  

            except KeyboardInterrupt:
                print("\nClient terminated by user.")


if __name__ == "__main__":
    client = OgnClient()
    client.runClient()
