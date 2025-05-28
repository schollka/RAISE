import socket
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone, time
from statistics import mean
from math import radians, sin, cos, sqrt, atan2
import select
from auxillaryFunctions import safeFloat, safeInt
from databankHandler import saveTrack

class OgnClient:
    # System parameters
    BUFFER_SECONDS = 300
    LANDING_LOOKBACK_SECONDS = 15
    AIRPORT_ALTITUDE = 266
    ALTITUDE_TOLERANCE = 25
    MAX_ON_GROUND_SPEED = 25 / 3.6
    STATE_DETECTION_TIME_WINDOW = 30
    AIRPORT_LATITUDE = 49.002222
    AIRPORT_LONGITUDE = 9.086389
    ON_GROUND_POSITION_RADIUS = 750
    MAXIMUM_MESSAGES_IN_BUFFER = 10
    DEBOUNCE_TIME = 5
    AIRCRAFT_LOST_TIME = 30
    AIRCRAFT_HEARBEAT_MISSING_TIME = 10
    REALTIME_MODE = False
    NUMBER_OF_DATA_POINTS_FOR_STATE_ESTIMATION = 5

    def __init__(self, host="127.0.0.1", port=50001):
        self.host = host
        self.port = port
        self.time = self.TimeManager()

        #Initialize aircraft tracks dictionary
        self.aircraftTracks = defaultdict(lambda: {
            "track": deque(maxlen=100000), #OGN message data

            #aircraft states
            "flightState": "unknown", #current calculated aicraft state
            "flightSubState": None, #substate for the "airborne" flightState
            "stableState": "unknown", #as stable determined aicraft state
            "prevStableState": "unknown", #previos stable aicraft state
            "lastStateChange": self.time.getSystemTime(), #time of last state change
            "lastAirborneTime": None, #last time when the aicraft was stable airborne

            #meta data
            "landedSaved": False, #flag if track data was saved into database
            "hasBeenAirborne": False, #flag if aircraft hast been airborne before

            #signal states
            "receptionState": "normal" #state of the signal reception
        })

    class TimeManager:
        def __init__(self):
            self.time = datetime.now(timezone.utc)
        
        def setSystemTime(self):
            self.time = datetime.now(timezone.utc)

        def setSystemTimeAsynchronousMode(self, asyntime: int, referenceDate: datetime = None):
            #this functions demodulates a timestamp in the format HHMMSS into a vaild datetime system time
            hh = asyntime // 10000
            mm = (asyntime // 100) % 100
            ss = asyntime % 100

            #Specify the date
            if referenceDate is None:
                referenceDate = datetime.now(timezone.utc)

            #Combine date and time
            asynSysTime = datetime.combine(referenceDate.date(), time(hh, mm, ss), tzinfo=timezone.utc)
            self.time = asynSysTime #set asynchrone system time

        def getSystemTime(self):
            return self.time

    '''
    Regex expression for OGN message decode
    Example message: 0.936sec:868.370MHz: 1:2:DD9A70 142236: [ +49.00106,  +9.07859]deg   268m  +0.0m/s   0.0m/s 180.0deg  +0.0deg/s __1 04x04m O :01f__-30.02kHz 42.8/52.5dB/0  0e 0.1km 285.8deg -4.6deg + !
    Message blocks: 
        - recieving time of ogn-decode
        - frequency
        - network ID level
        - aicraft ID
        - GNSS time
        - position [Lat, Long]
        - GPS altitude
        - vertical speed
        - ground speed
        - heading
        - turn rate
        - aircraft type
        - aicraft dimension
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
            if not self.REALTIME_MODE:
                self.time.setSystemTimeAsynchronousMode(asyntime=d["time"]) #create a timestamp based on the time in the recieved message
            d["timestamp"] = self.time.getSystemTime()
            d["aicraftStates"] = {}
        except Exception as e:
            print(f"OGN message parsing error: {e}")
            return None
        return d

    @staticmethod
    def haversineDistance(lat1, lon1, lat2, lon2):
        #Tool for the calculation of the distance between two points on earths surface
        R = 6371000  # Earth radius in meters
        phi1 = radians(lat1) #convert to radians
        phi2 = radians(lat2) #convert to radians
        dPhi = radians(lat2 - lat1) #compute latitude delta
        dLambda = radians(lon2 - lon1) #compute longitude delta

        a = sin(dPhi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dLambda / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c #compute distance
        return distance
    
    def distanceToAirport(self, lat, lon):
        distanceToAirport = self.haversineDistance(lat, lon, self.AIRPORT_LATITUDE, self.AIRPORT_LONGITUDE)
        return distanceToAirport
    
    def connectToOgnServer(self, sock):
        print(f"Connecting to {self.host}:{self.port}...")
        sock.connect((self.host, self.port))
        print("Connected. Waiting for OGN data...\n")

    def debounceStateMachine(self, aircraftID, stateKey, stateValue):
        print("a")


    def stateMachine(self, aircraftId):
        aircraft = self.aircraftTracks[aircraftId]
        track = aircraft["track"]

        if not track:
            return
        
        lastDataPoint = track[-1]

        altitude = lastDataPoint.get("alt", 0)
        speed = lastDataPoint.get("speed", 0)
        distance = lastDataPoint.get("distanceToAirport", 0)

        minAlt = self.AIRPORT_ALTITUDE - self.ALTITUDE_TOLERANCE
        maxAlt = self.AIRPORT_ALTITUDE + self.ALTITUDE_TOLERANCE
        maxSpeed = self.MAX_ON_GROUND_SPEED
        maxDist = self.ON_GROUND_POSITION_RADIUS

        flgHeightGroundLevel = minAlt <= altitude <= maxAlt
        flgSpeedValidGound = speed <= maxSpeed
        flgInsideAirportBoundaries = distance <= maxDist

        if flgHeightGroundLevel and flgSpeedValidGound and flgInsideAirportBoundaries:
            flightState = "onGround"
        elif not flgHeightGroundLevel and not flgSpeedValidGound:
            flightState = "airborne"
        elif flgHeightGroundLevel and flgInsideAirportBoundaries and not flgSpeedValidGound:
            flightState = "transitionAirGrnd"
        else:
            now = self.time.getSystemTime()
            windowStart = now - timedelta(seconds=self.STATE_DETECTION_TIME_WINDOW)
            recentPoints = [p for p in track if p["timestamp"] >= windowStart]
            
            avgAlt = mean(p["alt"] for p in recentPoints)
            avgSpeed = mean(p["speed"] for p in recentPoints)
            avgDist = mean(p["distanceToAirport"] for p in recentPoints)

            flgAvrHeightGroundLevel = minAlt <= avgAlt <= maxAlt
            flgAvrSpeedValidGound = avgSpeed <= maxSpeed
            flgAvrInsideAirportBoundaries = avgDist <= maxDist

            if flgAvrHeightGroundLevel and flgAvrSpeedValidGound and flgAvrInsideAirportBoundaries:
                flightState = "onGround"
            elif not flgAvrHeightGroundLevel and not flgAvrSpeedValidGound:
                flightState = "airborne"
            elif flgAvrHeightGroundLevel and flgAvrInsideAirportBoundaries and not flgAvrSpeedValidGound:
                flightState = "transitionAirGrnd"
            else:
                flightState = "unknown"

        aircraft["flighState"] = flightState
        lastDataPoint["aircraftStates"] = lastDataPoint.get("aircraftStates", {})
        lastDataPoint["aircraftStates"]["flightState"] = flightState

    def detectFlightState(self, aircraftId):
        aircraft = self.aircraftTracks[aircraftId]
        track = aircraft["track"]

        if not track:
            return "unknown"

        now = self.time.getSystemTime()
        windowStart = now - timedelta(seconds=self.STATE_DETECTION_TIME_WINDOW)
        recentPoints = [p for p in track if p["timestamp"] >= windowStart]

        if len(recentPoints) < self.NUMBER_OF_DATA_POINTS_FOR_STATE_ESTIMATION:
            if aircraft["stableState"] == "onGround":
                return "onGround"
            else:
                return "unknown"

        avgAlt = mean(p["alt"] for p in recentPoints)
        avgSpeed = mean(p["speed"] for p in recentPoints)
        avgLat = mean(p["lat"] for p in recentPoints)
        avgLon = mean(p["lon"] for p in recentPoints)
        avgDist = mean(p["distanceToAirport"] for p in recentPoints)

        minAltThres = self.AIRPORT_ALTITUDE - self.ALTITUDE_TOLERANCE
        maxAltThres = self.AIRPORT_ALTITUDE + self.ALTITUDE_TOLERANCE

        if minAltThres <= avgAlt <= maxAltThres and avgSpeed <= self.MAX_ON_GROUND_SPEED: # and distanceToAirport <= self.ON_GROUND_POSITION_RADIUS
            return "onGround"
        elif avgSpeed > self.MAX_ON_GROUND_SPEED:
            return "airborne"
        else:
            return "unknown"


    def removeOldTracks(self):
        '''
        Remove all data points that are older then the maximum set time.
        The additional data is not necessary for the landing detection.
        Free up valuable RAM.
        '''
        now = self.time.getSystemTime() #current time
        cutoff = now - timedelta(seconds=self.BUFFER_SECONDS) #cutoff time, all older message will be deleted
        for aircraftId, data in list(self.aircraftTracks.items()):
            track = data["track"]
            while track and track[0]["timestamp"] < cutoff:
                track.popleft() #delete data
            if not track:
                del self.aircraftTracks[aircraftId] #delete aicraft entry when no data points are left

    def dumpDataToDatabase(self, aircraftId, track):
        dbPath = "flightData.db"
        saveTrack(track, dbPath)
        print(f"\n[DB] Dumping {len(track)} points for {aircraftId} to database.")
        # TODO: Replace with actual DB logic

    def debounceState(self, aircraftId, newState):
        entry = self.aircraftTracks[aircraftId]
        now = self.time.getSystemTime()

        # Falls sich der Zustand geändert hat (z. B. onGround → airborne)
        if newState != entry["stableState"]:
            timeInCurrentState = now - entry["lastStateChange"]

            if timeInCurrentState >= timedelta(seconds=self.DEBOUNCE_TIME):
                entry["prevStableState"] = entry["stableState"]
                entry["stableState"] = newState
                entry["lastStateChange"] = now

                # Wenn stabiler Zustand jetzt 'airborne' ist → Zeit merken
                if newState == "airborne":
                    entry["lastAirborneTime"] = now

                if entry["track"]:
                    entry["track"][-1]["stableFlightState"] = newState
            else:
                if entry["track"]:
                    entry["track"][-1]["stableFlightState"] = entry["stableState"]

                return True

        else:
            # Zustand gleich geblieben → Zeitstempel aktualisieren
            entry["lastStateChange"] = now

            if entry["track"]:
                entry["track"][-1]["stableFlightState"] = newState

        return False

    
    def detectFlightSubState(self, track):
        """
        Platzhalter für spätere ML oder Regelbasierte Klassifikation.
        """
        return "cruise"  # Dummy – später ersetzt durch echte Logik oder ML

    
    def updateFlightState(self, aircraftId):
        aircraft = self.aircraftTracks[aircraftId]
        track = aircraft["track"]

        currentState = self.detectFlightState(aircraftId)
        aircraft["flightState"] = currentState
        if track:
            track[-1]["flightState"] = currentState

        stableChanged = self.debounceState(aircraftId, currentState)
        if not stableChanged:
            return

        prevState = aircraft["prevStableState"]
        newState = aircraft["stableState"]

        #print(f"[FSM-DEBUG] {aircraftId}: prev={prevState}, new={newState}")

        # 🛫 Takeoff-Erkennung
        if newState == "airborne":
            if prevState == "onGround" or (
                prevState == "unknown" and aircraft.get("prevStableState") == "onGround"
            ):
                aircraft["hasBeenAirborne"] = True
                aircraft["landedSaved"] = False
                aircraft["flightSubState"] = "takeoff"
                print(f"[FSM] {aircraftId}: takeoff detected")

        # 🛬 Landung robust erkennen, auch nach unknown
        elif newState == "onGround":
            lastAirborne = aircraft.get("lastAirborneTime")
            if lastAirborne and (self.time.getSystemTime() - lastAirborne) < timedelta(minutes=15):
                if aircraft["hasBeenAirborne"] and not aircraft["landedSaved"]:
                    self.dumpDataToDatabase(aircraftId, track)
                    aircraft["landedSaved"] = True
                    aircraft["flightSubState"] = None
                    duration = (self.time.getSystemTime() - lastAirborne).total_seconds()
                    print(f"[FSM] {aircraftId}: landed after {duration:.0f}s airborne")

        # ✋ alle anderen Zustände
        else:
            aircraft["flightSubState"] = None

        # 🧠 Substatus (Platzhalter)
        if newState == "airborne":
            aircraft["flightSubState"] = self.detectFlightSubState(track)
            if track:
                track[-1]["flightSubState"] = aircraft["flightSubState"]

    
    def processMessageLine(self, line):
        parsed = self.parseOgnLine(line)
        if not parsed:
            return

        aircraftId = parsed["aircraft"]
        self.aircraftTracks[aircraftId]["track"].append(parsed)
        self.updateFlightState(aircraftId)

    def printInfos(self):
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
        try:
            # Sicher konvertieren
            data["recvTime"] = safeFloat(data.get("recvTime"))
            data["freq"] = safeFloat(data.get("frequency"))
            data["time"] = safeInt(data.get("ognTime"))
            data["lat"] = safeFloat(data["lat"])
            data["lon"] = safeFloat(data["lon"])
            data["alt"] = safeInt(data["altitude"])
            data["vs"] = safeFloat(data["climbRate"])
            data["speed"] = safeFloat(data["groundSpeed"])
            data["track"] = safeFloat(data["track"])
            data["turnRate"] = safeFloat(data["turnRate"])
            data["snr"] = safeFloat(data.get("snr"))
            data["rssi"] = safeFloat(data.get("rssi"))
            data["errCount"] = safeInt(data.get("errCount"))
            data["eStatus"] = safeInt(data.get("eStatus"))
            data["distance"] = safeFloat(data.get("distance"))
            data["bearing"] = safeFloat(data.get("bearing"))
            data["elevAngle"] = safeFloat(data.get("elevAngle"))
            data["relayed"] = bool(data.get("relayed", False))
            data["reducedDataConfidence"] = data.get("flagged") == "!"
            data["distanceToAirport"] = self.distanceToAirport(data["lat"], data["lon"])

            if not self.REALTIME_MODE:
                self.time.setSystemTimeAsynchronousMode(asyntime=data["time"])
            data["timestamp"] = self.time.getSystemTime()
            data["aicraftStates"] = {}

        except Exception as e:
            print(f"Fehler beim Verarbeiten der OGN-Daten: {e}")
            return

        aircraftId = data["aircraft"]
        self.aircraftTracks[aircraftId]["track"].append(data)
        self.updateFlightState(aircraftId)
        self.stateMachine(aircraftId=aircraftId)


    def monitorSignalReception(self):
        '''
        Finite state machine for the signal reception status.
        Classifies if messages are recieved normal or if the time since the last message is to long
        '''
        now = self.time.getSystemTime() #current time
        for aircraftId, data in list(self.aircraftTracks.items()): #loop over aircrafts
            track = data["track"] #get track data
            if not track:
                continue #now data available

            lastTimestamp = track[-1]['timestamp'] #get last timestamp

            heartbeatCutoff = now - timedelta(seconds=self.AIRCRAFT_HEARBEAT_MISSING_TIME) #cutoff time for missing heartbeat
            lostCutoff = now - timedelta(seconds=self.AIRCRAFT_LOST_TIME) #cutoff time for lost aircraft

            if lastTimestamp < lostCutoff:
                newReceptionState = "aircraftLost" #cutoff time exceeded, aicraft lost
            elif lastTimestamp < heartbeatCutoff:
                newReceptionState = "heartbeatMissing" #cutoff time exceeded, aicraft not lost but missing
            else:
                newReceptionState = "normal" #normal

            if newReceptionState != data.get("receptionState", "normal"):
                prevState = data["receptionState"]
                data["receptionState"] = newReceptionState #write the new state
    
    def systemLoop(self):
        #Loop that executes while no new message is processed => maintanance
        if self.REALTIME_MODE:
            self.time.setSystemTime() #set system time
        self.removeOldTracks() #remove old track data
        self.monitorSignalReception() #monitor the signal reception from all aircrafts

    def runClient(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            self.connectToOgnServer(sock)
            buffer = ""
            
            try:
                while True:
                    ready, _, _ = select.select([sock], [], [], 0)

                    if ready:
                        data = sock.recv(4096)
                        if not data:
                            print("Connection closed by server.")
                            break

                        buffer += data.decode(errors='ignore')
                        processedCount = 0  #Counter for number of recieved messages

                        while '\n' in buffer and processedCount < self.MAXIMUM_MESSAGES_IN_BUFFER:
                            processedCount += 1
                            if self.REALTIME_MODE:
                                self.time.setSystemTime() #set system time

                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            if not line or not line[0].isdigit():
                                continue
                            
                            self.processMessageLine(line)
                            self.removeOldTracks()

                        if '\n' in buffer:
                            buffer = '' #delete remaining buffer contents

                    else:
                        #this executes, when no new messages are processed
                        self.systemLoop() #call defined maintanance functions
  

            except KeyboardInterrupt:
                print("\nClient terminated by user.")


if __name__ == "__main__":
    client = OgnClient()
    client.runClient()
