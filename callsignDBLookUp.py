'''
              __|__
       --------(_)--------       
              RAISE         
 Runway Approach Identification for Silent Entries
------------------------------------------------------
    Tracking • Prediction • Silent Entry Detection     
------------------------------------------------------

callsign database handling and look up script
'''

import csv
import threading
import requests
import random
from datetime import datetime, timezone

DDB_URL = "https://ddb.glidernet.org/download/"

class DDBLookup:
    def __init__(self, refreshIntervall):
        self.lookup = {}
        self.lastLoaded = None
        self.refreshIntervall = refreshIntervall
        self._loadDdb()
        self._scheduleReload(refreshIntervall)

    def _loadDdb(self):
        try:
            response = requests.get(DDB_URL, timeout=30)
            response.raise_for_status()

            for delimiter in [',', ';']:
                lines = response.text.splitlines()
                if lines and lines[0].startswith("#"):
                    lines[0] = lines[0][1:]
                reader = csv.DictReader(lines, delimiter=delimiter)

                if 'DEVICE_ID' in reader.fieldnames and 'REGISTRATION' in reader.fieldnames:
                    print(f"[Callsign DB] Using delimiter '{delimiter}' with fields: {reader.fieldnames}")
                    break
            else:
                print(f"[Callsign DB] No valid column names found: {reader.fieldnames}")
                self._retryLoadDdbInBackground()
                return

            lookup_tmp = {}
            for row in reader:
                icao = row.get('DEVICE_ID', '').strip().strip("'").upper()
                tracked = row.get('TRACKED', 'Y').strip().strip("'").upper()
                callsign = row.get('REGISTRATION', '').strip().strip("'")
                if icao:
                    lookup_tmp[icao] = {
                        'callsign': callsign if tracked == 'Y' else None,
                        'tracked': tracked == 'Y'
                    }

            self.lookup = lookup_tmp
            self.lastLoaded = datetime.now(timezone.utc)
            print(f"[Callsign DB] Loaded {len(self.lookup)} entries at {self.lastLoaded}")

        except Exception as e:
            print(f"[Callsign DB] Failed to load: {e}")
            self._retryLoadDdbInBackground()

    def _retryLoadDdbInBackground(self):
        retryIn = random.randint(5, 15)
        print(f"[Callsign DB] Retry scheduled in {retryIn} seconds")
        threading.Timer(retryIn, self._loadDdb).start()

    def _scheduleReload(self, refreshIntervall):
        threading.Timer(refreshIntervall, lambda: self._reload(refreshIntervall)).start()

    def _reload(self, refreshIntervall):
        print("[Callsign DB] Scheduled reload...")
        self._loadDdb()
        self._scheduleReload(refreshIntervall)

    def getCallsign(self, icao: str) -> str:
        entry = self.lookup.get(icao.upper())
        if not entry:
            return "XXXXX"  
        if not entry['tracked'] or not entry['callsign']:
            return "XXXXX"
        return entry['callsign']
