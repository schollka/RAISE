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
from datetime import datetime

DDB_URL = "https://ddb.glidernet.org/download/"
RELOAD_INTERVAL_SECONDS = 43200  # 12h

class DDBLookup:
    def __init__(self, refreshIntervall):
        self.lookup = {}
        self.lastLoaded = None
        self._loadDdb()
        self._scheduleReload(refreshIntervall)

    def _loadDdb(self):
        try:
            response = requests.get(DDB_URL, timeout=30)
            response.raise_for_status()

            for delimiter in [',', ';']:
                lines = response.text.splitlines()
                if lines and lines[0].startswith("#"):
                    lines[0] = lines[0][1:]  # '#' am Anfang entfernen
                reader = csv.DictReader(lines, delimiter=delimiter)

                if 'DEVICE_ID' in reader.fieldnames and 'REGISTRATION' in reader.fieldnames:
                    print(f"[DDB] Using delimiter '{delimiter}' with fields: {reader.fieldnames}")
                    break
            else:
                print(f"[DDB] Keine gültigen Spaltennamen in Header gefunden: {reader.fieldnames}")
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
            self.lastLoaded = datetime.utcnow()
            print(f"[DDB] Loaded {len(self.lookup)} entries at {self.lastLoaded}")

        except Exception as e:
            print(f"[DDB] Failed to load: {e}")

    def _scheduleReload(self, refreshIntervall):
        threading.Timer(refreshIntervall, self._reload).start()

    def _reload(self):
        self._loadDdb()
        self._scheduleReload()

    def getCallsign(self, icao: str) -> str:
        entry = self.lookup.get(icao.upper())
        if not entry:
            return "XXXXX"  
        if not entry['tracked'] or not entry['callsign']:
            return "XXXXX"
        return entry['callsign']
