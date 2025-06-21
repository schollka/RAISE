# RAISE – Runway Approach Identification for Silent Entries

TODO


# 1. Gruppe setzen: NGINX = www-data
sudo chown -R pi:www-data /home/pi/RAISE/web

# 2. Verzeichnisrechte: pi darf alles, www-data darf rein
sudo find /home/pi/RAISE/web -type d -exec chmod 750 {} \;

# 3. Dateirechte: pi darf schreiben, www-data darf lesen
sudo find /home/pi/RAISE/web -type f -exec chmod 640 {} \;

# 4. NGINX darf den Pfad betreten (aber keine anderen Inhalte sehen)
sudo chmod o+x /home
sudo chmod o+x /home/pi
sudo chmod o+x /home/pi/RAISE




server {
    listen 80;
    server_name _;

    root /home/pi/RAISE/web;
    index index.html;

    location /api/ {
        proxy_pass http://127.0.0.1:8181;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}



# Link in sites-enabled erstellen (falls nicht schon aktiv)
sudo ln -s /etc/nginx/sites-available/raise /etc/nginx/sites-enabled/raise

# Konfiguration prüfen und reload
sudo nginx -t
sudo systemctl reload nginx


#Python 3.10
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev


# VENV
pyhton <3.11 = > 3.10.11
python3.10 -m venv RAISE 
source ./RAISEenv/Scripts/activate
pip install --upgrade pip
pip install -r venvRequirements.txt


WIN: python.exe -m pip install --upgrade pip


## License and Data Attribution

### Map Data – OpenStreetMap

The map tiles and geodata used in this project are provided by [OpenStreetMap](https://www.openstreetmap.org/).

- © OpenStreetMap contributors
- License: [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/)
- Tile service (default): [tile.openstreetmap.org](https://tile.openstreetmap.org/)
- Attribution is shown in the web interface in accordance with [OSM's attribution requirements](https://www.openstreetmap.org/copyright).

If you plan to run this application in a high-traffic or commercial setting, consider hosting your own tile server or using a third-party tile provider to reduce load on the public OpenStreetMap infrastructure.

### Callsign Lookup – DDB (Distributed DataBase)

Callsigns and aircraft metadata are resolved based on the DDB database provided by the Open Glider Network (OGN) community:

- Source: [https://ddb.glidernet.org/download/](https://ddb.glidernet.org/download/)
- Maintained by: Open Glider Network and volunteer contributors
- Intended for: Non-commercial, aviation-related use only
- License: Use permitted for non-commercial purposes; redistribution and usage must comply with the conditions described on [https://ddb.glidernet.org/about.html](https://ddb.glidernet.org/about.html)

Aircraft identifiers in this project are matched using the downloaded CSV files containing FLARM or ICAO hexadecimal IDs with associated metadata (e.g., callsign, aircraft type, competition ID).

### General Notes

This project is intended for local, non-commercial deployment, such as in gliding clubs or airfield operations rooms. It does not provide guaranteed accuracy, and aircraft data is presented as-is from the decoded data stream.





