# RAISE – Runway Approach Identification for Silent Entries
RAISE – Runway Approach Identification for Silent Entries is an open-source project aimed at improving airspace monitoring at uncontrolled airfields. It processes real-time position data from aircraft equipped with FLARM devices and displays this data on a web-based map interface.

The system is designed to autonomously record approach patterns and use them to train a machine learning model. This model identifies aircraft in the vicinity of the airfield that are likely about to land. Aircraft currently on approach are visually highlighted to draw attention.

RAISE aims to improve safety by giving ground personnel a reliable tool to monitor aircraft activity near the airfield, with a special focus on identifying and highlighting landing approaches. In cases where radio communication is disrupted, RAISE acts as a fallback system to ensure that situational awareness on the ground is maintained.

## Use Case and System Requirements

### System Overview

RAISE integrates into existing Open Glider Network (OGN) receiver setups, commonly found at airfields and often based on Raspberry Pi hardware. The system passively listens to decoded FLARM messages in real time and identifies approaching aircraft that are likely to land.

It operates in two distinct phases:

1. **Data Collection Phase**  
   - RAISE logs real-time aircraft movement data via the `ogn-decode` service.  
   - The landing prediction is **not yet active** in this phase.  
   - Approach trajectories are stored in a local database for later manual labeling and model training.

2. **Machine Learning Phase**  
   - After enough data is collected, it must be manually labeled on a more powerful machine with a graphical interface.  
   - A machine learning model is then trained on this labeled data.  
   - The resulting model is deployed back to the Raspberry Pi to enable real-time landing prediction.
   - The machine leatning phase is the desired long-time operation mode

### Hardware Requirements

- **Raspberry Pi (recommended)**  
  - RAISE is optimized to run on a Raspberry Pi that also serves as an Open Glider Network receiver.  
  - It listens to aircraft data via the local socket provided by the `ogn-decode` service.  
  - OGN receiver installation instructions:  
    [http://wiki.glidernet.org/wiki:raspberry-pi-installation](http://wiki.glidernet.org/wiki:raspberry-pi-installation)

- **Alternative Platforms**  
  - The system can be adapted to other platforms that provide access to a running `ogn-decode` instance.

- **Training Machine**  
  - A standard PC or laptop with:
    - A graphical user interface (for manual data labeling)  
    - A Python environment with the required ML libraries, see installation guide below  

## Installation on Raspberry Pi

To install RAISE on your Raspberry Pi, follow these steps:

---

### 1. Clone the repository

1. **Move into the desired installation directory**  
   For example:
   ```bash
   cd /home/pi/
   ```

2. **Clone the RAISE repository from GitHub**
   ```bash
   git clone https://github.com/YOUR_USERNAME/raise.git
   cd raise
   ```

> 📌 Replace `YOUR_USERNAME` with your GitHub username or organization name.

---

### 2. Prepare the system

RAISE requires a specific Python environment in order to run, especially to support TensorFlow Lite:

- Python **3.10.x** is required  
- `tflite-runtime` only works with **NumPy < 2.0**

To set up the system:

1. **Update your Raspberry Pi**
   ```bash
   sudo apt update
   sudo apt upgrade
   ```

2. **Make the setup scripts executable**
   ```bash
   chmod +x installCompileSystemPackages.sh installPython310OPS.sh
   ```

3. **Install required system packages to compile Python 3.10**
   ```bash
   ./installCompileSystemPackages.sh
   ```

4. **Compile and install Python 3.10**
   ```bash
   ./installPython310OPS.sh
   ```

After these steps, your system will have the correct Python version installed and ready to set up the RAISE environment.

---

### 3. Set up the Python environment

After installing Python 3.10, make sure you are inside the RAISE source directory:

```bash
cd /home/pi/RAISE
```

Then run the following command to create the virtual Python environment called RAISE:

```bash
python3.10 -m venv RAISE 
```

Then activate the virtual environment and install the required packages:
```bash
source ./RAISE/bin/activate
pip install --upgrade pip
pip install -r venvRequirementsOPS.txt
```

This will:

- Activate the Python 3.10 virtual environment
- Upgrade `pip` to the latest version
- Install all required Python packages listed in `venvRequirementsOPS.txt`, including `tflite-runtime` (compatible with Python 3.10 and NumPy < 2.0)

> ⚠️ Make sure the `RAISEenv` virtual environment was created using Python 3.10. If it doesn't exist yet, see the setup instructions or troubleshooting section.

---

### 4. Customize Project Parameters

Before running RAISE, you need to configure the project-specific settings.

1. **Make sure you are in the RAISE source code directory**:
   ```bash
   cd /home/pi/raise
   ```

2. **Copy the default parameter file** to create your own editable version:
   ```bash
   cp defaultParameters.yaml parameters.yaml
   ```

3. **Edit `parameters.yaml`** to match your system setup:
   ```bash
   nano parameters.yaml
   ```

#### 🔧 Adjusting the OGN socket port

In the section `systemParameters` of `parameters.yaml`, you may need to change the `PORT` value to match your local `ogn-decode` service.

By default, this is usually port `50001`. You can check if this port is active by running:

```bash
netcat localhost 50001
```

> 📦 If `netcat` is not installed on your system, you can install it using:
```bash
sudo apt install netcat
```

- If the terminal connects, you should see the output of the `ogn-decode` service and port 50001 is open and active.  
- If you get an error like *connection refused*, your setup may be using a different port.

Update the `PORT` field accordingly in your `parameters.yaml` if your system does not use port 50001.


#### 🗂️ Setting the database path

In the `databaseParameters` section of your `parameters.yaml` file, you need to specify the `DATABASE_PATH`. This defines where RAISE will store the collected flight data.

1. **Create a directory to hold the database**  
   For example:
   ```bash
   mkdir /home/pi/RAISE_DB
   ```

2. **Set the `DATABASE_PATH`** in `parameters.yaml` to point to a file in that directory:
   ```yaml
   databaseParameters:
     DATABASE_PATH: /home/pi/RAISE_DB/flightData.db
   ```

Make sure that the directory exists and is writable by the user running RAISE. The `.db` file will be created automatically when data collection starts.

#### 📍 Setting the airport reference location

In the `airportParameters` section of your `parameters.yaml` file, you must define the location of your airfield.

The following fields need to be set:

```yaml
airportParameters:
  AIRPORT_ALTITUDE: <altitude_in_meters>
  AIRPORT_LATITUDE: <latitude_in_decimal_degrees>
  AIRPORT_LONGITUDE: <longitude_in_decimal_degrees>
```

- **AIRPORT_ALTITUDE**: Elevation above sea level in meters  
- **AIRPORT_LATITUDE** and **AIRPORT_LONGITUDE**: Coordinates of the reference point

> 📌 **Tip**: Use the **midpoint of your main runway** as the reference location. This provides the most accurate frame of reference for approach pattern detection and landing classification.

You can find the required coordinates using online maps (e.g., Google Maps) or aviation databases.

#### ⚙️ Other parameters

All other parameters in `parameters.yaml` are documented with inline comments inside the file.  
In most cases, you will not need to modify them for standard operation.

Only adjust advanced settings if you understand their purpose or have a specific use case.

---

## First Test Run

Once you've completed the setup and configuration, you can test whether RAISE starts correctly.

### ✅ Steps to run RAISE

1. **Navigate to the RAISE source code directory**:
   ```bash
   cd /home/pi/raise
   ```

2. **Activate the virtual environment**:
   ```bash
   source ./RAISEenv/Scripts/activate
   ```

3. **Run the main script**:
   ```bash
   python ognClient.py
   ```

If the script starts without any errors, your environment is correctly installed and all required parameters are set properly.

You should see log output indicating that RAISE is connected to the `ogn-decode` service and ready to receive aircraft data.

---

### ❌ Troubleshooting

If any errors occur during startup:

- Double-check that all required Python packages are installed.
- Ensure your `parameters.yaml` file contains valid paths and values.
- Make sure the `ogn-decode` socket is running and accessible on the specified port.
- Use tools like `netcat` to verify socket connectivity.
- You can also use ChatGPT or other AI tools to help resolve errors based on the terminal output.

---

#### ⚙️ Setting up the `ognclient` systemd service

To ensure the RAISE OGN client starts automatically on boot, set up a `systemd` service that launches the script using its virtual environment.

1. **Create the service file**  
   Open a new file with root privileges:

   ```bash
   sudo nano /etc/systemd/system/ognclient.service
   ```

2. **Paste the following content** into the file:  
   Replace paths if your project structure is different.

   ```ini
   [Unit]
   Description=RAISE OGN client autostart
   After=network.target

   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/RAISE
   ExecStart=/home/pi/RAISE/RAISE/bin/python /home/pi/RAISE/ognClient.py
   Restart=always
   RestartSec=2
   Environment=PYTHONUNBUFFERED=1
   StandardOutput=journal
   StandardError=journal

   [Install]
   WantedBy=multi-user.target
   ```

3. **Reload systemd and enable the service**

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable ognclient.service
   ```

4. **Start the service**

   ```bash
   sudo systemctl start ognclient.service
   ```

5. **Check the service status**

   ```bash
   sudo systemctl status ognclient.service
   ```

   You should see that the service is `active (running)` if everything worked correctly.

6. **View live logs from the service**

   ```bash
   journalctl -u ognclient.service -f
   ```

   Press `Ctrl+C` to stop viewing logs.

7. **(Optional) Restart, stop, or disable the service**

   ```bash
   sudo systemctl restart ognclient.service
   sudo systemctl stop ognclient.service
   sudo systemctl disable ognclient.service
   ```


#### 🌐 Installing and Preparing the NGINX Web Server

The RAISE frontend is served via NGINX and communicates with the backend through a local proxy. Follow the steps below to install NGINX, set the correct file permissions, and configure it for the RAISE web interface.

---

1. **Install NGINX (if not already installed)**  
   On Debian-based systems:

   ```bash
   sudo apt update
   sudo apt install nginx
   ```

   Check if NGINX is running:

   ```bash
   systemctl status nginx
   ```

   If it’s not active, start it:

   ```bash
   sudo systemctl start nginx
   ```

2. **Enable NGINX to start on boot**

   ```bash
   sudo systemctl enable nginx
   ```

3. **Set correct file permissions for NGINX access**  
   These steps allow NGINX (user `www-data`) to access the web content without exposing other parts of the system:

   ```bash
   # 1. Set group to www-data for all web content
   sudo chown -R pi:www-data /home/pi/RAISE/web

   # 2. Directories: full access for pi, read/execute for www-data
   sudo find /home/pi/RAISE/web -type d -exec chmod 750 {} \;

   # 3. Files: writable by pi, readable by www-data
   sudo find /home/pi/RAISE/web -type f -exec chmod 640 {} \;

   # 4. Allow traversal through parent directories
   sudo chmod o+x /home
   sudo chmod o+x /home/pi
   sudo chmod o+x /home/pi/RAISE
   ```

4. **Create NGINX site configuration**

   Create a file:

   ```bash
   sudo nano /etc/nginx/sites-available/raise
   ```

   Paste the following configuration:

   ```nginx
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
   ```

5. **Activate the site and reload NGINX**

   ```bash
   # Create symlink to enable the site
   sudo ln -s /etc/nginx/sites-available/raise /etc/nginx/sites-enabled/raise

   # Test configuration
   sudo nginx -t

   # Reload NGINX
   sudo systemctl reload nginx
   ```

---

NGINX is now set up to serve the RAISE frontend from `/home/pi/RAISE/web`, and forwards `/api/` requests to the backend running on port `8181`.







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
pip install --no-build-isolation --no-use-pep517 -r venvRequirementsOPS.txt


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





