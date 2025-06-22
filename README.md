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
   - The machine learning phase is the desired long-time operation mode.

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

## Installation on Raspberry Pi (OGN Reciever)

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
   git clone https://github.com/YOUR_USERNAME/RAISE.git
   cd RAISE
   ```

> TODO: Replace `YOUR_USERNAME` with your GitHub username or organization name.

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

---

### 4. Customize Project Parameters

Before running RAISE, you need to configure the project-specific settings.

1. **Make sure you are in the RAISE source code directory**:
   ```bash
   cd /home/pi/RAISE
   ```

2. **Copy the default parameter file** to create your own editable version:
   ```bash
   cp defaultParameters.yaml parameters.yaml
   ```

3. **Edit `parameters.yaml`** to match your system setup:
   ```bash
   nano parameters.yaml
   ```

#### Adjusting the OGN socket port

In the section `systemParameters` of `parameters.yaml`, you may need to change the `PORT` value to match your local `ogn-decode` service.

By default, this is usually port `50001`. You can check if this port is active by running:

```bash
netcat localhost 50001
```

If `netcat` is not installed on your system:

```bash
sudo apt install netcat
```

If the terminal connects, you should see the output of the `ogn-decode` service and port 50001 is open and active.  
If you get an error like *connection refused*, your setup may be using a different port.

Update the `PORT` field accordingly in your `parameters.yaml`.

#### Setting the database path

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

#### Setting the airport reference location

In the `airportParameters` section of your `parameters.yaml` file, you must define the location of your airfield.

```yaml
airportParameters:
  AIRPORT_ALTITUDE: <altitude_in_meters>
  AIRPORT_LATITUDE: <latitude_in_decimal_degrees>
  AIRPORT_LONGITUDE: <longitude_in_decimal_degrees>
```

Use the midpoint of your main runway as the reference location.  
You can find the required coordinates using online maps or aviation databases.

#### Other parameters

All other parameters in `parameters.yaml` are documented with inline comments inside the file.  
In most cases, you will not need to modify them for standard operation.

---

### 5. First Test Run

Once you've completed the setup and configuration, you can test whether RAISE starts correctly.

#### Steps to run RAISE

1. **Navigate to the RAISE source code directory**:
   ```bash
   cd /home/pi/raise
   ```

2. **Activate the virtual environment**:
   ```bash
   source ./RAISE/bin/activate
   ```

3. **Run the main script**:
   ```bash
   python ognClient.py
   ```

You should see log output indicating that RAISE is connected to the `ogn-decode` service and ready to receive aircraft data.

---

#### Troubleshooting

If any errors occur during startup:

- Double-check that all required Python packages are installed.
- Ensure your `parameters.yaml` file contains valid paths and values.
- Make sure the `ogn-decode` socket is running and accessible on the specified port.
- Use `netcat` to verify socket connectivity.

---

### 6. Setting up the `ognclient` systemd service

To ensure the RAISE OGN Client starts automatically on boot, set up a `systemd` service that launches the script using its virtual environment.

1. **Create the service file**  
   ```bash
   sudo nano /etc/systemd/system/ognclient.service
   ```

2. **Paste the following content**:

   ```ini
   [Unit]
   Description=RAISE OGN Client autostart
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

3. **Enable and start the service**

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable ognclient.service
   sudo systemctl start ognclient.service
   ```

4. **Check logs**

   ```bash
   journalctl -u ognclient.service -f
   ```

---

### 7. Installing and Preparing the NGINX Web Server

1. **Install NGINX**

   ```bash
   sudo apt update
   sudo apt install nginx
   ```

2. **Enable it on boot**

   ```bash
   sudo systemctl enable nginx
   ```

3. **Set file permissions**\
    You may need to change the paths, if you installed RAISE in a different directory.

   ```bash
   sudo chown -R pi:www-data /home/pi/RAISE/web
   sudo find /home/pi/RAISE/web -type d -exec chmod 750 {} \;
   sudo find /home/pi/RAISE/web -type f -exec chmod 640 {} \;
   sudo chmod o+x /home
   sudo chmod o+x /home/pi
   sudo chmod o+x /home/pi/RAISE
   ```

4. **Configure NGINX**

   ```bash
   sudo nano /etc/nginx/sites-available/raise
   ```

   Copy the settings into the configutation file:

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

5. **Activate site and reload**

   ```bash
   sudo ln -s /etc/nginx/sites-available/raise /etc/nginx/sites-enabled/raise
   sudo nginx -t
   sudo systemctl reload nginx
   ```

### 8. Accessing RAISE

NGINX now serves the RAISE frontend on port 80 of your device.  
You can access the interface on the same device by opening the following URL in a browser:

```
http://localhost:80
```

However, the usual use case is to access the RAISE frontend from a different device on the airfield, such as a tablet or laptop.

To do this:

1. Ensure both devices are connected to the same local network.
2. In the browser on the monitoring device, either:
   - Enter the **IP address** of your Raspberry Pi running RAISE,  
     e.g., `http://192.168.0.42`
   - Or enter the **hostname** of the device,  
     e.g., `http://ogn`

> RAISE is designed to run **entirely within your local airfield network**.  
> **Do not expose the system to the public internet.**  
> Access should remain restricted to the internal network for safety and privacy reasons.



# API BACKEND PORT 8181


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


## Disclosure

The RAISE system relies entirely on the presence and quality of an OGN receiver and its antenna installation. Depending on the receiver's location, the type and quality of the antenna, and the performance of the RF module, it is possible that data packets from an aircraft are not received, even when the aircraft is in close proximity to the airfield.

RAISE is **not** a certified or commercial airspace monitoring system. It is an open-source academic project that aims to utilize existing infrastructure to provide **additional safety** at small airfields operating in unmonitored airspace, such as gliding fields. The aircraft position data is presented as-is from the decoded data stream. 

Aircraft that are **not equipped with compatible broadcast systems** (e.g. FLARM, ADS-B, etc.) that transmit their position to the local OGN receiver **cannot** be displayed by RAISE.

The project originates from an academic context and offers **no warranty**, and **no guarantee of correctness** or **certified integrity** of the displayed information.

The machine learning-based landing detection depends entirely on the quality and quantity of training data. A model can only predict patterns that it has seen during training. It may:

- Produce **false positives**
- **Miss landings** that do not match the trained approach pattern
- Is trained primarily on the **most common local aircraft types** (e.g. gliders)

To ensure meaningful predictions, collect **sufficient approach data** over an extended period. The model will reflect the **most common type of traffic** at your airfield (e.g. gliders). Unusual traffic, such as police or EMS helicopters, may be displayed if their onboard equipment broadcasts to the OGN receiver, but these will **not** be classified as approaching by the machine learning algorithm, due to their different approach pattern.

RAISE is intended to support situational awareness, not to replace certified systems.


## License and Data Attribution
RAISE is released as open-source software. The full license text can be found in the [LICENSE](./LICENSE) file included in this repository.

### Map Data – OpenStreetMap

The map tiles and geodata used in this project are provided by [OpenStreetMap](https://www.openstreetmap.org/).

- © OpenStreetMap contributors
- License: [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/)
- Tile service: [tile.openstreetmap.org](https://tile.openstreetmap.org/)
- Attribution is shown in the web interface 

If you plan to run this application in a high-traffic or commercial setting, consider hosting your own tile server or using a third-party tile provider to reduce load on the public OpenStreetMap infrastructure.

### Callsign Lookup – DDB (Distributed DataBase)

Callsigns and aircraft metadata are resolved based on the DDB database provided by the Open Glider Network (OGN) community:

- Source: [https://ddb.glidernet.org/download/](https://ddb.glidernet.org/download/)
- Maintained by: Open Glider Network and volunteer contributors
- Intended for: Non-commercial, aviation-related use only
- License: Use permitted for non-commercial purposes; redistribution and usage must comply with the conditions described on [https://ddb.glidernet.org/about.html](https://ddb.glidernet.org/about.html)

Aircraft identifiers in this project are matched using the downloaded CSV files containing FLARM or ICAO hexadecimal IDs with associated metadata (e.g., callsign, aircraft type, competition ID).





