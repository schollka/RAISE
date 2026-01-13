# RAISE – Runway Approach Identification for Silent Entries

RAISE – Runway Approach Identification for Silent Entries is an open-source project aimed at improving airspace monitoring at uncontrolled airfields. It processes real-time position data from aircraft equipped with FLARM devices and displays this data on a web-based map interface.

The system is designed to autonomously record approach patterns and use them to train a machine learning model. This model identifies aircraft in the vicinity of the airfield that are likely about to land. Aircraft currently on approach are visually highlighted to draw attention.

RAISE aims to improve safety by giving ground personnel a reliable tool to monitor aircraft activity near the airfield, with a special focus on identifying and highlighting landing approaches. In cases where radio communication is disrupted, RAISE acts as a fallback system to ensure that situational awareness on the ground is maintained.

This documentation provides detailed step-by-step instructions to ensure correct installation and configuration of RAISE. While more experienced users may find some sections overly detailed, the target audience includes users with limited experience in handling source code or working with Linux-based systems. The goal is to make RAISE accessible and usable without requiring deep technical expertise.

## Use Case and System Requirements

### Use Case
RAISE is designed to be a localy hosted airspace monitoring tool. It can be used at small uncontrolled airfields. RAISE will monitor the position of aircrafts in the vicinity of the airfield and display them on a web interface. The machine lerning algorithm will detect approaching aircraft and mark them accordingly. The web interface can be seen below, with a marked aircraft.

![Initial View](/doc/GUI_Landing_2.png)

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

   You need to have Git installed on your system for this step. If it is not already installed, you can install it using:

   ```bash
   sudo apt install git
   ```

   Clone the respository into your directory:
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
- Install all required Python packages listed in [`venvRequirementsOPS.txt`](venvRequirementsOPS.txt), including `tflite-runtime` (compatible with Python 3.10 and NumPy < 2.0)

---

### 4. Customize Project Parameters

Before running RAISE, you need to configure the project-specific settings.

1. **Make sure you are in the RAISE source code directory**:
   ```bash
   cd /home/pi/RAISE
   ```

2. Copy the default parameter file [`defaultParameters.yaml`](defaultParameters.yaml) to create your own editable version:
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
   proxy_cache_path /var/cache/nginx/tiles levels=1:2 keys_zone=tilecache:10m max_size=1g inactive=10d use_temp_path=off;

   server {
      listen 80;
      server_name _;

      root /var/www/raise;
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

      location /tiles/ {
         proxy_pass https://tile.openstreetmap.org/;
         proxy_cache tilecache;
         proxy_cache_valid 200 302 10d;
         proxy_cache_use_stale error timeout;
         proxy_set_header Host tile.openstreetmap.org;
         proxy_set_header User-Agent "RAISE-MapTileProxy/1.0";
      }
   }
   ```

5. **Activate site and reload**

   ```bash
   sudo ln -s /etc/nginx/sites-available/raise /etc/nginx/sites-enabled/raise
   sudo mkdir -p /var/cache/nginx/tiles
   sudo chown -R www-data:www-data /var/cache/nginx
   sudo nginx -t
   sudo systemctl reload nginx
   ```

6. **Configure Hostname for Tile Caching**

   The **RAISE** web server uses **NGINX** to cache OpenStreetMap tiles locally. This reduces network load and minimizes repeated requests to the OpenStreetMap server.

   To ensure the tile requests work correctly on the Raspberry Pi itself, you need to define the hostname `ogn` by adding the following entry to the system's hosts file:

   ```bash
   sudo nano /etc/hosts
   ```

   Add this line, if not already present:
   `127.0.1.1 ogn`

   This tells the system that ogn refers to the local machine.

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

### 9. API Backend Port

The API that provides data from RAISE to the web interface runs on port `8181` by default.  
If this port is already in use on your system, you can change it.

To do so:

1. **Change the backend port**  
   In your `parameters.yaml`, set a new value for `API_PORT` under the `webServerParameters` section:

   ```yaml
   webServerParameters:
     API_PORT: 8080
   ```

2. **Update the frontend to match**  
   You must manually update the frontend source code as well.  
   Open the file [`web/scripts.js`](web/script.js) and modify the following line to use the new port:

   ```javascript
   const API_PORT = yourPortNumber;
   ```

Make sure both the backend and frontend use the same port so that communication between the web interface and the API works correctly.

## Data Acquisition Phase

After installation, RAISE runs in data acquisition mode, which is enabled by default but can be disabled by the user. In this mode, it will display nearby air traffic but will not yet predict landings.

RAISE continuously stores:

- Every detected approach
- A percentage of departures
- Random short in-flight sequences near the airfield

Data acquisition should be long rather than short.  
It is recommended to collect **at least 200 approaches** after installation.  
Make sure the dataset covers **all possible runway directions and approach paths** relevant to your airfield.

Once enough data has been collected, it must be manually labeled on a machine with a graphical user interface.  
Since model training is computationally intensive, this step should be performed on a PC or laptop rather than the Raspberry Pi.

The following chapter explains how to install RAISE in model training mode on a Windows machine.

## Installing RAISE on a Windows Machine (Model Training Mode)

To prepare RAISE for model training, you need to install Python 3.10, download the source code, and set up a virtual environment.

### 1. Install Python 3.10 (or 3.10.11)

If Python 3.10 is not already installed on your system:

1. Go to the official download page:  
   [https://www.python.org/downloads/release/python-31011/](https://www.python.org/downloads/release/python-31011/)

2. Download the **Windows installer (64-bit)** for Python 3.10.11.

3. Run the installer with the following options:
   - **Check the box**: “Add Python 3.10 to PATH”
   - Click “Customize installation” and keep all default options
   - On the advanced options screen, also check:  
     “Install for all users” and “Precompile standard library”
   - Finish the installation

To verify that Python 3.10 was installed correctly, open a Command Prompt (`Win + R`, type `cmd`) and run:

```bash
py -3.10 --version
```

If that does not work, try:

```bash
python --version
```

You should see something like:

```
Python 3.10.11
```

> Make sure Python 3.10 is installed and available in your system PATH. If neither `py -3.10` nor `python` returns version 3.10.x, check your installation settings or reinstall using the official installer with "Add Python to PATH" enabled.


### 2. Download the RAISE Source Code

You can either clone the repository using Git, or download it manually:

#### Option A: Clone with Git

> Requires Git to be installed. If Git is not available, use Option B.

Open a terminal (`cmd` or Git Bash), navigate to the folder where you want to store RAISE, and run:

```bash
git clone https://github.com/YOUR_USERNAME/raise.git
cd raise
```

Replace `YOUR_USERNAME` with your GitHub username or organization.

#### Option B: Download as ZIP

1. Visit the repository in your browser:  
   `https://github.com/YOUR_USERNAME/raise`

2. Click **"Code" > "Download ZIP"**

3. Extract the ZIP archive into a folder of your choice.

4. Open a terminal and navigate into the extracted folder.

---

### 3. Create Python Environment

Once you're inside the `RAISE` source folder, follow these steps to prepare the Python environment for model training:

1. **Create and activate the virtual environment**

   ```bash
   py -3.10 -m venv RAISE
   RAISE\Scripts\activate
   ```

   If activation is successful, your terminal will now show:

   ```
   (RAISE) C:\[...]\RAISE>
   ```

   > If `py -3.10` does not work, try `python -m venv RAISE` instead.

2. **Upgrade pip**

   Once the virtual environment is active, upgrade `pip`:

   ```bash
   python -m pip install --upgrade pip
   ```

3. **Install required Python packages**

   Use the provided [`venvRequirementsDEV.txt`](venvRequirementsDEV.txt) file to install all required packages:

   ```bash
   pip install -r venvRequirementsDEV.txt
   ```

   This will install all dependencies needed to run the labeling tools and train the machine learning model.  
   Make sure you are in the RAISE source code directory when running this command.

   > If you encounter errors during installation, ensure that your virtual environment is activated and that you are using Python 3.10.

## Train Approach Detection Model

Once enough flight data has been collected by the Raspberry Pi running RAISE, you can begin training the machine learning model that detects landing approaches.

### 1. Copy the database to your development machine

The path to the database is defined in `parameters.yaml` under the key `DATABASE_PATH`.

To transfer the database file:

- Option 1: Use a USB stick
- Option 2: Use `scp` (secure copy) from your development machine:

```bash
scp pi@ogn:/home/pi/RAISE_DB/flightData.db C:\Users\YourUser\Downloads
```

>Replace `/home/pi/RAISE_DB/flightData.db` with your actual `DATABASE_PATH`, and adjust the destination path for your OS.

### 2. Prepare the training environment

1. Move the `.db` file to a known, convenient directory on your development machine.

2. **In the RAISE source code directory**, copy the default parameters file `defaultParameters.yaml` and rename it to `parameters.yaml`. This can be done either via the Windows File Explorer or using a terminal.

3. **Edit `parameters.yaml`** and set `DATABASE_PATH` to the full path of the `.db` file on your current machine.

### 3. Activate the virtual environment

In a terminal:

```bash
cd /path/to/raise
source RAISE/Scripts/activate
```

### 4. Run Labeling Script

To properly train the machine learning model, it is necessary to review the stored approach patterns and manually mark the point at which the pilot likely made the decision to land.

This is done using an interactive labeling script that visually displays each stored approach trajectory. The user selects the most appropriate waypoint by clicking on the plot.

> ⚠️ **Important:**  
> Do **not** mark the official start of the downwind leg.  
> Instead, select the point where the **pilot likely made the mental decision to begin landing preparations** — this is often earlier and more subjective.

---

To run the labeling tool, make sure your virtual environment is active, then execute:

```bash
python manualLandingLabeler.py
```

You will be presented with a plot similar to the one below:

#### Initial View (Before Selection)
![Initial View](/doc/GUI_landingNotMarked_2.png)

Click on the waypoint where the pilot likely made the mental decision to start the approach.

After your selection, the plot will update to highlight all waypoints that follow the selected point:

#### Marked View (After Selection)
![Marked View](/doc/GUI_landingMarked_2.png)

You can:
- Change your selection by clicking a different point.
- Click **`Done`** to confirm the label and proceed to the next flight.
- If the trajectory appears unusual (e.g., from a police or EMS helicopter), **do not select anything** and click **`Remove flight`**. It will be deleted from the databasse

### 5. Prepare Training Data

Once the data is labeled, it can be processed into training data for the machine learning algorithm.

To do this, run the following script (with your virtual environment still active):

```bash
python prepareTrainingDataML.py
```

This script reads the manually labeled flight data from the database and generates two output files:

- `X.npy` — the input features for training
- `Y.npy` — the corresponding target labels

These files are saved in the directory of the database.

### 6. Train Model

Once the training data has been prepared, you can train the machine learning model by running:

```bash
python trainModel.py
```

This will generate two model files:

- `landingClassifier.keras` — the full Keras model (not compatible with Raspberry Pi)
- `landingClassifierLite.tflite` — a TensorFlow Lite version optimized for use on the Raspberry Pi

Copy the `.tflite` model to your Raspberry Pi, for example:

```bash
scp landingClassifierLite.tflite pi@ogn:/home/pi/RAISE_DB/landingClassifierLite.tflite
```

---

## Execute Model

After copying the model, open your `parameters.yaml` on the Raspberry Pi and update the following fields:

```yaml
machineLearningParameters:
  MODEL_PATH: /home/pi/RAISE_DB/landingClassifierLite.tflite
  ENABLE_MODEL: True

databaseParameters:
  ENABLE_DATABASE: False  # Optional: disable flight path logging
```

> Note: Logging to the database is no longer required, but can be re-enabled if desired.

Once you've updated and saved the parameter file, restart the RAISE service:

```bash
sudo systemctl restart ognclient.service
```

Then check the status:

```bash
systemctl status ognclient.service
```

> Loading TensorFlow Lite and the model may slightly delay the startup time.

Once the model is running, **RAISE will automatically highlight aircraft predicted to be on approach**.

- ✈️ When an aircraft is classified as likely to land soon, its **icon will turn red** on the live map.
- 🔴 This **visual marker helps alert ground personnel** to aircraft with an imminent landing, even if no radio communication is received.


## Disclaimer

The RAISE system relies entirely on the presence and quality of an OGN receiver and its antenna installation. Depending on the receiver's location, the type and quality of the antenna, the performance of the RF module, and the relative position of the aircraft to the receiver, it is possible that data packets from an aircraft are not received, even when the aircraft is in close proximity to the airfield.

Since RAISE solely relies on received data packets, it will not display or update the aircraft's position if those packets are missing — even if the aircraft is physically nearby.

RAISE is not a certified or commercial airspace monitoring system. It is an open-source academic project that aims to utilize existing infrastructure to provide **additional safety** at small airfields operating in unmonitored airspace, such as gliding fields. The aircraft position data is presented as-is from the decoded data stream. 

Aircraft that are **not equipped with compatible broadcast systems** (FLARM) that transmit their position to the local OGN receiver **cannot** be displayed by RAISE.

The project originates from an academic context and offers no warranty, and no guarantee of correctness or certified integrity of the displayed information.

The machine learning-based landing detection depends entirely on the quality and quantity of training data. A model can only predict patterns that it has seen during training. It may:

- Produce **false positives**
- **Miss landings** that do not match the trained approach pattern
- Is trained primarily on the **most common local aircraft types** (e.g. gliders)

To ensure meaningful predictions, collect sufficient approach data over an extended period. The model will reflect the most common type of traffic at your airfield (e.g. gliders). Unusual traffic, such as police or EMS helicopters, may be displayed if their onboard equipment broadcasts to the OGN receiver, but these will not be classified as approaching by the machine learning algorithm, due to their different approach pattern.

RAISE is intended to enhance situational awareness at small airfields. It is not a substitute for certified monitoring systems or for human visual surveillance of the surrounding airspace.

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





