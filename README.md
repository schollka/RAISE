# RAISE – Runway Approach Identification for Silent Entries

TODO

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





