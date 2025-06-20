let map;
const SERVER = location.hostname;
const API_PORT = 8181;

const callsignCache = {};
const aircraftMarkers = {};
const aircraftTracks = {};
const aircraftTrackPoints = {};
let selectedAircraft = null;
let selectedTrackRefreshInterval = null;

async function getCallsign(id) {
  if (callsignCache[id]) return callsignCache[id];
  try {
    const res = await fetch(`http://${SERVER}:${API_PORT}/callsign/${id}`);
    const data = await res.json();
    const callsign = data.callsign || id;
    callsignCache[id] = callsign;
    return callsign;
  } catch (e) {
    return id; // fallback
  }
}

function createRotatedIconHtml(imageUrl, heading) {
  return `<img src="${imageUrl}" style="width:32px; height:32px; transform: rotate(${heading}deg); transform-origin: center;">`;
}

function updateMarkerVisual(marker, imageUrl, heading) {
  const img = marker.getElement()?.querySelector('img');

  if (img) {
    img.src = imageUrl;
    img.style.transform = `rotate(${heading}deg)`;
  } else {
    // Marker-Element noch nicht verfügbar → setIcon neu setzen
    const icon = L.divIcon({
      className: '',
      html: `<img src="${imageUrl}" style="width:32px; height:32px; transform: rotate(${heading}deg); transform-origin: center;">`,
      iconSize: [32, 32],
      iconAnchor: [16, 16]
    });
    marker.setIcon(icon);
  }
}


async function initMapCenteredOnAirport() {
  const res = await fetch(`http://${SERVER}:${API_PORT}/config`);
  const config = await res.json();
  const airportLat = config.airport.lat;
  const airportLon = config.airport.lon;
  const defaultZoom = config.mapZoom || 12;

  map = L.map('map', { zoomControl: false }).setView([airportLat, airportLon], defaultZoom);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    attribution: 'Map data from © <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>'
  }).addTo(map);
  L.control.zoom({ position: 'bottomright' }).addTo(map);

  const airportIcon = L.icon({
    iconUrl: 'assets/location_marker.svg',
    iconSize: [32, 32],
    iconAnchor: [16, 32]
  });
  L.marker([airportLat, airportLon], { icon: airportIcon }).addTo(map);

  startApplication();
}

async function loadAircrafts() {
  const res = await fetch(`http://${SERVER}:${API_PORT}/aircrafts`);
  const aircrafts = await res.json();

  for (const ac of aircrafts) {
    const callsign = await getCallsign(ac.id);
    const heading = ac.heading || 0;
    const imageUrl = (ac.flightState === 'landing') ? 'assets/aircraft_landing.svg' : 'assets/aircraft.svg';

    if (!aircraftMarkers[ac.id]) {
      const icon = L.divIcon({
        className: '',
        html: createRotatedIconHtml(imageUrl, heading),
        iconSize: [32, 32],
        iconAnchor: [16, 16]
      });

      const marker = L.marker([ac.lat, ac.lon], { icon: icon });
      marker.bindTooltip(callsign, { permanent: true, className: 'aircraft-label', direction: 'top' });
      marker.on('click', () => onSelectAircraft(ac.id));
      marker.addTo(map);
      aircraftMarkers[ac.id] = marker;
    } else {
      const marker = aircraftMarkers[ac.id];
      marker.setLatLng([ac.lat, ac.lon]);
      updateMarkerVisual(marker, imageUrl, heading);

      const tooltip = marker.getTooltip();
      if (tooltip && tooltip._content !== callsign) {
        marker.setTooltipContent(callsign);
      }
    }
  }
}

async function loadAndUpdateTrack(id) {
  const res = await fetch(`http://${SERVER}:${API_PORT}/aircrafts/${id}/track`);
  const track = await res.json();
  aircraftTrackPoints[id] = track.map(p => ({ lat: p.lat, lon: p.lon }));
  const cleanTrack = aircraftTrackPoints[id].map(p => [p.lat, p.lon]);

  if (aircraftTracks[id]) {
    aircraftTracks[id].setLatLngs(cleanTrack);
  } else {
    aircraftTracks[id] = L.polyline(cleanTrack, { color: 'blue' }).addTo(map);
  }
}

async function onSelectAircraft(id) {
  if (selectedAircraft && aircraftTracks[selectedAircraft]) {
    map.removeLayer(aircraftTracks[selectedAircraft]);
  }

  if (selectedTrackRefreshInterval) {
    clearInterval(selectedTrackRefreshInterval);
    selectedTrackRefreshInterval = null;
  }

  selectedAircraft = id;
  await loadAndUpdateTrack(id);

  if (aircraftTracks[id] && !map.hasLayer(aircraftTracks[id])) {
    aircraftTracks[id].addTo(map);
  }

  selectedTrackRefreshInterval = setInterval(() => {
    if (selectedAircraft) {
      loadAndUpdateTrack(selectedAircraft);
    }
  }, 30000);
}

function startApplication() {
  map.on('click', () => {
    if (selectedAircraft) {
      if (aircraftTracks[selectedAircraft]) {
        map.removeLayer(aircraftTracks[selectedAircraft]);
      }
      selectedAircraft = null;
      if (selectedTrackRefreshInterval) {
        clearInterval(selectedTrackRefreshInterval);
        selectedTrackRefreshInterval = null;
      }
    }
  });

  const socket = new WebSocket(`ws://${SERVER}:${API_PORT}/ws`);
  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'positionUpdate') {
      const id = data.aircraftId;
      const lat = data.lat;
      const lon = data.lon;
      const heading = data.heading || 0;
      const imageUrl = (data.flightState === 'landing') ? 'assets/aircraft_landing.svg' : 'assets/aircraft.svg';

      if (aircraftMarkers[id]) {
        aircraftMarkers[id].setLatLng([lat, lon]);
        updateMarkerVisual(aircraftMarkers[id], imageUrl, heading);
      }

      if (selectedAircraft === id) {
        if (!aircraftTrackPoints[id]) aircraftTrackPoints[id] = [];
        aircraftTrackPoints[id].push({ lat: lat, lon: lon });

        const latlngs = aircraftTrackPoints[id].map(p => [p.lat, p.lon]);
        if (aircraftTracks[id]) {
          aircraftTracks[id].setLatLngs(latlngs);
        } else {
          aircraftTracks[id] = L.polyline(latlngs, { color: 'blue' }).addTo(map);
        }
      }
    }
  };

  loadAircrafts();
  setInterval(loadAircrafts, 5000);
}

initMapCenteredOnAirport();
