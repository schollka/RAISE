let map;
const SERVER = location.hostname;
const API_PORT = 8000;

const callsignCache = {};

async function getCallsign(id) {
  if (callsignCache[id]) return callsignCache[id];
  try {
    const res = await fetch(`http://${SERVER}:${API_PORT}/callsign/${id}`);
    const data = await res.json();
    const callsign = data.callsign || id;
    callsignCache[id] = callsign;
    return callsign;
  } catch (e) {
    return id;  // fallback
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

  startApplication();
}

const aircraftMarkers = {};
const aircraftTracks = {};
const aircraftTrackPoints = {};
let selectedAircraft = null;
let selectedTrackRefreshInterval = null;

const blueIcon = L.icon({
  iconUrl: 'https://cdn-icons-png.flaticon.com/512/684/684908.png',
  iconSize: [24, 24],
  iconAnchor: [12, 12]
});

const orangeIcon = L.icon({
  iconUrl: 'https://cdn-icons-png.flaticon.com/512/684/684908.png',
  iconSize: [24, 24],
  iconAnchor: [12, 12],
  className: 'selected-icon'
});

const style = document.createElement('style');
style.innerHTML = `.selected-icon img { filter: hue-rotate(45deg) brightness(1.2); }`;
document.head.appendChild(style);

async function loadAircrafts() {
  const res = await fetch(`http://${SERVER}:${API_PORT}/aircrafts`);
  const aircrafts = await res.json();

  for (const ac of aircrafts) {
    const callsign = await getCallsign(ac.id);
    if (!aircraftMarkers[ac.id]) {
      const marker = L.marker([ac.lat, ac.lon], { icon: blueIcon });
      marker.bindTooltip(callsign, { permanent: true, className: 'aircraft-label', direction: 'top' });
      marker.addTo(map);
      marker.on('click', () => onSelectAircraft(ac.id));
      aircraftMarkers[ac.id] = marker;
    } else {
      aircraftMarkers[ac.id].setLatLng([ac.lat, ac.lon]);
      const tooltip = aircraftMarkers[ac.id].getTooltip();
      if (tooltip && tooltip._content !== callsign) {
        aircraftMarkers[ac.id].setTooltipContent(callsign);
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
  if (selectedAircraft && aircraftMarkers[selectedAircraft]) {
    aircraftMarkers[selectedAircraft].setIcon(blueIcon);
    if (aircraftTracks[selectedAircraft]) {
      map.removeLayer(aircraftTracks[selectedAircraft]);
    }
    if (selectedTrackRefreshInterval) {
      clearInterval(selectedTrackRefreshInterval);
      selectedTrackRefreshInterval = null;
    }
  }

  selectedAircraft = id;
  aircraftMarkers[id].setIcon(orangeIcon);

  await loadAndUpdateTrack(id);

  // ensure track is shown again if it was removed before
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
      if (aircraftMarkers[selectedAircraft]) {
        aircraftMarkers[selectedAircraft].setIcon(blueIcon);
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

      if (aircraftMarkers[id]) {
        aircraftMarkers[id].setLatLng([lat, lon]);
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
