const SERVER = location.hostname;
const API_PORT = 8181;

const callsignCache = {};
const aircraftMarkers = {};
const aircraftTracks = {};
const aircraftTrackPoints = {};
let selectedAircraft = null;
let selectedTrackRefreshInterval = null;
let latestAircrafts = [];

let infoBox, infoCallsign, infoAlt, infoSpeed, infoLastTimeSeen;

document.addEventListener('DOMContentLoaded', async () => {
  infoBox = document.getElementById('aircraft-info');
  infoCallsign = document.getElementById('info-callsign');
  infoAlt = document.getElementById('info-alt');
  infoSpeed = document.getElementById('info-speed');
  infoLastTimeSeen = document.getElementById('info-lastTimeSeen');

  const config = await fetch(`http://${SERVER}:${API_PORT}/api/config`).then(r => r.json());
  const map = L.map('map', {
    zoomControl: false
  }).setView([config.airport.lat, config.airport.lon], config.mapZoom || 12);

  L.control.zoom({ position: 'bottomright' }).addTo(map);

  L.tileLayer('http://ogn/tiles/{z}/{x}/{y}.png', {
  //L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    attribution: 'Map data from © <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>'
  }).addTo(map);

  const airportIcon = L.icon({
    iconUrl: 'assets/location_marker.svg',
    iconSize: [32, 32],
    iconAnchor: [16, 32]
  });
  L.marker([config.airport.lat, config.airport.lon], { icon: airportIcon }).addTo(map);

  map.on('click', () => {
    if (selectedAircraft) {
      if (aircraftTracks[selectedAircraft]) map.removeLayer(aircraftTracks[selectedAircraft]);
      selectedAircraft = null;
      if (selectedTrackRefreshInterval) clearInterval(selectedTrackRefreshInterval);
      infoBox.style.display = 'none';

      const marker = aircraftMarkers[selectedAircraft];
      if (marker) marker.closeTooltip();
    }
  });

  function createRotatedIconHtml(imageUrl, heading) {
    return `<img src="${imageUrl}" style="width:32px;height:32px;transform:rotate(${heading}deg);transform-origin:center;">`;
  }

  function updateMarkerVisual(marker, imageUrl, heading) {
    const img = marker.getElement()?.querySelector('img');
    if (img) {
      img.src = imageUrl;
      img.style.transform = `rotate(${heading}deg)`;
    } else {
      const icon = L.divIcon({
        className: '',
        html: createRotatedIconHtml(imageUrl, heading),
        iconSize: [32, 32],
        iconAnchor: [16, 16]
      });
      marker.setIcon(icon);
    }
  }

  function formatLastTimeSeen(isoString) {
    try {
      const dt = new Date(isoString);
      return dt.toLocaleTimeString('en-GB', { hour12: false });
    } catch {
      return '-';
    }
  }

  async function getCallsign(id) {
    if (callsignCache[id]) return callsignCache[id];
    try {
      const res = await fetch(`http://${SERVER}:${API_PORT}/callsign/${id}`);
      const data = await res.json();
      const callsign = data.callsign || id;
      callsignCache[id] = callsign;
      return callsign;
    } catch (e) {
      return id;
    }
  }

  async function loadAircrafts() {
    const aircrafts = await fetch(`http://${SERVER}:${API_PORT}/aircrafts`).then(r => r.json());
    latestAircrafts = aircrafts;

    for (const ac of aircrafts) {
      const callsign = await getCallsign(ac.id);
      const heading = ac.heading || 0;

      let imageUrl = 'assets/aircraft.svg';
      if (ac.receptionState === 'aircraftLost') {
        imageUrl = 'assets/aircraft_lost.svg';
      } else if (ac.flightState === 'landing') {
        imageUrl = 'assets/aircraft_landing.svg';
      }

      if (!aircraftMarkers[ac.id]) {
        const icon = L.divIcon({
          className: '',
          html: createRotatedIconHtml(imageUrl, heading),
          iconSize: [32, 32],
          iconAnchor: [16, 16]
        });

        const marker = L.marker([ac.lat, ac.lon], { icon });
        marker.on('mouseover', () => marker.bindTooltip(callsign, { className: 'aircraft-label', direction: 'top' }).openTooltip());
        marker.on('mouseout', () => {
          if (selectedAircraft !== ac.id) marker.closeTooltip();
        });
        marker.on('click', () => onSelectAircraft(ac.id));
        marker.addTo(map);
        aircraftMarkers[ac.id] = marker;
      } else {
        const marker = aircraftMarkers[ac.id];
        marker.setLatLng([ac.lat, ac.lon]);
        updateMarkerVisual(marker, imageUrl, heading);
      }
    }
  }

  async function onSelectAircraft(id) {
    if (selectedAircraft && aircraftTracks[selectedAircraft]) map.removeLayer(aircraftTracks[selectedAircraft]);
    if (selectedTrackRefreshInterval) clearInterval(selectedTrackRefreshInterval);

    selectedAircraft = id;
    await loadAndUpdateTrack(id);
    if (aircraftTracks[id] && !map.hasLayer(aircraftTracks[id])) aircraftTracks[id].addTo(map);

    selectedTrackRefreshInterval = setInterval(() => {
      if (selectedAircraft) loadAndUpdateTrack(selectedAircraft);
    }, 30000);

    const aircraft = latestAircrafts.find(ac => ac.id === id);
    if (aircraft) {
      const callsign = await getCallsign(id);
      infoCallsign.textContent = callsign;
      infoAlt.textContent = aircraft.alt ?? '-';
      infoSpeed.textContent = aircraft.speed != null
        ? Math.round(aircraft.speed * 3.6)
        : '-';
      infoLastTimeSeen.textContent = formatLastTimeSeen(aircraft.lastTimeSeen);

      infoBox.style.display = 'block';

      const marker = aircraftMarkers[id];
      if (marker) marker.bindTooltip(callsign, {
        permanent: true,
        className: 'aircraft-label',
        direction: 'top'
      }).openTooltip();
    }
  }

  async function loadAndUpdateTrack(id) {
    const track = await fetch(`http://${SERVER}:${API_PORT}/aircrafts/${id}/track`).then(r => r.json());
    aircraftTrackPoints[id] = track.map(p => ({ lat: p.lat, lon: p.lon }));
    const cleanTrack = aircraftTrackPoints[id].map(p => [p.lat, p.lon]);
    if (aircraftTracks[id]) {
      aircraftTracks[id].setLatLngs(cleanTrack);
    } else {
      aircraftTracks[id] = L.polyline(cleanTrack, { color: 'blue' }).addTo(map);
    }
  }

  const socket = new WebSocket(`ws://${SERVER}:${API_PORT}/ws`);
  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'positionUpdate') {
        const id = data.aircraftId;
        const heading = data.heading || 0;

        let imageUrl = 'assets/aircraft.svg';
        if (data.receptionState === 'aircraftLost') {
          imageUrl = 'assets/aircraft_lost.svg';
        } else if (data.flightState === 'landing') {
          imageUrl = 'assets/aircraft_landing.svg';
        }

        if (aircraftMarkers[id]) {
          aircraftMarkers[id].setLatLng([data.lat, data.lon]);
          updateMarkerVisual(aircraftMarkers[id], imageUrl, heading);
        }

        if (selectedAircraft === id) {
          if (!aircraftTrackPoints[id]) aircraftTrackPoints[id] = [];
          aircraftTrackPoints[id].push({ lat: data.lat, lon: data.lon });
          const latlngs = aircraftTrackPoints[id].map(p => [p.lat, p.lon]);
          if (aircraftTracks[id]) {
            aircraftTracks[id].setLatLngs(latlngs);
          } else {
            aircraftTracks[id] = L.polyline(latlngs, { color: 'blue' }).addTo(map);
          }

          if (infoBox.style.display === 'block') {
            infoAlt.textContent = data.alt ?? '-';
            infoSpeed.textContent = data.speed != null
              ? Math.round(data.speed * 3.6)
              : '-';
            infoLastTimeSeen.textContent = formatLastTimeSeen(data.lastTimeSeen);
          }
        }
      }
    } catch (e) {
      console.error('WebSocket error:', e);
    }
  };

  loadAircrafts();
  setInterval(loadAircrafts, 5000);
});
