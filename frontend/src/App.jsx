import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity, ArrowUpRight, Check, ChevronLeft,
  ChevronRight, CircleAlert, Clock3, Crosshair, Database,
  LoaderCircle, MapPin, Navigation, RefreshCw, Route, Search, Sparkles,
  Wifi,
} from 'lucide-react'
import { CircleMarker, MapContainer, Polyline, TileLayer, Tooltip, useMap } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')
const PARTITION = 'student_test'
const PAGE_SIZE = 40

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options)
  if (!response.ok) {
    let detail = `Request failed (${response.status})`
    try {
      const body = await response.json()
      detail = body.detail || body.message || detail
    } catch { /* use the status text */ }
    throw new Error(detail)
  }
  return response.json()
}

function extractTrips(payload) {
  const values = Array.isArray(payload) ? payload : payload?.trips
  if (!Array.isArray(values)) return []
  return values.map((trip) => typeof trip === 'string' ? trip : trip?.trip_id ?? trip?.id)
    .filter((tripId) => typeof tripId === 'string' && tripId.length > 0)
}

function gpsToLatLngs(coordinates) {
  if (!Array.isArray(coordinates)) return []
  return coordinates.map((point) => {
    if (Array.isArray(point)) return [Number(point[1]), Number(point[0])]
    return [Number(point?.lat), Number(point?.lon)]
  }).filter(([lat, lon]) => Number.isFinite(lat) && Number.isFinite(lon))
}

function geometryToLines(geometry) {
  if (geometry?.type !== 'MultiLineString' || !Array.isArray(geometry.coordinates)) return []
  return geometry.coordinates.map((line) => gpsToLatLngs(line)).filter((line) => line.length > 0)
}

function FitMap({ gps, lines }) {
  const map = useMap()
  useEffect(() => {
    const fit = () => {
      const positions = [...gps, ...lines.flat()]
      if (positions.length === 1) map.setView(positions[0], 15)
      else if (positions.length > 1) map.fitBounds(positions, { padding: [36, 36], maxZoom: 16 })
    }
    fit()
    window.addEventListener('pathsnap-fit-map', fit)
    return () => window.removeEventListener('pathsnap-fit-map', fit)
  }, [gps, lines, map])
  return null
}

function Metric({ icon: Icon, label, value, helper, accent = false }) {
  return (
    <div className={`metric-card${accent ? ' metric-card-accent' : ''}`}>
      <div className="metric-top"><span className="metric-icon"><Icon size={16} /></span><span>{label}</span></div>
      <div className="metric-value">{value}</div>
      <div className="metric-helper">{helper}</div>
    </div>
  )
}

function App() {
  const [trips, setTrips] = useState([])
  const [selectedTrip, setSelectedTrip] = useState('')
  const [tripInfo, setTripInfo] = useState(null)
  const [match, setMatch] = useState(null)
  const [apiStatus, setApiStatus] = useState('checking')
  const [listLoading, setListLoading] = useState(true)
  const [infoLoading, setInfoLoading] = useState(false)
  const [matchLoading, setMatchLoading] = useState(false)
  const [listError, setListError] = useState('')
  const [tripError, setTripError] = useState('')
  const [search, setSearch] = useState('')
  const [routePage, setRoutePage] = useState(0)
  const [tablePage, setTablePage] = useState(0)

  const loadTrips = useCallback(async () => {
    setListLoading(true)
    setListError('')
    try {
      const payload = await apiRequest(`/trips?partition=${PARTITION}`)
      const tripIds = extractTrips(payload)
      setTrips(tripIds)
      setSelectedTrip((current) => tripIds.includes(current) ? current : tripIds[0] || '')
    } catch (error) {
      setListError(error.message)
    } finally {
      setListLoading(false)
    }
  }, [])

  useEffect(() => {
    let active = true
    const checkHealth = async () => {
      try {
        await apiRequest('/health')
        if (active) setApiStatus('connected')
      } catch {
        if (active) setApiStatus('offline')
      }
    }
    checkHealth()
    const timer = window.setInterval(checkHealth, 15000)
    loadTrips()
    return () => { active = false; window.clearInterval(timer) }
  }, [loadTrips])

  useEffect(() => {
    if (!selectedTrip) {
      setTripInfo(null)
      setMatch(null)
      return undefined
    }
    const controller = new AbortController()
    setTripInfo(null)
    setMatch(null)
    setTablePage(0)
    setTripError('')
    setInfoLoading(true)
    apiRequest(`/trips/${encodeURIComponent(selectedTrip)}/info?partition=${PARTITION}`, { signal: controller.signal })
      .then(setTripInfo)
      .catch((error) => { if (error.name !== 'AbortError') setTripError(error.message) })
      .finally(() => { if (!controller.signal.aborted) setInfoLoading(false) })
    return () => controller.abort()
  }, [selectedTrip])

  const filteredTrips = useMemo(() => trips.filter((tripId) => tripId.toLowerCase().includes(search.trim().toLowerCase())), [trips, search])
  useEffect(() => setRoutePage(0), [search])
  const pageCount = Math.max(1, Math.ceil(filteredTrips.length / PAGE_SIZE))
  const visibleTrips = filteredTrips.slice(routePage * PAGE_SIZE, (routePage + 1) * PAGE_SIZE)

  const gps = useMemo(() => gpsToLatLngs(match?.gps_coordinates ?? tripInfo?.gps_coordinates), [match, tripInfo])
  const lines = useMemo(() => geometryToLines(match?.route_geometry), [match])
  const predictionRows = useMemo(() => {
    if (!match) return []
    const arcs = Array.isArray(match.predicted_arc_ids) ? match.predicted_arc_ids : []
    const confidence = Array.isArray(match.confidence_scores) ? match.confidence_scores : []
    return arcs.map((arcId, index) => ({ index, arcId, confidence: confidence[index] }))
  }, [match])

  const observedCount = match?.gps_coordinates?.length ?? tripInfo?.gps_observations ?? gps.length
  const predictedRows = predictionRows.filter((row) => row.arcId !== null && row.arcId !== undefined).length
  const uniqueArcs = new Set(predictionRows.filter((row) => row.arcId !== null && row.arcId !== undefined).map((row) => row.arcId)).size
  const missingPredictions = Math.max(0, Number(observedCount || 0) - predictedRows)
  const confidenceValues = predictionRows.map(({ confidence }) => Number(confidence)).filter((value) => Number.isFinite(value))
  const averageConfidence = confidenceValues.length
    ? `${(confidenceValues.reduce((sum, value) => sum + value, 0) / confidenceValues.length * 100).toFixed(1)}%`
    : '—'

  async function runMatching() {
    if (!selectedTrip) return
    setMatchLoading(true)
    setTripError('')
    try {
      const result = await apiRequest(`/trips/${encodeURIComponent(selectedTrip)}/match?partition=${PARTITION}`, { method: 'POST' })
      setMatch(result)
      setTripInfo((current) => current ? { ...current, gps_coordinates: [] } : current)
    } catch (error) {
      setTripError(error.message)
    } finally {
      setMatchLoading(false)
    }
  }

  const mapCenter = gps[0] || [37.7749, -122.4194]
  const statusText = apiStatus === 'connected' ? 'API connected' : apiStatus === 'offline' ? 'API unavailable' : 'Connecting to API'

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="PathSnap home">
          <span className="brand-mark"><Route size={19} strokeWidth={2.5} /></span>
          <span className="brand-name">path<span>snap</span><small>MAP MATCHING EVALUATOR</small></span>
        </a>
        <div className="topbar-right">
          <div className="workspace-label"><span className="live-dot" /> EVALUATION WORKSPACE</div>
          <div className={`connection-pill connection-${apiStatus}`}><span className="connection-dot" />{statusText}</div>
        </div>
      </header>

      <main id="top" className="dashboard">
        <section className="intro-row">
          <div>
            <div className="eyebrow"><span className="eyebrow-line" /> STUDENT TEST PARTITION</div>
            <h1>Route evaluator<span className="title-period">.</span></h1>
            <p className="intro-copy">Inspect GPS traces and classifier predictions, one trip at a time.</p>
          </div>
          <div className="partition-badge"><Database size={15} /><span>student_test</span><b>{listLoading ? '…' : `${trips.length} trips`}</b></div>
        </section>

        <div className="workspace-grid">
          <aside className="route-panel panel">
            <div className="panel-heading route-heading">
              <div><div className="section-kicker">TRIP EXPLORER</div><h2>Test routes <span className="count-bubble">{trips.length}</span></h2></div>
              <button className="icon-button" type="button" aria-label="Refresh trips" onClick={loadTrips} disabled={listLoading}><RefreshCw size={16} className={listLoading ? 'spin' : ''} /></button>
            </div>
            <label className="search-box"><Search size={16} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Find a trip…" aria-label="Search test routes" /></label>
              <div className="route-list-label"><span>AVAILABLE TRIPS</span><span>{filteredTrips.length ? `${routePage * PAGE_SIZE + 1}–${Math.min((routePage + 1) * PAGE_SIZE, filteredTrips.length)} OF ${filteredTrips.length}` : '0 TRIPS'}</span></div>
            <div className="route-list" aria-label="Student test trips">
              {listLoading && <div className="list-message"><LoaderCircle className="spin" size={19} /> Loading trips…</div>}
              {!listLoading && listError && <div className="list-message list-error"><CircleAlert size={18} /><span>{listError}<button onClick={loadTrips}>Retry</button></span></div>}
              {!listLoading && !listError && visibleTrips.map((tripId, index) => (
                <button key={tripId} className={`trip-row${selectedTrip === tripId ? ' selected' : ''}`} onClick={() => setSelectedTrip(tripId)} aria-current={selectedTrip === tripId ? 'true' : undefined}>
                  <span className="trip-index">{String(routePage * PAGE_SIZE + index + 1).padStart(2, '0')}</span>
                  <span className="trip-route-icon"><Navigation size={15} /></span>
                  <span className="trip-details"><strong>{tripId}</strong><small>{selectedTrip === tripId && tripInfo ? `${tripInfo.gps_observations} GPS observations` : 'Student test route'}</small></span>
                  {selectedTrip === tripId && <span className="selected-check"><Check size={14} /></span>}
                  {selectedTrip !== tripId && <ChevronRight className="trip-chevron" size={15} />}
                </button>
              ))}
              {!listLoading && !listError && filteredTrips.length === 0 && <div className="list-message">{trips.length ? 'No trips match your search.' : 'No trips returned by the API.'}</div>}
            </div>
            <div className="route-panel-footer">
              <span><span className="footer-status-dot" /> Live API data</span>
              <div className="pagination"><button aria-label="Previous page" disabled={routePage === 0} onClick={() => setRoutePage((value) => Math.max(0, value - 1))}><ChevronLeft size={15} /></button><span>{routePage + 1} / {pageCount}</span><button aria-label="Next page" disabled={routePage + 1 >= pageCount} onClick={() => setRoutePage((value) => Math.min(pageCount - 1, value + 1))}><ChevronRight size={15} /></button></div>
            </div>
          </aside>

          <section className="main-column">
            <div className="selected-trip-bar panel">
              <div className="selected-trip-id"><span className="trip-pin"><MapPin size={17} /></span><div><span className="section-kicker">SELECTED TRIP</span><strong>{selectedTrip || 'Select a trip'}</strong></div></div>
              <div className="trip-meta"><span className="meta-item"><Clock3 size={14} /> {infoLoading ? 'Loading…' : tripInfo ? `${tripInfo.gps_observations} observations` : '— observations'}</span><span className="meta-divider" /><span className="partition-mini">STUDENT TEST</span></div>
              <button className="run-button" onClick={runMatching} disabled={!selectedTrip || matchLoading || apiStatus !== 'connected'}>
                {matchLoading ? <LoaderCircle size={16} className="spin" /> : <Sparkles size={16} />}
                {matchLoading ? 'Matching…' : 'Run map matching'}<ArrowUpRight size={15} className="run-arrow" />
              </button>
            </div>

            {tripError && <div className="inline-error"><CircleAlert size={17} /><span>{tripError}</span><button onClick={() => setTripError('')} aria-label="Dismiss error">×</button></div>}

            <div className="metrics-grid">
              <Metric icon={Crosshair} label="GPS observations" value={observedCount ?? '—'} helper="Raw trajectory points" />
              <Metric icon={Activity} label="Predicted observations" value={match ? predictedRows : '—'} helper={match ? `of ${observedCount} GPS points` : 'Run matching to populate'} />
              <Metric icon={Route} label="Unique predicted arcs" value={match ? uniqueArcs : '—'} helper="Distinct road segments" />
              <Metric icon={CircleAlert} label="Missing predictions" value={match ? missingPredictions : '—'} helper="Observations without an arc" />
              <Metric icon={Sparkles} label="Average confidence" value={match ? averageConfidence : '—'} helper="Classifier score · not accuracy" accent />
            </div>

            <section className="map-panel panel">
              <div className="map-header">
                <div><div className="section-kicker">TRAJECTORY VIEW</div><h2>Route map</h2></div>
                <div className="map-header-right">{match && <span className="classifier-tag"><span className="tag-dot" /> CLASSIFIER ONLY</span>}<button className="icon-button map-reset" aria-label="Fit route to map" title="Fit route to map" onClick={() => window.dispatchEvent(new Event('pathsnap-fit-map'))}><Crosshair size={16} /></button></div>
              </div>
              <div className="map-wrap">
                {gps.length > 0 ? (
                  <MapContainer key={selectedTrip} center={mapCenter} zoom={13} scrollWheelZoom className="leaflet-map">
                    <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                    <FitMap gps={gps} lines={lines} />
                    <Polyline positions={gps} pathOptions={{ color: '#3385e2', weight: 4, opacity: 0.92, lineCap: 'round', lineJoin: 'round' }} />
                    {lines.map((line, index) => <Polyline key={`arc-line-${index}`} positions={line} pathOptions={{ color: '#16a978', weight: 5, opacity: 0.88, lineCap: 'round', lineJoin: 'round' }} />)}
                    <CircleMarker center={gps[0]} radius={8} pathOptions={{ color: '#fff', weight: 3, fillColor: '#20a778', fillOpacity: 1 }}><Tooltip>Trip start</Tooltip></CircleMarker>
                    <CircleMarker center={gps[gps.length - 1]} radius={8} pathOptions={{ color: '#fff', weight: 3, fillColor: '#ef6b57', fillOpacity: 1 }}><Tooltip>Trip end</Tooltip></CircleMarker>
                  </MapContainer>
                ) : (
                  <div className="map-empty"><span className="empty-map-art"><MapPin size={25} /><span /></span><strong>{infoLoading ? 'Loading trajectory' : selectedTrip ? 'No GPS coordinates available' : 'Choose a test route'}</strong><p>{infoLoading ? 'Fetching trip points from the API…' : 'Select a route to preview its GPS trace on the map.'}</p>{infoLoading && <LoaderCircle size={18} className="spin" />}</div>
                )}
                <div className="map-legend"><div className="legend-title">MAP LEGEND</div><div><span className="legend-line gps-line" />GPS trajectory</div><div><span className="legend-line predicted-line" />Predicted road geometry</div><div><span className="legend-marker start-marker" />Start point</div><div><span className="legend-marker end-marker" />End point</div></div>
                {match && lines.length === 0 && <div className="geometry-note"><CircleAlert size={14} /> No predicted road geometry returned</div>}
              </div>
              <div className="map-caption"><span><span className="tile-dot" /> OpenStreetMap</span><span>{gps.length ? `${gps.length.toLocaleString()} GPS points` : 'Awaiting trip data'}{match ? ` · ${lines.length.toLocaleString()} geometry segments` : ''}</span></div>
            </section>

            <section className="details-panel panel">
              <div className="details-heading"><div><div className="section-kicker">OBSERVATION-LEVEL OUTPUT</div><h2>Prediction details</h2></div>{match && <span className="rows-count">{predictionRows.length.toLocaleString()} observations</span>}</div>
              {!match ? <div className="table-empty"><span className="table-empty-icon"><Activity size={18} /></span><span><strong>No predictions yet</strong><small>Run map matching to inspect each observation.</small></span></div> : (
                <>
                  <div className="table-scroll"><table><thead><tr><th>OBSERVATION</th><th>PREDICTED ARC ID</th><th>CONFIDENCE <span className="th-note">· NOT ACCURACY</span></th></tr></thead><tbody>
                    {predictionRows.slice(tablePage * PAGE_SIZE, tablePage * PAGE_SIZE + PAGE_SIZE).map(({ index, arcId, confidence }) => <tr key={index}><td><span className="observation-index">{String(index).padStart(3, '0')}</span></td><td>{arcId === null || arcId === undefined ? <span className="null-value">No prediction</span> : <span className="arc-chip">{arcId}</span>}</td><td>{confidence === null || confidence === undefined || !Number.isFinite(Number(confidence)) ? <span className="null-value">—</span> : <span className="confidence-cell"><span className="confidence-track"><span style={{ width: `${Math.max(0, Math.min(1, Number(confidence))) * 100}%` }} /></span>{(Number(confidence) * 100).toFixed(1)}%</span>}</td></tr>)}
                    {!predictionRows.length && <tr><td colSpan="3" className="table-no-rows">The API returned no observation predictions.</td></tr>}
                  </tbody></table></div>
                  {predictionRows.length > PAGE_SIZE && <div className="table-pagination"><span>Showing {tablePage * PAGE_SIZE + 1}–{Math.min((tablePage + 1) * PAGE_SIZE, predictionRows.length)} of {predictionRows.length.toLocaleString()}</span><div><button aria-label="Previous observations" onClick={() => setTablePage((value) => Math.max(0, value - 1))} disabled={tablePage === 0}><ChevronLeft size={14} /></button><button aria-label="Next observations" onClick={() => setTablePage((value) => Math.min(Math.ceil(predictionRows.length / PAGE_SIZE) - 1, value + 1))} disabled={tablePage + 1 >= Math.ceil(predictionRows.length / PAGE_SIZE)}><ChevronRight size={14} /></button></div></div>}
                </>
              )}
              {match && <div className="status-strip"><span className="status-strip-label"><Check size={14} /> INFERENCE STATUS</span><span className="status-chip"><b>Method</b>{match.prediction_method || 'unknown'}</span><span className="status-chip"><b>Decoder</b>{match.decoder_status || 'unknown'}</span><span className="classifier-disclaimer"><Sparkles size={13} /> Classifier-only output; HMM/Viterbi is not implemented.</span></div>}
            </section>
            <footer className="page-footer"><span>PathSnap <b>·</b> Map matching evaluation</span><span><Wifi size={13} /> Connected to {API_BASE}</span></footer>
          </section>
        </div>
      </main>
    </div>
  )
}

export default App
