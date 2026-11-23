import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, useMap, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Player } from '@remotion/player';
import {
    Plane, TrainFront, Car, Search, Trash2, ArrowUp, ArrowDown, ImagePlus, X, Film,
    Download, ArrowLeft, Loader2, MapPin, Pencil,
} from 'lucide-react';
import { searchPlaces, reverseLookup } from './geocode';
import { TravelReel } from './TravelReel';
import { REEL, MODES, buildTimeline, distanceKm } from './geo';
import { loadWorld } from './world';

const MODE_ICONS = { flight: Plane, train: TrainFront, car: Car };
const MAX_STOPS = 20;
const MAX_PHOTOS = 4;
const DRAFT_KEY = 'travel-reel-draft-v1';
let nextId = 1;
const newId = () => `${Date.now().toString(36)}-${nextId++}`;

// ---- Drafts: stops and title are kept in this browser; photos are not ----------

function loadDraft() {
    try {
        const d = JSON.parse(localStorage.getItem(DRAFT_KEY) || 'null');
        if (d && Array.isArray(d.stops)) {
            return { title: d.title || 'My trip', stops: d.stops.map((s) => ({ ...s, id: newId(), photos: [] })) };
        }
    } catch {
        // ignore a broken or blocked draft
    }
    return { title: 'My trip', stops: [] };
}

function saveDraft(title, stops) {
    try {
        localStorage.setItem(DRAFT_KEY, JSON.stringify({
            title,
            stops: stops.map(({ name, lat, lng, mode }) => ({ name, lat, lng, mode })),
        }));
    } catch {
        // storage full or blocked: the editor still works
    }
}

// Shrink photos to at most 1600 px so previews and exports stay fast.
async function preparePhoto(file) {
    const bitmap = await createImageBitmap(file);
    const scale = Math.min(1, 1600 / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close?.();
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.88));
    return { id: newId(), url: URL.createObjectURL(blob) };
}

// ---- Map -------------------------------------------------------------------

const markerIcon = (n, active) => L.divIcon({
    className: '',
    iconSize: [34, 34],
    iconAnchor: [17, 17],
    html: `<div style="width:34px;height:34px;border-radius:17px;background:${active ? '#f0c34e' : '#16233f'};color:${active ? '#0a1324' : '#f4f7fc'};border:3px solid #f4f7fc;display:flex;align-items:center;justify-content:center;font:700 14px system-ui;box-shadow:0 4px 12px rgba(0,0,0,.4)">${n}</div>`,
});

const ClickToAdd = ({ onPick, disabled }) => {
    useMapEvents({
        click(e) {
            if (!disabled) onPick(e.latlng.lat, e.latlng.lng);
        },
    });
    return null;
};

const FitToStops = ({ stops, focus }) => {
    const map = useMap();
    const count = stops.length;
    useEffect(() => {
        if (focus) {
            map.flyTo([focus.lat, focus.lng], Math.max(map.getZoom(), 6), { duration: 0.8 });
        } else if (count > 1) {
            map.fitBounds(stops.map((s) => [s.lat, s.lng]), { padding: [60, 60], maxZoom: 9 });
        }
        // Only re-frame when a stop is added or removed, not on every rename
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [count, focus]);
    return null;
};

const TripMap = ({ stops, onPick, adding, focus }) => (
    <MapContainer center={[20, 78]} zoom={4} worldCopyJump className="h-full w-full" style={{ background: '#0a1324' }}>
        <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            maxZoom={18}
        />
        <ClickToAdd onPick={onPick} disabled={adding || stops.length >= MAX_STOPS} />
        <FitToStops stops={stops} focus={focus} />
        {stops.slice(1).map((s, i) => (
            <Polyline
                key={s.id}
                positions={[[stops[i].lat, stops[i].lng], [s.lat, s.lng]]}
                pathOptions={{ color: MODES[s.mode]?.color || '#fff', weight: 4, dashArray: s.mode === 'flight' ? '8 8' : undefined }}
            />
        ))}
        {stops.map((s, i) => (
            <Marker key={s.id} position={[s.lat, s.lng]} icon={markerIcon(i + 1, i === stops.length - 1)} title={s.name} />
        ))}
    </MapContainer>
);

// ---- Small controls ----------------------------------------------------------

const ModePicker = ({ value, onChange, label }) => (
    <div role="radiogroup" aria-label={label} className="flex gap-1">
        {Object.entries(MODES).map(([key, m]) => {
            const Icon = MODE_ICONS[key];
            const on = value === key;
            return (
                <button
                    key={key}
                    type="button"
                    role="radio"
                    aria-checked={on}
                    onClick={() => onChange(key)}
                    className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-bold border transition-colors ${on ? 'text-night' : 'text-snow/80 border-frost/20 hover:border-frost/50'}`}
                    style={on ? { background: m.color, borderColor: m.color } : undefined}
                >
                    <Icon size={14} aria-hidden="true" /> {m.label}
                </button>
            );
        })}
    </div>
);

const StopCard = ({ stop, index, total, onChange, onRemove, onMove, onAddPhotos, onRemovePhoto, prev }) => {
    const fileRef = useRef(null);
    const km = prev ? distanceKm(prev, stop) : 0;
    return (
        <li className="rounded-2xl border border-frost/15 bg-night-card p-3">
            {index > 0 && (
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-frost/10 pb-3">
                    <span className="text-xs text-text-muted">From stop {index} · {Math.round(km).toLocaleString('en-IN')} km by</span>
                    <ModePicker value={stop.mode} onChange={(mode) => onChange({ mode })} label={`How you travelled to stop ${index + 1}`} />
                </div>
            )}
            <div className="flex items-start gap-3">
                <span className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gold text-sm font-extrabold text-night">{index + 1}</span>
                <div className="min-w-0 flex-1">
                    <label className="sr-only" htmlFor={`name-${stop.id}`}>Name of stop {index + 1}</label>
                    <div className="flex items-center gap-1.5">
                        <input
                            id={`name-${stop.id}`}
                            value={stop.name}
                            onChange={(e) => onChange({ name: e.target.value })}
                            className="w-full rounded-lg bg-transparent px-1 py-0.5 font-bold text-snow outline-none focus:bg-night"
                        />
                        <Pencil size={12} className="shrink-0 text-text-muted" aria-hidden="true" />
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2">
                        {stop.photos.map((p) => (
                            <div key={p.id} className="group relative h-14 w-14 overflow-hidden rounded-lg">
                                <img src={p.url} alt="" className="h-full w-full object-cover" />
                                <button
                                    type="button"
                                    onClick={() => onRemovePhoto(p.id)}
                                    className="absolute right-0.5 top-0.5 rounded-full bg-night/80 p-0.5 text-snow"
                                    aria-label="Remove photo"
                                >
                                    <X size={12} />
                                </button>
                            </div>
                        ))}
                        {stop.photos.length < MAX_PHOTOS && (
                            <button
                                type="button"
                                onClick={() => fileRef.current?.click()}
                                className="flex h-14 w-14 flex-col items-center justify-center rounded-lg border border-dashed border-frost/30 text-[10px] text-text-muted hover:border-gold hover:text-gold"
                            >
                                <ImagePlus size={18} aria-hidden="true" /> Photo
                            </button>
                        )}
                        <input
                            ref={fileRef}
                            type="file"
                            accept="image/*"
                            multiple
                            hidden
                            onChange={(e) => { onAddPhotos([...e.target.files]); e.target.value = ''; }}
                        />
                    </div>
                </div>
                <div className="flex shrink-0 flex-col gap-1">
                    <button type="button" onClick={() => onMove(-1)} disabled={index === 0} className="rounded p-1 text-text-muted hover:text-snow disabled:opacity-30" aria-label={`Move stop ${index + 1} up`}><ArrowUp size={16} /></button>
                    <button type="button" onClick={() => onMove(1)} disabled={index === total - 1} className="rounded p-1 text-text-muted hover:text-snow disabled:opacity-30" aria-label={`Move stop ${index + 1} down`}><ArrowDown size={16} /></button>
                    <button type="button" onClick={onRemove} className="rounded p-1 text-text-muted hover:text-holly" aria-label={`Remove stop ${index + 1}`}><Trash2 size={16} /></button>
                </div>
            </div>
        </li>
    );
};

// ---- Reel preview and export ------------------------------------------------

async function pickFormat(renderer) {
    try {
        const mp4 = await renderer.getEncodableVideoCodecs('mp4');
        if (mp4.includes('h264')) return { container: 'mp4', videoCodec: 'h264', ext: 'mp4' };
    } catch {
        // fall through
    }
    const webm = await renderer.getEncodableVideoCodecs('webm');
    const codec = ['vp9', 'vp8', 'av1'].find((c) => webm.includes(c));
    if (!codec) throw new Error('This browser cannot encode video. Try a recent Chrome, Edge or Safari.');
    return { container: 'webm', videoCodec: codec, ext: 'webm' };
}

const ReelView = ({ title, stops, onBack }) => {
    const [ready, setReady] = useState(false);
    const [exporting, setExporting] = useState(null); // { progress } while rendering
    const [download, setDownload] = useState(null);
    const [error, setError] = useState('');
    const abortRef = useRef(null);

    const inputProps = useMemo(() => ({
        title,
        stops: stops.map((s) => ({ name: s.name, lat: s.lat, lng: s.lng, mode: s.mode, photos: s.photos.map((p) => p.url) })),
    }), [title, stops]);
    const duration = useMemo(() => buildTimeline(inputProps.stops).durationInFrames, [inputProps]);

    useEffect(() => {
        loadWorld().then(() => setReady(true)).catch(() => setError('Could not load the world map.'));
    }, []);

    useEffect(() => () => {
        abortRef.current?.abort();
        if (download) URL.revokeObjectURL(download.url);
    }, [download]);

    const startExport = async () => {
        setError('');
        setDownload(null);
        const controller = new AbortController();
        abortRef.current = controller;
        setExporting({ progress: 0 });
        try {
            const renderer = await import('@remotion/web-renderer');
            const format = await pickFormat(renderer);
            const result = await renderer.renderMediaOnWeb({
                composition: {
                    id: 'travel-reel',
                    component: TravelReel,
                    width: REEL.width,
                    height: REEL.height,
                    fps: REEL.fps,
                    durationInFrames: duration,
                },
                inputProps,
                container: format.container,
                videoCodec: format.videoCodec,
                muted: true,
                videoBitrate: 'high',
                signal: controller.signal,
                licenseKey: 'free-license',
                onProgress: (p) => setExporting({ progress: p.progress }),
            });
            const blob = await result.getBlob();
            const safe = title.trim().replace(/[^\w-]+/g, '-').replace(/^-+|-+$/g, '').toLowerCase() || 'trip';
            setDownload({ url: URL.createObjectURL(blob), name: `${safe}-reel.${format.ext}`, size: blob.size });
        } catch (e) {
            if (e?.name !== 'AbortError' && !controller.signal.aborted) setError(e?.message || 'Export failed.');
        } finally {
            setExporting(null);
            abortRef.current = null;
        }
    };

    return (
        <div className="mx-auto flex max-w-5xl flex-col items-center gap-6 px-4 py-6 lg:flex-row lg:items-start">
            <div className="w-full max-w-[380px] shrink-0 overflow-hidden rounded-3xl border border-frost/15 bg-night-card shadow-2xl">
                {ready ? (
                    <Player
                        component={TravelReel}
                        inputProps={inputProps}
                        durationInFrames={duration}
                        fps={REEL.fps}
                        compositionWidth={REEL.width}
                        compositionHeight={REEL.height}
                        style={{ width: '100%', aspectRatio: '9 / 16' }}
                        controls
                        autoPlay
                        loop
                        acknowledgeRemotionLicense
                    />
                ) : (
                    <div className="flex aspect-[9/16] items-center justify-center text-text-muted">
                        <Loader2 className="animate-spin" aria-hidden="true" /> <span className="ml-2">Loading map…</span>
                    </div>
                )}
            </div>

            <div className="w-full flex-1 space-y-4">
                <button type="button" onClick={onBack} className="flex items-center gap-2 text-sm font-bold text-frost hover:text-snow">
                    <ArrowLeft size={16} aria-hidden="true" /> Back to editing
                </button>
                <h2 className="font-display text-3xl font-bold text-snow">{title}</h2>
                <p className="text-text-muted">
                    {stops.length} stops · {(duration / REEL.fps).toFixed(0)} second reel · 1080 × 1920, ready for Reels, Shorts and Stories.
                </p>

                {!exporting && (
                    <button
                        type="button"
                        onClick={startExport}
                        disabled={!ready}
                        className="flex items-center gap-2 rounded-full bg-holly px-6 py-3 font-bold text-snow hover:bg-holly-deep disabled:opacity-50"
                    >
                        <Film size={18} aria-hidden="true" /> Create video file
                    </button>
                )}

                {exporting && (
                    <div className="space-y-2 rounded-2xl border border-frost/15 bg-night-card p-4" aria-live="polite">
                        <div className="flex items-center justify-between text-sm text-snow">
                            <span className="flex items-center gap-2"><Loader2 size={16} className="animate-spin" aria-hidden="true" /> Rendering your reel…</span>
                            <span className="tabular-nums">{Math.round(exporting.progress * 100)}%</span>
                        </div>
                        <div className="h-2 overflow-hidden rounded-full bg-night">
                            <div className="h-full rounded-full bg-gold transition-[width]" style={{ width: `${exporting.progress * 100}%` }} />
                        </div>
                        <p className="text-xs text-text-muted">Keep this tab open. Everything happens on your device; nothing is uploaded.</p>
                        <button type="button" onClick={() => abortRef.current?.abort()} className="text-xs font-bold text-holly hover:underline">Cancel</button>
                    </div>
                )}

                {download && (
                    <a
                        href={download.url}
                        download={download.name}
                        className="inline-flex items-center gap-2 rounded-full bg-pine px-6 py-3 font-bold text-snow hover:bg-pine-deep"
                    >
                        <Download size={18} aria-hidden="true" /> Download {download.name} ({(download.size / 1e6).toFixed(1)} MB)
                    </a>
                )}

                {error && <p role="alert" className="rounded-xl bg-holly/15 p-3 text-sm text-[#ff9aa8]">{error}</p>}
            </div>
        </div>
    );
};

// ---- Page --------------------------------------------------------------------

export default function TravelEditorPage() {
    const initial = useMemo(loadDraft, []);
    const [title, setTitle] = useState(initial.title);
    const [stops, setStops] = useState(initial.stops);
    const [nextMode, setNextMode] = useState('car');
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [searching, setSearching] = useState(false);
    const [adding, setAdding] = useState(false);
    const [message, setMessage] = useState('');
    const [focus, setFocus] = useState(null);
    const [view, setView] = useState('edit');

    useEffect(() => { document.title = 'Trip reel maker · Adarsh Nair'; }, []);
    useEffect(() => { saveDraft(title, stops); }, [title, stops]);

    const addStop = useCallback((place) => {
        setStops((prev) => {
            if (prev.length >= MAX_STOPS) return prev;
            return [...prev, { id: newId(), name: place.name, lat: place.lat, lng: place.lng, mode: prev.length ? nextMode : null, photos: [] }];
        });
        setFocus({ lat: place.lat, lng: place.lng, t: Date.now() });
        setResults([]);
        setQuery('');
        setMessage(`Added ${place.name}.`);
    }, [nextMode]);

    const onMapPick = useCallback(async (lat, lng) => {
        setAdding(true);
        try {
            const name = await reverseLookup(lat, lng);
            addStop({ name, lat, lng });
        } finally {
            setAdding(false);
        }
    }, [addStop]);

    const onSearch = async (e) => {
        e.preventDefault();
        if (!query.trim()) return;
        setSearching(true);
        setMessage('');
        try {
            const found = await searchPlaces(query.trim());
            setResults(found);
            if (!found.length) setMessage('No places found. Try another name, or click the map.');
        } catch {
            setMessage('Place search is unavailable right now. You can still click the map to add places.');
        } finally {
            setSearching(false);
        }
    };

    const update = (id, patch) => setStops((prev) => prev.map((s) => (s.id === id ? { ...s, ...patch } : s)));
    const remove = (id) => setStops((prev) => {
        const gone = prev.find((s) => s.id === id);
        gone?.photos.forEach((p) => URL.revokeObjectURL(p.url));
        const next = prev.filter((s) => s.id !== id);
        if (next[0]) next[0] = { ...next[0], mode: null };
        return next;
    });
    const move = (index, dir) => setStops((prev) => {
        const j = index + dir;
        if (j < 0 || j >= prev.length) return prev;
        const next = [...prev];
        [next[index], next[j]] = [next[j], next[index]];
        // The first stop has no way of arriving; give a moved-down first stop a mode
        return next.map((s, k) => ({ ...s, mode: k === 0 ? null : s.mode || nextMode }));
    });
    const addPhotos = async (id, files) => {
        const stop = stops.find((s) => s.id === id);
        const room = MAX_PHOTOS - (stop?.photos.length || 0);
        const prepared = [];
        for (const f of files.slice(0, room)) {
            try { prepared.push(await preparePhoto(f)); } catch { setMessage(`Could not read ${f.name}.`); }
        }
        setStops((prev) => prev.map((s) => (s.id === id ? { ...s, photos: [...s.photos, ...prepared].slice(0, MAX_PHOTOS) } : s)));
    };
    const removePhoto = (id, photoId) => setStops((prev) => prev.map((s) => {
        if (s.id !== id) return s;
        s.photos.filter((p) => p.id === photoId).forEach((p) => URL.revokeObjectURL(p.url));
        return { ...s, photos: s.photos.filter((p) => p.id !== photoId) };
    }));
    const clearAll = () => {
        stops.forEach((s) => s.photos.forEach((p) => URL.revokeObjectURL(p.url)));
        setStops([]);
        setMessage('Trip cleared.');
    };

    const totalKm = stops.slice(1).reduce((sum, s, i) => sum + distanceKm(stops[i], s), 0);

    return (
        <div className="min-h-screen bg-night text-snow">
            <header className="flex items-center justify-between gap-3 border-b border-frost/10 bg-night-soft px-4 py-3">
                <a href="/" className="flex items-center gap-2 text-sm font-bold text-frost hover:text-snow">
                    <ArrowLeft size={16} aria-hidden="true" /> adarshnair.online
                </a>
                <h1 className="font-display text-lg font-bold md:text-xl">Trip reel maker</h1>
                <span className="hidden text-xs text-text-muted sm:block">Photos stay on your device</span>
            </header>

            {view === 'reel' ? (
                <ReelView title={title} stops={stops} onBack={() => setView('edit')} />
            ) : (
                <div className="grid md:h-[calc(100vh-57px)] md:grid-cols-[400px_1fr]">
                    <aside className="order-2 flex flex-col gap-4 overflow-y-auto p-4 md:order-1">
                        <div>
                            <label htmlFor="trip-title" className="text-xs font-bold uppercase tracking-widest text-text-muted">Trip name</label>
                            <input
                                id="trip-title"
                                value={title}
                                maxLength={40}
                                onChange={(e) => setTitle(e.target.value)}
                                className="mt-1 w-full rounded-xl border border-frost/15 bg-night-card px-3 py-2 font-bold outline-none focus:border-gold"
                            />
                        </div>

                        <form onSubmit={onSearch} className="space-y-2">
                            <label htmlFor="place-search" className="text-xs font-bold uppercase tracking-widest text-text-muted">
                                {stops.length ? 'Add the next place' : 'Where did your trip start?'}
                            </label>
                            <div className="flex gap-2">
                                <input
                                    id="place-search"
                                    value={query}
                                    onChange={(e) => setQuery(e.target.value)}
                                    placeholder="Search a city or place"
                                    className="min-w-0 flex-1 rounded-xl border border-frost/15 bg-night-card px-3 py-2 outline-none focus:border-gold"
                                    disabled={stops.length >= MAX_STOPS}
                                />
                                <button type="submit" disabled={searching || stops.length >= MAX_STOPS} className="rounded-xl bg-gold px-3 text-night disabled:opacity-50" aria-label="Search">
                                    {searching ? <Loader2 size={18} className="animate-spin" /> : <Search size={18} />}
                                </button>
                            </div>
                            {stops.length > 0 && (
                                <div className="flex flex-wrap items-center gap-2">
                                    <span className="text-xs text-text-muted">Getting there by</span>
                                    <ModePicker value={nextMode} onChange={setNextMode} label="How you travel to the next place" />
                                </div>
                            )}
                            <p className="flex items-center gap-1.5 text-xs text-text-muted">
                                <MapPin size={12} aria-hidden="true" /> Or click anywhere on the map.{adding ? ' Looking up the place…' : ''}
                            </p>
                        </form>

                        {results.length > 0 && (
                            <ul className="space-y-1" aria-label="Search results">
                                {results.map((r, i) => (
                                    <li key={`${r.lat},${r.lng},${i}`}>
                                        <button type="button" onClick={() => addStop(r)} className="w-full rounded-xl px-3 py-2 text-left hover:bg-night-card">
                                            <div className="font-bold">{r.name}</div>
                                            <div className="truncate text-xs text-text-muted">{r.detail}</div>
                                        </button>
                                    </li>
                                ))}
                            </ul>
                        )}

                        <p aria-live="polite" className="min-h-[1rem] text-xs text-frost">{message}</p>

                        {stops.length > 0 && (
                            <>
                                <div className="flex items-center justify-between text-xs text-text-muted">
                                    <span>{stops.length} stop{stops.length > 1 ? 's' : ''} · {Math.round(totalKm).toLocaleString('en-IN')} km</span>
                                    <button type="button" onClick={clearAll} className="font-bold hover:text-holly">Clear trip</button>
                                </div>
                                <ol className="space-y-3">
                                    {stops.map((s, i) => (
                                        <StopCard
                                            key={s.id}
                                            stop={s}
                                            prev={stops[i - 1]}
                                            index={i}
                                            total={stops.length}
                                            onChange={(patch) => update(s.id, patch)}
                                            onRemove={() => remove(s.id)}
                                            onMove={(dir) => move(i, dir)}
                                            onAddPhotos={(files) => addPhotos(s.id, files)}
                                            onRemovePhoto={(pid) => removePhoto(s.id, pid)}
                                        />
                                    ))}
                                </ol>
                            </>
                        )}

                        <button
                            type="button"
                            onClick={() => setView('reel')}
                            disabled={stops.length < 2}
                            className="sticky bottom-0 mt-auto flex items-center justify-center gap-2 rounded-full bg-holly px-6 py-3 font-bold text-snow shadow-lg hover:bg-holly-deep disabled:cursor-not-allowed disabled:opacity-40"
                        >
                            <Film size={18} aria-hidden="true" /> {stops.length < 2 ? 'Add at least 2 places' : 'Done, make my reel'}
                        </button>
                    </aside>

                    <div className="order-1 h-[50vh] md:order-2 md:h-full">
                        <TripMap stops={stops} onPick={onMapPick} adding={adding} focus={focus} />
                    </div>
                </div>
            )}
        </div>
    );
}
