import { useMemo } from 'react';
import { AbsoluteFill, Img, interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { geoMercator, geoPath, geoGraticule10 } from 'd3-geo';
import { Plane, TrainFront, Car, MapPin } from 'lucide-react';
import {
    REEL, MODES, TIMING, buildTimeline, cameraAt, segmentAt, legProgress, legPath,
    toPathD, headingDeg, project, clamp01, pointBack,
} from './geo';
import { getWorld } from './world';

const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';
const C = {
    sea: '#0b1628',
    land: '#1c2b45',
    border: '#2e4366',
    grid: '#13213a',
    text: '#f4f7fc',
    muted: '#a6b4cc',
    gold: '#f0c34e',
    panel: 'rgba(10, 19, 36, 0.82)',
};
const ICONS = { flight: Plane, train: TrainFront, car: Car };

const fmtKm = (km) => `${Math.round(km).toLocaleString('en-IN')} km`;

// ---------------------------------------------------------------------------
// Map layer: land and borders, clipped to the frame so each frame stays light.

const MapLayer = ({ cam }) => {
    const world = getWorld();
    const paths = useMemo(() => {
        const centerLng = (cam.cx * 180) / Math.PI;
        const projection = geoMercator()
            .rotate([-centerLng, 0])
            .scale(cam.scale)
            .translate([REEL.width / 2, REEL.height / 2 + cam.cy * cam.scale])
            .clipExtent([[-10, -10], [REEL.width + 10, REEL.height + 10]])
            .precision(cam.scale > 5000 ? 0.2 : 0.6);
        const path = geoPath(projection);
        return {
            grid: path(geoGraticule10()),
            land: world ? path(world.land) : null,
            borders: world ? path(world.borders) : null,
        };
    }, [cam.cx, cam.cy, cam.scale, world]);

    return (
        <svg width={REEL.width} height={REEL.height} style={{ position: 'absolute', inset: 0 }}>
            <rect width={REEL.width} height={REEL.height} fill={C.sea} />
            {paths.grid && <path d={paths.grid} fill="none" stroke={C.grid} strokeWidth={1.5} />}
            {paths.land && <path d={paths.land} fill={C.land} />}
            {paths.borders && <path d={paths.borders} fill="none" stroke={C.border} strokeWidth={1.5} />}
        </svg>
    );
};

// ---------------------------------------------------------------------------
// Route layer: finished legs, the leg in progress, stop markers and the vehicle.

const RouteLayer = ({ timeline, frame, cam }) => {
    const { segments, stops } = timeline;
    const idx = segmentAt(segments, frame);
    const current = segments[idx];
    const isOutro = current.type === 'outro';

    const legs = segments.filter((s) => s.type === 'leg');
    const drawn = [];
    legs.forEach((leg) => {
        let progress;
        if (isOutro || frame >= leg.start + leg.dur) progress = 1;
        else if (frame < leg.start) progress = 0;
        else progress = legProgress(leg, frame);
        if (progress <= 0) return;
        const pts = legPath(cam, leg, progress);
        drawn.push({ leg, d: toPathD(pts) });
    });

    // A stop is "reached" once the vehicle gets there
    const reached = stops.map((_, i) => {
        if (i === 0) return frame >= (segments[1]?.start ?? 0);
        const leg = legs.find((l) => l.index === i);
        return isOutro || frame >= leg.start + leg.dur * 0.9;
    });
    const activeStop = current.type === 'stop' ? current.index : null;

    return (
        <svg width={REEL.width} height={REEL.height} style={{ position: 'absolute', inset: 0 }}>
            {drawn.map(({ leg, d }) => {
                const color = MODES[leg.mode].color;
                return (
                    <g key={leg.index}>
                        <path d={d} fill="none" stroke="#000" strokeOpacity={0.35} strokeWidth={14} strokeLinecap="round" strokeLinejoin="round" />
                        {leg.mode === 'flight' && (
                            <path d={d} fill="none" stroke={color} strokeWidth={7} strokeLinecap="round" strokeLinejoin="round" />
                        )}
                        {leg.mode === 'train' && (
                            <>
                                {/* sleepers across the track, then the two rails */}
                                <path d={d} fill="none" stroke={color} strokeOpacity={0.55} strokeWidth={22} strokeDasharray="5 13" />
                                <path d={d} fill="none" stroke={color} strokeWidth={12} strokeLinecap="round" strokeLinejoin="round" />
                                <path d={d} fill="none" stroke="#0b1628" strokeWidth={5} strokeLinecap="round" strokeLinejoin="round" />
                            </>
                        )}
                        {leg.mode === 'car' && (
                            <path d={d} fill="none" stroke={color} strokeWidth={8} strokeLinecap="round" strokeLinejoin="round" />
                        )}
                    </g>
                );
            })}

            {stops.map((s, i) => {
                const [x, y] = project(cam, s.lng, s.lat);
                if (x < -60 || x > REEL.width + 60 || y < -60 || y > REEL.height + 60) return null;
                const on = reached[i];
                const active = activeStop === i;
                const pulse = active ? 1 + 0.25 * Math.sin((frame - current.start) / 5) : 1;
                return (
                    <g key={i} transform={`translate(${x} ${y})`}>
                        {active && <circle r={44 * pulse} fill={C.gold} opacity={0.22} />}
                        <circle r={26} fill={on ? C.gold : '#22324f'} stroke={on ? '#fff' : '#4c6187'} strokeWidth={4} />
                        <text y={10} textAnchor="middle" fontSize={28} fontWeight={800} fontFamily={FONT} fill={on ? '#0b1628' : C.muted}>{i + 1}</text>
                    </g>
                );
            })}
        </svg>
    );
};

// ---- Vehicles ------------------------------------------------------------------
// A "lub-dub" heartbeat (about one beat a second) that makes the vehicle feel alive,
// with a ring rippling out on every beat.

const BEAT_FRAMES = 30;
const bump = (t, centre, width) => Math.exp(-(((t - centre) / width) ** 2));
const heartbeat = (frame) => {
    const t = (frame % BEAT_FRAMES) / BEAT_FRAMES;
    return {
        scale: 1 + 0.2 * (bump(t, 0.08, 0.05) + 0.65 * bump(t, 0.3, 0.05)),
        ripple: t, // 0..1 through the beat
    };
};

const Ripple = ({ beat, color, base = 62 }) => (
    <circle r={base + beat.ripple * 70} fill="none" stroke={color} strokeWidth={8 * (1 - beat.ripple)} opacity={0.7 * (1 - beat.ripple)} />
);

// Cartoon car seen from the side, facing right. Wheels spin, body bobs.
const CarArt = ({ frame }) => {
    const bob = Math.sin(frame * 0.9) * 1.6;
    const spin = (frame * 24) % 360;
    const wheel = (cx) => (
        <g transform={`translate(${cx} 16)`}>
            <circle r={11} fill="#1d2433" stroke="#0b1628" strokeWidth={3} />
            <circle r={4.5} fill="#cfd8e3" />
            <g transform={`rotate(${spin})`} stroke="#cfd8e3" strokeWidth={2.4} strokeLinecap="round">
                <line x1={-7} y1={0} x2={7} y2={0} />
                <line x1={0} y1={-7} x2={0} y2={7} />
            </g>
        </g>
    );
    return (
        <g>
            <ellipse cx={2} cy={29} rx={40} ry={5} fill="#000" opacity={0.25} />
            <g transform={`translate(0 ${bob.toFixed(2)})`}>
                <path d="M-42 8 Q-43 -5 -31 -7 L-19 -7 L-9 -23 Q-6 -27 -1 -27 L17 -27 Q23 -27 27 -21 L36 -7 Q45 -6 45 4 L45 12 Q45 17 40 17 L-37 17 Q-42 17 -42 12 Z"
                    fill={MODES.car.color} stroke="#0b1628" strokeWidth={3.5} strokeLinejoin="round" />
                <path d="M-5 -21 L1 -21 L1 -9 L-13 -9 Z" fill="#bfe6ff" stroke="#0b1628" strokeWidth={2.5} strokeLinejoin="round" />
                <path d="M5 -21 L16 -21 Q20 -21 23 -16 L27 -9 L5 -9 Z" fill="#bfe6ff" stroke="#0b1628" strokeWidth={2.5} strokeLinejoin="round" />
                <path d="M-36 -2 L30 -2" stroke="#fff" strokeOpacity={0.45} strokeWidth={3} strokeLinecap="round" />
                <circle cx={41} cy={2} r={4} fill={C.gold} stroke="#0b1628" strokeWidth={2} />
                <rect x={-43} y={4} width={7} height={5} rx={2} fill="#ffd2d8" />
            </g>
            {wheel(-22)}
            {wheel(25)}
        </g>
    );
};

// Cartoon plane seen from above, nose pointing right.
const PlaneArt = () => (
    <g>
        <path d="M2 -8 L-14 -46 Q-16 -50 -11 -50 L-5 -50 L24 -8 Z" fill={C.gold} stroke="#0b1628" strokeWidth={3} strokeLinejoin="round" />
        <path d="M2 8 L-14 46 Q-16 50 -11 50 L-5 50 L24 8 Z" fill={C.gold} stroke="#0b1628" strokeWidth={3} strokeLinejoin="round" />
        <path d="M-30 -5 L-42 -21 L-35 -22 L-20 -5 Z" fill={C.gold} stroke="#0b1628" strokeWidth={2.5} strokeLinejoin="round" />
        <path d="M-30 5 L-42 21 L-35 22 L-20 5 Z" fill={C.gold} stroke="#0b1628" strokeWidth={2.5} strokeLinejoin="round" />
        <path d="M-42 0 Q-42 -9 -30 -10 L28 -10 Q46 -9 48 0 Q46 9 28 10 L-30 10 Q-42 9 -42 0 Z" fill="#f4f7fc" stroke="#0b1628" strokeWidth={3.5} />
        {[-20, -11, -2, 7, 16].map((wx) => <circle key={wx} cx={wx} cy={0} r={2.6} fill="#5aa9e6" />)}
        <path d="M32 -6 Q42 -4 43 0 Q42 4 32 6 Z" fill="#5aa9e6" stroke="#0b1628" strokeWidth={2} />
    </g>
);

const Badge = ({ x, y, color, beat, appear, children }) => (
    <g transform={`translate(${x.toFixed(1)} ${y.toFixed(1)})`} opacity={appear}>
        <Ripple beat={beat} color={color} />
        <g transform={`scale(${(beat.scale * (0.6 + 0.4 * appear)).toFixed(3)})`}>
            <circle cy={6} r={60} fill="#000" opacity={0.3} />
            <circle r={60} fill={color} stroke="#fff" strokeWidth={6} />
            <circle r={48} fill="#fff" opacity={0.22} />
            {children}
        </g>
    </g>
);

// A cartoon train (engine plus three colourful carriages) that follows the curve of the track.
const TRAIN_PARTS = [{ len: 112, engine: true }, { len: 90, roof: C.gold }, { len: 90, roof: MODES.car.color }, { len: 90, roof: '#4cc38a' }];
const TrainSprite = ({ pts, frame, appear }) => {
    const items = [];
    let back = 0;
    TRAIN_PARTS.forEach((part) => {
        items.push({ ...part, ...pointBack(pts, back + part.len / 2) });
        back += part.len + 12;
    });
    const engine = items[0];
    const rad = (engine.angle * Math.PI) / 180;
    const beat = heartbeat(frame);
    return (
        <svg width={REEL.width} height={REEL.height} style={{ position: 'absolute', inset: 0, opacity: appear }}>
            <g transform={`translate(${engine.x.toFixed(1)} ${engine.y.toFixed(1)})`}>
                <Ripple beat={beat} color={MODES.train.color} base={66} />
            </g>
            {[0, 1, 2].map((k) => {
                // Smoke puffs from the chimney, drifting up and fading
                const phase = ((frame * 0.9 + k * 13) % 39) / 39;
                const x = engine.x - Math.cos(rad) * 40 + phase * 16;
                const y = engine.y - Math.sin(rad) * 40 - 10 - phase * 80;
                return <circle key={k} cx={x} cy={y} r={12 + phase * 18} fill="#f4f7fc" stroke="#cfd8e3" strokeWidth={2} opacity={(1 - phase) * 0.8} />;
            })}
            {items.slice(1).reverse().map((it, i) => (
                <g key={i} transform={`translate(${it.x.toFixed(1)} ${it.y.toFixed(1)}) rotate(${it.angle.toFixed(1)})`}>
                    <rect x={-45} y={-23} width={90} height={46} rx={13} fill="#f4f7fc" stroke="#0b1628" strokeWidth={4.5} />
                    <rect x={-37} y={-14} width={74} height={28} rx={9} fill={it.roof} />
                    {[-24, -8, 8, 24].map((wx) => <circle key={wx} cx={wx} cy={0} r={4} fill="#fff" />)}
                </g>
            ))}
            <g transform={`translate(${engine.x.toFixed(1)} ${engine.y.toFixed(1)}) rotate(${engine.angle.toFixed(1)}) scale(${beat.scale.toFixed(3)})`}>
                <path d="M56 -26 L70 0 L56 26 Z" fill={C.gold} stroke="#0b1628" strokeWidth={4} strokeLinejoin="round" />
                <rect x={-56} y={-28} width={116} height={56} rx={20} fill={MODES.train.color} stroke="#0b1628" strokeWidth={5} />
                <rect x={-50} y={-20} width={34} height={40} rx={9} fill="#2a5d8f" stroke="#0b1628" strokeWidth={3} />
                <rect x={-44} y={-12} width={22} height={24} rx={5} fill="#bfe6ff" />
                <circle cx={14} cy={0} r={13} fill="#1d2433" stroke="#0b1628" strokeWidth={3} />
                <circle cx={14} cy={0} r={6} fill="#4b5568" />
                <circle cx={46} cy={0} r={9} fill={C.gold} stroke="#0b1628" strokeWidth={3} />
                <circle cx={46} cy={0} r={4} fill="#fff" />
                <path d="M-8 -22 L36 -22" stroke="#fff" strokeOpacity={0.5} strokeWidth={4} strokeLinecap="round" />
            </g>
        </svg>
    );
};

const Vehicle = ({ timeline, frame, cam }) => {
    const seg = timeline.segments[segmentAt(timeline.segments, frame)];
    if (seg.type !== 'leg') return null;
    const p = legProgress(seg, frame);
    const pts = legPath(cam, seg, Math.max(p, 0.001));
    const appear = clamp01((frame - seg.start - 4) / 8);
    if (seg.mode === 'train') return <TrainSprite pts={pts} frame={frame} appear={appear} />;

    const [x, y] = pts[pts.length - 1];
    const heading = headingDeg(pts);
    const beat = heartbeat(frame);
    // The plane points along its arc and banks gently; the car stays upright and faces its direction
    const art = seg.mode === 'flight'
        ? <g transform={`rotate(${(heading + Math.sin(frame * 0.15) * 4).toFixed(1)}) scale(0.95)`}><PlaneArt /></g>
        : <g transform={`scale(${Math.abs(heading) > 90 ? -1 : 1} 1)`}><CarArt frame={frame} /></g>;
    return (
        <svg width={REEL.width} height={REEL.height} style={{ position: 'absolute', inset: 0 }}>
            <Badge x={x} y={y} color={seg.mode === 'flight' ? '#2a5d8f' : '#fff4e0'} beat={beat} appear={appear}>
                {art}
            </Badge>
        </svg>
    );
};

// ---------------------------------------------------------------------------
// Text and photo overlays.

const TopBand = ({ title, timeline, frame }) => {
    const { segments, stops } = timeline;
    const seg = segments[segmentAt(segments, frame)];
    if (seg.type === 'intro') return null;
    const reachedCount = seg.type === 'outro' ? stops.length
        : seg.type === 'stop' ? seg.index + 1 : seg.index;
    return (
        <div style={{ position: 'absolute', top: 70, left: 60, right: 60, fontFamily: FONT }}>
            <div style={{ color: C.muted, fontSize: 30, fontWeight: 700, letterSpacing: 6, textTransform: 'uppercase' }}>{title}</div>
            <div style={{ display: 'flex', gap: 10, marginTop: 18 }}>
                {stops.map((_, i) => (
                    <div key={i} style={{ flex: 1, height: 10, borderRadius: 5, background: i < reachedCount ? C.gold : 'rgba(255,255,255,0.18)' }} />
                ))}
            </div>
        </div>
    );
};

const PlaceTitle = ({ stop, number, appear }) => (
    <div style={{
        position: 'absolute', top: 190, left: 60, right: 60, fontFamily: FONT,
        opacity: appear, transform: `translateY(${(1 - appear) * 30}px)`,
    }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 14, background: C.panel, borderRadius: 28, padding: '22px 34px' }}>
            <MapPin size={52} color={C.gold} strokeWidth={2.6} />
            <div>
                <div style={{ color: C.muted, fontSize: 28, fontWeight: 700, letterSpacing: 3 }}>STOP {number}</div>
                <div style={{ color: C.text, fontSize: 64, fontWeight: 800, lineHeight: 1.05 }}>{stop.name}</div>
            </div>
        </div>
    </div>
);

const PhotoCard = ({ src, local, index, total }) => {
    // local: frames since this photo started
    const { fps } = useVideoConfig();
    const enter = spring({ frame: local, fps, config: { damping: 14, mass: 0.8 } });
    const tilt = index % 2 === 0 ? -3 : 3;
    const zoom = interpolate(local, [0, TIMING.perPhoto * 2], [1, 1.12], { extrapolateRight: 'clamp' });
    return (
        <div style={{
            position: 'absolute', left: 130, top: 760, width: 820, padding: 22, paddingBottom: 26,
            background: '#fbf4e4', borderRadius: 18, boxShadow: '0 30px 60px rgba(0,0,0,0.5)',
            transform: `translateY(${(1 - enter) * 700}px) rotate(${tilt * enter}deg)`,
        }}>
            <div style={{ width: 776, height: 800, overflow: 'hidden', borderRadius: 8, background: '#ddd' }}>
                <Img src={src} style={{ width: '100%', height: '100%', objectFit: 'cover', transform: `scale(${zoom})` }} />
            </div>
            <div style={{ marginTop: 14, color: '#2b1d14', fontFamily: FONT, fontSize: 28, fontWeight: 700, textAlign: 'right' }}>
                {index + 1} / {total}
            </div>
        </div>
    );
};

const StatsBand = ({ timeline, frame }) => {
    const { segments } = timeline;
    const seg = segments[segmentAt(segments, frame)];
    if (seg.type === 'intro' || seg.type === 'outro') return null;
    let km = seg.kmBefore;
    let mode = null;
    if (seg.type === 'leg') {
        km += seg.km * legProgress(seg, frame);
        mode = seg.mode;
    }
    const Icon = mode ? ICONS[mode] : null;
    return (
        <div style={{
            position: 'absolute', left: 60, right: 60, bottom: 90, display: 'flex', alignItems: 'center',
            justifyContent: 'space-between', fontFamily: FONT,
        }}>
            <div style={{ background: C.panel, borderRadius: 26, padding: '20px 30px' }}>
                <div style={{ color: C.muted, fontSize: 26, fontWeight: 700, letterSpacing: 3 }}>TRAVELLED</div>
                <div style={{ color: C.text, fontSize: 60, fontWeight: 800 }}>{fmtKm(km)}</div>
            </div>
            {Icon && (
                <div style={{ background: C.panel, borderRadius: 26, padding: '20px 30px', display: 'flex', alignItems: 'center', gap: 16 }}>
                    <Icon size={48} color={MODES[mode].color} strokeWidth={2.4} />
                    <div style={{ color: C.text, fontSize: 40, fontWeight: 700 }}>{MODES[mode].label}</div>
                </div>
            )}
        </div>
    );
};

const Intro = ({ title, timeline, frame }) => {
    const seg = timeline.segments[0];
    if (frame >= seg.dur + 10) return null;
    const inT = clamp01(frame / 18);
    const outT = clamp01((frame - (seg.dur - 12)) / 18);
    const o = inT * (1 - outT);
    return (
        <AbsoluteFill style={{ background: `rgba(7, 14, 28, ${0.55 * o})`, justifyContent: 'center', alignItems: 'center', fontFamily: FONT }}>
            <div style={{ opacity: o, transform: `translateY(${(1 - inT) * 40}px)`, textAlign: 'center', padding: '0 70px' }}>
                <div style={{ color: C.gold, fontSize: 34, fontWeight: 800, letterSpacing: 8 }}>MY TRIP</div>
                <div style={{ color: C.text, fontSize: 108, fontWeight: 900, lineHeight: 1.02, marginTop: 18 }}>{title}</div>
                <div style={{ color: C.muted, fontSize: 40, fontWeight: 600, marginTop: 26 }}>
                    {timeline.totals.places} places · {fmtKm(timeline.totals.km)}
                </div>
            </div>
        </AbsoluteFill>
    );
};

const Outro = ({ title, timeline, frame }) => {
    const { fps } = useVideoConfig();
    const seg = timeline.segments[timeline.segments.length - 1];
    if (frame < seg.start) return null;
    const local = frame - seg.start;
    const enter = spring({ frame: local - 18, fps, config: { damping: 16 } });
    const rows = Object.entries(timeline.totals.byMode).filter(([, km]) => km > 0);
    return (
        <AbsoluteFill style={{ justifyContent: 'flex-end', alignItems: 'center', fontFamily: FONT, paddingBottom: 150 }}>
            <div style={{
                width: 900, background: C.panel, borderRadius: 36, padding: '44px 54px',
                opacity: clamp01(enter), transform: `translateY(${(1 - enter) * 120}px)`,
            }}>
                <div style={{ color: C.gold, fontSize: 30, fontWeight: 800, letterSpacing: 6 }}>THAT'S A WRAP</div>
                <div style={{ color: C.text, fontSize: 66, fontWeight: 900, marginTop: 8, lineHeight: 1.05 }}>{title}</div>
                <div style={{ display: 'flex', gap: 60, marginTop: 30 }}>
                    <div>
                        <div style={{ color: C.text, fontSize: 64, fontWeight: 800 }}>{timeline.totals.places}</div>
                        <div style={{ color: C.muted, fontSize: 28, fontWeight: 600 }}>places</div>
                    </div>
                    <div>
                        <div style={{ color: C.text, fontSize: 64, fontWeight: 800 }}>{fmtKm(timeline.totals.km)}</div>
                        <div style={{ color: C.muted, fontSize: 28, fontWeight: 600 }}>travelled</div>
                    </div>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 18, marginTop: 30 }}>
                    {rows.map(([mode, km]) => {
                        const Icon = ICONS[mode];
                        return (
                            <div key={mode} style={{ display: 'flex', alignItems: 'center', gap: 12, background: 'rgba(255,255,255,0.08)', borderRadius: 20, padding: '12px 20px' }}>
                                <Icon size={36} color={MODES[mode].color} strokeWidth={2.4} />
                                <span style={{ color: C.text, fontSize: 30, fontWeight: 700 }}>{fmtKm(km)}</span>
                            </div>
                        );
                    })}
                </div>
            </div>
            <div style={{ color: C.muted, fontSize: 26, marginTop: 30, opacity: clamp01(enter) }}>adarshnair.online/travel_edit/route</div>
        </AbsoluteFill>
    );
};

// ---------------------------------------------------------------------------

export const TravelReel = ({ title = 'My trip', stops = [] }) => {
    const frame = useCurrentFrame();
    const timeline = useMemo(() => buildTimeline(stops), [stops]);
    const cam = cameraAt(timeline.segments, frame);
    const seg = timeline.segments[segmentAt(timeline.segments, frame)];

    let stopOverlay = null;
    if (seg.type === 'stop') {
        const stop = timeline.stops[seg.index];
        const local = frame - seg.start;
        const appear = clamp01(local / 12) * (1 - clamp01((local - (seg.dur - 10)) / 10));
        const photoIndex = Math.floor((local - TIMING.stopBase * 0.4) / TIMING.perPhoto);
        stopOverlay = (
            <>
                <PlaceTitle stop={stop} number={seg.index + 1} appear={appear} />
                {seg.photos.map((src, k) => {
                    if (k > photoIndex || k < photoIndex - 1) return null;
                    const startAt = TIMING.stopBase * 0.4 + k * TIMING.perPhoto;
                    return <PhotoCard key={k} src={src} local={local - startAt} index={k} total={seg.photos.length} />;
                })}
            </>
        );
    }

    return (
        <AbsoluteFill style={{ background: C.sea, overflow: 'hidden' }}>
            <MapLayer cam={cam} />
            <RouteLayer timeline={timeline} frame={frame} cam={cam} />
            <Vehicle timeline={timeline} frame={frame} cam={cam} />
            <TopBand title={title} timeline={timeline} frame={frame} />
            {stopOverlay}
            <StatsBand timeline={timeline} frame={frame} />
            <Intro title={title} timeline={timeline} frame={frame} />
            <Outro title={title} timeline={timeline} frame={frame} />
        </AbsoluteFill>
    );
};
