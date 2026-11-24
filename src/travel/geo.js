// Geometry and timing for the travel reel. Pure functions, no React.

export const REEL = { width: 1080, height: 1920, fps: 30 };

export const MODES = {
    flight: { label: 'Flight', color: '#f0c34e' },
    train: { label: 'Train', color: '#7cc4f5' },
    car: { label: 'Car', color: '#ef5b6f' },
};

const R_EARTH_KM = 6371;
const rad = (d) => (d * Math.PI) / 180;

export function distanceKm(a, b) {
    const dLat = rad(b.lat - a.lat);
    const dLng = rad(b.lng - a.lng);
    const h = Math.sin(dLat / 2) ** 2 + Math.cos(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.sin(dLng / 2) ** 2;
    return 2 * R_EARTH_KM * Math.asin(Math.min(1, Math.sqrt(h)));
}

// Web Mercator in radians: x = longitude, y grows northwards.
export const mercX = (lng) => rad(lng);
export const mercY = (lat) => {
    const clamped = Math.max(-85, Math.min(85, lat));
    return Math.log(Math.tan(Math.PI / 4 + rad(clamped) / 2));
};

// Camera = centre in Mercator units + scale in pixels per radian.
const MIN_SCALE = 170; // whole world
const MAX_SCALE = 60000; // roughly 110 km across the reel
const MAP_TOP = 260; // space kept clear for the title band
const MAP_BOTTOM = 300; // and for the stats band

export function fitCamera(points, { pad = 160, minSpanKm = 40, maxScale = MAX_SCALE } = {}) {
    const xs = points.map((p) => mercX(p.lng));
    const ys = points.map((p) => mercY(p.lat));
    let minX = Math.min(...xs), maxX = Math.max(...xs);
    let minY = Math.min(...ys), maxY = Math.max(...ys);
    // Never zoom in tighter than minSpanKm, so a short hop still shows some land around it
    const minSpan = minSpanKm / R_EARTH_KM;
    if (maxX - minX < minSpan) { const c = (minX + maxX) / 2; minX = c - minSpan / 2; maxX = c + minSpan / 2; }
    if (maxY - minY < minSpan) { const c = (minY + maxY) / 2; minY = c - minSpan / 2; maxY = c + minSpan / 2; }
    const availW = REEL.width - 2 * pad;
    const availH = REEL.height - MAP_TOP - MAP_BOTTOM - 2 * pad;
    const scale = Math.max(MIN_SCALE, Math.min(maxScale, availW / (maxX - minX), availH / (maxY - minY)));
    // Centre the box in the space between the bands
    const cx = (minX + maxX) / 2;
    // Shift so the box is centred in the space between the bands (screen y grows downwards)
    const cy = (minY + maxY) / 2 - ((MAP_BOTTOM - MAP_TOP) / 2) / scale;
    return { cx, cy, scale };
}

export function project(cam, lng, lat) {
    return [
        REEL.width / 2 + (mercX(lng) - cam.cx) * cam.scale,
        REEL.height / 2 - (mercY(lat) - cam.cy) * cam.scale,
    ];
}

// Smooth camera blend: centre linear in Mercator, zoom linear in log space.
export function lerpCamera(a, b, t) {
    return {
        cx: a.cx + (b.cx - a.cx) * t,
        cy: a.cy + (b.cy - a.cy) * t,
        scale: Math.exp(Math.log(a.scale) + (Math.log(b.scale) - Math.log(a.scale)) * t),
    };
}

export const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
export const clamp01 = (t) => Math.max(0, Math.min(1, t));

// When a leg crosses the date line, move the end point by 360 degrees so the
// line takes the short way round instead of crossing the whole map.
export function unwrapLeg(a, b) {
    let lng = b.lng;
    while (lng - a.lng > 180) lng -= 360;
    while (lng - a.lng < -180) lng += 360;
    return { ...b, lng };
}

// ---- Timeline -------------------------------------------------------------

export const TIMING = {
    intro: 75,
    stopBase: 45,
    perPhoto: 48,
    maxPhotos: 4,
    outro: 120,
};

export const legFrames = (km) => Math.round(Math.min(150, 66 + 32 * Math.log10(1 + km / 20)));

/**
 * Build the reel's timeline from the stops.
 * Returns { segments, durationInFrames, totals } where every segment has
 * { type, start, dur, cam, ... } and legs carry their distance.
 */
export function buildTimeline(rawStops) {
    // Unwrap longitudes along the route once, so every leg takes the short way round
    const stops = [];
    rawStops.forEach((s, i) => stops.push(i ? unwrapLeg(stops[i - 1], s) : s));
    const segments = [];
    let t = 0;
    const overview = fitCamera(stops, { pad: 170 });
    const push = (seg) => { segments.push({ ...seg, start: t }); t += seg.dur; };

    push({ type: 'intro', dur: TIMING.intro, cam: overview });

    let travelled = 0;
    const byMode = { flight: 0, train: 0, car: 0 };
    stops.forEach((stop, i) => {
        if (i > 0) {
            const from = stops[i - 1];
            const to = stop;
            const km = distanceKm(from, stop);
            const mode = MODES[stop.mode] ? stop.mode : 'car';
            const legCam = fitCamera([from, to], { pad: mode === 'flight' ? 220 : 180 });
            push({ type: 'leg', index: i, from, to, mode, km, kmBefore: travelled, dur: legFrames(km), cam: legCam });
            travelled += km;
            byMode[mode] += km;
        }
        const photos = (stop.photos || []).slice(0, TIMING.maxPhotos);
        const prevCam = segments[segments.length - 1].cam;
        // Zoom in a little on arrival (never past city level) and put the stop in the
        // upper part of the frame, leaving the lower part for photo cards.
        const scale = Math.min(MAX_SCALE, Math.max(prevCam.scale * 1.5, 2500));
        const stopY = photos.length ? 560 : 800;
        const stopCam = {
            cx: mercX(stop.lng),
            cy: mercY(stop.lat) + (stopY - REEL.height / 2) / scale,
            scale,
        };
        push({ type: 'stop', index: i, photos, kmBefore: travelled, dur: TIMING.stopBase + TIMING.perPhoto * photos.length, cam: stopCam });
    });

    push({ type: 'outro', dur: TIMING.outro, cam: overview });

    return {
        stops,
        segments,
        durationInFrames: t,
        totals: { km: travelled, byMode, places: stops.length },
    };
}

export function segmentAt(segments, frame) {
    let lo = 0, hi = segments.length - 1;
    while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (segments[mid].start <= frame) lo = mid; else hi = mid - 1;
    }
    return lo;
}

// Camera at a frame: blend from the previous segment's camera during the first
// part of each segment.
export function cameraAt(segments, frame) {
    const i = segmentAt(segments, frame);
    const seg = segments[i];
    if (i === 0) return seg.cam;
    const blend = Math.min(36, seg.dur * 0.55);
    const t = easeInOut(clamp01((frame - seg.start) / blend));
    return lerpCamera(segments[i - 1].cam, seg.cam, t);
}

// Leg progress (0..1) at a frame: the vehicle waits for the camera, then moves.
export function legProgress(seg, frame) {
    const t = (frame - seg.start) / seg.dur;
    return easeInOut(clamp01((t - 0.18) / 0.72));
}

// ---- Leg shapes --------------------------------------------------------------
// Each leg's shape comes from a seed made from its end points, so the same trip
// always draws the same curves, in the editor, the preview and the exported video.

function seededRandom(seed) {
    let s = seed >>> 0 || 1;
    return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
    };
}

export const legSeed = (a, b) =>
    Math.round(a.lat * 1e4) * 73856093 ^ Math.round(a.lng * 1e4) * 19349663 ^
    Math.round(b.lat * 1e4) * 83492791 ^ Math.round(b.lng * 1e4) * 2654435761;

/**
 * Points from p0 to p1 (any 2D space) shaped by travel mode:
 *  - flight: one smooth parabolic arc bulging "up"
 *  - car: a winding road, several bends of decreasing size
 *  - train: a gentle sweeping curve, like a rail line
 * Bends taper to zero at both ends so the line always meets its stops.
 * upIsNegativeY: true on screen (y grows down), false for latitude/longitude.
 */
export function curvePoints(p0, p1, mode, seed, { samples = 120, upIsNegativeY = true } = {}) {
    const dx = p1[0] - p0[0], dy = p1[1] - p0[1];
    const len = Math.hypot(dx, dy);
    if (len < 1e-9) return [p0, p1];
    let nx = -dy / len, ny = dx / len;
    const pts = [];
    if (mode === 'flight') {
        if (upIsNegativeY ? ny > 0 : ny < 0) { nx = -nx; ny = -ny; }
        const h = len * 0.32;
        for (let k = 0; k <= samples; k++) {
            const u = k / samples;
            const off = 4 * h * u * (1 - u); // parabola: 0 at both ends, h in the middle
            pts.push([p0[0] + dx * u + nx * off, p0[1] + dy * u + ny * off]);
        }
        return pts;
    }
    const rnd = seededRandom(seed);
    const waves = mode === 'train'
        ? [[1, 0.1], [2, 0.03]] // long, gentle sweeps
        : [[1.2, 0.08], [2.6, 0.035], [5, 0.008]]; // a road's big bends, smaller turns and a little texture
    const parts = waves.map(([f, amp]) => ({
        f: f * (0.8 + 0.4 * rnd()),
        amp: amp * (rnd() < 0.5 ? -1 : 1),
        ph: rnd() * Math.PI * 2,
    }));
    for (let k = 0; k <= samples; k++) {
        const u = k / samples;
        const taper = Math.sin(Math.PI * u);
        const off = len * taper * parts.reduce((acc, w) => acc + w.amp * Math.sin(2 * Math.PI * w.f * u + w.ph), 0);
        pts.push([p0[0] + dx * u + nx * off, p0[1] + dy * u + ny * off]);
    }
    return pts;
}

// Cut a polyline at a fraction of its length (so vehicles move at an even pace).
export function trimByLength(pts, fraction) {
    if (fraction >= 1) return pts;
    const seglen = [];
    let total = 0;
    for (let i = 1; i < pts.length; i++) {
        const l = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
        seglen.push(l);
        total += l;
    }
    let target = Math.max(0, fraction) * total;
    const out = [pts[0]];
    for (let i = 1; i < pts.length; i++) {
        if (target <= seglen[i - 1]) {
            const t = seglen[i - 1] ? target / seglen[i - 1] : 0;
            out.push([pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * t, pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * t]);
            return out;
        }
        target -= seglen[i - 1];
        out.push(pts[i]);
    }
    return out;
}

// Position and heading at a distance (in the polyline's units) back from its end.
export function pointBack(pts, dist) {
    let remaining = dist;
    for (let i = pts.length - 1; i > 0; i--) {
        const [x2, y2] = pts[i];
        const [x1, y1] = pts[i - 1];
        const l = Math.hypot(x2 - x1, y2 - y1);
        const angle = (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI;
        if (remaining <= l) {
            const t = l ? remaining / l : 0;
            return { x: x2 - (x2 - x1) * t, y: y2 - (y2 - y1) * t, angle };
        }
        remaining -= l;
    }
    const [x1, y1] = pts[0];
    const [x2, y2] = pts[Math.min(1, pts.length - 1)];
    return { x: x1, y: y1, angle: (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI };
}

// Screen-space points of a leg, drawn up to `progress` of its length.
export function legPath(cam, seg, progress) {
    const a = project(cam, seg.from.lng, seg.from.lat);
    const b = project(cam, seg.to.lng, seg.to.lat);
    const full = curvePoints(a, b, seg.mode, legSeed(seg.from, seg.to));
    return trimByLength(full, progress);
}

export const toPathD = (pts) => pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join('');

export function headingDeg(pts) {
    if (pts.length < 2) return 0;
    // Look back far enough along the path that tiny final steps don't make the angle jitter
    const [x2, y2] = pts[pts.length - 1];
    let i = pts.length - 2;
    while (i > 0 && Math.hypot(x2 - pts[i][0], y2 - pts[i][1]) < 30) i -= 1;
    const [x1, y1] = pts[i];
    return (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI;
}
