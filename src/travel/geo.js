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

// Points along a leg in screen space. Flights bulge into an arc; land legs are straight.
export function legPath(cam, seg, progress, samples = 48) {
    const a = project(cam, seg.from.lng, seg.from.lat);
    const b = project(cam, seg.to.lng, seg.to.lat);
    const pts = [];
    const n = Math.max(2, Math.ceil(samples * progress));
    if (seg.mode === 'flight') {
        const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
        const dx = b[0] - a[0], dy = b[1] - a[1];
        const len = Math.hypot(dx, dy) || 1;
        // Bulge perpendicular to the line, always towards the top of the frame
        let nx = -dy / len, ny = dx / len;
        if (ny > 0) { nx = -nx; ny = -ny; }
        const c = [mx + nx * len * 0.28, my + ny * len * 0.28];
        for (let k = 0; k <= n; k++) {
            const u = (k / n) * progress;
            pts.push([
                (1 - u) ** 2 * a[0] + 2 * (1 - u) * u * c[0] + u * u * b[0],
                (1 - u) ** 2 * a[1] + 2 * (1 - u) * u * c[1] + u * u * b[1],
            ]);
        }
    } else {
        for (let k = 0; k <= n; k++) {
            const u = (k / n) * progress;
            pts.push(project(cam, seg.from.lng + (seg.to.lng - seg.from.lng) * u, seg.from.lat + (seg.to.lat - seg.from.lat) * u));
        }
    }
    return pts;
}

export const toPathD = (pts) => pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join('');

export function headingDeg(pts) {
    if (pts.length < 2) return 0;
    const [x1, y1] = pts[pts.length - 2];
    const [x2, y2] = pts[pts.length - 1];
    return (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI;
}
