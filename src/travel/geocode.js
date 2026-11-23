// Place search and reverse lookup with OpenStreetMap's Nominatim service.
// Usage policy: at most one request per second, no autocomplete-per-keystroke,
// and attribution to OpenStreetMap. Searches run only when the user submits.
const BASE = 'https://nominatim.openstreetmap.org';

let last = 0;
async function politeFetch(url, signal) {
    const wait = Math.max(0, last + 1100 - Date.now());
    if (wait) await new Promise((r) => setTimeout(r, wait));
    last = Date.now();
    const res = await fetch(url, { signal, headers: { Accept: 'application/json' } });
    if (!res.ok) throw new Error(`Place search failed (${res.status})`);
    return res.json();
}

const lang = () => (typeof navigator !== 'undefined' && navigator.language) || 'en';

const shortName = (r) => {
    const a = r.address || {};
    return r.name || a.city || a.town || a.village || a.suburb || a.county || a.state || (r.display_name || '').split(',')[0] || 'Unnamed place';
};

export async function searchPlaces(query, signal) {
    const url = `${BASE}/search?format=jsonv2&addressdetails=1&limit=6&accept-language=${encodeURIComponent(lang())}&q=${encodeURIComponent(query)}`;
    const rows = await politeFetch(url, signal);
    return rows.map((r) => ({
        name: shortName(r),
        detail: r.display_name,
        lat: Number(r.lat),
        lng: Number(r.lon),
    }));
}

export async function reverseLookup(lat, lng, signal) {
    const url = `${BASE}/reverse?format=jsonv2&zoom=10&addressdetails=1&accept-language=${encodeURIComponent(lang())}&lat=${lat}&lon=${lng}`;
    try {
        const r = await politeFetch(url, signal);
        if (r && !r.error) return shortName(r);
    } catch {
        // fall through to coordinates
    }
    return `${lat.toFixed(3)}, ${lng.toFixed(3)}`;
}
