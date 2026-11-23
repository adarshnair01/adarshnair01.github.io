// World map for the reel: country shapes and borders from Natural Earth (via world-atlas),
// loaded on demand and kept in memory so the reel can draw it synchronously.
import { feature, mesh } from 'topojson-client';

let world = null;
let loading = null;

export function loadWorld() {
    if (world) return Promise.resolve(world);
    loading ??= import('world-atlas/countries-50m.json').then((mod) => {
        const topo = mod.default || mod;
        world = {
            land: feature(topo, topo.objects.countries),
            borders: mesh(topo, topo.objects.countries, (a, b) => a !== b),
        };
        return world;
    });
    return loading;
}

export const getWorld = () => world;
