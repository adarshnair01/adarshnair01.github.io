import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';

/* Shared Christmas & New Year decorative pieces used across sections. */

// Full-screen falling snow on a canvas (pauses for reduced-motion users)
export const Snowfall = ({ density = 110 }) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        let w = 0, h = 0, raf = 0;

        const resize = () => {
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            w = window.innerWidth; h = window.innerHeight;
            canvas.width = w * dpr; canvas.height = h * dpr;
            canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        };
        resize();
        window.addEventListener('resize', resize);

        const count = Math.round(density * Math.min(1, w / 1400) + 30);
        const flakes = Array.from({ length: count }, () => ({
            x: Math.random() * w,
            y: Math.random() * h,
            r: Math.random() * 2.2 + 0.6,
            vy: Math.random() * 0.6 + 0.3,
            drift: Math.random() * Math.PI * 2,
            o: Math.random() * 0.5 + 0.35,
        }));

        const draw = () => {
            ctx.clearRect(0, 0, w, h);
            for (const f of flakes) {
                if (!reduce) {
                    f.drift += 0.01;
                    f.y += f.vy;
                    f.x += Math.sin(f.drift) * 0.35;
                    if (f.y > h + 5) { f.y = -5; f.x = Math.random() * w; }
                    if (f.x > w + 5) f.x = -5;
                    if (f.x < -5) f.x = w + 5;
                }
                ctx.beginPath();
                ctx.arc(f.x, f.y, f.r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(244, 247, 252, ${f.o})`;
                ctx.fill();
            }
            if (!reduce) raf = requestAnimationFrame(draw);
        };
        draw();

        return () => {
            cancelAnimationFrame(raf);
            window.removeEventListener('resize', resize);
        };
    }, [density]);

    return <canvas ref={canvasRef} className="fixed inset-0 pointer-events-none z-[5]" aria-hidden="true" />;
};

// String of twinkling fairy lights drooping between two points
const BULB_COLORS = ['#d4243b', '#f0c34e', '#1f8a55', '#a9d4f5', '#f4f7fc'];
export const FairyLights = ({ count = 18, className = '' }) => (
    <div className={`relative w-full h-8 ${className}`} aria-hidden="true">
        <svg viewBox="0 0 1000 30" preserveAspectRatio="none" className="absolute inset-0 w-full h-full">
            {Array.from({ length: 4 }).map((_, i) => (
                <path key={i} d={`M${i * 250} 2 Q ${i * 250 + 125} 26 ${i * 250 + 250} 2`} fill="none" stroke="#2c3a2f" strokeWidth="1.5" />
            ))}
        </svg>
        <div className="absolute inset-0 flex justify-between px-[1%]">
            {Array.from({ length: count }).map((_, i) => {
                // follow the drooping wire: four arcs across the width
                const t = ((i + 0.5) / count) * 4 % 1;
                const drop = Math.sin(t * Math.PI) * 18;
                const color = BULB_COLORS[i % BULB_COLORS.length];
                return (
                    <span
                        key={i}
                        className="block w-2.5 h-3.5 rounded-b-full rounded-t-sm animate-twinkle"
                        style={{
                            marginTop: `${drop}px`,
                            backgroundColor: color,
                            boxShadow: `0 0 10px 2px ${color}`,
                            animationDelay: `${(i % 5) * 0.35}s`,
                        }}
                    />
                );
            })}
        </div>
    </div>
);

// Six-armed snowflake glyph
export const Snowflake = ({ className = '' }) => (
    <svg viewBox="0 0 40 40" className={className} aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
        {[0, 60, 120].map((a) => (
            <g key={a} transform={`rotate(${a} 20 20)`}>
                <path d="M20 2 V38" />
                <path d="M20 8 L15 4 M20 8 L25 4 M20 32 L15 36 M20 32 L25 36" />
            </g>
        ))}
    </svg>
);

// Christmas bauble
export const Bauble = ({ color = '#d4243b', className = '' }) => (
    <svg viewBox="0 0 40 52" className={className} aria-hidden="true">
        <path d="M20 0 V8" stroke="#b8891c" strokeWidth="1.5" />
        <rect x="14" y="7" width="12" height="6" rx="1.5" fill="#f0c34e" />
        <circle cx="20" cy="32" r="18" fill={color} />
        <path d="M5 30 Q 20 38 35 30" stroke="#ffffff" strokeOpacity="0.45" strokeWidth="2" fill="none" />
        <ellipse cx="13" cy="24" rx="4" ry="6" fill="#ffffff" opacity="0.35" transform="rotate(-25 13 24)" />
    </svg>
);

// Holly sprig used as a divider centrepiece
export const HollyDivider = ({ className = '' }) => (
    <div className={`flex items-center justify-center gap-3 ${className}`} aria-hidden="true">
        <span className="h-px w-16 md:w-28 bg-gradient-to-r from-transparent to-gold/70" />
        <svg viewBox="0 0 80 36" className="w-20 h-9">
            <path d="M40 22 C 28 6, 12 8, 4 16 C 10 16, 12 20, 10 24 C 16 22, 20 26, 18 30 C 26 26, 34 28, 40 22 Z" fill="#1f8a55" stroke="#0f5132" strokeWidth="1" />
            <path d="M40 22 C 52 6, 68 8, 76 16 C 70 16, 68 20, 70 24 C 64 22, 60 26, 62 30 C 54 26, 46 28, 40 22 Z" fill="#1f8a55" stroke="#0f5132" strokeWidth="1" />
            <circle cx="36" cy="20" r="4.5" fill="#d4243b" />
            <circle cx="44" cy="20" r="4.5" fill="#d4243b" />
            <circle cx="40" cy="14" r="4.5" fill="#e0384e" />
            <circle cx="38.5" cy="12.5" r="1.3" fill="#fff" opacity="0.7" />
        </svg>
        <span className="h-px w-16 md:w-28 bg-gradient-to-l from-transparent to-gold/70" />
    </div>
);

// Section heading: eyebrow + title + script flourish + divider
export const FestiveHeading = ({ eyebrow, title, script, subtitle, dark = false }) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="text-center mb-14"
    >
        {eyebrow && <p className={`section-eyebrow mb-3 ${dark ? '!text-holly-deep/80' : ''}`}>{eyebrow}</p>}
        <h2 className={`text-4xl md:text-5xl font-bold tracking-wide mb-1 ${dark ? 'text-cream-ink' : 'text-snow'}`}>
            {title}
        </h2>
        {script && <p className={`font-script text-3xl md:text-4xl ${dark ? 'text-holly' : 'text-gold'}`}>{script}</p>}
        <HollyDivider className="mt-3" />
        {subtitle && <p className={`mt-4 text-lg ${dark ? 'text-cream-ink/80' : 'text-text-muted'}`}>{subtitle}</p>}
    </motion.div>
);

// Next 1 January (local time). During the first week of January we celebrate instead of counting down.
const getNewYearState = () => {
    const now = new Date();
    const year = now.getFullYear();
    if (now.getMonth() === 0 && now.getDate() <= 7) {
        return { celebrating: true, year, diff: 0 };
    }
    const target = new Date(year + 1, 0, 1);
    return { celebrating: false, year: year + 1, diff: target.getTime() - now.getTime() };
};

export const NewYearCountdown = ({ className = '' }) => {
    const [state, setState] = useState(getNewYearState);

    useEffect(() => {
        const id = setInterval(() => setState(getNewYearState()), 1000);
        return () => clearInterval(id);
    }, []);

    if (state.celebrating) {
        return (
            <div className={`text-center ${className}`}>
                <p className="font-script text-4xl md:text-5xl text-gold">Happy New Year {state.year}!</p>
            </div>
        );
    }

    const s = Math.max(0, Math.floor(state.diff / 1000));
    const parts = [
        { label: 'Days', value: Math.floor(s / 86400) },
        { label: 'Hours', value: Math.floor((s % 86400) / 3600) },
        { label: 'Minutes', value: Math.floor((s % 3600) / 60) },
        { label: 'Seconds', value: s % 60 },
    ];

    return (
        <div className={`text-center ${className}`} aria-live="off">
            <p className="section-eyebrow mb-3">Counting down to {state.year}</p>
            <div className="flex justify-center gap-2 sm:gap-4">
                {parts.map((p) => (
                    <div key={p.label} className="w-16 sm:w-20 py-2 rounded-xl bg-night-card/80 border border-gold/30 shadow-[0_0_20px_rgba(240,195,78,0.12)]">
                        <div className="font-display text-2xl sm:text-3xl text-snow tabular-nums">{String(p.value).padStart(2, '0')}</div>
                        <div className="text-[10px] uppercase tracking-widest text-gold/80">{p.label}</div>
                    </div>
                ))}
            </div>
        </div>
    );
};
