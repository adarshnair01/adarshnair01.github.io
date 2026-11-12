import { motion } from 'framer-motion';

/* Shared Ramayana-style decorative pieces used across sections. */

// Surya disc — emblem of the Suryavansha (solar dynasty) of Rama
export const SunDisc = ({ className = '', rays = 24 }) => (
    <svg viewBox="0 0 200 200" className={className} aria-hidden="true">
        <defs>
            <radialGradient id="sun-core" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#ffe39a" />
                <stop offset="55%" stopColor="#f0a83a" />
                <stop offset="100%" stopColor="#c8601a" />
            </radialGradient>
        </defs>
        <g fill="#e2b33c" opacity="0.85">
            {Array.from({ length: rays }).map((_, i) => (
                <path
                    key={i}
                    d="M100 4 L106 40 L100 34 L94 40 Z"
                    transform={`rotate(${(360 / rays) * i} 100 100)`}
                />
            ))}
        </g>
        <circle cx="100" cy="100" r="54" fill="none" stroke="#e2b33c" strokeWidth="1.5" opacity="0.7" />
        <circle cx="100" cy="100" r="46" fill="url(#sun-core)" />
        <circle cx="100" cy="100" r="38" fill="none" stroke="#7a3a0e" strokeWidth="1" opacity="0.5" strokeDasharray="2 4" />
    </svg>
);

// Mandala ring — used as slow-rotating background ornament
export const Mandala = ({ className = '' }) => (
    <svg viewBox="0 0 400 400" className={className} aria-hidden="true" fill="none" stroke="currentColor">
        <circle cx="200" cy="200" r="190" strokeWidth="0.75" />
        <circle cx="200" cy="200" r="150" strokeWidth="0.75" strokeDasharray="3 6" />
        <circle cx="200" cy="200" r="110" strokeWidth="0.75" />
        {Array.from({ length: 16 }).map((_, i) => (
            <g key={i} transform={`rotate(${22.5 * i} 200 200)`}>
                <path d="M200 10 C 220 50, 220 70, 200 90 C 180 70, 180 50, 200 10 Z" strokeWidth="0.75" />
                <path d="M200 90 C 212 110, 212 120, 200 130 C 188 120, 188 110, 200 90 Z" strokeWidth="0.6" />
                <circle cx="200" cy="40" r="3" strokeWidth="0.6" />
            </g>
        ))}
    </svg>
);

// Lotus glyph
export const Lotus = ({ className = '' }) => (
    <svg viewBox="0 0 64 40" className={className} aria-hidden="true" fill="currentColor">
        <path d="M32 2 C 38 12, 38 24, 32 36 C 26 24, 26 12, 32 2 Z" />
        <path d="M32 36 C 24 26, 16 20, 6 18 C 10 28, 20 36, 32 36 Z" opacity="0.8" />
        <path d="M32 36 C 40 26, 48 20, 58 18 C 54 28, 44 36, 32 36 Z" opacity="0.8" />
        <path d="M32 36 C 22 32, 10 32, 2 34 C 10 38, 22 39, 32 36 Z" opacity="0.6" />
        <path d="M32 36 C 42 32, 54 32, 62 34 C 54 38, 42 39, 32 36 Z" opacity="0.6" />
    </svg>
);

// Bow-and-arrow divider (Kodanda)
export const BowDivider = ({ className = '' }) => (
    <div className={`flex items-center justify-center gap-3 text-gold ${className}`} aria-hidden="true">
        <span className="h-px w-16 md:w-28 bg-gradient-to-r from-transparent to-gold/70" />
        <svg viewBox="0 0 120 40" className="w-24 h-8">
            <path d="M20 4 Q 60 40 100 4" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
            <path d="M20 4 L100 4" stroke="currentColor" strokeWidth="0.75" opacity="0.7" />
            <path d="M60 30 L60 -2" stroke="#f08a24" strokeWidth="1.5" />
            <path d="M55 3 L60 -4 L65 3 Z" fill="#f08a24" />
            <path d="M57 28 L60 33 L63 28" stroke="#b3261e" strokeWidth="1.5" fill="none" />
        </svg>
        <span className="h-px w-16 md:w-28 bg-gradient-to-l from-transparent to-gold/70" />
    </div>
);

// Diya (oil lamp) with flickering flame
export const Diya = ({ className = '' }) => (
    <div className={`relative inline-flex flex-col items-center ${className}`} aria-hidden="true">
        <div className="absolute -top-6 w-10 h-10 rounded-full bg-saffron/40 blur-xl" />
        <svg viewBox="0 0 20 30" className="w-4 h-6 origin-bottom animate-flicker relative">
            <path d="M10 0 C 16 10, 16 20, 10 30 C 4 20, 4 10, 10 0 Z" fill="#ffb347" />
            <path d="M10 10 C 13 16, 13 22, 10 28 C 7 22, 7 16, 10 10 Z" fill="#fff1c1" />
        </svg>
        <svg viewBox="0 0 60 20" className="w-12 h-4 -mt-1">
            <path d="M2 4 Q 30 26 58 4 Q 30 10 2 4 Z" fill="#b5651d" stroke="#7a3a0e" strokeWidth="1" />
        </svg>
    </div>
);

// Section heading: Kanda name + title + Devanagari subtitle
export const KandaHeading = ({ kanda, title, sanskrit, subtitle, dark = false }) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="text-center mb-14"
    >
        {kanda && <p className={`kanda-label mb-3 ${dark ? '!text-palm-ink/70' : ''}`}>{kanda}</p>}
        <h2 className={`text-4xl md:text-5xl tracking-wide mb-2 ${dark ? 'text-palm-ink' : 'text-transparent bg-clip-text bg-gradient-to-b from-[#ffe7a8] via-gold to-saffron'}`}>
            {title}
        </h2>
        {sanskrit && (
            <p className={`font-sanskrit text-lg ${dark ? 'text-kumkum/80' : 'text-saffron/80'}`} lang="sa">{sanskrit}</p>
        )}
        <BowDivider className="mt-4" />
        {subtitle && <p className={`mt-4 font-serif italic text-lg ${dark ? 'text-palm-ink/80' : 'text-text-muted'}`}>{subtitle}</p>}
    </motion.div>
);
