import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Lines inspired by the Valmiki Ramayana (paraphrased translations)
const SAYINGS = [
    "Enthusiasm is the greatest strength; nothing is impossible for the enthusiastic.",
    "Mother and motherland are greater than heaven itself.",
    "Truth is the root of all; there is nothing higher than truth.",
    "A bridge can be built across any ocean, one stone at a time.",
    "Courage, cleverness and perseverance — Hanuman crossed the sea with these.",
    "Even a small squirrel helped build the Setu. No contribution is too small.",
    "The one who acts without delay achieves the goal.",
    "Do not be deceived by what glitters — not every golden deer is real.",
    "Patience and dharma win the longest wars.",
    "Anger destroys wisdom; the calm mind sees the path.",
];

// Hand-drawn golden deer (the illusory Maricha that lured Rama away)
const GoldenDeer = () => (
    <svg viewBox="0 0 160 160" className="w-32 h-32 md:w-40 md:h-40 drop-shadow-[0_0_18px_rgba(226,179,60,0.6)]" aria-hidden="true">
        <defs>
            <linearGradient id="deer-gold" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#ffe7a0" />
                <stop offset="50%" stopColor="#e2b33c" />
                <stop offset="100%" stopColor="#b07a12" />
            </linearGradient>
        </defs>
        {/* Antlers */}
        <g stroke="#a87a12" strokeWidth="3" fill="none" strokeLinecap="round">
            <path d="M108 38 L112 14 M110 26 L120 18 M111 20 L104 10" />
            <path d="M98 38 L92 16 M94 28 L84 22 M93 20 L98 9" />
        </g>
        {/* Body */}
        <path d="M30 88 C 30 68, 60 62, 90 66 C 100 56, 98 46, 102 40 C 110 36, 118 42, 118 50 C 118 58, 110 62, 106 70 C 112 82, 108 96, 96 100 L 40 102 C 30 100, 28 94, 30 88 Z" fill="url(#deer-gold)" />
        {/* Tail */}
        <path d="M32 84 C 22 80, 20 72, 26 70 C 28 76, 30 80, 34 82 Z" fill="#e2b33c" />
        {/* Ear */}
        <path d="M110 42 C 122 36, 128 40, 124 46 C 120 48, 114 46, 110 44 Z" fill="#c8941e" />
        {/* Eye */}
        <circle cx="111" cy="50" r="2.2" fill="#3b2412" />
        {/* Silver/jewel spots — Valmiki describes the deer as studded with gems */}
        <g fill="#fff6d6" opacity="0.9">
            <circle cx="52" cy="78" r="3" /><circle cx="66" cy="74" r="2.5" /><circle cx="80" cy="80" r="3" />
            <circle cx="60" cy="90" r="2.2" /><circle cx="74" cy="92" r="2.6" /><circle cx="90" cy="86" r="2.2" />
        </g>
        {/* Legs */}
        <g stroke="#b07a12" strokeWidth="5" strokeLinecap="round">
            <path d="M44 100 L40 132" /><path d="M56 101 L58 134" />
            <path d="M86 100 L84 132" /><path d="M96 98 L102 130" />
        </g>
        <g fill="#3b2412">
            <circle cx="40" cy="134" r="3" /><circle cx="58" cy="136" r="3" />
            <circle cx="84" cy="134" r="3" /><circle cx="102" cy="132" r="3" />
        </g>
    </svg>
);

const GoldenDeerMascot = () => {
    const [isVisible, setIsVisible] = useState(false);
    const [isVanishing, setIsVanishing] = useState(false);
    const [saying, setSaying] = useState('');
    const [isHovered, setIsHovered] = useState(false);
    const [side, setSide] = useState({ right: '6%' });

    const nextSpawnRef = useRef(null);                 // set on mount: first appearance after ~6s
    const cooldownRef = useRef(3 * 60 * 1000);        // then 3 min, doubling each time

    const spawn = useCallback(() => {
        const onRight = Math.random() > 0.5;
        const offset = `${Math.round(Math.random() * 15 + 4)}%`;
        setSide(onRight ? { right: offset } : { left: offset });
        setSaying(SAYINGS[Math.floor(Math.random() * SAYINGS.length)]);
        setIsVanishing(false);
        setIsVisible(true);
    }, []);

    useEffect(() => {
        if (nextSpawnRef.current === null) nextSpawnRef.current = Date.now() + 6000;
        const id = setInterval(() => {
            if (!isVisible && !isVanishing && Date.now() >= nextSpawnRef.current) spawn();
        }, 3000);
        return () => clearInterval(id);
    }, [isVisible, isVanishing, spawn]);

    const handleClick = () => {
        setIsVanishing(true);
        nextSpawnRef.current = Date.now() + cooldownRef.current;
        cooldownRef.current *= 2;
        setTimeout(() => {
            setIsVisible(false);
            setIsVanishing(false);
        }, 1400);
    };

    return (
        <div className="fixed bottom-0 z-[60] pointer-events-none" style={side}>
            <AnimatePresence>
                {isVisible && (
                    <motion.button
                        type="button"
                        initial={{ opacity: 0, y: 60 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className="relative pointer-events-auto group block"
                        onClick={handleClick}
                        onMouseEnter={() => setIsHovered(true)}
                        onMouseLeave={() => setIsHovered(false)}
                        aria-label="A golden deer. Click to see through the illusion."
                    >
                        {!isVanishing && (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ delay: 0.5 }}
                                className="absolute bottom-full mb-3 left-1/2 -translate-x-1/2 w-56 palm-leaf text-palm-ink p-3 rounded-md border border-gold-deep text-sm font-serif italic text-center shadow-[0_0_20px_rgba(240,138,36,0.35)]"
                            >
                                “{isHovered ? "Am I real, or Maricha's illusion? Click to find out." : saying}”
                                <span className="absolute top-full left-1/2 -translate-x-1/2 border-8 border-transparent border-t-palm-deep" />
                            </motion.div>
                        )}

                        <div className="absolute inset-0 m-auto w-[120%] h-[80%] bg-gold/30 blur-[40px] rounded-full pointer-events-none animate-pulse" />

                        <motion.div
                            animate={
                                isVanishing
                                    ? { opacity: 0, filter: 'blur(14px) hue-rotate(-60deg)', scale: 1.15 }
                                    : { y: [0, -4, 0] }
                            }
                            transition={isVanishing ? { duration: 0.5 } : { duration: 2.2, repeat: Infinity }}
                            className="relative"
                        >
                            <GoldenDeer />
                        </motion.div>

                        {isVanishing && (
                            <motion.p
                                initial={{ opacity: 0, y: 0 }}
                                animate={{ opacity: [0, 1, 0], y: -60 }}
                                transition={{ duration: 1.4 }}
                                className="absolute bottom-1/2 left-1/2 -translate-x-1/2 whitespace-nowrap font-epic text-saffron text-lg drop-shadow"
                            >
                                It was Maricha!
                            </motion.p>
                        )}
                    </motion.button>
                )}
            </AnimatePresence>

            {/* Golden sparkles scatter when the illusion breaks */}
            {isVanishing && (
                <div className="absolute bottom-16 left-1/2 w-0 h-0">
                    {Array.from({ length: 22 }).map((_, i) => {
                        const angle = (Math.PI * 2 * i) / 22;
                        const dist = 80 + (i % 4) * 30;
                        return (
                            <motion.span
                                key={i}
                                className="absolute w-2 h-2 rounded-full bg-gold shadow-[0_0_8px_#e2b33c]"
                                initial={{ x: 0, y: 0, opacity: 1 }}
                                animate={{ x: Math.cos(angle) * dist, y: Math.sin(angle) * dist - 40, opacity: 0 }}
                                transition={{ duration: 1.1, ease: 'easeOut' }}
                            />
                        );
                    })}
                </div>
            )}
        </div>
    );
};

export default GoldenDeerMascot;
