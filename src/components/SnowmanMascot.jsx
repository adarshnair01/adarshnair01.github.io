import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const WISHES = [
    "Wishing you a merry Christmas and a model with zero overfitting!",
    "May your New Year be full of signal and very little noise.",
    "New year, new features: may all your pipelines run green.",
    "Warm wishes and a cosy holiday season to you and yours!",
    "My resolution: more data, better questions.",
    "May your gradients descend smoothly into the new year.",
    "Even a snowman knows: small flakes add up to something big.",
    "Here's to a year of fewer bugs and more breakthroughs!",
    "Peace on earth and p-values below 0.05.",
    "Thanks for stopping by. Have a wonderful holiday!",
];

// Hand-drawn snowman with scarf and top hat
const Snowman = () => (
    <svg viewBox="0 0 140 170" className="w-28 h-36 md:w-32 md:h-40 drop-shadow-[0_0_18px_rgba(169,212,245,0.45)]" aria-hidden="true">
        {/* Body */}
        <circle cx="70" cy="128" r="38" fill="#f4f7fc" />
        <circle cx="70" cy="76" r="27" fill="#f4f7fc" />
        <circle cx="70" cy="128" r="38" fill="none" stroke="#c9d6ea" strokeWidth="2" />
        <circle cx="70" cy="76" r="27" fill="none" stroke="#c9d6ea" strokeWidth="2" />
        {/* Stick arms */}
        <path d="M44 106 L14 88 M24 94 L18 82" stroke="#5a3522" strokeWidth="3" strokeLinecap="round" />
        <path d="M96 106 L126 86 M116 92 L124 80" stroke="#5a3522" strokeWidth="3" strokeLinecap="round" />
        {/* Buttons */}
        <circle cx="70" cy="112" r="3.5" fill="#2b1d14" />
        <circle cx="70" cy="126" r="3.5" fill="#2b1d14" />
        <circle cx="70" cy="140" r="3.5" fill="#2b1d14" />
        {/* Face */}
        <circle cx="61" cy="70" r="3" fill="#2b1d14" />
        <circle cx="79" cy="70" r="3" fill="#2b1d14" />
        <path d="M70 76 L92 80 L70 82 Z" fill="#f08a24" />
        <path d="M60 87 Q 70 94 80 87" stroke="#2b1d14" strokeWidth="2" fill="none" strokeLinecap="round" strokeDasharray="1 4" />
        {/* Scarf */}
        <path d="M46 96 Q 70 106 94 96 L94 104 Q 70 114 46 104 Z" fill="#d4243b" />
        <path d="M82 104 L88 128 L78 128 L74 106 Z" fill="#d4243b" />
        <path d="M80 112 L86 112 M81 118 L87 118" stroke="#f4f7fc" strokeWidth="2" />
        {/* Top hat */}
        <rect x="46" y="48" width="48" height="6" rx="2" fill="#1b1b24" />
        <rect x="54" y="18" width="32" height="32" rx="3" fill="#1b1b24" />
        <rect x="54" y="40" width="32" height="6" fill="#1f8a55" />
        <circle cx="62" cy="43" r="3" fill="#d4243b" />
    </svg>
);

const SnowmanMascot = () => {
    const [isVisible, setIsVisible] = useState(false);
    const [isMelting, setIsMelting] = useState(false);
    const [wish, setWish] = useState('');
    const [isHovered, setIsHovered] = useState(false);
    const [side, setSide] = useState({ right: '6%' });

    const nextSpawnRef = useRef(null);                // set on mount: first appearance after ~6s
    const cooldownRef = useRef(3 * 60 * 1000);        // then 3 min, doubling each time

    const spawn = useCallback(() => {
        const onRight = Math.random() > 0.5;
        const offset = `${Math.round(Math.random() * 15 + 4)}%`;
        setSide(onRight ? { right: offset } : { left: offset });
        setWish(WISHES[Math.floor(Math.random() * WISHES.length)]);
        setIsMelting(false);
        setIsVisible(true);
    }, []);

    useEffect(() => {
        if (nextSpawnRef.current === null) nextSpawnRef.current = Date.now() + 6000;
        const id = setInterval(() => {
            if (!isVisible && !isMelting && Date.now() >= nextSpawnRef.current) spawn();
        }, 3000);
        return () => clearInterval(id);
    }, [isVisible, isMelting, spawn]);

    const handleClick = () => {
        setIsMelting(true);
        nextSpawnRef.current = Date.now() + cooldownRef.current;
        cooldownRef.current *= 2;
        setTimeout(() => {
            setIsVisible(false);
            setIsMelting(false);
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
                        aria-label="A friendly snowman. Click to say goodbye."
                    >
                        {!isMelting && (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ delay: 0.5 }}
                                className={`absolute bottom-full mb-3 w-56 ${side.right ? 'right-0' : 'left-0'} bg-cream text-cream-ink p-3 rounded-xl border-2 border-holly text-sm text-center shadow-[0_0_20px_rgba(212,36,59,0.3)]`}
                            >
                                {isHovered ? "Brr! Click me and I'll melt away… see you next time!" : wish}
                                <span className={`absolute top-full ${side.right ? 'right-12' : 'left-12'} border-8 border-transparent border-t-holly`} />
                            </motion.div>
                        )}

                        <motion.div
                            animate={
                                isMelting
                                    ? { scaleY: 0.1, scaleX: 1.3, opacity: 0, y: 40 }
                                    : { rotate: [-2, 2, -2] }
                            }
                            transition={isMelting ? { duration: 0.8, ease: 'easeIn' } : { duration: 3, repeat: Infinity }}
                            className="relative origin-bottom"
                        >
                            <Snowman />
                        </motion.div>

                        {isMelting && (
                            <motion.p
                                initial={{ opacity: 0, y: 0 }}
                                animate={{ opacity: [0, 1, 0], y: -60 }}
                                transition={{ duration: 1.4 }}
                                className="absolute bottom-1/2 left-1/2 -translate-x-1/2 whitespace-nowrap font-script text-gold text-3xl drop-shadow"
                            >
                                Happy holidays!
                            </motion.p>
                        )}
                    </motion.button>
                )}
            </AnimatePresence>

            {/* Snowflakes burst out as he melts */}
            {isMelting && (
                <div className="absolute bottom-16 left-1/2 w-0 h-0">
                    {Array.from({ length: 22 }).map((_, i) => {
                        const angle = (Math.PI * 2 * i) / 22;
                        const dist = 80 + (i % 4) * 30;
                        return (
                            <motion.span
                                key={i}
                                className="absolute text-snow text-sm"
                                initial={{ x: 0, y: 0, opacity: 1 }}
                                animate={{ x: Math.cos(angle) * dist, y: Math.sin(angle) * dist - 40, opacity: 0, rotate: 180 }}
                                transition={{ duration: 1.1, ease: 'easeOut' }}
                            >
                                ❄
                            </motion.span>
                        );
                    })}
                </div>
            )}
        </div>
    );
};

export default SnowmanMascot;
