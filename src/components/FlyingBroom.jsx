import { motion, AnimatePresence } from 'framer-motion';
import { useEffect, useState, useRef } from 'react';

const FlyingBroom = () => {
    const [isVisible, setIsVisible] = useState(false);
    const [isChasing, setIsChasing] = useState(false);
    const [randomY, setRandomY] = useState(100);

    // Snitch & Broom Chase Coordinates
    const [snitchPos, setSnitchPos] = useState({ x: 0, y: 0 });
    const [broomPos, setBroomPos] = useState({ x: -300, y: 100 });
    const chaseInterval = useRef(null);

    // Passive Flight Logic
    useEffect(() => {
        if (isChasing) return; // Disable passive flight during chase

        const triggerFlight = () => {
            const y = Math.floor(Math.random() * (window.innerHeight * 0.4)) + 100;
            setRandomY(y);
            setIsVisible(true);

            // Hide after flight (Slowed down: 7.5s)
            setTimeout(() => {
                setIsVisible(false);
            }, 7500);

            // Schedule next flight (Fast Paced: 3s - 10s)
            const nextFlightDelay = Math.random() * 7000 + 3000;
            setTimeout(triggerFlight, nextFlightDelay);
        };

        const initialTimer = setTimeout(triggerFlight, 1000);
        return () => clearTimeout(initialTimer);
    }, [isChasing]);


    // Chase Logic
    const startChase = (e) => {
        e.stopPropagation(); // Prevent other clicks
        if (isChasing) return;

        setIsChasing(true);
        setIsVisible(true); // Ensure visible

        // Initialize positions
        setSnitchPos({ x: Math.random() * (window.innerWidth - 100), y: Math.random() * (window.innerHeight - 100) });

        // Chase Loop
        let chaseCount = 0;
        const maxChaseMoves = 20; // Lasts ~10 seconds (20 * 500ms)

        // Chase Loop: Snitch moves first, then Broom reacts
        chaseInterval.current = setInterval(() => {
            if (chaseCount >= maxChaseMoves) {
                endChase();
                return;
            }

            // 1. Move Snitch to a NEW random Position
            const snitchX = Math.max(50, Math.random() * (window.innerWidth - 100));
            const snitchY = Math.max(50, Math.random() * (window.innerHeight - 100));
            setSnitchPos({ x: snitchX, y: snitchY });

            // 2. Schedule Broom to move to that position with a slight delay/variance (Reaction time)
            // setTimeout creates the visible "chase" effect where broom follows the path
            setTimeout(() => {
                if (!chaseInterval.current) return; // check if cancelled
                setBroomPos({
                    x: snitchX - 120 + (Math.random() * 80 - 40), // Stay slightly behind
                    y: snitchY + 40
                });
            }, 100); // 100ms reaction delay

            chaseCount++;
        }, 600); // Slower updates to make the chase readable
    };

    const endChase = () => {
        clearInterval(chaseInterval.current);
        setIsChasing(false);
        setIsVisible(false); // Disappear after chase
    };

    // Clean up chase on unmount
    useEffect(() => {
        return () => clearInterval(chaseInterval.current);
    }, []);

    // If not visible (passive) and not chasing, don't render anything
    if (!isVisible && !isChasing) return null;

    return (
        <>
            <AnimatePresence>
                {/* GOLDEN SNITCH (Only visible during chase) */}
                {isChasing && (
                    <motion.div
                        initial={{ scale: 0, x: snitchPos.x, y: snitchPos.y }}
                        animate={{
                            x: snitchPos.x,
                            y: snitchPos.y,
                            scale: 1,
                            rotate: [0, 10, -10, 0] // Jitter
                        }}
                        exit={{ scale: 0, opacity: 0 }}
                        transition={{ duration: 0.4, ease: "easeInOut" }}
                        className="fixed z-[41] pointer-events-none drop-shadow-[0_0_15px_rgba(255,215,0,0.8)]"
                    >
                        <svg width="60" height="60" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="50" cy="50" r="15" fill="url(#goldGradient)" />
                            {/* Wings */}
                            <path d="M65 50 C 75 30, 95 20, 95 10" stroke="#F0F0F0" strokeWidth="2" fill="none">
                                <animate attributeName="d" values="M65 50 C 75 30, 95 20, 95 10; M65 50 C 75 70, 95 80, 95 90; M65 50 C 75 30, 95 20, 95 10" dur="0.1s" repeatCount="indefinite" />
                            </path>
                            <path d="M35 50 C 25 30, 5 20, 5 10" stroke="#F0F0F0" strokeWidth="2" fill="none">
                                <animate attributeName="d" values="M35 50 C 25 30, 5 20, 5 10; M35 50 C 25 70, 5 80, 5 90; M35 50 C 25 30, 5 20, 5 10" dur="0.1s" repeatCount="indefinite" />
                            </path>
                            <defs>
                                <radialGradient id="goldGradient" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(50 50) rotate(90) scale(15)">
                                    <stop stopColor="#FFD700" />
                                    <stop offset="1" stopColor="#B8860B" />
                                </radialGradient>
                            </defs>
                        </svg>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* BROOMSTICK CONTAINER */}
            <motion.div
                onClick={startChase} // Removed 'layout' prop to fix jitter
                // Passive Props
                initial={!isChasing ? { x: -300, y: randomY, rotate: 5, scale: 0.8 } : {}}

                // Animate based on Mode
                animate={isChasing ? {
                    x: broomPos.x,
                    y: broomPos.y,
                    rotate: [5, 25, 5],
                    scale: 0.6
                } : {
                    x: window.innerWidth + 300,
                    y: [randomY, randomY - 20, randomY + 10, randomY],
                    rotate: [5, 0, 5]
                }}

                transition={isChasing ? {
                    duration: 0.4,
                    ease: "backOut"
                } : {
                    duration: 7,
                    ease: "easeInOut",
                }}
                className={`fixed z-[100] drop-shadow-2xl cursor-pointer group ${isChasing ? 'pointer-events-none' : ''}`}
            >
                {/* Magic Cloud / Text Hint */}
                {!isChasing && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute -top-12 left-1/2 -translate-x-1/2 whitespace-nowrap bg-white/10 backdrop-blur-md border border-white/20 text-white px-3 py-1 rounded-full text-xs font-harry tracking-widest pointer-events-none select-none"
                    >
                        Catch the broom to win the Snitch!
                    </motion.div>
                )}

                {/* Visual Group moved to center rotation correctly */}
                <div className="relative w-[250px] h-[80px]">

                    {/* Tail Glitter Emitter (Back of Broom) */}
                    <div className="absolute top-1/2 left-[10px] -translate-y-1/2">
                        <motion.div
                            animate={{ opacity: [0.5, 1, 0.5], scale: [1, 1.2, 1] }}
                            transition={{ duration: 0.5, repeat: Infinity }}
                            className="w-4 h-4 rounded-full bg-yellow-400 blur-sm absolute"
                        />
                        {/* Sparkle Particles Trail */}
                        {[...Array(5)].map((_, i) => (
                            <motion.div
                                key={i}
                                animate={{ x: [-10, -50], y: [0, (Math.random() - 0.5) * 30], opacity: [1, 0] }}
                                transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.1 }}
                                className="absolute w-1 h-1 bg-white rounded-full"
                            />
                        ))}
                    </div>

                    {/* Nimbus 2000 SVG */}
                    <svg width="250" height="80" viewBox="0 0 300 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <g filter="url(#speed-blur)">
                            <path d="M280 45 C 200 48, 150 42, 100 50 C 80 53, 70 55, 60 55" stroke="#5D4037" strokeWidth="6" strokeLinecap="round" />
                            <path d="M275 44 C 200 46, 150 40, 105 48" stroke="#8D6E63" strokeWidth="2" strokeLinecap="round" opacity="0.6" />
                            <rect x="55" y="48" width="8" height="14" rx="2" fill="#FFC107" transform="rotate(-5 59 55)" />
                            <path d="M60 55 C 40 55, 20 40, 5 35 C 10 50, 5 70, 10 80 C 30 70, 50 60, 60 55" fill="#3E2723" />
                            <path d="M58 55 L 10 40" stroke="#5D4037" strokeWidth="1" />
                            <path d="M56 58 L 8 50" stroke="#5D4037" strokeWidth="1" />
                            <path d="M58 56 L 12 70" stroke="#5D4037" strokeWidth="1" />
                            <path d="M55 54 L 15 60" stroke="#5D4037" strokeWidth="1" />
                            <path d="M180 46 L 185 55" stroke="#FFC107" strokeWidth="2" />
                        </g>
                        <defs>
                            <filter id="speed-blur" x="-10" y="-10" width="320" height="120">
                                <feGaussianBlur in="SourceGraphic" stdDeviation="0.5" />
                            </filter>
                        </defs>
                    </svg>
                </div>
            </motion.div>
        </>
    );
};

export default FlyingBroom;
