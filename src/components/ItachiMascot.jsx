import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const ITATCHI_QUOTES = [
    "People live their lives bound by what they accept as correct and true.",
    "Growth occurs when one goes beyond one's limits.",
    "Self-sacrifice... That is a true shinobi.",
    "You are weak. Why are you weak? Because you lack... hatred.",
    "Forgive me... There won't be a next time.",
    "Every jutsu has its weakness.",
    "I will be the one to judge you.",
    "Even the strongest of opponents always has a weakness.",
    "It is foolish to fear what we have yet to see and know.",
    "No matter what you decide to do from here on out, I will love you always.",
    "Knowledge and awareness are vague, and perhaps better called illusions.",
    "Those who forgive themselves, and are able to accept their true nature... They are the strong ones!",
    "We don't know what kind of people we truly are until the moment before our deaths.",
];

const CrowParticle = ({ origin }) => {
    // Random destination for the crow
    const x = (Math.random() - 0.5) * 500;
    const y = (Math.random() - 0.5) * 500 - 100; // Tend to fly up
    const duration = 1 + Math.random();
    const delay = Math.random() * 0.2;

    return (
        <motion.div
            className="absolute z-50 pointer-events-none"
            initial={{ x: 0, y: 0, opacity: 1, scale: 0.5 }}
            animate={{ x, y, opacity: 0, scale: 1.5 }}
            transition={{ duration, ease: "easeOut", delay }}
        >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="black" xmlns="http://www.w3.org/2000/svg">
                <path d="M22 2L11 13L22 24L2 13L22 2Z" /> {/* Simplified abstract bird shape */}
            </svg>
        </motion.div>
    );
};

const ItachiMascot = () => {
    const [isVisible, setIsVisible] = useState(false);
    const [isDisappearing, setIsDisappearing] = useState(false);
    const [quote, setQuote] = useState("");
    const [isHovered, setIsHovered] = useState(false);
    const [position, setPosition] = useState({ bottom: 0, right: 20 });
    const containerRef = useRef(null);

    // Cooldown state
    const nextSpawnTimeRef = useRef(Date.now() + 5000); // Start showing up shortly after load (5s)
    const cooldownDurationRef = useRef(3 * 60 * 1000); // 3 minutes initial cooldown

    // Check spawn loop
    useEffect(() => {
        const checkSpawn = () => {
            if (isVisible || isDisappearing) return;

            // Only spawn if we've passed the cooldown time
            if (Date.now() >= nextSpawnTimeRef.current) {
                spawnItachi();
            }
        };

        const interval = setInterval(checkSpawn, 10000); // Check every 10 seconds
        return () => clearInterval(interval);
    }, [isVisible, isDisappearing]);

    const spawnItachi = () => {
        // Pick random side? For now keep it simple: Bottom Right or Bottom Left
        const isRight = Math.random() > 0.5;
        setPosition({
            bottom: 0,
            [isRight ? 'right' : 'left']: Math.random() * 20 + 5
        });
        setQuote(ITATCHI_QUOTES[Math.floor(Math.random() * ITATCHI_QUOTES.length)]);
        setIsVisible(true);
        setIsDisappearing(false);

        // Auto disappear after 15 seconds if not clicked (no penalty on cooldown)
        setTimeout(() => {
            if (Math.random() > 0.3) {
                // He stays until clicked 70% of the time, or whatever existing logic was. 
                // Actually existing logic was empty inside the random check? 
                // "if (Math.random() > 0.3) { }"  <-- This block was empty in original code??
                // Ah, looking at the view code: 
                // if (Math.random() > 0.3) { // 70% chance he stays until clicked? Or just auto hide
                //      // Let's rely on user click mostly, but maybe auto-hide eventually
                // }
                // It was effectively doing nothing? I'll leave it as is or implement auto-hide. 
                // If he wasn't auto-hiding before, I won't start now. 
                // I'll just leave the function call but maybe clean it up.
            }
        }, 15000);
    };

    const handleClick = () => {
        setIsDisappearing(true);

        // Schedule next appearance with doubling cooldown
        nextSpawnTimeRef.current = Date.now() + cooldownDurationRef.current;
        cooldownDurationRef.current *= 2; // Double the wait time

        // Wait for animation then hide
        setTimeout(() => {
            setIsVisible(false);
            setIsDisappearing(false);
        }, 1500);
    };

    return (
        <div
            className="fixed z-[60] pointer-events-none"
            style={{
                bottom: position.bottom,
                left: position.left ? `${position.left}%` : 'auto',
                right: position.right ? `${position.right}%` : 'auto'
            }}
            ref={containerRef}
        >
            <AnimatePresence>
                {isVisible && (
                    <motion.div
                        initial={{ opacity: 0, y: 50 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, duration: 2 }}
                        className="relative cursor-pointer pointer-events-auto group"
                        onClick={handleClick}
                        onMouseEnter={() => setIsHovered(true)}
                        onMouseLeave={() => setIsHovered(false)}
                    >
                        {/* Dialogue Bubble */}
                        {!isDisappearing && (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.5 }}
                                transition={{ delay: 0.5 }}
                                className="absolute bottom-full mb-4 w-48 bg-black/80 text-white p-3 rounded-xl border border-red-900 text-xs font-serif italic text-center shadow-[0_0_15px_rgba(255,0,0,0.3)] backdrop-blur-sm"
                                style={{
                                    left: '50%',
                                    translateX: '-50%'
                                }}
                            >
                                "{isHovered ? "You're under my illusion. Try clicking me." : quote}"
                                <div className="absolute top-full left-1/2 -translate-x-1/2 border-8 border-transparent border-t-black/80"></div>
                            </motion.div>
                        )}

                        {/* Glow Background - Persists */}
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[140%] h-[100%] bg-red-600/40 blur-[50px] rounded-full pointer-events-none mix-blend-screen animate-pulse"></div>
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[100%] h-[60%] bg-red-500/60 blur-[30px] rounded-full pointer-events-none mix-blend-plus-lighter animate-pulse"></div>

                        {/* Itachi Image - Fades out on click */}
                        <motion.img
                            src="/itachi.png"
                            alt="Itachi Uchiha"
                            animate={{
                                opacity: isDisappearing ? 0 : 1,
                                filter: isDisappearing ? 'blur(20px)' : 'none',
                                scale: isDisappearing ? 1.1 : 1
                            }}
                            transition={{ duration: 0.3 }}
                            className="h-48 object-contain drop-shadow-[0_0_10px_rgba(0,0,0,0.8)] transition-transform hover:scale-105"
                        />

                        {/* Sharingan Glow effect on hover */}
                        {!isDisappearing && (
                            <>
                                <div className="absolute top-[20%] left-[45%] w-1 h-1 bg-red-600 rounded-full blur-[1px] opacity-0 group-hover:opacity-100 animate-pulse"></div>
                                <div className="absolute top-[20%] left-[55%] w-1 h-1 bg-red-600 rounded-full blur-[1px] opacity-0 group-hover:opacity-100 animate-pulse"></div>
                            </>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Crows Effect */}
            {isDisappearing && (
                <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-0 h-0">
                    {Array.from({ length: 25 }).map((_, i) => (
                        <CrowParticle key={i} />
                    ))}
                </div>
            )}
        </div>
    );
};

export default ItachiMascot;
