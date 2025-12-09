import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const MouseFollower = () => {
    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
    const [sparkles, setSparkles] = useState([]);

    const [isClicking, setIsClicking] = useState(false);

    const [magicBursts, setMagicBursts] = useState([]);

    useEffect(() => {
        const updateMousePosition = (e) => {
            setMousePosition({ x: e.clientX, y: e.clientY });

            if (Math.random() > 0.8) {
                const newSparkle = {
                    id: Math.random(),
                    x: e.clientX,
                    y: e.clientY,
                    size: Math.random() * 4 + 2,
                    isBurst: false
                };
                setSparkles(prev => [...prev.slice(-15), newSparkle]); // slightly increased trail limit
            }
        };

        const handleMouseDown = (e) => {
            setIsClicking(true);

            // Create a burst of magical particles
            const burstCount = 12;
            const newBurstParticles = [];
            for (let i = 0; i < burstCount; i++) {
                const angle = (Math.PI * 2 * i) / burstCount;
                const distance = Math.random() * 50 + 30; // Random spread distance
                newBurstParticles.push({
                    id: Date.now() + i,
                    x: e.clientX,
                    y: e.clientY,
                    destX: e.clientX + Math.cos(angle) * distance,
                    destY: e.clientY + Math.sin(angle) * distance,
                    size: Math.random() * 6 + 3,
                    color: i % 2 === 0 ? '#87CEEB' : '#FFFFFF', // Alternating Sky Blue and White
                    isBurst: true
                });
            }
            // Add to persistent bursts list or reuse sparkles?
            // Reusing sparkles state for simplicity but distinguishing them
            setSparkles(prev => [...prev, ...newBurstParticles]);
        };

        const handleMouseUp = () => setIsClicking(false);

        window.addEventListener("mousemove", updateMousePosition);
        window.addEventListener("mousedown", handleMouseDown);
        window.addEventListener("mouseup", handleMouseUp);
        return () => {
            window.removeEventListener("mousemove", updateMousePosition);
            window.removeEventListener("mousedown", handleMouseDown);
            window.removeEventListener("mouseup", handleMouseUp);
        };
    }, []);

    return (
        <>
            {/* Main "Lumos" Wand Tip - Position Follower */}
            <motion.div
                className="fixed top-0 left-0 w-6 h-6 z-[100] pointer-events-none"
                animate={{
                    x: mousePosition.x - 12,
                    y: mousePosition.y - 12,
                }}
                transition={{
                    type: "spring",
                    stiffness: 800,
                    damping: 28,
                }}
            >
                {/* Inner Pulsing Glow - Heartbeat */}
                <motion.div
                    className="w-full h-full rounded-full blur-sm mix-blend-screen"
                    animate={{
                        scale: isClicking ? 1.6 : [1, 1.3, 1, 1.3, 1],
                        opacity: isClicking ? 1 : [0.6, 1, 0.6, 1, 0.6]
                    }}
                    transition={{
                        scale: { duration: 0.2 },
                        default: {
                            duration: 3,
                            repeat: Infinity,
                            ease: "easeInOut",
                            times: [0, 0.1, 0.2, 0.3, 1] // Lub-dub...... Lub-dub effect
                        }
                    }}
                    style={{
                        background: isClicking
                            ? "radial-gradient(circle, rgba(135, 206, 235, 1) 0%, rgba(0, 191, 255, 0) 70%)" // Sky Blue
                            : "radial-gradient(circle, rgba(255,255,255,1) 0%, rgba(255,255,255,0) 70%)",
                        boxShadow: isClicking
                            ? "0 0 30px 10px rgba(135, 206, 235, 0.8), 0 0 60px 30px rgba(0, 191, 255, 0.2)"
                            : "0 0 20px 5px rgba(255, 255, 255, 0.6), 0 0 40px 20px rgba(99, 102, 241, 0.1)",
                    }}
                />
            </motion.div>

            {/* Sparkle Trail & Magic Bursts */}
            <AnimatePresence>
                {sparkles.map(sparkle => (
                    <motion.div
                        key={sparkle.id}
                        className="fixed rounded-full z-[100] pointer-events-none mix-blend-screen"
                        initial={{
                            opacity: 1,
                            scale: 0.5,
                            x: sparkle.x,
                            y: sparkle.y
                        }}
                        animate={{
                            opacity: 0,
                            scale: 0,
                            x: sparkle.isBurst ? sparkle.destX : sparkle.x, // Movement
                            y: sparkle.isBurst ? sparkle.destY : sparkle.y + 20 // Fall down if not burst
                        }}
                        exit={{ opacity: 0 }}
                        transition={{
                            duration: sparkle.isBurst ? 0.6 : 1, // Faster burst
                            ease: "easeOut"
                        }}
                        style={{
                            width: sparkle.size,
                            height: sparkle.size,
                            backgroundColor: sparkle.color || 'white',
                            boxShadow: `0 0 ${sparkle.isBurst ? '15px' : '10px'} 2px ${sparkle.color || 'rgba(255,255,255,0.8)'}`
                        }}
                        onAnimationComplete={() => {
                            setSparkles(prev => prev.filter(s => s.id !== sparkle.id));
                        }}
                    />
                ))}
            </AnimatePresence>
        </>
    );
};

export default MouseFollower;
