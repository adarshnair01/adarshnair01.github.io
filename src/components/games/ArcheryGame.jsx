import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RotateCcw, Target } from 'lucide-react';

const ArcheryGame = () => {
    const [gameState, setGameState] = useState('start'); // start, playing, gameover
    const [score, setScore] = useState(0);
    const [arrows, setArrows] = useState(10);

    // Game Objects
    const [arrowState, setArrowState] = useState('idle'); // idle, charging, flying, hit
    const [aimY, setAimY] = useState(50); // Vertical aiming percentage
    const [power, setPower] = useState(0); // 0 to 100

    // Visually arrow follows aimY when idle/charging, but follows trajectory when flying
    const [arrowPos, setArrowPos] = useState({ x: 10, y: 50 });
    const [targetPos, setTargetPos] = useState({ x: 90, y: 50 });

    const [lastHitScore, setLastHitScore] = useState(null);

    const containerRef = useRef(null);
    const rafRef = useRef(null);

    // Refs for loop state
    const powerDir = useRef(1); // 1 = up, -1 = down
    const flightVelocity = useRef({ x: 0, y: 0 });
    const isPerfectShot = useRef(false);

    // Constants
    // State for dynamic difficulty
    const [difficultyLevel, setDifficultyLevel] = useState(1);

    // Constants
    const TARGET_SPEED_BASE = 0.5;
    const POWER_SPEED_BASE = 1.5;
    const GRAVITY = 0.4;

    // Game Loop
    useEffect(() => {
        if (gameState !== 'playing') return;

        let startTime = Date.now();

        const loop = () => {
            const now = Date.now();
            const time = (now - startTime) / 1000;

            // 1. Move Target (Stationary)
            const newTargetY = 50;
            setTargetPos(prev => ({ ...prev, y: newTargetY }));

            // 2. Handle Power Charging
            if (arrowState === 'charging') {
                setPower(prev => {
                    // Speed increases with difficulty (score)
                    // Every 500 points = +0.5 speed
                    const currentSpeed = POWER_SPEED_BASE + (score / 500) * 0.5;

                    let next = prev + currentSpeed * powerDir.current;
                    if (next >= 100) {
                        next = 100;
                        powerDir.current = -1;
                    } else if (next <= 0) {
                        next = 0;
                        powerDir.current = 1;
                    }
                    return next;
                });
            }

            // 3. Handle Arrow Flight
            if (arrowState === 'flying') {
                setArrowPos(prev => {
                    // Update Velocity with Gravity
                    // ONLY apply gravity if it's NOT a perfect shot
                    if (!isPerfectShot.current) {
                        flightVelocity.current.y += GRAVITY * 0.1;
                    }

                    const newX = prev.x + flightVelocity.current.x;
                    const newY = prev.y + flightVelocity.current.y;

                    // Hit Detection
                    if (newX >= 90) { // Target Plane
                        // Calculate Distance to Target Center
                        const distY = Math.abs(newY - newTargetY);

                        let hitScore = 0;
                        // Precision Hit Boxes
                        if (distY < 4) hitScore = 50;      // Bullseye
                        else if (distY < 8) hitScore = 20; // Inner
                        else if (distY < 15) hitScore = 10;// Outer

                        setTimeout(() => handleHit(hitScore, newY), 0);
                        return { x: 90, y: newY }; // Freeze at hit
                    }

                    // Miss Detection (Past screen)
                    if (newX > 105 || newY > 105 || newY < -5) {
                        setTimeout(() => handleHit(0, newY), 0);
                        return { x: 105, y: newY };
                    }

                    return { x: newX, y: newY };
                });
            } else if (arrowState === 'idle' || arrowState === 'charging') {
                // Determine Arrow Position based on Aim
                setArrowPos({ x: 10, y: aimY });
            }

            rafRef.current = requestAnimationFrame(loop);
        };

        rafRef.current = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(rafRef.current);
    }, [gameState, arrowState, aimY]); // aimY dependency ensures update when moving mouse

    // Mouse Controls for Aiming
    const handleMouseMove = (e) => {
        // Stop aiming if charging (steady aim)
        if (gameState !== 'playing' || arrowState === 'flying' || arrowState === 'hit' || arrowState === 'charging') return;
        if (!containerRef.current) return;

        const rect = containerRef.current.getBoundingClientRect();
        const yPercent = ((e.clientY - rect.top) / rect.height) * 100;
        // Clamp Aim
        setAimY(Math.max(5, Math.min(95, yPercent)));
    };

    // Charging Start
    const handleMouseDown = () => {
        if (gameState !== 'playing' || arrowState !== 'idle') return;
        setArrowState('charging');
        setPower(0);
        powerDir.current = 1;
    };

    // Release to Shoot
    const handleMouseUp = () => {
        if (gameState !== 'playing' || arrowState !== 'charging') return;

        // Shoot!
        setArrowState('flying');

        // Calculate Flight Physics based on Power
        // Perfect Power = 85 (Green Zone: 80 - 90)
        // If Power < 85: Drops (Gravity affects more relative to speed? No, simpler: Initial Trajectory dip)
        // Actually physically: Less power = Less X Velocity = Gravity acts longer = Drops more.

        // User request: "release logic: green mark -> bullseye"
        // Interpretation: If Aim is reasonable and Power is Green, it hits well.
        // Let's map Power directly to Vertical Drift.

        const GREEN_ZONE_MIN = 80;
        const GREEN_ZONE_MAX = 90;

        // Velocity X is constant-ish for feeling
        flightVelocity.current.x = 2.0;

        // Velocity Y (Initial Launch Angle/Drift)
        // If Green: 0 Drift (Perfect)
        // If Low Power: Positive Drift (Downwards/Gravity wins)
        // If Over Power: Negative Drift (Overshoot/Upwards) -> Archery doesn't really work like that but typically "Power" = "Distance".
        // In this 2D side view, "Power" usually means "Flat Trajectory".

        let driftY = 0;
        if (power >= GREEN_ZONE_MIN && power <= GREEN_ZONE_MAX) {
            driftY = 0; // Perfect, flies straight
            flightVelocity.current.x = 2.5; // Fast
            isPerfectShot.current = true; // No Gravity
        } else if (power < GREEN_ZONE_MIN) {
            // Weak shot, gravity pulls it down faster effectively? 
            // Or we simulate initial downward velocity?
            driftY = (GREEN_ZONE_MIN - power) * 0.05; // Increased drop for weak shots
            flightVelocity.current.x = 1.5; // Slower
            isPerfectShot.current = false;
        } else {
            // Overpower -> Maybe flies erratic or slightly up (recoil)
            driftY = (power - GREEN_ZONE_MAX) * -0.05; // Up
            flightVelocity.current.x = 2.8;
            isPerfectShot.current = false;
        }

        flightVelocity.current.y = driftY;
    };

    const handleHit = (hitPoints, impactY) => {
        setArrowState('hit');

        if (hitPoints > 0) {
            setScore(s => s + hitPoints);
            setLastHitScore(hitPoints);
        } else {
            setLastHitScore(0); // Miss
        }

        setTimeout(() => setLastHitScore(null), 1000);

        setArrows(prev => {
            const left = prev - 1;
            if (left <= 0) {
                setTimeout(() => setGameState('gameover'), 800);
                return 0;
            }
            // Reset for next arrow
            setTimeout(() => {
                setArrowState('idle');
                setPower(0);
                // Reset pos is handled by idle state in loop
            }, 800);
            return left;
        });
    };

    const startGame = () => {
        setScore(0);
        setArrows(10);
        setGameState('playing');
        setArrowState('idle');
        setAimY(50);
        setPower(0);
    };

    // SVG Path for the Bow (Simple recurve shape)
    // Drawn via SVG path.
    // Dynamic String: calculated in render based on 'power'

    return (
        <div
            ref={containerRef}
            className="flex flex-col items-center justify-center w-full h-full relative p-4 bg-[#1a1a2e] overflow-hidden select-none"
            onMouseMove={handleMouseMove}
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onTouchStart={handleMouseDown}
            onTouchEnd={handleMouseUp}
            style={{ cursor: 'none' }} // Hide default cursor for immersion
        >

            {/* Custom Aim Cursor (if used to replace default) - Optional */}
            {/* Keeping it simple with existing mechanics for now, but steady aim requested.
                "Aim should not be moving it should be steady" -> This implies that when CHARGING, we might 
                want to lock the aim or slow it down? Or user means the *oscillation* of target is tricky?
                "Initially power upgrade will be show but later it will speed up" -> This refers to the power bar speed increasing.
             */}

            {/* Header */}
            <div className="absolute top-4 left-0 w-full flex justify-between px-8 z-10 pointer-events-none">
                <div className="text-2xl font-black text-white font-harry">Score: <span className="text-yellow-400">{score}</span></div>
                <div className="text-2xl font-black text-white font-harry">Arrows: <span className="text-orange-400">{arrows}</span></div>
            </div>

            {/* Target */}
            <div
                className="absolute w-24 h-24 flex items-center justify-center pointer-events-none"
                style={{
                    left: `${targetPos.x}%`,
                    top: `${targetPos.y}%`,
                    transform: 'translate(-50%, -50%)',
                }}
            >
                {/* Visual Target */}
                <div className="w-full h-full bg-white rounded-full border-4 border-slate-300 relative shadow-xl">
                    <div className="absolute inset-0 m-auto w-16 h-16 bg-blue-600 rounded-full border-4 border-white"></div>
                    <div className="absolute inset-0 m-auto w-8 h-8 bg-red-600 rounded-full border-4 border-white"></div>
                    <div className="absolute inset-0 m-auto w-3 h-3 bg-yellow-400 rounded-full"></div>
                </div>
            </div>


            {/* Player Container (Bow + Arrow + Power) */}
            {/* This follows aimY. If charging, we can clamp or smooth aimY if "steady" means software stabilization.
                But usually "steady aim" means the PLAYER must hold steady.
                If user meant "the aim should not be moving" during charging -> we stop updating aimY in handleMouseMove if charging.
            */}

            <div
                className="absolute left-8 w-32 h-64 pointer-events-none flex items-center"
                style={{
                    top: `${aimY}%`,
                    transform: 'translateY(-50%)',
                    transition: 'top 0.1s linear' // Smooth visual follow
                }}
            >
                {/* SVG Bow */}
                {/* We use an SVG to draw the bow and the dynamic string */}
                {/* SVG Bow (Mirrored to face Right) */}
                <svg width="100%" height="100%" viewBox="0 0 100 200" className="overflow-visible">
                    <defs>
                        <linearGradient id="bowGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#8B4513" />
                            <stop offset="50%" stopColor="#A0522D" />
                            <stop offset="100%" stopColor="#5D4037" />
                        </linearGradient>
                    </defs>

                    {/* Bow Riser (Handle) - Flipped X: 30->70, 40->60 */}
                    <path
                        d="M 70,80 Q 60,100 70,120"
                        stroke="#5D4037" strokeWidth="6" fill="none" strokeLinecap="round"
                    />

                    {/* Upper Limb - Flipped X: 30->70, 10->90, 40->60 */}
                    <path
                        d="M 70,80 C 90,70 90,30 60,10"
                        stroke="url(#bowGradient)" strokeWidth="5" fill="none" strokeLinecap="round"
                    />

                    {/* Lower Limb - Flipped X */}
                    <path
                        d="M 70,120 C 90,130 90,170 60,190"
                        stroke="url(#bowGradient)" strokeWidth="5" fill="none" strokeLinecap="round"
                    />

                    {/* String */}
                    {/* Tips at 60,10 and 60,190 */}
                    {(() => {
                        const stringRestX = 60;
                        const pullAmount = (power / 100) * 45; // Pull back moves Left (minus X)
                        const nockX = stringRestX - pullAmount;

                        // If flying, string snaps back to rest
                        const currentNockX = (arrowState === 'charging') ? nockX : stringRestX;

                        return (
                            <polyline
                                points={`60,10 ${currentNockX},100 60,190`}
                                stroke="#DDDDDD"
                                strokeWidth="2"
                                fill="none"
                            />
                        );
                    })()}
                </svg>

                {/* Arrow (Visual only, relative to bow group) */}
                {(arrowState === 'idle' || arrowState === 'charging') && (
                    <motion.div
                        className="absolute top-1/2 w-40 h-2 origin-left"
                        style={{
                            // Align Nock (Left) with String Rest (60% of 100 units)
                            // Container is w-32 (128px approx), SVG viewbox 100.
                            // So left should be approx 60%.
                            left: '60%',
                            x: arrowState === 'charging' ? -(power / 100) * 50 : 0, // Pull moves left (negative x)
                            y: '-50%'
                        }}
                    >
                        <div className="w-full h-full relative" style={{ transform: 'translateX(-2px)' }}> {/* Slight nudge to seat nock */}
                            {/* Shaft */}
                            <div className="absolute inset-y-0 left-0 right-2 bg-gradient-to-r from-neutral-300 to-neutral-400 h-1 my-auto rounded-full"></div>
                            {/* Fletching */}
                            <div className="absolute left-0 top-1/2 -translate-y-1/2 w-4 h-3 bg-red-600 rounded-sm -skew-x-12"></div>
                            {/* Point */}
                            <div className="absolute right-0 top-1/2 -translate-y-1/2 w-0 h-0 border-l-[12px] border-l-stone-400 border-y-[4px] border-y-transparent"></div>
                        </div>
                    </motion.div>
                )}

                {/* Trajectory Guide (Dotted Line) */}
                {/* Shows predicted path based on current power */}
                <svg className="absolute top-1/2 left-0 w-96 h-96 overflow-visible pointer-events-none" style={{ transform: 'translateY(-50%)', zIndex: -1 }}>
                    {(() => {
                        if (arrowState !== 'charging') return null;

                        // Calculate trajectory points
                        // Simulate physics same as loop
                        const points = [];
                        let simX = 0; // Relative to bow center (approx)
                        let simY = 0;

                        let simVelX = 2.0;
                        // Logic from handleMouseUp prediction
                        const GREEN_ZONE_MIN = 80;
                        const GREEN_ZONE_MAX = 90;
                        let driftY = 0;
                        let simIsPerfect = false;

                        if (power >= GREEN_ZONE_MIN && power <= GREEN_ZONE_MAX) {
                            driftY = 0;
                            simVelX = 2.5;
                            simIsPerfect = true;
                        } else if (power < GREEN_ZONE_MIN) {
                            driftY = (GREEN_ZONE_MIN - power) * 0.05; // Matches new weak drop
                            simVelX = 1.5;
                        } else {
                            driftY = (power - GREEN_ZONE_MAX) * -0.05;
                            simVelX = 2.8;
                        }

                        let simVelY = driftY;

                        // Simulate 30 steps
                        for (let i = 0; i < 40; i++) {
                            // Only apply gravity if NOT perfect
                            if (!simIsPerfect) {
                                simVelY += GRAVITY * 0.1;
                            }
                            simX += simVelX * 15; // Scale up for visual distance (since loop is per frame)
                            simY += simVelY * 15;
                            points.push(`${simX},${simY}`);
                        }

                        return (
                            <polyline
                                points={points.join(' ')}
                                fill="none"
                                stroke={simIsPerfect ? "rgba(74, 222, 128, 0.6)" : "rgba(255, 255, 255, 0.3)"} // Color hint
                                strokeWidth="2"
                                strokeDasharray="5,5"
                            />
                        );
                    })()}
                </svg>

                {/* Power Meter (Attached to Bow) */}
                <AnimatePresence>
                    {(arrowState === 'charging') && (
                        <motion.div
                            initial={{ opacity: 0, scaleY: 0 }}
                            animate={{ opacity: 1, scaleY: 1 }}
                            exit={{ opacity: 0 }}
                            className="absolute -left-8 top-1/2 -translate-y-1/2 w-2 h-40 bg-gray-900/50 rounded-full border border-white/20 overflow-hidden"
                        >
                            {/* Green Zone Indicator */}
                            <div className="absolute bottom-[80%] h-[10%] w-full bg-green-400/80 z-10 shadow-[0_0_10px_rgba(74,222,128,0.5)]"></div>

                            {/* Fill Bar */}
                            <div
                                className={`absolute bottom-0 w-full transition-all duration-75 ease-linear ${power >= 80 && power <= 90 ? 'bg-green-500 shadow-[0_0_15px_#22c55e]' :
                                    power > 90 ? 'bg-red-500' : 'bg-yellow-400'
                                    }`}
                                style={{ height: `${power}%` }}
                            ></div>
                        </motion.div>
                    )}
                </AnimatePresence>

            </div>

            {/* Global Flying Arrow (Independent of Bow Group) */}
            {/* We map game 0-100 coordinates to viewport for standard look */}
            {(arrowState === 'flying' || arrowState === 'hit') && (
                <div
                    className="absolute h-2 w-40 pointer-events-none"
                    style={{
                        left: `${arrowPos.x}%`,
                        top: `${arrowPos.y}%`,
                        transform: `translate(-20%, -50%) rotate(${Math.atan2(flightVelocity.current.y, flightVelocity.current.x) * (180 / Math.PI)}deg)`,
                        // We calculate rotation based on velocity vector!
                    }}
                >
                    <div className="w-full h-full relative">
                        {/* Shaft */}
                        <div className="absolute inset-y-0 left-0 right-2 bg-gradient-to-r from-neutral-300 to-neutral-400 h-1 my-auto rounded-full shadow-sm"></div>
                        {/* Fletching */}
                        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-4 h-3 bg-red-600 rounded-sm -skew-x-12"></div>
                        {/* Point */}
                        <div className="absolute right-0 top-1/2 -translate-y-1/2 w-0 h-0 border-l-[12px] border-l-stone-400 border-y-[4px] border-y-transparent"></div>
                    </div>
                </div>
            )}

            {/* Hit Score Popup */}
            <AnimatePresence>
                {lastHitScore !== null && (
                    <motion.div
                        initial={{ opacity: 0, y: 0, scale: 0.5 }}
                        animate={{ opacity: 1, y: -50, scale: 1.5 }}
                        exit={{ opacity: 0 }}
                        className="absolute top-1/2 left-1/2 font-black text-4xl text-yellow-400 drop-shadow-[0_0_10px_rgba(250,204,21,0.8)] pointer-events-none z-50"
                    >
                        {lastHitScore === 0 ? 'MISS!' : `+${lastHitScore}`}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Game Over / Start Screen */}
            <AnimatePresence>
                {gameState !== 'playing' && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute inset-0 bg-black/80 backdrop-blur-md flex flex-col items-center justify-center z-50 p-6 text-center"
                    >
                        <h2 className="text-5xl font-black text-white mb-2 font-harry tracking-widest">
                            {gameState === 'start' ? 'ARCHERY' : 'GAME OVER'}
                        </h2>
                        {gameState === 'gameover' && (
                            <div className="text-3xl text-yellow-400 mb-6 font-bold">Score: {score}</div>
                        )}
                        <p className="text-slate-400 mb-8 max-w-xs leading-relaxed">
                            1. Move mouse to Aim.<br />
                            2. <b>Hold Click</b> to Charge Power.<br />
                            3. Release when bar is <b>GREEN</b>.
                        </p>

                        <button
                            onClick={(e) => { e.stopPropagation(); startGame(); }}
                            className="px-8 py-3 bg-white text-slate-900 font-bold rounded-full hover:bg-slate-200 transition-colors flex items-center gap-2 shadow-lg scale-110"
                        >
                            {gameState === 'start' ? <Target size={20} /> : <RotateCcw size={20} />}
                            {gameState === 'start' ? 'Start Shooting' : 'Try Again'}
                        </button>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default ArcheryGame;
