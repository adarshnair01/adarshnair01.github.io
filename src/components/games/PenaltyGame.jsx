import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const PenaltyGame = () => {
    const [score, setScore] = useState({ goals: 0, attempts: 0 });
    const [gameState, setGameState] = useState('start'); // start, aiming, power, kicking, result, end

    // Game metrics
    const [aimX, setAimX] = useState(50); // 0-100 (50 is center)
    const [power, setPower] = useState(0);  // 0-100

    // Objects
    const [ballPos, setBallPos] = useState({ x: 50, y: 80, scale: 1 });
    const [goaliePos, setGoaliePos] = useState(50);

    // Refs for optimization/physics
    const aimDirection = useRef(1); // 1 = right, -1 = left
    const aimRef = useRef(50);
    const powerDir = useRef(1);
    const powerRef = useRef(0);
    const rafRef = useRef(null);

    // Initial Start
    const startGame = () => {
        setScore({ goals: 0, attempts: 0 });
        resetRound();
    };

    // Reset Game Round
    const resetRound = () => {
        setBallPos({ x: 50, y: 80, scale: 1 });
        setGoaliePos(50);
        setGameState('aiming');
        setAimX(50);
        setPower(0);
        aimRef.current = 50;
        powerRef.current = 0;
        aimDirection.current = 1;
        powerDir.current = 1;
    };

    // Main Loop
    useEffect(() => {
        if (gameState === 'kicking' || gameState === 'result' || gameState === 'start' || gameState === 'end') return;

        let lastTime = performance.now();

        const loop = (time) => {
            const dt = (time - lastTime) / 1000;
            lastTime = time;

            if (gameState === 'aiming') {
                // oscillate aim
                // Speed: crosses screen in ~1.5 sec
                const speed = 70; // % per second
                let newAim = aimRef.current + speed * dt * aimDirection.current;

                if (newAim > 95) { newAim = 95; aimDirection.current = -1; }
                if (newAim < 5) { newAim = 5; aimDirection.current = 1; }

                aimRef.current = newAim;
                setAimX(newAim);
            } else if (gameState === 'power') {
                // Charge power
                const speed = 100; // 1 sec to full
                let newPower = powerRef.current + speed * dt * powerDir.current;

                if (newPower > 100) { newPower = 100; powerDir.current = -1; }
                if (newPower < 0) { newPower = 0; powerDir.current = 1; }

                powerRef.current = newPower;
                setPower(newPower);
            }

            rafRef.current = requestAnimationFrame(loop);
        };

        rafRef.current = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(rafRef.current);
    }, [gameState]);

    const handleAction = () => {
        if (gameState === 'start' || gameState === 'end') return; // Handled by buttons

        if (gameState === 'aiming') {
            setGameState('power');
        } else if (gameState === 'power') {
            shoot();
        }
    };

    const shoot = () => {
        setGameState('kicking');
        const finalAim = aimRef.current;
        const finalPower = powerRef.current;

        // Calculate Shot Logic
        // Final X Target with some randomness based on power (overpower = less accurate)
        // High power (>90) adds random spread
        let spread = 0;
        if (finalPower > 90) spread = (Math.random() - 0.5) * 20; // Big spread
        else if (finalPower > 70) spread = (Math.random() - 0.5) * 5; // Slight spread

        let targetX = finalAim + spread;

        // Clamp to physical bounds maybe? Or allow missing
        // Goal is roughly 20% to 80% on screen?
        // Let's say goal posts are at 15% and 85%.
        const POST_LEFT = 15;
        const POST_RIGHT = 85;

        // Goalie Reaction
        // Goalie tries to go to targetX but with error/limitations
        // If power is high, goalie might be too slow

        // Goalie guess: Goalie picks a spot near targetX but maybe delayed
        let goalieDest = targetX + (Math.random() - 0.5) * 20; // Goalie isn't perfect

        // Clamp goalie
        if (goalieDest < 20) goalieDest = 20;
        if (goalieDest > 80) goalieDest = 80;

        setGoaliePos(goalieDest);

        // Animate Ball
        setBallPos({ x: targetX, y: 28, scale: 0.5 }); // Move to goal line

        setTimeout(() => {
            // Collision / Result Logic

            // 1. Did it miss?
            if (targetX < POST_LEFT || targetX > POST_RIGHT) {
                // Hit post? within 2%?
                if (Math.abs(targetX - POST_LEFT) < 2 || Math.abs(targetX - POST_RIGHT) < 2) {
                    // Post
                }
                finishShot('MISS');
                return;
            }

            // 2. Did goalie save?
            // Goalie width covers +/- 8 units?
            const ballDist = Math.abs(targetX - goalieDest);
            // High power reduces save chance (goalie "can't reach" effectively modeled by narrower save window?)
            // Or just check overlap
            let saveRadius = 10;
            if (finalPower > 85) saveRadius = 6; // Harder to save power shots

            if (ballDist < saveRadius) {
                finishShot('SAVED');
            } else {
                finishShot('GOAL');
                setScore(s => ({ ...s, goals: s.goals + 1 }));
            }

            setScore(s => ({ ...s, attempts: s.attempts + 1 }));

        }, 600);
    };

    // Result message state
    const [resultMessage, setResultMessage] = useState(null);

    const finishShot = (msg) => {
        setResultMessage(msg);
        setGameState('result');

        // Auto-next round logic
        setTimeout(() => {
            if (score.attempts + 1 >= 5) {
                setGameState('end');
            } else {
                resetRound();
            }
        }, 1500); // 1.5s delay before next kick
    };

    return (
        <div
            onClick={handleAction}
            className="relative w-full h-full flex flex-col items-center justify-center bg-gradient-to-b from-slate-900 to-slate-800 overflow-hidden cursor-pointer select-none"
        >

            {/* Stadium Atmosphere */}
            <div className="absolute inset-0 opacity-20 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-blue-900 to-black pointer-events-none"></div>

            {/* Scoreboard */}
            <div className="absolute top-8 left-1/2 -translate-x-1/2 flex items-center justify-center gap-8 bg-black/60 backdrop-blur-md px-8 py-3 rounded-full border border-white/10 shadow-2xl z-20 pointer-events-none">
                <div className="flex flex-col items-center">
                    <span className="text-xs font-bold text-green-400 uppercase tracking-widest">Goals</span>
                    <span className="text-4xl font-black text-white">{score.goals}</span>
                </div>
                <div className="w-px h-10 bg-white/20"></div>
                <div className="flex flex-col items-center">
                    <span className="text-xs font-bold text-gray-400 uppercase tracking-widest">Kick</span>
                    <span className="text-4xl font-black text-gray-300">{score.attempts + 1}<span className="text-xl text-gray-500">/5</span></span>
                </div>
            </div>

            {/* Field */}
            <div className="relative w-full max-w-5xl aspect-[16/9] flex flex-col justify-end perspective-1000 group">

                {/* Grass */}
                <div className="absolute bottom-0 w-full h-[60%] bg-gradient-to-t from-emerald-800 to-emerald-600 transform origin-bottom rotate-x-12 opacity-90 border-t border-emerald-400/30 overflow-hidden">
                    {/* Markings */}
                    <div className="absolute top-[10%] left-[20%] right-[20%] h-px bg-white/40"></div> {/* Goal Line roughly */}
                    <div className="absolute top-[10%] left-[20%] w-px h-full bg-white/20"></div>
                    <div className="absolute top-[10%] right-[20%] w-px h-full bg-white/20"></div>
                    <div className="absolute bottom-[20%] left-1/2 -translate-x-1/2 w-3 h-3 bg-white/80 rounded-full box-content border-2 border-emerald-900/20"></div>
                </div>

                {/* Goal Post */}
                <div className="absolute top-[20%] left-[20%] right-[20%] h-[40%] pointer-events-none z-10 perspective-goal">
                    <div className="absolute top-0 left-0 right-0 h-3 bg-slate-300 shadow-lg z-20 rounded-full"></div>
                    <div className="absolute top-0 bottom-0 left-0 w-3 bg-slate-300 shadow-lg z-20 rounded-full"></div>
                    <div className="absolute top-0 bottom-0 right-0 w-3 bg-slate-300 shadow-lg z-20 rounded-full"></div>
                    <div className="absolute inset-2 border-white/5 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-white/10 to-transparent bg-[length:10px_10px] bg-repeat opacity-30"></div>
                </div>

                {/* Goalie */}
                <motion.div
                    className="absolute top-[35%] w-16 md:w-20 h-28 md:h-36 left-[50%] -translate-x-1/2 z-15 flex flex-col items-center justify-end"
                    animate={{ left: `${goaliePos}%` }}
                    transition={{ type: "spring", stiffness: 100, damping: 20 }}
                >
                    <div className="w-full h-full bg-yellow-500 rounded-t-xl relative shadow-2xl skew-x-[-2deg]">
                        <div className="absolute top-0 w-full h-8 bg-black/10"></div>
                        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 text-3xl font-black text-black/30">1</div>
                        {/* Arms hint */}
                        <div className="absolute top-8 -left-2 w-4 h-16 bg-yellow-600 rounded-full origin-top rotate-12"></div>
                        <div className="absolute top-8 -right-2 w-4 h-16 bg-yellow-600 rounded-full origin-top -rotate-12"></div>
                    </div>
                </motion.div>

                {/* Ball */}
                <motion.div
                    className="absolute w-10 h-10 md:w-14 md:h-14 z-30 flex items-center justify-center left-[50%] -translate-x-1/2 shadow-xl rounded-full bg-white"
                    initial={{ left: '50%', top: '80%' }}
                    animate={{ left: `${ballPos.x}%`, top: `${ballPos.y}%`, scale: ballPos.scale }}
                    transition={{ duration: gameState === 'kicking' ? 0.6 : 0.5, ease: "circOut" }}
                >
                    <div className="w-full h-full bg-[url('https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Soccerball.svg/1024px-Soccerball.svg.png')] bg-cover bg-center opacity-90 animate-spin-slow"></div>
                </motion.div>

                {/* Result Text */}
                <AnimatePresence>
                    {(gameState === 'result') && (
                        <motion.div
                            initial={{ opacity: 0, scale: 3, y: -100 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            className="absolute inset-0 flex items-center justify-center z-50 pointer-events-none"
                        >
                            <h2 className={`text-9xl font-black italic drop-shadow-[0_0_30px_rgba(0,0,0,1)] stroke-2 stroke-white ${resultMessage === 'GOAL' ? 'text-green-500' : 'text-red-500'
                                }`}>
                                {resultMessage}!
                            </h2>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Start / End Screens */}
                <AnimatePresence>
                    {(gameState === 'start' || gameState === 'end') && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="absolute inset-0 bg-black/80 backdrop-blur-md flex flex-col items-center justify-center z-50 p-8 text-center"
                        >
                            <h2 className="text-6xl font-black text-white italic tracking-tighter mb-4">
                                {gameState === 'start' ? 'PENALTY SHOOTOUT' : 'GAME OVER'}
                            </h2>
                            {gameState === 'end' && (
                                <div className="text-4xl text-green-400 font-bold mb-8">
                                    Final Score: {score.goals} / 5
                                </div>
                            )}
                            <button
                                onClick={(e) => { e.stopPropagation(); startGame(); }}
                                className="px-8 py-4 bg-white text-black font-black text-xl hover:scale-105 transition-transform rounded-full"
                            >
                                {gameState === 'start' ? 'START KICKING' : 'PLAY AGAIN'}
                            </button>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {/* FIFA Style Mechanics UI */}
            <div className="absolute bottom-10 w-full flex flex-col items-center gap-4 pointer-events-none">

                {/* Aim Bar */}
                <div className="relative w-64 h-4 bg-gray-900/80 rounded-full border border-white/20 overflow-hidden">
                    <div className="absolute inset-0 flex justify-center">
                        <div className="w-1 h-full bg-white/20"></div> {/* Center mark */}
                    </div>
                    {/* Goal Zone Hints */}
                    <div className="absolute top-0 bottom-0 left-[20%] width-[60%] bg-green-500/10"></div>

                    {/* Cursor */}
                    <motion.div
                        className="absolute top-0 bottom-0 w-2 bg-yellow-400 shadow-[0_0_10px_#facc15]"
                        style={{ left: `${aimX}%`, transform: 'translateX(-50%)' }}
                    ></motion.div>
                </div>

                {/* Aim Label */}
                <div className={`text-xs font-bold tracking-widest ${gameState === 'aiming' ? 'text-yellow-400 animate-pulse' : 'text-gray-500'}`}>
                    {gameState === 'aiming' ? 'CLICK TO LOCK AIM' : 'AIM LOCKED'}
                </div>

                {/* Power Bar */}
                <div className="relative w-64 h-6 bg-gray-900/80 rounded-full border border-white/20 overflow-hidden mt-1">
                    {/* Gradient Filling */}
                    <div
                        className={`h-full transition-all duration-75 ${power > 90 ? 'bg-red-500' :
                                power > 70 ? 'bg-green-500' :
                                    'bg-blue-500'
                            }`}
                        style={{ width: `${power}%` }}
                    ></div>

                    {/* Tick Marks for Sweet Spot */}
                    <div className="absolute top-0 bottom-0 right-[20%] w-1 bg-white/50 z-10"></div> {/* 80% mark */}
                    <div className="absolute top-0 bottom-0 right-[10%] w-1 bg-white/50 z-10"></div> {/* 90% mark */}
                </div>

                {/* Power Label */}
                <div className={`text-xs font-bold tracking-widest ${gameState === 'power' ? 'text-blue-400 animate-pulse' : 'text-gray-500'}`}>
                    {gameState === 'power' ? 'CLICK TO SHOOT' : (gameState === 'aiming' ? 'WAITING...' : 'POWER LOCKED')}
                </div>

            </div>
        </div>
    );
};

export default PenaltyGame;
