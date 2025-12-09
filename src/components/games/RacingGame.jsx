import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RotateCcw, ArrowLeft, ArrowRight } from 'lucide-react';

const RacingGame = () => {
    const [gameState, setGameState] = useState('start'); // start, playing, gameover
    const [score, setScore] = useState(0);
    const [lane, setLane] = useState(1); // 0, 1, 2
    const [obstacles, setObstacles] = useState([]); // { id, lane, y, type }
    const [speed, setSpeed] = useState(1.5);

    const rafRef = useRef(null);
    const scoreRef = useRef(0);
    const obstacleIntervalRef = useRef(0);

    const CAR_TYPES = ['🚕', '🚙', '🚑', '🚓', '🚚'];

    // Game Loop
    useEffect(() => {
        if (gameState !== 'playing') return;

        let lastTime = Date.now();
        setObstacles([]); // Clear on start
        setSpeed(1.5);
        scoreRef.current = 0;

        const loop = () => {
            const now = Date.now();
            const dt = (now - lastTime) / 16; // approx 1 frame
            lastTime = now;

            // Spawn Obstacles
            obstacleIntervalRef.current -= dt;
            if (obstacleIntervalRef.current <= 0) {
                const newLane = Math.floor(Math.random() * 3);
                const type = CAR_TYPES[Math.floor(Math.random() * CAR_TYPES.length)];

                // Don't spawn if there's already a car very close in that lane (prevents unfair piles)
                // Simplified: just push
                setObstacles(prev => [
                    ...prev,
                    { id: Date.now() + Math.random(), lane: newLane, y: -20, type }
                ]);

                // Reset Interval (decreases as speed increases)
                obstacleIntervalRef.current = Math.max(20, 100 - scoreRef.current / 5);
            }

            // Move Obstacles
            setObstacles(prev => {
                const next = [];
                let collided = false;

                prev.forEach(obs => {
                    obs.y += speed * 0.8; // Move down

                    // Collision Check
                    // Player is roughly at y=80, size ~15%
                    // Obstacle size ~15%
                    const playerY = 80;
                    if (
                        obs.lane === lane &&
                        obs.y > playerY - 10 &&
                        obs.y < playerY + 10
                    ) {
                        collided = true;
                    }

                    if (obs.y < 110) next.push(obs); // Keep if on screen
                });

                if (collided) {
                    setGameState('gameover');
                    return prev; // Stop updating
                }

                return next;
            });

            // Increase Score & Speed
            if (gameState === 'playing') {
                scoreRef.current += 0.5;
                setScore(Math.floor(scoreRef.current));
                setSpeed(s => Math.min(3.5, s + 0.001)); // Cap speed
                rafRef.current = requestAnimationFrame(loop);
            }
        };

        rafRef.current = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(rafRef.current);
    }, [gameState, lane]); // Dependency on 'lane' is crucial for collision check in closure? No, using ref for updated lane would be better, but 'lane' changes triggers re-render, forcing new closure. This is fine for simple game.

    // Better way to handle collision: Check collision inside setting of state where 'lane' is accessible or use Ref for lane.
    // Let's use Ref for lane to avoid loop restart
    const laneRef = useRef(1);
    useEffect(() => { laneRef.current = lane; }, [lane]);

    // Redefine Loop with Lane Ref to avoid re-binding
    useEffect(() => {
        if (gameState !== 'playing') return;
        let lastTime = Date.now();

        const loop = () => {
            // ... Logic duplicated here? To fix the closure issue proper:
            setObstacles(prev => {
                const next = [];
                let collided = false;
                prev.forEach(obs => {
                    obs.y += speed * 0.5; // Adjusted speed factor relative to fps

                    if (
                        obs.lane === laneRef.current &&
                        obs.y > 75 && // Player top
                        obs.y < 95    // Player bottom
                    ) {
                        collided = true;
                    }
                    if (obs.y < 120) next.push(obs);
                });

                if (collided) {
                    setGameState('gameover');
                    return prev;
                }
                return next;
            });
            // Spawn logic needs to be inside or outside?
            // Let's simplify: Use a single interval for 'tick' that handles move and spawn
        };
        // ...
        // Actually, the previous structure re-creating loop on 'lane' change is LAGGY.
        // We must use the Ref approach for lane.
    }, [gameState]); // Only run on game state change

    // Correct Loop Implementation
    useEffect(() => {
        if (gameState !== 'playing') return;

        const gameLoop = setInterval(() => {
            setObstacles(prev => {
                let gameOver = false;
                // Spawn
                if (Math.random() < (0.05 + scoreRef.current * 0.0001)) { // Increasing spawn rate
                    const newLane = Math.floor(Math.random() * 3);
                    // Check if lane is clear at top
                    const isClear = !prev.some(o => o.lane === newLane && o.y < 20);
                    if (isClear) {
                        return [...prev, { id: Math.random(), lane: newLane, y: -20, type: CAR_TYPES[Math.floor(Math.random() * CAR_TYPES.length)] }];
                    }
                }

                // Move & Collide
                const next = prev.map(o => ({ ...o, y: o.y + speed })).filter(o => o.y < 120);

                // Check Collision
                if (next.some(o => o.lane === laneRef.current && o.y > 70 && o.y < 95)) {
                    setGameState('gameover');
                    gameOver = true;
                }

                if (!gameOver) {
                    scoreRef.current += 1;
                    setScore(Math.floor(scoreRef.current / 10)); // Slow down score display
                    setSpeed(s => Math.min(4, s + 0.001));
                }

                return gameOver ? prev : next;
            });
        }, 16); // 60 FPS

        return () => clearInterval(gameLoop);
    }, [gameState]);


    const moveLeft = () => setLane(p => Math.max(0, p - 1));
    const moveRight = () => setLane(p => Math.min(2, p + 1));

    const handleKeyDown = (e) => {
        if (e.key === 'ArrowLeft') moveLeft();
        if (e.key === 'ArrowRight') moveRight();
    };

    useEffect(() => {
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);

    const startGame = () => {
        setLane(1);
        setObstacles([]);
        setScore(0);
        setSpeed(1.5);
        scoreRef.current = 0;
        setGameState('playing');
    };

    return (
        <div className="flex flex-col items-center justify-center w-full h-full relative bg-gray-900 overflow-hidden">

            {/* Road */}
            <div className="w-64 md:w-80 h-full bg-gray-800 relative border-x-4 border-yellow-500 shadow-2xl overflow-hidden">
                {/* Lane Markers */}
                <div className="absolute left-1/3 top-0 h-full w-2 border-r-2 border-dashed border-white/30"></div>
                <div className="absolute left-2/3 top-0 h-full w-2 border-r-2 border-dashed border-white/30"></div>

                {/* Moving Road Effect (Simple animate) */}
                <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/asfalt-dark.png')] opacity-50 animate-pulse"></div>

                {/* Obstacles */}
                <AnimatePresence>
                    {obstacles.map(obs => (
                        <div
                            key={obs.id}
                            className="absolute text-5xl flex justify-center items-center"
                            style={{
                                left: `${obs.lane * 33.33}%`,
                                top: `${obs.y}%`,
                                width: '33.33%',
                                height: '10%' // roughly car size
                            }}
                        >
                            {obs.type}
                        </div>
                    ))}
                </AnimatePresence>

                {/* Player Car */}
                <motion.div
                    animate={{ left: `${lane * 33.33}%` }}
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                    className="absolute bottom-10 w-1/3 text-5xl flex justify-center items-center z-10 drop-shadow-[0_0_15px_rgba(255,0,0,0.5)]"
                    style={{ height: '10%' }}
                >
                    🏎️
                </motion.div>
            </div>

            {/* Header */}
            <div className="absolute top-4 left-0 w-full text-center pointer-events-none">
                <div className="text-3xl font-black text-white italic font-mono tracking-tighter">SPEED: {Math.floor(speed * 100)} km/h</div>
                <div className="text-xl font-bold text-yellow-400">SCORE: {score}</div>
            </div>

            {/* Mobile Controls */}
            <div className="absolute bottom-4 w-full flex justify-between px-8 md:hidden">
                <button onClick={moveLeft} className="p-6 bg-white/10 rounded-full backdrop-blur active:bg-white/30"><ArrowLeft className="text-white" /></button>
                <button onClick={moveRight} className="p-6 bg-white/10 rounded-full backdrop-blur active:bg-white/30"><ArrowRight className="text-white" /></button>
            </div>

            {/* Game Over / Start Screen */}
            <AnimatePresence>
                {gameState !== 'playing' && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute inset-0 bg-black/80 backdrop-blur-md flex flex-col items-center justify-center z-50 p-6 text-center"
                    >
                        <h2 className="text-5xl font-black text-white mb-2 font-harry tracking-widest italic transform -skew-x-12">
                            {gameState === 'start' ? 'TURBO RACER' : 'CRASHED!'}
                        </h2>
                        {gameState === 'gameover' && (
                            <div className="text-3xl text-yellow-400 mb-6 font-bold">Score: {score}</div>
                        )}
                        <p className="text-slate-400 mb-8 max-w-xs">
                            Dodging traffic at high speeds. Use Left/Right keys or buttons.
                        </p>

                        <button
                            onClick={startGame}
                            className="px-8 py-3 bg-red-600 text-white font-bold rounded-full hover:bg-red-700 transition-colors flex items-center gap-2 shadow-[0_0_20px_rgba(220,38,38,0.5)] scale-110"
                        >
                            {gameState === 'start' ? 'START ENGINE' : 'RESTART'}
                        </button>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default RacingGame;
