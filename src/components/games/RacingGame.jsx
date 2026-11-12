import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RotateCcw, ArrowLeft, ArrowRight } from 'lucide-react';

const CAR_TYPES = ['🚕', '🚙', '🚑', '🚓', '🚚'];

const RacingGame = () => {
    const [gameState, setGameState] = useState('start'); // start, playing, gameover
    const [score, setScore] = useState(0);
    const [lane, setLane] = useState(1); // 0, 1, 2
    const [obstacles, setObstacles] = useState([]); // { id, lane, y, type }

    // Refs let the single game loop read the latest values without restarting
    const laneRef = useRef(1);
    const speedRef = useRef(1.5);
    const scoreRef = useRef(0);
    const obstaclesRef = useRef([]);

    useEffect(() => { laneRef.current = lane; }, [lane]);

    // Single game loop: spawn, move, collide, score (~60 fps)
    useEffect(() => {
        if (gameState !== 'playing') return;

        const gameLoop = setInterval(() => {
            let next = obstaclesRef.current;

            // Spawn a car (spawn rate grows with score), unless the top of that lane is busy
            if (Math.random() < 0.05 + scoreRef.current * 0.0001) {
                const newLane = Math.floor(Math.random() * 3);
                if (!next.some(o => o.lane === newLane && o.y < 20)) {
                    next = [...next, { id: Math.random(), lane: newLane, y: -20, type: CAR_TYPES[Math.floor(Math.random() * CAR_TYPES.length)] }];
                }
            }

            // Move cars down and drop the ones that left the screen
            next = next.map(o => ({ ...o, y: o.y + speedRef.current })).filter(o => o.y < 120);

            // Collision with the player's car
            if (next.some(o => o.lane === laneRef.current && o.y > 70 && o.y < 95)) {
                setGameState('gameover');
                return;
            }

            obstaclesRef.current = next;
            setObstacles(next);
            scoreRef.current += 1;
            setScore(Math.floor(scoreRef.current / 10)); // slower-moving score display
            speedRef.current = Math.min(4, speedRef.current + 0.001); // gradually speed up
        }, 16);

        return () => clearInterval(gameLoop);
    }, [gameState]);

    const moveLeft = () => setLane(p => Math.max(0, p - 1));
    const moveRight = () => setLane(p => Math.min(2, p + 1));

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'ArrowLeft') setLane(p => Math.max(0, p - 1));
            if (e.key === 'ArrowRight') setLane(p => Math.min(2, p + 1));
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);

    const startGame = () => {
        setLane(1);
        laneRef.current = 1;
        obstaclesRef.current = [];
        setObstacles([]);
        setScore(0);
        speedRef.current = 1.5;
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
                <div className="absolute inset-0 opacity-50 animate-pulse bg-[radial-gradient(rgba(255,255,255,0.06)_1px,transparent_1px)] bg-[length:6px_6px]"></div>

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
                <div className="text-3xl font-black text-white italic font-mono tracking-tighter">SPEED: {Math.floor(Math.min(4, 1.5 + score * 0.01) * 100)} km/h</div>
                <div className="text-xl font-bold text-yellow-400">SCORE: {score}</div>
            </div>

            {/* Mobile Controls */}
            <div className="absolute bottom-4 w-full flex justify-between px-8 md:hidden">
                <button onClick={moveLeft} aria-label="Steer left" className="p-6 bg-white/10 rounded-full backdrop-blur active:bg-white/30"><ArrowLeft className="text-white" /></button>
                <button onClick={moveRight} aria-label="Steer right" className="p-6 bg-white/10 rounded-full backdrop-blur active:bg-white/30"><ArrowRight className="text-white" /></button>
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
                        <h2 className="text-5xl font-black text-white mb-2 font-display tracking-widest italic transform -skew-x-12">
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
