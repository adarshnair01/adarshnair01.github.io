import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowUp, ArrowDown, ArrowLeft, ArrowRight, RotateCcw } from 'lucide-react';

const NarutoMazeGame = () => {
    // 0: Path, 1: Wall, 2: Start (Naruto), 3: Goal (Ramen), 4: Trap (Shuriken)
    const [maze, setMaze] = useState([]);
    const [playerPos, setPlayerPos] = useState({ x: 0, y: 0 });
    const [gameState, setGameState] = useState('playing'); // playing, won
    const [level, setLevel] = useState(1);
    const [message, setMessage] = useState('');

    const ROWS = 15;
    const COLS = 15;

    // Maze Generation (Recursive Backtracking)
    const generateMaze = useCallback(() => {
        // Initialize grid with walls (1)
        const grid = Array(ROWS).fill().map(() => Array(COLS).fill(1));

        const dirs = [
            [0, -2], [0, 2], [-2, 0], [2, 0]
        ];

        const isValid = (x, y) => x > 0 && x < COLS - 1 && y > 0 && y < ROWS - 1;

        const carve = (x, y) => {
            grid[y][x] = 0; // Mark as path

            // Shuffle directions
            const shuffledDirs = dirs.sort(() => Math.random() - 0.5);

            for (let [dx, dy] of shuffledDirs) {
                const nx = x + dx;
                const ny = y + dy;

                if (isValid(nx, ny) && grid[ny][nx] === 1) {
                    // Remove wall between
                    grid[y + dy / 2][x + dx / 2] = 0;
                    carve(nx, ny);
                }
            }
        };

        // Start carving from (1,1)
        carve(1, 1);

        // CREATE LOOPS: Randomly remove some internal walls to ensure multiple paths
        // This prevents traps from blocking the only solution
        const extraPaths = Math.floor((ROWS * COLS) * 0.1); // 10% of cells
        for (let i = 0; i < extraPaths; i++) {
            const rx = Math.floor(Math.random() * (COLS - 2)) + 1;
            const ry = Math.floor(Math.random() * (ROWS - 2)) + 1;
            if (grid[ry][rx] === 1) {
                // Ensure we don't remove outer walls
                if (rx > 0 && rx < COLS - 1 && ry > 0 && ry < ROWS - 1) {
                    // Only remove if it connects two open spaces
                    let neighbors = 0;
                    if (grid[ry - 1][rx] === 0) neighbors++;
                    if (grid[ry + 1][rx] === 0) neighbors++;
                    if (grid[ry][rx - 1] === 0) neighbors++;
                    if (grid[ry][rx + 1] === 0) neighbors++;

                    if (neighbors >= 2) {
                        grid[ry][rx] = 0;
                    }
                }
            }
        }

        // Set Start (2)
        grid[1][1] = 2;

        // Set Goal (3) - Find a distant empty spot
        // Simple heuristic: bottom-right-ish
        let goalX = -1, goalY = -1;
        let goalSet = false;
        for (let y = ROWS - 2; y > 0; y--) {
            for (let x = COLS - 2; x > 0; x--) {
                if (grid[y][x] === 0) {
                    grid[y][x] = 3;
                    goalX = x;
                    goalY = y;
                    goalSet = true;
                    break;
                }
            }
            if (goalSet) break;
        }

        // ENSURE SOLVABILITY: Find a valid path from Start (1,1) to Goal
        // We will prevent traps from landing on this "Golden Path"
        const findPath = () => {
            const resultPath = new Set();
            const visited = new Set();
            const queue = [[1, 1, []]]; // x, y, pathHistory

            visited.add('1,1');

            while (queue.length > 0) {
                const [cx, cy, path] = queue.shift();
                const currentPath = [...path, `${cx},${cy}`];

                if (cx === goalX && cy === goalY) {
                    currentPath.forEach(p => resultPath.add(p));
                    return resultPath;
                }

                const moves = [[0, 1], [0, -1], [1, 0], [-1, 0]];
                for (const [dx, dy] of moves) {
                    const nx = cx + dx;
                    const ny = cy + dy;
                    if (isValid(nx, ny) && grid[ny][nx] !== 1 && !visited.has(`${nx},${ny}`)) {
                        visited.add(`${nx},${ny}`);
                        queue.push([nx, ny, currentPath]);
                    }
                }
            }
            return new Set(); // Should not happen with recursive backtracker
        };

        const solutionPath = findPath();

        // Add Traps (4) based on Level
        const trapCount = 2 + level * 2;
        let placedTraps = 0;
        let attempts = 0;
        while (placedTraps < trapCount && attempts < 200) {
            const rx = Math.floor(Math.random() * (COLS - 2)) + 1;
            const ry = Math.floor(Math.random() * (ROWS - 2)) + 1;

            // Allow placing trap ONLY if it's not on the logical solution path
            // This guarantees at least one winnable route exists
            if (grid[ry][rx] === 0 && !solutionPath.has(`${rx},${ry}`)) {
                // Also ensure not directly adjacent to start to give a chance
                if (rx + ry > 4) {
                    grid[ry][rx] = 4;
                    placedTraps++;
                }
            }
            attempts++;
        }

        return grid;
    }, [level]);

    // Initialize Game
    useEffect(() => {
        startLevel();
    }, [level]); // Re-run when level changes

    const startLevel = () => {
        const newMaze = generateMaze();
        setMaze(newMaze);

        // Find start (should be 1,1 but let's be safe)
        // DFS guarantees connectivity so (1,1) is fine.
        setPlayerPos({ x: 1, y: 1 });
        setGameState('playing');
        setMessage('');
    };

    const nextLevel = () => {
        setLevel(prev => prev + 1);
    };

    const move = (dx, dy) => {
        if (gameState !== 'playing') return;
        if (maze.length === 0) return;

        const newX = playerPos.x + dx;
        const newY = playerPos.y + dy;

        // Check bounds
        if (newY < 0 || newY >= ROWS || newX < 0 || newX >= COLS) return;

        const cell = maze[newY][newX];

        // Wall Collision
        if (cell === 1) return;

        // Trap Collision
        if (cell === 4) {
            // Hit a trap! Reset to start
            setMessage('IT WAS A TRAP! DATTEBAYO!');
            setPlayerPos({ x: 1, y: 1 }); // Reset to start
            setTimeout(() => setMessage(''), 1500);
            return;
        }

        // Move
        setPlayerPos({ x: newX, y: newY });

        // Goal Collision
        if (cell === 3) {
            setGameState('won');
        }
    };

    // Keyboard
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
                e.preventDefault(); // Stop scroll
            }
            if (e.key === 'ArrowUp') move(0, -1);
            if (e.key === 'ArrowDown') move(0, 1);
            if (e.key === 'ArrowLeft') move(-1, 0);
            if (e.key === 'ArrowRight') move(1, 0);
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [playerPos, gameState, maze]);

    if (maze.length === 0) return <div className="text-white">Loading Maze...</div>;

    return (
        <div className="flex flex-col items-center justify-center w-full h-full relative p-4 bg-[#1a1a2e]">

            {/* Header */}
            <div className="flex items-center justify-between w-full max-w-md mb-4 px-4">
                <h2 className="text-2xl font-black text-orange-500 font-harry tracking-wider drop-shadow-lg flex items-center gap-2">
                    <span className="text-3xl">🍥</span> LVL {level}
                </h2>
                {message && (
                    <span className="text-red-500 font-bold animate-pulse text-sm">{message}</span>
                )}
            </div>

            {/* Maze Container */}
            <div className="relative bg-[#0f172a] p-2 rounded-lg border-4 border-orange-600 shadow-2xl">
                <div
                    className="grid gap-0.5" // Tighter gap
                    style={{
                        gridTemplateColumns: `repeat(${COLS}, minmax(0, 1fr))`
                    }}
                >
                    {maze.map((row, y) => (
                        row.map((cell, x) => {
                            const isPlayer = playerPos.x === x && playerPos.y === y;
                            let cellColor = 'bg-slate-800'; // Path (0)
                            if (cell === 1) cellColor = 'bg-slate-600'; // Wall (1)

                            // Trap visibility? Let's make them visible "Shurikens"

                            return (
                                <div
                                    key={`${x}-${y}`}
                                    className={`w-6 h-6 sm:w-8 sm:h-8 rounded-sm ${cellColor} flex items-center justify-center relative`}
                                >
                                    {/* Wall Texture */}
                                    {cell === 1 && <div className="absolute inset-0 bg-opacity-20 bg-[url('https://www.transparenttextures.com/patterns/dark-matter.png')]"></div>}

                                    {/* Trap: Shuriken */}
                                    {cell === 4 && <span className="text-sm z-10 animate-spin opacity-80">✴️</span>}

                                    {/* Goal: Ramen */}
                                    {cell === 3 && <span className="text-xl z-20 animate-bounce">🍜</span>}

                                    {/* Player: Naruto Icon */}
                                    {isPlayer && (
                                        <motion.div
                                            layoutId="naruto"
                                            className="absolute inset-0 flex items-center justify-center z-30"
                                            transition={{ type: "spring", stiffness: 500, damping: 30 }}
                                        >
                                            <span className="text-xl drop-shadow-[0_0_5px_rgba(249,115,22,0.8)]">🦊</span>
                                        </motion.div>
                                    )}
                                </div>
                            );
                        })
                    ))}
                </div>

                {/* Victory Overlay */}
                <AnimatePresence>
                    {gameState === 'won' && (
                        <motion.div
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0 }}
                            className="absolute inset-0 flex flex-col items-center justify-center bg-black/90 backdrop-blur-sm rounded-lg z-40 p-6 text-center"
                        >
                            <h3 className="text-3xl font-black text-orange-500 mb-2 font-harry tracking-widest">LEVEL {level} CLEAR!</h3>
                            <div className="text-6xl mb-4 animate-bounce">🍜</div>
                            <p className="text-slate-300 mb-6 text-sm">Delicious! Ready for harder training?</p>

                            <button
                                onClick={nextLevel}
                                className="px-8 py-3 bg-orange-600 text-white font-bold rounded-full hover:bg-orange-700 transition-colors flex items-center gap-2 shadow-[0_0_20px_rgba(234,88,12,0.5)] transform hover:scale-105 active:scale-95"
                            >
                                <ArrowUp size={20} /> Next Level
                            </button>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {/* Mobile Controls (D-Pad) */}
            <div className="mt-6 grid grid-cols-3 gap-2 md:hidden">
                <div></div>
                <button onClick={() => move(0, -1)} className="p-4 bg-slate-700 rounded-lg active:bg-orange-600 transition-colors pointer-events-auto shadow-md">
                    <ArrowUp className="text-white" />
                </button>
                <div></div>
                <button onClick={() => move(-1, 0)} className="p-4 bg-slate-700 rounded-lg active:bg-orange-600 transition-colors pointer-events-auto shadow-md">
                    <ArrowLeft className="text-white" />
                </button>
                <button onClick={() => move(0, 1)} className="p-4 bg-slate-700 rounded-lg active:bg-orange-600 transition-colors pointer-events-auto shadow-md">
                    <ArrowDown className="text-white" />
                </button>
                <button onClick={() => move(1, 0)} className="p-4 bg-slate-700 rounded-lg active:bg-orange-600 transition-colors pointer-events-auto shadow-md">
                    <ArrowRight className="text-white" />
                </button>
            </div>

            <p className="mt-4 text-slate-400 text-sm font-mono hidden md:block">
                Watch out for <span className="text-red-500 font-bold">Shurikens ✴️</span>!
            </p>
        </div>
    );
};

export default NarutoMazeGame;
