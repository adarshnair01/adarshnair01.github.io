import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RotateCcw } from 'lucide-react';

const TicTacToeGame = () => {
    const [board, setBoard] = useState(Array(9).fill(null));
    const [isXNext, setIsXNext] = useState(true); // Player is X
    const [winner, setWinner] = useState(null);
    const [winningLine, setWinningLine] = useState(null);
    const [isDraw, setIsDraw] = useState(false);

    // Check for win
    const calculateWinner = (squares) => {
        const lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], // Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8], // Cols
            [0, 4, 8], [2, 4, 6]             // Diagonals
        ];
        for (let i = 0; i < lines.length; i++) {
            const [a, b, c] = lines[i];
            if (squares[a] && squares[a] === squares[b] && squares[a] === squares[c]) {
                return { winner: squares[a], line: lines[i] };
            }
        }
        return null;
    };

    // AI Move (Basic Logic: Win -> Block -> Center -> Random)
    useEffect(() => {
        if (!isXNext && !winner && !isDraw) {
            const timer = setTimeout(() => {
                makeCpuMove();
            }, 600); // Slight delay for realism
            return () => clearTimeout(timer);
        }
    }, [isXNext, winner, isDraw]);

    const makeCpuMove = () => {
        // 1. Try to Win
        for (let i = 0; i < 9; i++) {
            if (!board[i]) {
                const temp = [...board];
                temp[i] = 'O';
                if (calculateWinner(temp)) {
                    handleClick(i);
                    return;
                }
            }
        }

        // 2. Block Player
        for (let i = 0; i < 9; i++) {
            if (!board[i]) {
                const temp = [...board];
                temp[i] = 'X';
                if (calculateWinner(temp)) {
                    handleClick(i);
                    return;
                }
            }
        }

        // 3. Take Center
        if (!board[4]) {
            handleClick(4);
            return;
        }

        // 4. Random available
        const available = board.map((val, idx) => val === null ? idx : null).filter(val => val !== null);
        if (available.length > 0) {
            const random = available[Math.floor(Math.random() * available.length)];
            handleClick(random);
        }
    };

    const handleClick = (index) => {
        if (board[index] || winner || isDraw) return;

        const newBoard = [...board];
        newBoard[index] = isXNext ? 'X' : 'O';
        setBoard(newBoard);

        const winInfo = calculateWinner(newBoard);
        if (winInfo) {
            setWinner(winInfo.winner);
            setWinningLine(winInfo.line);
        } else if (!newBoard.includes(null)) {
            setIsDraw(true);
        } else {
            setIsXNext(!isXNext);
        }
    };

    const resetGame = () => {
        setBoard(Array(9).fill(null));
        setIsXNext(true);
        setWinner(null);
        setWinningLine(null);
        setIsDraw(false);
    };

    return (
        <div className="flex flex-col items-center justify-center w-full h-full p-4 bg-[#1a1a2e]">

            <div className="flex flex-col items-center mb-8">
                <h2 className="text-3xl font-black text-white font-harry tracking-wider drop-shadow-lg mb-2">
                    <span className="text-blue-500">X</span> vs <span className="text-red-500">O</span>
                </h2>
                <div className="text-slate-400 font-mono text-sm">
                    {winner ? (
                        <span className={winner === 'X' ? 'text-blue-500 font-bold' : 'text-red-500 font-bold'}>
                            {winner === 'X' ? 'YOU WON!' : 'CPU WON!'}
                        </span>
                    ) : isDraw ? (
                        <span className="text-purple-400 font-bold">DRAW!</span>
                    ) : (
                        <span>{isXNext ? 'YOUR TURN' : 'CPU THINKING...'}</span>
                    )}
                </div>
            </div>

            {/* Board */}
            <div className="relative bg-[#0f172a] p-4 rounded-2xl shadow-2xl border-4 border-slate-700">
                <div className="grid grid-cols-3 gap-3">
                    {board.map((cell, index) => (
                        <motion.button
                            key={index}
                            whileHover={{ scale: !cell && !winner ? 1.05 : 1 }}
                            whileTap={{ scale: !cell && !winner ? 0.95 : 1 }}
                            onClick={() => isXNext && handleClick(index)}
                            disabled={!!cell || !!winner || !isXNext}
                            className={`
                                w-20 h-20 sm:w-24 sm:h-24 rounded-xl flex items-center justify-center text-5xl font-black shadow-inner
                                ${cell === 'X' ? 'bg-blue-500/10 text-blue-500' :
                                    cell === 'O' ? 'bg-red-500/10 text-red-500' :
                                        'bg-slate-800 hover:bg-slate-700'}
                                ${winningLine?.includes(index) ? (cell === 'X' ? 'bg-blue-500 text-white ring-4 ring-blue-400' : 'bg-red-500 text-white ring-4 ring-red-400') : ''}
                                transition-colors duration-300
                            `}
                        >
                            {cell && (
                                <motion.span
                                    initial={{ scale: 0, rotate: -45 }}
                                    animate={{ scale: 1, rotate: 0 }}
                                    transition={{ type: "spring", stiffness: 300, damping: 20 }}
                                >
                                    {cell}
                                </motion.span>
                            )}
                        </motion.button>
                    ))}
                </div>
            </div>

            {/* Reset Button */}
            <AnimatePresence>
                {(winner || isDraw) && (
                    <motion.button
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 20 }}
                        onClick={resetGame}
                        className="mt-8 px-8 py-3 bg-white text-slate-900 font-bold rounded-full hover:bg-slate-200 transition-colors flex items-center gap-2 shadow-lg"
                    >
                        <RotateCcw size={20} /> Play Again
                    </motion.button>
                )}
            </AnimatePresence>
        </div>
    );
};

export default TicTacToeGame;
