import { motion } from 'framer-motion';
import { useState } from 'react';
import FullScreenOverlay from './FullScreenOverlay';
import TableTennisGame from './games/TableTennisGame';
import PenaltyGame from './games/PenaltyGame';
import NarutoMazeGame from './games/NarutoMazeGame';
import TicTacToeGame from './games/TicTacToeGame';
import ArcheryGame from './games/ArcheryGame';
import RacingGame from './games/RacingGame';

const QuickGames = () => {
    const [activeGame, setActiveGame] = useState(null);

    const openGame = (game) => setActiveGame(game);

    return (
        <section id="games" className="py-10 bg-transparent">
            <div className="container mx-auto px-4">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="text-center mb-12"
                >
                    <h2 className="text-3xl md:text-4xl font-bold text-white mb-4 font-harry tracking-wider">Quick Games</h2>
                    <p className="mt-4 text-gray-400 font-medium">Take a break and test your skills.</p>
                </motion.div>

                {/* Compact Grid Container */}
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 px-2">
                    {[
                        { id: 'tennis', icon: '🏓', title: 'Table Tennis', desc: 'Pong Arcade', hover: 'hover:border-orange-500', bg: 'bg-orange-900/30', text: 'text-orange-500' },
                        { id: 'penalty', icon: '⚽', title: 'Penalty Kick', desc: 'Test Your Aim', hover: 'hover:border-green-500', bg: 'bg-green-900/30', text: 'text-green-500' },
                        { id: 'naruto', icon: '🍥', title: 'Naruto Maze', desc: 'Find Ramen', hover: 'hover:border-orange-500', bg: 'bg-orange-900/30', text: 'text-orange-500' },
                        { id: 'tictactoe', icon: '❌', title: 'Tic Tac Toe', desc: 'Strategy', hover: 'hover:border-blue-500', bg: 'bg-blue-900/30', text: 'text-blue-500' },
                        { id: 'archery', icon: '🏹', title: 'Archery', desc: 'Bullseye', hover: 'hover:border-yellow-500', bg: 'bg-yellow-900/30', text: 'text-yellow-500' },
                        { id: 'racing', icon: '🏎️', title: 'Turbo Racer', desc: 'Dodge Traffic', hover: 'hover:border-red-500', bg: 'bg-red-900/30', text: 'text-red-500' }
                    ].map((game) => (
                        <motion.button
                            key={game.id}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => openGame(game.id)}
                            className={`w-full bg-[#1e1e1e] border border-black p-4 rounded-xl text-left ${game.hover} transition-colors group shadow-lg flex flex-col items-center text-center`}
                        >
                            <div className={`w-12 h-12 ${game.bg} ${game.text} rounded-xl flex items-center justify-center mb-3 text-2xl group-hover:scale-110 transition-transform`}>
                                {game.icon}
                            </div>
                            <h5 className="font-bold text-sm md:text-base text-gray-200 mb-1 leading-tight">{game.title}</h5>
                            <p className="text-[10px] md:text-xs text-gray-500">{game.desc}</p>
                        </motion.button>
                    ))}
                </div>
            </div>

            {/* Game Fullscreen Overlays */}
            <FullScreenOverlay
                isOpen={activeGame === 'tennis'}
                onClose={() => setActiveGame(null)}
                title="🏓 Table Tennis"
            >
                <TableTennisGame />
            </FullScreenOverlay>

            <FullScreenOverlay
                isOpen={activeGame === 'penalty'}
                onClose={() => setActiveGame(null)}
                title="⚽ Penalty Shootout"
            >
                <PenaltyGame />
            </FullScreenOverlay>

            <FullScreenOverlay
                isOpen={activeGame === 'naruto'}
                onClose={() => setActiveGame(null)}
                title="🍥 Naruto Maze"
            >
                <NarutoMazeGame />
            </FullScreenOverlay>

            <FullScreenOverlay
                isOpen={activeGame === 'tictactoe'}
                onClose={() => setActiveGame(null)}
                title="⭕ Tic Tac Toe"
            >
                <TicTacToeGame />
            </FullScreenOverlay>

            <FullScreenOverlay
                isOpen={activeGame === 'archery'}
                onClose={() => setActiveGame(null)}
                title="🏹 Archery Master"
            >
                <ArcheryGame />
            </FullScreenOverlay>

            <FullScreenOverlay
                isOpen={activeGame === 'racing'}
                onClose={() => setActiveGame(null)}
                title="🏎️ Turbo Racer"
            >
                <RacingGame />
            </FullScreenOverlay>
        </section>
    );
};

export default QuickGames;
