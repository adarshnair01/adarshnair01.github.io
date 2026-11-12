import { motion } from 'framer-motion';
import { useState } from 'react';
import FullScreenOverlay from './FullScreenOverlay';
import TableTennisGame from './games/TableTennisGame';
import PenaltyGame from './games/PenaltyGame';
import SantaMazeGame from './games/SantaMazeGame';
import TicTacToeGame from './games/TicTacToeGame';
import ArcheryGame from './games/ArcheryGame';
import RacingGame from './games/RacingGame';
import { FestiveHeading } from './Festive';

const games = [
    { id: 'santa', icon: '🎅', title: 'Santa Maze', desc: 'Deliver the gift', Component: SantaMazeGame },
    { id: 'archery', icon: '🏹', title: 'Archery', desc: 'Hit the star', Component: ArcheryGame },
    { id: 'tictactoe', icon: '⛄', title: 'Tic Tac Toe', desc: 'A game of strategy', Component: TicTacToeGame },
    { id: 'tennis', icon: '🏓', title: 'Table Tennis', desc: 'Pong arcade', Component: TableTennisGame },
    { id: 'penalty', icon: '⚽', title: 'Penalty Kick', desc: 'Test your aim', Component: PenaltyGame },
    { id: 'racing', icon: '🏎️', title: 'Turbo Racer', desc: 'Dodge the traffic', Component: RacingGame },
];

const QuickGames = () => {
    const [activeGame, setActiveGame] = useState(null);
    const current = games.find((g) => g.id === activeGame);

    return (
        <section id="games" className="py-20 bg-transparent">
            <div className="container mx-auto px-4">
                <FestiveHeading
                    eyebrow="Holiday Break"
                    title="Quick Games"
                    script="fun by the fireplace"
                    subtitle="Take a break and test your skills."
                />

                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 px-2">
                    {games.map((game) => (
                        <motion.button
                            key={game.id}
                            whileHover={{ y: -4 }}
                            whileTap={{ scale: 0.96 }}
                            onClick={() => setActiveGame(game.id)}
                            className="snow-cap w-full bg-night-card border border-frost/15 hover:border-gold p-5 rounded-2xl transition-colors group shadow-lg flex flex-col items-center text-center"
                        >
                            <div className="w-14 h-14 bg-holly/15 border border-holly/30 rounded-full flex items-center justify-center mb-3 text-2xl group-hover:scale-110 group-hover:bg-holly/25 transition-transform">
                                {game.icon}
                            </div>
                            <h5 className="text-base font-bold text-snow mb-1 leading-tight">{game.title}</h5>
                            <p className="text-[11px] md:text-xs text-text-muted">{game.desc}</p>
                        </motion.button>
                    ))}
                </div>
            </div>

            <FullScreenOverlay
                isOpen={!!current}
                onClose={() => setActiveGame(null)}
                title={current ? `${current.icon} ${current.title}` : ''}
            >
                {current && <current.Component />}
            </FullScreenOverlay>
        </section>
    );
};

export default QuickGames;
