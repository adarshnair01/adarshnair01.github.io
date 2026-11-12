import { motion } from 'framer-motion';
import { useState } from 'react';
import FullScreenOverlay from './FullScreenOverlay';
import TableTennisGame from './games/TableTennisGame';
import PenaltyGame from './games/PenaltyGame';
import LankaMazeGame from './games/LankaMazeGame';
import TicTacToeGame from './games/TicTacToeGame';
import ArcheryGame from './games/ArcheryGame';
import RacingGame from './games/RacingGame';
import { KandaHeading } from './Ornaments';

const games = [
    { id: 'archery', icon: '🏹', title: 'Archery', desc: 'Draw the bow, find the mark', Component: ArcheryGame },
    { id: 'lanka', icon: '🐒', title: 'Lanka Maze', desc: 'Lead the vanara to Sita', Component: LankaMazeGame },
    { id: 'tictactoe', icon: '🪔', title: 'Tic Tac Toe', desc: 'A game of strategy', Component: TicTacToeGame },
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
                <KandaHeading
                    kanda="Kiṣkindhā Kāṇḍa · Games of the Vanaras"
                    title="Quick Games"
                    sanskrit="॥ किष्किन्धाकाण्ड ॥"
                    subtitle="Take a break and test your skills."
                />

                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 px-2">
                    {games.map((game) => (
                        <motion.button
                            key={game.id}
                            whileHover={{ y: -4 }}
                            whileTap={{ scale: 0.96 }}
                            onClick={() => setActiveGame(game.id)}
                            className="w-full bg-ink-card border border-gold/20 hover:border-saffron p-5 rounded-t-[60px] rounded-b-md transition-colors group shadow-lg flex flex-col items-center text-center"
                        >
                            <div className="w-14 h-14 bg-saffron/10 border border-gold/30 rounded-full flex items-center justify-center mb-3 text-2xl group-hover:scale-110 group-hover:bg-saffron/20 transition-transform">
                                {game.icon}
                            </div>
                            <h5 className="text-base text-palm mb-1 leading-tight">{game.title}</h5>
                            <p className="text-[11px] md:text-xs text-text-muted font-serif italic">{game.desc}</p>
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
