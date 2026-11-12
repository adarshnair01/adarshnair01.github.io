import { motion } from 'framer-motion';
import { HollyDivider, NewYearCountdown, Bauble } from './Festive';

// A row of snowy pine trees along the horizon
const PineHorizon = () => {
    const trees = [
        { x: 40, h: 90 }, { x: 120, h: 130 }, { x: 190, h: 80 }, { x: 260, h: 115 },
        { x: 900, h: 120 }, { x: 980, h: 85 }, { x: 1050, h: 140 }, { x: 1140, h: 95 },
        { x: 520, h: 60 }, { x: 600, h: 75 }, { x: 680, h: 55 },
    ];
    return (
        <svg viewBox="0 0 1200 170" preserveAspectRatio="none" className="w-full h-28 md:h-44" aria-hidden="true">
            {/* Snow drifts */}
            <path d="M0 150 Q 150 120 300 140 T 600 135 T 900 140 T 1200 130 V170 H0 Z" fill="#dfe8f5" />
            <path d="M0 160 Q 200 140 400 155 T 800 150 T 1200 152 V170 H0 Z" fill="#f4f7fc" />
            {trees.map((t, i) => (
                <g key={i} transform={`translate(${t.x} ${150 - t.h})`}>
                    <path d={`M0 ${t.h} L ${t.h * 0.32} ${t.h} L 0 0 L ${-t.h * 0.32} ${t.h} Z`} fill="#0f5132" />
                    <path d={`M0 0 L ${t.h * 0.12} ${t.h * 0.22} L ${-t.h * 0.12} ${t.h * 0.22} Z`} fill="#f4f7fc" />
                    <path d={`M${-t.h * 0.2} ${t.h * 0.55} L ${t.h * 0.2} ${t.h * 0.55} L ${t.h * 0.1} ${t.h * 0.45} L 0 ${t.h * 0.5} L ${-t.h * 0.1} ${t.h * 0.45} Z`} fill="#f4f7fc" opacity="0.85" />
                    <rect x={-3} y={t.h} width={6} height={8} fill="#3b2a1e" />
                </g>
            ))}
            {/* Cosy cabin with a lit window */}
            <g transform="translate(380 92)">
                <path d="M0 30 L40 0 L80 30 Z" fill="#f4f7fc" />
                <rect x="6" y="28" width="68" height="34" fill="#5a3522" />
                <rect x="16" y="38" width="14" height="12" fill="#f0c34e" />
                <rect x="46" y="38" width="14" height="24" fill="#3b2a1e" />
                <rect x="58" y="6" width="8" height="14" fill="#5a3522" />
            </g>
        </svg>
    );
};

const Hero = () => {
    return (
        <section className="min-h-screen flex flex-col items-center justify-center pt-28 relative overflow-hidden">
            {/* Moon */}
            <div className="hidden md:block absolute top-24 right-[8%] w-24 h-24 md:w-32 md:h-32 rounded-full bg-[radial-gradient(circle_at_35%_35%,#fffbe8,#e9e3c8_60%,#c9c2a2)] shadow-[0_0_80px_20px_rgba(255,251,232,0.18)] -z-10" aria-hidden="true" />

            {/* Hanging baubles */}
            <div className="absolute top-14 left-[6%] hidden md:flex gap-6 -z-0" aria-hidden="true">
                {[['#d4243b', 80], ['#f0c34e', 130], ['#1f8a55', 60]].map(([c, len], i) => (
                    <div key={i} className="flex flex-col items-center origin-top animate-sway" style={{ animationDelay: `${i * 0.6}s` }}>
                        <span className="w-px bg-gold/50" style={{ height: len }} />
                        <Bauble color={c} className="w-10 h-12 -mt-1" />
                    </div>
                ))}
            </div>

            <div className="container mx-auto px-4 text-center z-10 flex-1 flex flex-col justify-center">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.9 }}
                >
                    <p className="font-script text-gold text-4xl md:text-5xl mb-2">Season's Greetings</p>
                    <p className="section-eyebrow mb-6">Merry Christmas &amp; Happy New Year</p>

                    <h1 className="text-6xl md:text-8xl font-display font-extrabold text-snow mb-4 drop-shadow-[0_4px_24px_rgba(169,212,245,0.25)] leading-tight">
                        Adarsh Nair
                    </h1>

                    <HollyDivider className="mb-6" />

                    <h3 className="text-2xl md:text-4xl text-snow/85 mb-8 font-display">
                        Data Science Manager <span className="text-gold">&amp;</span>{' '}
                        <br className="md:hidden" />
                        <span className="text-holly italic font-bold">ML Engineer</span>
                    </h3>

                    <p className="max-w-2xl mx-auto text-text-muted text-lg mb-10 leading-relaxed">
                        I lead teams that turn data into gifts worth unwrapping — predictive models and
                        recommendation engines that make every customer feel like it's Christmas morning.
                    </p>

                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.5, duration: 0.5 }}
                        className="flex flex-col sm:flex-row justify-center gap-4 sm:gap-6 mb-12"
                    >
                        <a
                            href="#projects"
                            className="px-8 py-3 bg-holly hover:bg-holly-deep text-snow rounded-full font-bold transition-all hover:shadow-[0_0_25px_rgba(212,36,59,0.55)] shadow-lg uppercase tracking-[0.15em] text-sm"
                        >
                            Unwrap My Work
                        </a>
                        <a
                            href="#contact"
                            className="px-8 py-3 border-2 border-pine text-snow hover:bg-pine/20 rounded-full font-bold transition-all uppercase tracking-[0.15em] text-sm"
                        >
                            Send a Greeting
                        </a>
                    </motion.div>

                    <NewYearCountdown />
                </motion.div>
            </div>

            <div className="relative w-full mt-10">
                <PineHorizon />
            </div>
        </section>
    );
};

export default Hero;
