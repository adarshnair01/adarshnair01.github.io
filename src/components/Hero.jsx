import { motion } from 'framer-motion';
import { SunDisc, BowDivider, Diya } from './Ornaments';

// Silhouette of Ayodhya's temple spires along the horizon
const AyodhyaSkyline = () => (
    <svg viewBox="0 0 1200 160" preserveAspectRatio="none" className="w-full h-28 md:h-40" aria-hidden="true">
        <defs>
            <linearGradient id="skyline" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3a210f" />
                <stop offset="100%" stopColor="#140c08" />
            </linearGradient>
        </defs>
        <path
            fill="url(#skyline)"
            d="M0 160 V120 H60 V100 H80 L95 70 L110 100 H130 V115 H190 V95 L215 40 L240 95 V115 H300 V105 H330 L345 80 L360 105 H400 V90
               L430 60 L445 20 L460 60 L490 90 V110 H560 V100 L585 55 L600 10 L615 55 L640 100 V110 H700 V90 L730 60 L745 25 L760 60 L790 90
               V110 H850 V100 L870 75 L890 100 H920 V115 H980 V95 L1005 45 L1030 95 V115 H1080 V100 L1095 75 L1110 100 H1140 V120 H1200 V160 Z"
        />
        {/* Flags on the tallest spires */}
        <path d="M600 10 V-6 L620 0 L600 6" fill="#f08a24" />
        <path d="M445 20 V6 L462 11 L445 16" fill="#f08a24" />
        <path d="M745 25 V11 L762 16 L745 21" fill="#f08a24" />
    </svg>
);

const Hero = () => {
    return (
        <section className="min-h-screen flex flex-col items-center justify-center pt-24 relative overflow-hidden">
            {/* Rising sun behind the name (outer div centers, inner rotates) */}
            <div className="absolute top-[12%] left-1/2 -translate-x-1/2 -z-10 w-[520px] h-[520px] md:w-[680px] md:h-[680px]">
                <motion.div
                    className="w-full h-full"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 0.25, scale: 1, rotate: 360 }}
                    transition={{ opacity: { duration: 2 }, scale: { duration: 2 }, rotate: { duration: 160, repeat: Infinity, ease: 'linear' } }}
                >
                    <SunDisc className="w-full h-full" rays={32} />
                </motion.div>
            </div>
            <div className="absolute top-[20%] left-1/2 -translate-x-1/2 -z-10 w-[420px] h-[420px] rounded-full bg-saffron/20 blur-[120px]" />

            <div className="container mx-auto px-4 text-center z-10 flex-1 flex flex-col justify-center">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.9 }}
                >
                    <p className="font-sanskrit text-saffron/90 text-lg md:text-xl mb-4" lang="sa">॥ अथ प्रथमः सर्गः ॥</p>
                    <p className="kanda-label mb-6">Here begins the first chapter</p>

                    <h1 className="text-6xl md:text-8xl text-transparent bg-clip-text bg-gradient-to-b from-[#fff1c8] via-gold to-saffron mb-4 drop-shadow-[0_4px_20px_rgba(240,138,36,0.35)] leading-tight">
                        Adarsh Nair
                    </h1>

                    <BowDivider className="mb-6" />

                    <h3 className="text-2xl md:text-4xl text-palm/80 mb-8 font-serif">
                        Data Science Manager <span className="text-gold">&</span>{' '}
                        <br className="md:hidden" />
                        <span className="text-saffron font-semibold italic">ML Engineer</span>
                    </h3>

                    <p className="max-w-2xl mx-auto text-text-muted text-lg mb-10 leading-relaxed">
                        Like an archer who studies the wind before he draws, I lead teams that read the patterns in data —
                        building predictive models and recommendation engines that hit their mark.
                    </p>

                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.5, duration: 0.5 }}
                        className="flex flex-col sm:flex-row justify-center gap-4 sm:gap-6"
                    >
                        <a
                            href="#projects"
                            className="px-8 py-3 bg-gradient-to-r from-saffron to-kumkum text-palm rounded-sm font-semibold transition-all hover:shadow-[0_0_25px_rgba(240,138,36,0.5)] shadow-lg uppercase tracking-[0.2em] text-sm border border-gold/40"
                        >
                            Behold the Works
                        </a>
                        <a
                            href="#contact"
                            className="px-8 py-3 border border-gold/50 text-gold hover:bg-gold/10 hover:border-gold rounded-sm font-semibold transition-all uppercase tracking-[0.2em] text-sm"
                        >
                            Send a Message
                        </a>
                    </motion.div>
                </motion.div>
            </div>

            {/* Horizon with diyas */}
            <div className="relative w-full mt-10">
                <div className="absolute inset-x-0 -top-6 flex justify-around px-[10%] z-10">
                    {[0, 1, 2, 3, 4].map((i) => (
                        <Diya key={i} className={i % 2 ? 'hidden md:inline-flex' : ''} />
                    ))}
                </div>
                <AyodhyaSkyline />
            </div>

            {/* Scroll indicator — a falling arrow */}
            <motion.a
                href="#about"
                aria-label="Scroll to about"
                className="absolute bottom-6 left-1/2 -translate-x-1/2 text-gold/80 z-20"
                animate={{ y: [0, 10, 0] }}
                transition={{ duration: 1.6, repeat: Infinity }}
            >
                <svg width="16" height="44" viewBox="0 0 16 44" fill="none">
                    <path d="M8 0 V36" stroke="currentColor" strokeWidth="1.5" />
                    <path d="M2 30 L8 42 L14 30 Z" fill="#f08a24" />
                    <path d="M4 2 L8 7 L12 2 M4 7 L8 12 L12 7" stroke="#b3261e" strokeWidth="1.2" />
                </svg>
            </motion.a>
        </section>
    );
};

export default Hero;
