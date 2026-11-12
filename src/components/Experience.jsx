import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { GraduationCap, X } from 'lucide-react';
import { BowDivider } from './Ornaments';

// Rolling ocean waves (SVG band, animated sideways)
const Waves = ({ className = '', opacity = 0.35 }) => (
    <svg viewBox="0 0 1200 60" preserveAspectRatio="none" className={`w-[110%] h-10 animate-wave ${className}`} aria-hidden="true">
        <path
            d="M0 30 Q 50 10 100 30 T 200 30 T 300 30 T 400 30 T 500 30 T 600 30 T 700 30 T 800 30 T 900 30 T 1000 30 T 1100 30 T 1200 30 V60 H0 Z"
            fill={`rgba(160, 220, 230, ${opacity})`}
        />
    </svg>
);

// A floating stone of the bridge, inscribed with "राम"
const SetuStone = ({ className = '', label = 'राम' }) => (
    <div className={`flex items-center justify-center rounded-[45%_55%_50%_50%/55%_45%_55%_45%] bg-gradient-to-br from-[#8a8070] via-[#6b6255] to-[#4a4338] shadow-[inset_-6px_-8px_14px_rgba(0,0,0,0.45),0_8px_20px_rgba(0,0,0,0.4)] ${className}`}>
        <span className="font-sanskrit text-saffron drop-shadow-[0_0_6px_rgba(240,138,36,0.7)]" lang="sa">{label}</span>
    </div>
);

const experience = [
    {
        role: 'Senior Data Science Manager',
        company: 'Yum! Brands',
        period: '2022 - Present',
        desc: 'Manage a team of data scientists and analysts to build customer propensity and churn models, as well as a recommendation engine to enhance customer experience and revenue growth for a leading restaurant brand.',
    },
    {
        role: 'Data Science Manager',
        company: 'Yum! Brands',
        period: '2021 - 2022',
        desc: 'Led a team of data scientists and analysts to develop predictive models, segmentation strategies, and pricing optimization algorithms, resulting in a 10% increase in conversion rates through a propensity model and a 10% increase in marketing effectiveness through customer segmentation.',
    },
    {
        role: 'Principal Data Scientist',
        company: 'Kvantum Inc.',
        period: '2020 - 2021',
        desc: 'Managed end-to-end development of a sales forecasting system with cross-functional teams, improving forecast accuracy by 25%, while conducting data analysis and creating visualizations for technical and non-technical stakeholders.',
    },
    {
        role: 'Senior Data Scientist',
        company: 'Kvantum Inc.',
        period: '2019 - 2020',
        desc: 'Lead the end-to-end design, development, and deployment of time series predictive modeling solutions.',
    },
    {
        role: 'Data Scientist',
        company: 'Kvantum Inc.',
        period: '2017 - 2019',
        desc: 'Monitored technical aspects of the delivery for several projects and develop new NLP components.',
    },
    {
        role: 'Software Developer',
        company: 'MAQ Software',
        period: '2016 - 2017',
        desc: 'Design software architecture, Develop algorithms and Ensure quality of code.',
    }
];

const education = [
    {
        degree: "Master's Degree in Information Technology",
        school: "Indian Institute of Information Technology, Allahabad (2014 - 2016)",
    },
    {
        degree: "Bachelor's Degree in Information Technology",
        school: "Government Engineering College, Bhavnagar (2009 - 2013)",
    }
];

const Experience = () => {
    const [isSetuOpen, setIsSetuOpen] = useState(false);

    // Lock page scroll and allow ESC to close while the Setu is open
    useEffect(() => {
        if (!isSetuOpen) return;
        document.body.style.overflow = 'hidden';
        const onKey = (e) => e.key === 'Escape' && setIsSetuOpen(false);
        window.addEventListener('keydown', onKey);
        return () => {
            document.body.style.overflow = '';
            window.removeEventListener('keydown', onKey);
        };
    }, [isSetuOpen]);

    return (
        <section id="experience" className="relative overflow-hidden py-10">

            {/* INLINE TEASER — the ocean waiting to be crossed */}
            <motion.button
                type="button"
                onClick={() => setIsSetuOpen(true)}
                whileHover={{ scale: 1.005 }}
                className="relative w-full h-[340px] overflow-hidden bg-gradient-to-b from-[#0b2530] via-ocean to-[#0b2530] border-y-2 border-gold/40 flex flex-col items-center justify-center group shadow-[0_0_60px_rgba(0,0,0,0.6)]"
                aria-label="Open the Rama Setu: my professional journey"
            >
                <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(240,138,36,0.18),transparent_60%)]" />
                <div className="absolute bottom-0 left-0 w-full overflow-hidden"><Waves opacity={0.18} /></div>
                <div className="absolute bottom-3 left-0 w-full overflow-hidden"><Waves opacity={0.12} className="[animation-delay:-3s]" /></div>

                {/* Stones of the bridge bobbing on the water */}
                <div className="absolute bottom-10 inset-x-0 flex justify-center gap-4 md:gap-8">
                    {experience.map((_, i) => (
                        <motion.div
                            key={i}
                            animate={{ y: [0, -6, 0] }}
                            transition={{ duration: 3 + (i % 3) * 0.6, repeat: Infinity, delay: i * 0.3 }}
                            className={i > 3 ? 'hidden md:block' : ''}
                        >
                            <SetuStone className="relative w-12 h-9 md:w-16 md:h-11 text-sm md:text-base group-hover:brightness-110 transition" />
                        </motion.div>
                    ))}
                </div>

                <p className="kanda-label relative z-10 mb-3">Yuddha Kāṇḍa · The Crossing</p>
                <h2 className="relative z-10 text-4xl md:text-6xl text-transparent bg-clip-text bg-gradient-to-b from-[#fff1c8] via-gold to-saffron mb-3 tracking-wide drop-shadow-sm text-center">
                    The Rama Setu
                </h2>
                <p className="relative z-10 text-palm/80 font-serif italic text-lg md:text-xl mb-24 px-4 text-center">
                    Tap to cross the bridge of my journey
                </p>
            </motion.button>

            {/* FULL SCREEN SETU */}
            <AnimatePresence>
                {isSetuOpen && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.4 }}
                        className="fixed inset-0 z-[150] bg-gradient-to-b from-[#06161d] via-ocean to-[#06161d] overflow-y-auto"
                        role="dialog"
                        aria-modal="true"
                        aria-label="Rama Setu: professional journey"
                    >
                        {/* Sea shimmer */}
                        <div className="fixed inset-0 pointer-events-none bg-[radial-gradient(ellipse_at_top,rgba(240,138,36,0.18),transparent_55%)]" />
                        <div className="fixed bottom-0 left-0 w-full overflow-hidden pointer-events-none"><Waves opacity={0.15} /></div>

                        <button
                            onClick={() => setIsSetuOpen(false)}
                            className="fixed top-5 right-5 z-[160] p-3 rounded-full bg-ink/70 border border-gold/40 text-gold hover:bg-saffron hover:text-ink transition-colors"
                            aria-label="Close the Setu"
                        >
                            <X size={22} />
                        </button>

                        <div className="relative container mx-auto px-4 py-20 max-w-5xl">
                            <motion.div
                                initial={{ opacity: 0, y: -20 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="text-center mb-16"
                            >
                                <p className="font-sanskrit text-saffron text-xl mb-2" lang="sa">॥ सेतुबन्ध ॥</p>
                                <h2 className="text-5xl md:text-7xl text-transparent bg-clip-text bg-gradient-to-b from-[#fff1c8] via-gold to-saffron tracking-wide">
                                    The Rama Setu
                                </h2>
                                <p className="text-palm/80 font-serif italic text-lg md:text-xl mt-4 max-w-2xl mx-auto">
                                    Every stone that bore His name floated. Every role below is a stone I set upon the sea —
                                    together they form the bridge that carried me here.
                                </p>
                                <BowDivider className="mt-6" />
                            </motion.div>

                            {/* The bridge: a central causeway with stones alternating either side */}
                            <div className="relative">
                                <div className="absolute left-7 md:left-1/2 top-0 bottom-0 w-3 -translate-x-1/2 rounded-full bg-gradient-to-b from-[#7a7062] via-[#5c5448] to-[#3d372e] shadow-[0_0_20px_rgba(0,0,0,0.6)]" aria-hidden="true" />

                                <ol className="space-y-12 relative">
                                    {experience.map((item, index) => {
                                        const left = index % 2 === 0;
                                        return (
                                            <motion.li
                                                key={index}
                                                initial={{ opacity: 0, y: 60 }}
                                                whileInView={{ opacity: 1, y: 0 }}
                                                viewport={{ once: true, margin: '-60px' }}
                                                transition={{ duration: 0.6, type: 'spring', bounce: 0.35 }}
                                                className={`relative pl-20 md:pl-0 md:w-1/2 ${left ? 'md:pr-16 md:text-right' : 'md:ml-auto md:pl-16'}`}
                                            >
                                                {/* Stone on the causeway */}
                                                <SetuStone
                                                    className={`absolute z-10 top-4 w-14 h-10 text-base ${left ? 'left-0 md:left-auto md:-right-7' : 'left-0 md:-left-7'}`}
                                                />

                                                <div className="palm-leaf text-palm-ink p-6 md:p-7 rounded-sm shadow-[0_15px_40px_rgba(0,0,0,0.5)] border border-palm-deep relative">
                                                    <span className="absolute top-2 left-2 w-2 h-2 rounded-full bg-palm-ink/40" aria-hidden="true" />
                                                    <span className="absolute top-2 right-2 w-2 h-2 rounded-full bg-palm-ink/40" aria-hidden="true" />
                                                    <p className="text-kumkum text-xs font-semibold uppercase tracking-[0.25em] mb-1">{item.period}</p>
                                                    <h3 className="text-2xl text-palm-ink leading-tight mb-1">{item.role}</h3>
                                                    <p className="text-[#7a4a1a] font-semibold text-sm uppercase tracking-widest mb-3">{item.company}</p>
                                                    <p className="font-serif text-base md:text-lg leading-relaxed text-palm-ink/90">{item.desc}</p>
                                                </div>
                                            </motion.li>
                                        );
                                    })}
                                </ol>

                                {/* The shore where the bridge began: education */}
                                <div className="relative mt-20 pt-6 bg-gradient-to-b from-transparent to-[#06161d]">
                                    <h3 className="text-3xl text-gold mb-2 flex items-center justify-center gap-3 text-center">
                                        <GraduationCap className="text-saffron" size={30} /> The Gurukula
                                    </h3>
                                    <p className="text-center text-palm/70 font-serif italic mb-8">Where the first lessons of the bow were learned</p>
                                    <div className="grid md:grid-cols-2 gap-6">
                                        {education.map((item, index) => (
                                            <motion.div
                                                key={index}
                                                initial={{ opacity: 0, y: 20 }}
                                                whileInView={{ opacity: 1, y: 0 }}
                                                viewport={{ once: true }}
                                                transition={{ delay: index * 0.15 }}
                                                className="bg-ink-card p-6 rounded-sm border border-gold/30 shadow-lg"
                                            >
                                                <h4 className="text-xl text-palm mb-2">{item.degree}</h4>
                                                <p className="text-gold/80 text-sm font-semibold uppercase tracking-wider">{item.school}</p>
                                            </motion.div>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            <div className="mt-16 text-center">
                                <p className="font-sanskrit text-saffron text-2xl" lang="sa">॥ इति ॥</p>
                                <button
                                    onClick={() => setIsSetuOpen(false)}
                                    className="mt-4 px-6 py-2 border border-gold/50 text-gold font-serif italic hover:bg-gold/10 rounded-sm"
                                >
                                    Return to shore (Esc)
                                </button>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </section>
    );
};

export default Experience;
