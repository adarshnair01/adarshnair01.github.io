import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { GraduationCap, Flame } from 'lucide-react';

const Experience = () => {
    const [isMapOpen, setIsMapOpen] = useState(false);
    const [showHint, setShowHint] = useState(false);
    const canvasRef = useRef(null);
    const wandRef = useRef(null);
    const lastInteractionRef = useRef(0);

    // Initialize Canvas "Fog of War"
    useEffect(() => {
        if (isMapOpen) {
            lastInteractionRef.current = Date.now();
            setShowHint(false);

            if (canvasRef.current) {
                const canvas = canvasRef.current;
                const ctx = canvas.getContext('2d');
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;

                // Fill with parchment color to hide content initially
                ctx.fillStyle = '#d4c4a8';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
            }

            // Idle Check Interval
            const idleInterval = setInterval(() => {
                if (Date.now() - lastInteractionRef.current > 3000) {
                    setShowHint(true);
                }
            }, 1000);

            return () => clearInterval(idleInterval);
        }
    }, [isMapOpen]);

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

    return (
        <section id="experience" className="bg-[#121212] relative overflow-hidden">

            {/* INLINE TEASER (When Closed) */}
            <div className="pb-24 pt-0 relative w-full">
                <motion.div
                    className="relative w-full h-[300px] bg-[#d4c4a8] overflow-hidden cursor-pointer shadow-[0_0_50px_rgba(0,0,0,0.5)] border-y-4 border-[#5d4037] flex flex-col items-center justify-center group"
                    onDoubleClick={() => setIsMapOpen(true)}
                    whileHover={{ scale: 1.01 }}
                >
                    {/* Paper Texture */}
                    <div className="absolute inset-0 opacity-40 bg-[url('https://www.transparenttextures.com/patterns/aged-paper.png')] pointer-events-none mix-blend-multiply"></div>

                    <h2 className="text-4xl md:text-6xl font-bold text-[#3e2723] mb-4 font-harry tracking-widest uppercase drop-shadow-sm text-center relative z-10">
                        The Marauder's Map
                    </h2>
                    <p className="text-[#5d4037] font-harry tracking-widest text-lg animate-pulse relative z-10">
                        Double Tap to Reveal My Journey
                    </p>

                    {/* Footprints Decoration */}
                    <div className="absolute bottom-10 flex gap-8 opacity-40">
                        <div className="w-4 h-8 bg-[#3e2723] rounded-full rotate-12"></div>
                        <div className="w-4 h-8 bg-[#3e2723] rounded-full -rotate-12 translate-y-4"></div>
                        <div className="w-4 h-8 bg-[#3e2723] rounded-full rotate-12"></div>
                    </div>
                </motion.div>

                {/* Education Section (Now merged into Map - Hidden externally) */}
            </div>

            {/* FULL SCREEN OVERLAY (When Open) */}
            <AnimatePresence>
                {isMapOpen && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        transition={{ duration: 0.5 }}
                        className="fixed inset-0 z-50 bg-[#d4c4a8] overflow-hidden cursor-none" // Use cursor-none to rely on our custom wand
                        onDoubleClick={() => setIsMapOpen(false)}
                        onMouseMove={(e) => {
                            const container = e.currentTarget;
                            const rect = container.getBoundingClientRect();
                            const x = e.clientX - rect.left;
                            const y = e.clientY - rect.top;

                            // Visual Wand Tracking (Fix space typo)
                            container.style.setProperty('--x', `${x}px`);
                            container.style.setProperty('--y', `${y}px`);

                            // Move Physical Wand directly for performance
                            if (wandRef.current) {
                                wandRef.current.style.left = `${e.clientX}px`;
                                wandRef.current.style.top = `${e.clientY}px`;
                                wandRef.current.style.opacity = '1'; // Reveal on first move
                            }

                            // Reset Idle Timer
                            lastInteractionRef.current = Date.now();
                            if (showHint) setShowHint(false);

                            // Persistent Reveal Logic on Canvas
                            if (canvasRef.current) {
                                const ctx = canvasRef.current.getContext('2d');
                                ctx.globalCompositeOperation = 'destination-out';
                                ctx.beginPath();
                                // Create soft brush
                                const gradient = ctx.createRadialGradient(x, y, 50, x, y, 200);
                                gradient.addColorStop(0, 'rgba(0,0,0,1)');
                                gradient.addColorStop(1, 'rgba(0,0,0,0)');
                                ctx.fillStyle = gradient;
                                ctx.fillRect(x - 200, y - 200, 400, 400);
                            }
                        }}
                    >
                        {/* 1. Base Layer: Blank Parchment (Always Visible) */}
                        <div className="absolute inset-0 opacity-100 bg-[url('https://www.transparenttextures.com/patterns/aged-paper.png')] pointer-events-none z-0 mix-blend-multiply fixed"></div>

                        {/* Hint for idle users */}
                        <AnimatePresence>
                            {showHint && (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: 10 }}
                                    className="fixed bottom-32 left-1/2 -translate-x-1/2 z-[60] pointer-events-none"
                                >
                                    <div className="bg-[#3e2723] text-[#e6dcc3] px-6 py-3 rounded-full font-harry text-xl shadow-[0_0_20px_rgba(255,165,0,0.3)] border border-[#8d6e63] flex items-center gap-2">
                                        <Flame size={20} className="animate-pulse text-orange-400" />
                                        <span>"Lumos"... Use your wand to reveal the path</span>
                                        <Flame size={20} className="animate-pulse text-orange-400" />
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {/* 2. Persistent Cover Layer (Canvas) */}
                        <canvas
                            ref={canvasRef}
                            className="absolute inset-0 z-20 pointer-events-none"
                        />

                        {/* 4. The Wand Light Follower (Visual only, to help user see where they are) */}
                        <div
                            className="fixed w-[300px] h-[300px] bg-[#ffaa00] rounded-full blur-[100px] opacity-20 pointer-events-none mix-blend-color-dodge z-40"
                            style={{
                                left: 'var(--x, 50%)',
                                top: 'var(--y, 50%)',
                                transform: 'translate(-50%, -50%)'
                            }}
                        ></div>

                        {/* 5. The Physical Wand Cursor */}
                        <div
                            ref={wandRef}
                            className="fixed pointer-events-none z-[100] opacity-0 transition-opacity duration-300" // Start hidden
                            style={{
                                left: '0px',
                                top: '0px',
                                transform: 'translate(-5%, -5%) rotate(-15deg)', // Align tip to cursor
                            }}
                        >
                            {/* Wand SVG */}
                            <svg width="200" height="200" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" className="drop-shadow-2xl filter drop-shadow-[0_10px_10px_rgba(0,0,0,0.5)]">
                                {/* Wand Shaft */}
                                <path d="M5 5 L 80 80" stroke="#3e2723" strokeWidth="6" strokeLinecap="round" />
                                <path d="M5 5 L 80 80" stroke="#5d4037" strokeWidth="2" strokeLinecap="round" className="opacity-50" />

                                {/* Wand Handle Detail */}
                                <path d="M60 60 L 85 85" stroke="#2e1a16" strokeWidth="8" strokeLinecap="round" />
                                {/* Knobs/Details on handle */}
                                <circle cx="65" cy="65" r="3" fill="#1a0f0d" />
                                <circle cx="75" cy="75" r="3" fill="#1a0f0d" />
                            </svg>
                        </div>

                        {/* 3. The "Ink" Layer - Underlying content */}
                        <div
                            className="relative z-10 w-full h-full overflow-y-auto"
                        >
                            <div className="container mx-auto px-4 py-20 min-h-screen flex flex-col items-center">

                                {/* Close Hint (Always visible outside mask? No, let's keep it inside for effect) */}
                                <div className="absolute top-8 right-8 text-[#5d4037] font-harry text-sm animate-pulse cursor-pointer" onClick={() => setIsMapOpen(false)}>
                                    X (Close Map)
                                </div>

                                <motion.div
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    className="text-center mb-32"
                                >
                                    <h2 className="text-5xl md:text-7xl font-bold text-[#3e2723] mb-4 font-harry tracking-widest uppercase drop-shadow-sm">
                                        The Marauder's Map
                                    </h2>
                                    <p className="text-[#5d4037] font-serif italic text-xl">
                                        "I solemnly swear that I am up to no good"
                                    </p>
                                </motion.div>

                                {/* MAP GRID (EXPERIENCE) */}
                                <div className="flex flex-wrap justify-center gap-10 w-full max-w-6xl mb-20">
                                    {experience.map((item, index) => (
                                        <motion.div
                                            key={index}
                                            initial={{ opacity: 0, y: 50 }}
                                            whileInView={{ opacity: 1, y: 0 }}
                                            viewport={{ once: true }}
                                            transition={{ delay: index * 0.1 }}
                                            className="w-full md:w-[45%] lg:w-[30%] group"
                                        >
                                            <div className="bg-[#e6dcc3] p-8 rounded-sm shadow-[0_10px_30px_rgba(62,39,35,0.15)] border-2 border-[#8d6e63] relative overflow-hidden transition-all duration-300 group-hover:-translate-y-2 group-hover:shadow-[0_20px_40px_rgba(62,39,35,0.25)] h-full flex flex-col">

                                                {/* Decorative Corners */}
                                                <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-[#3e2723]"></div>
                                                <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-[#3e2723]"></div>
                                                <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-[#3e2723]"></div>
                                                <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-[#3e2723]"></div>

                                                <div className="text-center mb-6">
                                                    <h4 className="text-2xl font-bold text-[#3e2723] font-harry mb-2 leading-tight">{item.role}</h4>
                                                    <div className="w-16 h-0.5 bg-[#3e2723]/20 mx-auto my-2"></div>
                                                    <p className="text-[#5d4037] font-bold text-sm uppercase tracking-widest">{item.company}</p>
                                                    <p className="text-[#8d6e63] text-xs font-serif italic mt-1">{item.period}</p>
                                                </div>

                                                <p className="text-[#4e342e] text-sm font-serif leading-relaxed italic text-center">
                                                    {item.desc}
                                                </p>
                                            </div>
                                        </motion.div>
                                    ))}
                                </div>

                                {/* EDUCATION WITHIN MAP */}
                                <div className="text-center mb-16 w-full max-w-4xl">
                                    <h3 className="text-3xl font-bold text-[#3e2723] mb-8 flex items-center justify-center gap-3 font-harry tracking-wide opacity-80 decoration-double underline decoration-[#8d6e63]">
                                        <GraduationCap className="text-[#3e2723]" size={32} /> Academic Records
                                    </h3>
                                    <div className="flex flex-wrap justify-center gap-6">
                                        {education.map((item, index) => (
                                            <motion.div
                                                key={index}
                                                initial={{ opacity: 0, y: 10 }}
                                                whileInView={{ opacity: 1, y: 0 }}
                                                viewport={{ once: true }}
                                                className="w-full md:w-[45%] bg-[#ded1b6] p-6 rounded-sm border-2 border-[#8d6e63] shadow-md relative group"
                                            >
                                                {/* Corner Accents for Education */}
                                                <div className="absolute top-2 left-2 w-2 h-2 rounded-full bg-[#3e2723] opacity-20 group-hover:opacity-60 transition-opacity"></div>
                                                <div className="absolute bottom-2 right-2 w-2 h-2 rounded-full bg-[#3e2723] opacity-20 group-hover:opacity-60 transition-opacity"></div>

                                                <h4 className="text-xl font-bold text-[#3e2723] mb-2 font-harry">{item.degree}</h4>
                                                <p className="text-[#5d4037] text-sm font-bold uppercase tracking-wider">{item.school}</p>
                                            </motion.div>
                                        ))}
                                    </div>
                                </div>

                                <motion.div
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    transition={{ delay: 1 }}
                                    className="mt-12 mb-8 text-[#5d4037] font-harry text-xl md:text-2xl animate-pulse flex flex-col items-center gap-2 opacity-70"
                                >
                                    <span>Mischief Managed</span>
                                    <span className="text-sm font-serif italic opacity-60">(Double tap wand to close)</span>
                                </motion.div>
                            </div>
                        </div>

                        {/* 4. The Wand Light Follower (Visual only, to help user see where they are) */}
                        <div
                            className="fixed w-[300px] h-[300px] bg-[#ffaa00] rounded-full blur-[100px] opacity-10 pointer-events-none mix-blend-color-dodge z-20"
                            style={{
                                left: 'var(--x, 50%)',
                                top: 'var(--y, 50%)',
                                transform: 'translate(-50%, -50%)'
                            }}
                        ></div>

                    </motion.div>
                )}
            </AnimatePresence>
        </section>
    );
};

export default Experience;
