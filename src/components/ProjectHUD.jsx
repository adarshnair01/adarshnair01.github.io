import { motion, AnimatePresence } from 'framer-motion';
import { useState, useEffect } from 'react';
import { X, Cpu, Database, Globe, Activity, Terminal, Shield, Zap, Target, Layers, Github } from 'lucide-react';

// Matrix Boot Sequence Component
const MatrixBoot = ({ onComplete }) => {
    useEffect(() => {
        const canvas = document.getElementById('matrix-canvas');
        const ctx = canvas.getContext('2d');

        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        const katakana = 'アァカサタナハマヤャラワガザダバパイィキシチニヒミリヰギジヂビピウゥクスツヌフムユュルグズブヅプエェケセテネヘメレヱゲゼデベペオォコソトノホモヨョロヲゴゾドボポ1234567890';
        const latin = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
        const nums = '0123456789';
        const alphabet = katakana + latin + nums;

        const fontSize = 16;
        const columns = canvas.width / fontSize;

        const rainDrops = Array.from({ length: Math.ceil(columns) }).fill(1);

        const draw = () => {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = '#0F0'; // Matrix Green
            ctx.font = fontSize + 'px monospace';

            for (let i = 0; i < rainDrops.length; i++) {
                const text = alphabet.charAt(Math.floor(Math.random() * alphabet.length));
                ctx.fillText(text, i * fontSize, rainDrops[i] * fontSize);

                if (rainDrops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                    rainDrops[i] = 0;
                }
                rainDrops[i]++;
            }
        };

        const interval = setInterval(draw, 30);

        // End sequence after 2s
        const timeout = setTimeout(() => {
            clearInterval(interval);
            onComplete();
        }, 1500);

        return () => {
            clearInterval(interval);
            clearTimeout(timeout);
        };
    }, [onComplete]);

    return (
        <motion.div
            className="absolute inset-0 z-50 bg-black flex items-center justify-center overflow-hidden"
            exit={{ opacity: 0 }}
        >
            <canvas id="matrix-canvas" className="absolute inset-0 opacity-80" />
            <div className="relative z-10 text-[#0F0] font-mono font-bold text-2xl tracking-[0.5em] bg-black/80 px-8 py-4 border border-[#0F0] animate-pulse">
                INITIALIZING NEURAL LINK...
            </div>
        </motion.div>
    );
};

const ProjectHUD = ({ isOpen, onClose, project, onNext, onPrev }) => {
    const [isBooting, setIsBooting] = useState(true);

    // Reset boot state when modal opens
    useEffect(() => {
        if (isOpen) setIsBooting(true);
    }, [isOpen]);

    if (!project) return null;

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-[100] bg-black/95 backdrop-blur-xl flex items-center justify-center overflow-hidden font-mono"
                >
                    {/* Show Matrix Boot Sequence first */}
                    {isBooting ? (
                        <MatrixBoot onComplete={() => setIsBooting(false)} />
                    ) : (
                        <>
                            {/* Background Grid & Ambient Scanlines */}
                            <div className="absolute inset-0 z-0 pointer-events-none">
                                <div className="absolute inset-0 bg-[linear-gradient(rgba(0,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(0,255,255,0.03)_1px,transparent_1px)] bg-[size:40px_40px]"></div>
                                <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(0,243,255,0.1),transparent_70%)]"></div>
                            </div>

                            {/* JARVIS / HUD Container */}
                            <motion.div
                                initial={{ scale: 0.9, opacity: 0 }}
                                animate={{ scale: 1, opacity: 1 }}
                                exit={{ scale: 0.9, opacity: 0 }}
                                transition={{ duration: 0.4, ease: "easeOut" }}
                                className="relative z-10 w-[95%] h-[90%] border border-[#00f3ff]/30 bg-[#0a0a0a]/90 flex flex-col shadow-[0_0_50px_rgba(0,243,255,0.15)] rounded-lg overflow-hidden"
                            >
                                {/* 1. HUD HEADER */}
                                <div className="h-16 border-b border-[#00f3ff]/30 flex items-center justify-between px-6 bg-[#00f3ff]/5 relative">
                                    {/* Scanning Line Animation on Header */}
                                    <motion.div
                                        animate={{ x: ["-100%", "100%"] }}
                                        transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                                        className="absolute bottom-0 left-0 h-[1px] w-full bg-[#00f3ff] shadow-[0_0_10px_#00f3ff]"
                                    />

                                    <div className="flex items-center gap-4">
                                        <div className="p-2 border border-[#00f3ff] rounded-full">
                                            <Cpu className="text-[#00f3ff] animate-pulse" size={20} />
                                        </div>
                                        <div>
                                            <h2 className="text-xl md:text-2xl font-bold text-white tracking-widest uppercase">{project.title}</h2>
                                            <p className="text-[10px] text-[#00f3ff] tracking-[0.2em] uppercase">System Architecture Verified • {project.id || 'PRJ-X1'}</p>
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-4">
                                        <div className="flex mr-4">
                                            {project.link && project.link !== '#' && (
                                                <a
                                                    href={project.link}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="p-2 border border-[#00f3ff]/50 text-[#00f3ff] hover:bg-[#00f3ff] hover:text-black transition-all flex items-center gap-2 mr-2"
                                                >
                                                    <Github size={16} />
                                                    <span className="hidden md:inline">GITHUB</span>
                                                </a>
                                            )}
                                            <button
                                                onClick={(e) => { e.stopPropagation(); onPrev(); }}
                                                className="p-2 border border-[#00f3ff]/50 text-[#00f3ff] hover:bg-[#00f3ff] hover:text-black transition-all border-r-0"
                                            >
                                                &lt; PREV
                                            </button>
                                            <button
                                                onClick={(e) => { e.stopPropagation(); onNext(); }}
                                                className="p-2 border border-[#00f3ff]/50 text-[#00f3ff] hover:bg-[#00f3ff] hover:text-black transition-all"
                                            >
                                                NEXT &gt;
                                            </button>
                                        </div>

                                        <button
                                            onClick={onClose}
                                            className="group relative px-4 py-2 border border-[#00f3ff]/50 text-[#00f3ff] hover:bg-[#00f3ff] hover:text-black transition-all duration-300 uppercase text-xs tracking-widest font-bold"
                                        >
                                            <span className="absolute inset-0 border-t border-b border-[#00f3ff] scale-x-0 group-hover:scale-x-100 transition-transform origin-center"></span>
                                            Close Module
                                        </button>
                                    </div>
                                </div>

                                {/* 2. MAIN CONTENT GRID */}
                                <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-0 overflow-hidden relative min-h-0">

                                    {/* LEFT COLUMN: VISUALS & CORE STATS (4Cols) */}
                                    <div className="lg:col-span-4 border-r border-[#00f3ff]/20 bg-[#000]/40 p-6 flex flex-col gap-6 relative overflow-y-auto custom-scrollbar">
                                        {/* Image Display Frame */}
                                        <div className="relative aspect-video border border-[#00f3ff]/30 p-1 group">
                                            {/* Corner Markers */}
                                            <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-[#00f3ff]"></div>
                                            <div className="absolute top-0 right-0 w-2 h-2 border-t border-r border-[#00f3ff]"></div>
                                            <div className="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-[#00f3ff]"></div>
                                            <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-[#00f3ff]"></div>

                                            <img
                                                src={project.image}
                                                alt="Project Preview"
                                                className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity grayscale group-hover:grayscale-0"
                                            />

                                            <div className="absolute bottom-2 right-2 px-2 py-0.5 bg-[#00f3ff] text-black text-[10px] font-bold">
                                                IMG_SRC_01
                                            </div>
                                        </div>

                                        {/* Quick Stats Module */}
                                        <div className="space-y-4">
                                            <h3 className="text-[#00f3ff] text-xs uppercase tracking-widest border-b border-[#00f3ff]/20 pb-2 mb-4 flex items-center gap-2">
                                                <Activity size={14} /> System Diagnostics
                                            </h3>
                                            <div className="grid grid-cols-2 gap-4">
                                                <div className="bg-[#00f3ff]/5 p-3 border-l-2 border-[#00f3ff]">
                                                    <div className="text-[10px] text-gray-400 uppercase">Latency</div>
                                                    <div className="text-white font-bold text-lg"><Zap size={12} className="inline mr-1 text-yellow-400" />{project.latency || "N/A"}</div>
                                                </div>
                                                <div className="bg-[#00f3ff]/5 p-3 border-l-2 border-[#00f3ff]">
                                                    <div className="text-[10px] text-gray-400 uppercase">Accuracy</div>
                                                    <div className="text-white font-bold text-lg"><Target size={12} className="inline mr-1 text-green-400" />{project.accuracy || "N/A"}</div>
                                                </div>
                                                <div className="bg-[#00f3ff]/5 p-3 border-l-2 border-[#00f3ff] col-span-2">
                                                    <div className="text-[10px] text-gray-400 uppercase">Status</div>
                                                    <div className="text-[#00f3ff] font-bold tracking-wider animate-pulse">● SYSTEM ONLINE</div>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Tech Stack Tags */}
                                        <div>
                                            <h3 className="text-[#00f3ff] text-xs uppercase tracking-widest border-b border-[#00f3ff]/20 pb-2 mb-4 flex items-center gap-2">
                                                <Layers size={14} /> Stack Modules
                                            </h3>
                                            <div className="flex flex-wrap gap-2">
                                                {project.tags.map((tag, i) => (
                                                    <span key={i} className="px-2 py-1 bg-[#00f3ff]/10 text-[#00f3ff] text-xs border border-[#00f3ff]/30">
                                                        {tag}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    </div>

                                    {/* RIGHT COLUMN: DETAILED DATA (8Cols) */}
                                    <div className="lg:col-span-8 p-8 overflow-y-auto custom-scrollbar relative pb-20">
                                        {/* Decorative Background Elements in Right Panel */}
                                        <div className="absolute top-10 right-10 w-20 h-20 border border-[#00f3ff]/10 rounded-full border-t-[#00f3ff]/30 animate-spin-slow pointer-events-none"></div>

                                        {/* 1. Mission Brief (Description) */}
                                        <div className="mb-8">
                                            <div className="flex items-center gap-3 mb-4">
                                                <Terminal className="text-[#00f3ff]" size={18} />
                                                <h3 className="text-white text-lg font-bold tracking-wider uppercase">Mission Brief</h3>
                                            </div>
                                            <p className="text-gray-300 leading-relaxed text-sm md:text-base border-l-2 border-[#00f3ff]/50 pl-4">
                                                {project.technicalDesc || project.desc}
                                            </p>
                                        </div>

                                        {/* 2. Architecure & Features */}
                                        <div className="grid md:grid-cols-2 gap-8 mb-8">
                                            <div className="bg-[#0a0a0a] border border-[#333] p-5 relative overflow-hidden group hover:border-[#00f3ff]/50 transition-colors">
                                                <h4 className="text-[#00f3ff] font-bold text-sm uppercase mb-4 flex items-center gap-2">
                                                    <Database size={14} /> Data Pipeline
                                                </h4>
                                                <ul className="space-y-2">
                                                    {project.features?.slice(0, 3).map((feat, i) => (
                                                        <li key={i} className="text-gray-400 text-xs flex items-start gap-2">
                                                            <span className="text-[#00f3ff] mt-0.5">▹</span> {feat}
                                                        </li>
                                                    ))}
                                                </ul>
                                            </div>

                                            <div className="bg-[#0a0a0a] border border-[#333] p-5 relative overflow-hidden group hover:border-[#00f3ff]/50 transition-colors">
                                                <h4 className="text-[#00f3ff] font-bold text-sm uppercase mb-4 flex items-center gap-2">
                                                    <Shield size={14} /> Security & Scale
                                                </h4>
                                                <ul className="space-y-2">
                                                    {project.features?.slice(3).map((feat, i) => (
                                                        <li key={i} className="text-gray-400 text-xs flex items-start gap-2">
                                                            <span className="text-[#00f3ff] mt-0.5">▹</span> {feat}
                                                        </li>
                                                    ))}
                                                </ul>
                                            </div>
                                        </div>

                                        {/* 3. Key Challenges & Solutions */}
                                        <div className="mb-8">
                                            <h3 className="text-white text-sm font-bold tracking-wider uppercase mb-4 flex items-center gap-2">
                                                <Target className="text-red-500" size={16} /> Conflict Resolution
                                            </h3>
                                            <div className="space-y-4">
                                                <div className="flex gap-4 p-4 bg-red-500/5 border border-red-500/20 rounded-r-lg border-l-4 border-l-red-500">
                                                    <div className="text-xs text-gray-300">
                                                        <span className="text-red-400 font-bold block mb-1">CHALLENGE DETECTED</span>
                                                        {project.challenge || "High latency in real-time inference causing user drop-off."}
                                                    </div>
                                                </div>
                                                <div className="flex gap-4 p-4 bg-[#00f3ff]/5 border border-[#00f3ff]/20 rounded-r-lg border-l-4 border-l-[#00f3ff]">
                                                    <div className="text-xs text-gray-300">
                                                        <span className="text-[#00f3ff] font-bold block mb-1">SOLUTION DEPLOYED</span>
                                                        {project.solution || "Implemented optimized caching layer (Redis) and quantized model weights for 40% faster inference."}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        {/* 4. Impact Metrics */}
                                        <div className="border-t border-[#00f3ff]/20 pt-6">
                                            <h3 className="text-white text-sm font-bold tracking-wider uppercase mb-4">
                                                Project Impact
                                            </h3>
                                            <div className="grid grid-cols-3 gap-4">
                                                {project.metrics?.map((metric, i) => (
                                                    <div key={i} className="text-center">
                                                        <div className="text-2xl md:text-3xl font-bold text-[#00f3ff]">{metric.value}</div>
                                                        <div className="text-[10px] text-gray-500 uppercase tracking-widest">{metric.label}</div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>

                                    </div>
                                </div>

                                {/* 3. FOOTER */}
                                <div className="h-10 bg-[#000] border-t border-[#00f3ff]/30 flex items-center justify-between px-6 text-[10px] text-gray-500 uppercase tracking-widest">
                                    <div>SECURE CONNECTION ESTABLISHED</div>
                                    <div className="flex gap-4">
                                        <span><span className="text-[#00f3ff]">RAM:</span> 64GB</span>
                                        <span><span className="text-[#00f3ff]">CORE:</span> 12</span>
                                        <span className="animate-pulse text-[#00f3ff]">LIVE</span>
                                    </div>
                                </div>

                            </motion.div>
                        </>
                    )}
                </motion.div>
            )}
        </AnimatePresence>
    );
};

export default ProjectHUD;
