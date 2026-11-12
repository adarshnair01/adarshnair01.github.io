import { motion, AnimatePresence } from 'framer-motion';
import { useEffect } from 'react';
import { X, Github, ChevronLeft, ChevronRight, Zap, Target, Layers, Swords, Sparkles, ScrollText } from 'lucide-react';
import { BowDivider, Lotus, SunDisc } from './Ornaments';

// Small heading used inside the manuscript
const LeafHeading = ({ icon: Icon, children }) => (
    <h3 className="text-palm-ink text-lg mb-3 flex items-center gap-2 border-b border-palm-ink/20 pb-2">
        <Icon size={16} className="text-kumkum" /> {children}
    </h3>
);

const ProjectHUD = ({ isOpen, onClose, project, onNext, onPrev }) => {
    // Keyboard: Esc closes, arrows page through manuscripts; lock page scroll
    useEffect(() => {
        if (!isOpen) return;
        document.body.style.overflow = 'hidden';
        const onKey = (e) => {
            if (e.key === 'Escape') onClose();
            if (e.key === 'ArrowRight') onNext();
            if (e.key === 'ArrowLeft') onPrev();
        };
        window.addEventListener('keydown', onKey);
        return () => {
            document.body.style.overflow = '';
            window.removeEventListener('keydown', onKey);
        };
    }, [isOpen, onClose, onNext, onPrev]);

    return (
        <AnimatePresence>
            {isOpen && project && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-[150] bg-ink/90 backdrop-blur-md flex items-center justify-center p-3 md:p-8"
                    onClick={onClose}
                    role="dialog"
                    aria-modal="true"
                    aria-label={project.title}
                >
                    {/* Glowing sun behind the scroll */}
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none overflow-hidden">
                        <SunDisc className="w-[900px] h-[900px] opacity-10 animate-spin-slower" rays={36} />
                    </div>

                    <motion.div
                        key={project.id}
                        initial={{ scaleY: 0.05, opacity: 0 }}
                        animate={{ scaleY: 1, opacity: 1 }}
                        exit={{ scaleY: 0.05, opacity: 0 }}
                        transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
                        style={{ originY: 0.5 }}
                        onClick={(e) => e.stopPropagation()}
                        className="relative w-full max-w-6xl max-h-full flex flex-col"
                    >
                        {/* Top wooden rod of the scroll */}
                        <div className="h-5 mx-[-10px] rounded-full bg-gradient-to-b from-[#8b5a2b] via-[#5c3a1a] to-[#3b2412] shadow-lg relative z-10">
                            <span className="absolute -left-2 top-1/2 -translate-y-1/2 w-5 h-5 rounded-full bg-gold border-2 border-gold-deep" />
                            <span className="absolute -right-2 top-1/2 -translate-y-1/2 w-5 h-5 rounded-full bg-gold border-2 border-gold-deep" />
                        </div>

                        <div className="palm-leaf text-palm-ink flex-1 overflow-y-auto shadow-[0_30px_80px_rgba(0,0,0,0.7)] border-x-4 border-palm-deep">
                            {/* Header */}
                            <div className="sticky top-0 z-10 bg-[#e3cd98]/95 backdrop-blur-sm border-b-2 border-palm-ink/20 px-5 md:px-8 py-4 flex flex-wrap items-center justify-between gap-3">
                                <div>
                                    <p className="text-kumkum text-[11px] font-semibold uppercase tracking-[0.3em] flex items-center gap-2">
                                        <ScrollText size={12} /> Manuscript · {project.id}
                                    </p>
                                    <h2 className="text-2xl md:text-4xl text-palm-ink leading-tight">{project.title}</h2>
                                </div>
                                <div className="flex items-center gap-2">
                                    {project.link && project.link !== '#' && (
                                        <a
                                            href={project.link}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="px-3 py-2 border border-palm-ink/40 text-palm-ink hover:bg-palm-ink hover:text-palm transition-colors flex items-center gap-2 text-sm rounded-sm"
                                        >
                                            <Github size={16} /> <span className="hidden md:inline">Source</span>
                                        </a>
                                    )}
                                    <button onClick={onPrev} aria-label="Previous project" className="p-2 border border-palm-ink/40 hover:bg-palm-ink hover:text-palm transition-colors rounded-sm">
                                        <ChevronLeft size={18} />
                                    </button>
                                    <button onClick={onNext} aria-label="Next project" className="p-2 border border-palm-ink/40 hover:bg-palm-ink hover:text-palm transition-colors rounded-sm">
                                        <ChevronRight size={18} />
                                    </button>
                                    <button onClick={onClose} aria-label="Close" className="p-2 bg-kumkum text-palm hover:bg-palm-ink transition-colors rounded-sm">
                                        <X size={18} />
                                    </button>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 lg:grid-cols-12">
                                {/* Left column: image, stats, stack */}
                                <div className="lg:col-span-4 p-5 md:p-8 lg:border-r border-palm-ink/15 space-y-7">
                                    <div className="relative p-1.5 bg-palm-ink/10 border border-palm-ink/30 rounded-t-[999px] overflow-hidden">
                                        <img
                                            src={project.image}
                                            alt={`${project.title} preview`}
                                            className="w-full aspect-[4/3] object-cover rounded-t-[999px] sepia-[.35]"
                                        />
                                    </div>

                                    <div className="grid grid-cols-2 gap-3">
                                        <div className="bg-palm-ink/5 p-3 border-l-4 border-saffron">
                                            <div className="text-[10px] text-palm-ink/60 uppercase tracking-wider">Latency</div>
                                            <div className="text-palm-ink font-semibold text-lg flex items-center gap-1"><Zap size={14} className="text-saffron" />{project.latency || 'N/A'}</div>
                                        </div>
                                        <div className="bg-palm-ink/5 p-3 border-l-4 border-kumkum">
                                            <div className="text-[10px] text-palm-ink/60 uppercase tracking-wider">Accuracy</div>
                                            <div className="text-palm-ink font-semibold text-lg flex items-center gap-1"><Target size={14} className="text-kumkum" />{project.accuracy || 'N/A'}</div>
                                        </div>
                                    </div>

                                    <div>
                                        <LeafHeading icon={Layers}>Weapons Wielded</LeafHeading>
                                        <div className="flex flex-wrap gap-2">
                                            {project.tags.map((tag) => (
                                                <span key={tag} className="px-2 py-1 bg-palm-ink/10 text-palm-ink text-xs border border-palm-ink/25 rounded-sm">
                                                    {tag}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                </div>

                                {/* Right column: story */}
                                <div className="lg:col-span-8 p-5 md:p-8 space-y-8">
                                    <div>
                                        <LeafHeading icon={ScrollText}>The Tale</LeafHeading>
                                        <p className="font-serif text-lg leading-relaxed">
                                            {project.technicalDesc || project.desc}
                                        </p>
                                    </div>

                                    <div>
                                        <LeafHeading icon={Sparkles}>Feats of the Build</LeafHeading>
                                        <ul className="grid md:grid-cols-2 gap-x-6 gap-y-2">
                                            {project.features?.map((feat) => (
                                                <li key={feat} className="flex items-start gap-2 text-sm">
                                                    <Lotus className="w-4 h-3 mt-1 shrink-0 text-saffron" /> {feat}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>

                                    <div>
                                        <LeafHeading icon={Swords}>The Battle</LeafHeading>
                                        <div className="grid md:grid-cols-2 gap-4">
                                            <div className="p-4 bg-kumkum/10 border-l-4 border-kumkum">
                                                <span className="text-kumkum font-semibold text-xs uppercase tracking-widest block mb-1">The Rakshasa (Challenge)</span>
                                                <p className="text-sm">{project.challenge}</p>
                                            </div>
                                            <div className="p-4 bg-saffron/10 border-l-4 border-saffron">
                                                <span className="text-[#a3530e] font-semibold text-xs uppercase tracking-widest block mb-1">The Astra (Solution)</span>
                                                <p className="text-sm">{project.solution}</p>
                                            </div>
                                        </div>
                                    </div>

                                    <div>
                                        <BowDivider className="!text-gold-deep mb-5" />
                                        <div className="grid grid-cols-3 gap-4">
                                            {project.metrics?.map((metric) => (
                                                <div key={metric.label} className="text-center">
                                                    <div className="text-2xl md:text-4xl font-epic text-kumkum">{metric.value}</div>
                                                    <div className="text-[10px] md:text-xs text-palm-ink/60 uppercase tracking-widest">{metric.label}</div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Bottom wooden rod */}
                        <div className="h-5 mx-[-10px] rounded-full bg-gradient-to-b from-[#8b5a2b] via-[#5c3a1a] to-[#3b2412] shadow-lg relative z-10">
                            <span className="absolute -left-2 top-1/2 -translate-y-1/2 w-5 h-5 rounded-full bg-gold border-2 border-gold-deep" />
                            <span className="absolute -right-2 top-1/2 -translate-y-1/2 w-5 h-5 rounded-full bg-gold border-2 border-gold-deep" />
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};

export default ProjectHUD;
