import { motion, AnimatePresence } from 'framer-motion';
import { useRef } from 'react';
import useDialogFocus from '../hooks/useDialogFocus';
import { X, Github, ChevronLeft, ChevronRight, Zap, Target, Layers, Swords, Sparkles, Mail, Star } from 'lucide-react';
import { HollyDivider, Snowflake } from './Festive';

// Small heading used inside the letter
const LetterHeading = ({ icon: Icon, children }) => (
    <h3 className="text-cream-ink text-lg font-bold mb-3 flex items-center gap-2 border-b border-cream-ink/15 pb-2">
        <Icon size={16} className="text-holly" /> {children}
    </h3>
);

const ProjectHUD = ({ isOpen, onClose, project, onNext, onPrev }) => {
    // Esc closes, arrow keys page through projects, Tab stays inside the dialog
    const dialogRef = useRef(null);
    useDialogFocus(dialogRef, isOpen, onClose, { ArrowRight: onNext, ArrowLeft: onPrev });

    return (
        <AnimatePresence>
            {isOpen && project && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-[150] bg-night/90 backdrop-blur-md flex items-center justify-center p-3 md:p-8"
                    onClick={onClose}
                    ref={dialogRef}
                    tabIndex={-1}
                    role="dialog"
                    aria-modal="true"
                    aria-label={project.title}
                >
                    {/* Giant slow snowflake behind the letter */}
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none overflow-hidden">
                        <Snowflake className="w-[800px] h-[800px] text-frost/[0.06] animate-spin-slow" />
                    </div>

                    <motion.div
                        key={project.id}
                        initial={{ y: 80, rotate: -2, opacity: 0 }}
                        animate={{ y: 0, rotate: 0, opacity: 1 }}
                        exit={{ y: 80, rotate: 2, opacity: 0 }}
                        transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
                        onClick={(e) => e.stopPropagation()}
                        className="relative w-full max-w-6xl max-h-full flex flex-col rounded-xl overflow-hidden shadow-[0_30px_80px_rgba(0,0,0,0.7)]"
                    >
                        {/* Candy-cane border like a holiday envelope */}
                        <div className="h-3 candy-stripe shrink-0" aria-hidden="true" />

                        <div className="letter-paper text-cream-ink flex-1 overflow-y-auto">
                            {/* Header */}
                            <div className="sticky top-0 z-10 bg-cream/95 backdrop-blur-sm border-b-2 border-cream-ink/10 px-5 md:px-8 py-4 flex flex-wrap items-center justify-between gap-3">
                                <div>
                                    <p className="text-holly text-[11px] font-bold uppercase tracking-[0.3em] flex items-center gap-2">
                                        <Mail size={12} /> Project · {project.id}
                                    </p>
                                    <h2 className="text-2xl md:text-4xl font-bold text-cream-ink leading-tight">{project.title}</h2>
                                </div>
                                <div className="flex items-center gap-2">
                                    {project.note && (
                                        <span className="px-3 py-2 text-xs font-semibold text-cream-ink/70 border border-dashed border-cream-ink/30 rounded-full">
                                            {project.note}
                                        </span>
                                    )}
                                    {project.link && (
                                        <a
                                            href={project.link}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="px-3 py-2 border border-cream-ink/30 text-cream-ink hover:bg-pine hover:text-snow hover:border-pine transition-colors flex items-center gap-2 text-sm rounded-full"
                                        >
                                            <Github size={16} /> <span className="hidden md:inline">Source</span>
                                        </a>
                                    )}
                                    <button onClick={onPrev} aria-label="Previous project" className="p-2 border border-cream-ink/30 hover:bg-pine hover:text-snow hover:border-pine transition-colors rounded-full">
                                        <ChevronLeft size={18} />
                                    </button>
                                    <button onClick={onNext} aria-label="Next project" className="p-2 border border-cream-ink/30 hover:bg-pine hover:text-snow hover:border-pine transition-colors rounded-full">
                                        <ChevronRight size={18} />
                                    </button>
                                    <button onClick={onClose} aria-label="Close" className="p-2 bg-holly text-snow hover:bg-holly-deep transition-colors rounded-full">
                                        <X size={18} />
                                    </button>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 lg:grid-cols-12">
                                {/* Left column: photo, stats, stack */}
                                <div className="lg:col-span-4 p-5 md:p-8 lg:border-r border-cream-ink/10 space-y-7">
                                    <div className="relative p-2 pb-8 bg-white shadow-md -rotate-2 rounded-sm">
                                        <img
                                            src={project.image}
                                            alt={`${project.title} preview`}
                                            width="1024"
                                            height="1024"
                                            decoding="async"
                                            className="w-full aspect-[4/3] object-cover"
                                        />
                                        <span className="absolute bottom-1.5 left-0 right-0 text-center font-script text-xl text-holly">{project.title}</span>
                                    </div>

                                    <div className="grid grid-cols-2 gap-3">
                                        <div className="bg-pine/10 p-3 rounded-r-lg border-l-4 border-pine">
                                            <div className="text-[10px] text-cream-ink/60 uppercase tracking-wider">Latency</div>
                                            <div className="text-cream-ink font-bold text-lg flex items-center gap-1"><Zap size={14} className="text-pine" />{project.latency || 'N/A'}</div>
                                        </div>
                                        <div className="bg-holly/10 p-3 rounded-r-lg border-l-4 border-holly">
                                            <div className="text-[10px] text-cream-ink/60 uppercase tracking-wider">Accuracy</div>
                                            <div className="text-cream-ink font-bold text-lg flex items-center gap-1"><Target size={14} className="text-holly" />{project.accuracy || 'N/A'}</div>
                                        </div>
                                    </div>

                                    <div>
                                        <LetterHeading icon={Layers}>Tools in the Workshop</LetterHeading>
                                        <div className="flex flex-wrap gap-2">
                                            {project.tags.map((tag) => (
                                                <span key={tag} className="px-2 py-1 bg-pine/10 text-pine-deep text-xs font-semibold border border-pine/30 rounded-full">
                                                    {tag}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                </div>

                                {/* Right column: the story */}
                                <div className="lg:col-span-8 p-5 md:p-8 space-y-8">
                                    <div>
                                        <LetterHeading icon={Mail}>Dear Reader</LetterHeading>
                                        <p className="text-lg leading-relaxed">
                                            {project.technicalDesc || project.desc}
                                        </p>
                                    </div>

                                    <div>
                                        <LetterHeading icon={Sparkles}>What's Inside the Box</LetterHeading>
                                        <ul className="grid md:grid-cols-2 gap-x-6 gap-y-2">
                                            {project.features?.map((feat) => (
                                                <li key={feat} className="flex items-start gap-2 text-sm">
                                                    <Star size={14} className="mt-0.5 shrink-0 text-gold-deep fill-gold" /> {feat}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>

                                    <div>
                                        <LetterHeading icon={Swords}>Naughty &amp; Nice</LetterHeading>
                                        <div className="grid md:grid-cols-2 gap-4">
                                            <div className="p-4 bg-holly/10 rounded-r-lg border-l-4 border-holly">
                                                <span className="text-holly-deep font-bold text-xs uppercase tracking-widest block mb-1">Naughty: the challenge</span>
                                                <p className="text-sm">{project.challenge}</p>
                                            </div>
                                            <div className="p-4 bg-pine/10 rounded-r-lg border-l-4 border-pine">
                                                <span className="text-pine-deep font-bold text-xs uppercase tracking-widest block mb-1">Nice: the solution</span>
                                                <p className="text-sm">{project.solution}</p>
                                            </div>
                                        </div>
                                    </div>

                                    <div>
                                        <HollyDivider className="mb-5" />
                                        <div className="grid grid-cols-3 gap-4">
                                            {project.metrics?.map((metric) => (
                                                <div key={metric.label} className="text-center">
                                                    <div className="text-2xl md:text-4xl font-display font-bold text-holly">{metric.value}</div>
                                                    <div className="text-[10px] md:text-xs text-cream-ink/60 uppercase tracking-widest">{metric.label}</div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="h-3 candy-stripe shrink-0" aria-hidden="true" />
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};

export default ProjectHUD;
