import { motion, AnimatePresence } from "framer-motion";
import { X, Maximize2, Minimize2 } from "lucide-react";

const FullScreenOverlay = ({ isOpen, onClose, title, children }) => {
    // Lock body scroll when open
    if (typeof document !== 'undefined') {
        if (isOpen) document.body.style.overflow = 'hidden';
        else document.body.style.overflow = 'unset';
    }

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-[200] bg-slate-900 text-white flex flex-col"
                >
                    {/* Header Controls */}
                    <div className="absolute top-0 left-0 right-0 p-6 flex justify-between items-start z-50 pointer-events-none">
                        <div className="bg-black/40 backdrop-blur-md px-6 py-3 rounded-full pointer-events-auto border border-white/10">
                            <h3 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary">
                                {title}
                            </h3>
                        </div>

                        <button
                            onClick={onClose}
                            className="bg-red-500/20 hover:bg-red-500 hover:text-white backdrop-blur-md p-3 rounded-full transition-all pointer-events-auto border border-red-500/30 group"
                        >
                            <X size={28} />
                            <span className="absolute top-full right-0 mt-2 bg-black/80 text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                                Exit Game (ESC)
                            </span>
                        </button>
                    </div>

                    {/* Game Container - Centered & Scaled */}
                    <div className="flex-1 w-full h-full flex items-center justify-center p-4 md:p-8 overflow-hidden relative">
                        {/* Background Ambience */}
                        <div className="absolute inset-0 opacity-20 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-slate-800 to-slate-950"></div>

                        <div className="w-full h-full max-w-7xl max-h-[85vh] flex items-center justify-center relative z-10">
                            {children}
                        </div>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};

export default FullScreenOverlay;
