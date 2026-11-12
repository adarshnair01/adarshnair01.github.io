import { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";

const FullScreenOverlay = ({ isOpen, onClose, title, children }) => {
    // Lock body scroll and close on ESC while open
    useEffect(() => {
        if (!isOpen) return;
        document.body.style.overflow = 'hidden';
        const onKey = (e) => e.key === 'Escape' && onClose();
        window.addEventListener('keydown', onKey);
        return () => {
            document.body.style.overflow = '';
            window.removeEventListener('keydown', onKey);
        };
    }, [isOpen, onClose]);

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-[200] bg-ink text-palm flex flex-col"
                    role="dialog"
                    aria-modal="true"
                    aria-label={typeof title === 'string' ? title : undefined}
                >
                    {/* Header Controls */}
                    <div className="absolute top-0 left-0 right-0 p-4 md:p-6 flex justify-between items-start z-50 pointer-events-none">
                        <div className="bg-ink-card/80 backdrop-blur-md px-6 py-3 rounded-full pointer-events-auto border border-gold/30">
                            <h3 className="text-xl bg-clip-text text-transparent bg-gradient-to-r from-gold to-saffron">
                                {title}
                            </h3>
                        </div>

                        <button
                            onClick={onClose}
                            aria-label="Close (Esc)"
                            className="relative bg-kumkum/20 hover:bg-kumkum text-palm backdrop-blur-md p-3 rounded-full transition-all pointer-events-auto border border-kumkum/40 group"
                        >
                            <X size={28} />
                            <span className="absolute top-full right-0 mt-2 bg-ink/90 text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                                Close (ESC)
                            </span>
                        </button>
                    </div>

                    <div className="flex-1 w-full h-full flex items-center justify-center p-4 md:p-8 overflow-hidden relative">
                        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(240,138,36,0.10),transparent_65%)]"></div>

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
