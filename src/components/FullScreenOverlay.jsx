import { useRef } from "react";
import useDialogFocus from "../hooks/useDialogFocus";
import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";

const FullScreenOverlay = ({ isOpen, onClose, title, children }) => {
    const dialogRef = useRef(null);
    useDialogFocus(dialogRef, isOpen, onClose);

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-[200] bg-night text-snow flex flex-col"
                    ref={dialogRef}
                    tabIndex={-1}
                    role="dialog"
                    aria-modal="true"
                    aria-label={typeof title === 'string' ? title : undefined}
                >
                    {/* Header Controls */}
                    <div className="absolute top-0 left-0 right-0 p-4 md:p-6 flex justify-between items-start z-50 pointer-events-none">
                        <div className="bg-night-card/80 backdrop-blur-md px-6 py-3 rounded-full pointer-events-auto border border-frost/20">
                            <h3 className="text-xl font-bold text-gold">
                                {title}
                            </h3>
                        </div>

                        <button
                            onClick={onClose}
                            aria-label="Close (Esc)"
                            className="relative bg-holly/20 hover:bg-holly text-snow backdrop-blur-md p-3 rounded-full transition-all pointer-events-auto border border-holly/40 group"
                        >
                            <X size={28} />
                            <span className="absolute top-full right-0 mt-2 bg-night/90 text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                                Close (ESC)
                            </span>
                        </button>
                    </div>

                    <div className="flex-1 w-full h-full flex items-center justify-center p-4 md:p-8 overflow-hidden relative">
                        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(31,138,85,0.12),transparent_65%)]"></div>

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
