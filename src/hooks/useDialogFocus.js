import { useEffect, useRef } from 'react';

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

// While a dialog is open: lock page scroll, close on Escape, keep Tab focus
// inside the dialog, and return focus to whatever opened it when it closes.
export default function useDialogFocus(ref, isOpen, onClose, extraKeys) {
    // Keep the latest handlers without re-running the effect on every render
    const handlers = useRef({ onClose, extraKeys });
    useEffect(() => {
        handlers.current = { onClose, extraKeys };
    });

    useEffect(() => {
        if (!isOpen) return;
        const opener = document.activeElement;
        document.body.style.overflow = 'hidden';

        // Move focus into the dialog once it has rendered
        const focusTimer = setTimeout(() => {
            const first = ref.current?.querySelector(FOCUSABLE);
            (first || ref.current)?.focus();
        }, 50);

        const onKey = (e) => {
            const { onClose: close, extraKeys: keys } = handlers.current;
            if (e.key === 'Escape') { close(); return; }
            if (keys?.[e.key]) { keys[e.key](); return; }
            if (e.key !== 'Tab' || !ref.current) return;

            const items = [...ref.current.querySelectorAll(FOCUSABLE)].filter((el) => el.offsetParent !== null);
            if (items.length === 0) { e.preventDefault(); return; }
            const first = items[0];
            const last = items[items.length - 1];
            if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
            else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
            else if (!ref.current.contains(document.activeElement)) { e.preventDefault(); first.focus(); }
        };
        window.addEventListener('keydown', onKey);

        return () => {
            clearTimeout(focusTimer);
            document.body.style.overflow = '';
            window.removeEventListener('keydown', onKey);
            if (opener && typeof opener.focus === 'function') opener.focus();
        };
    }, [ref, isOpen]);
}
