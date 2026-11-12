import { useEffect, useRef, useState } from "react";

// Soft starlight glow that follows the cursor, with a small red/green/gold
// firework burst on click. Only runs on devices with a fine pointer (mouse)
// and when the visitor hasn't asked for reduced motion.
// Everything is drawn on one canvas, so mouse movement never re-renders React.

const BURST_COLORS = ['#d4243b', '#1f8a55', '#f0c34e'];

const useFancyCursorEnabled = () => {
    const query = '(hover: hover) and (pointer: fine) and (prefers-reduced-motion: no-preference)';
    const [enabled, setEnabled] = useState(() => typeof window !== 'undefined' && window.matchMedia(query).matches);
    useEffect(() => {
        const mql = window.matchMedia(query);
        const onChange = () => setEnabled(mql.matches);
        mql.addEventListener('change', onChange);
        return () => mql.removeEventListener('change', onChange);
    }, []);
    return enabled;
};

const MouseFollower = () => {
    const enabled = useFancyCursorEnabled();
    const canvasRef = useRef(null);

    useEffect(() => {
        if (!enabled) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        let w = 0, h = 0, raf = 0;
        const mouse = { x: -100, y: -100, gx: -100, gy: -100, down: false };
        let particles = [];

        const resize = () => {
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            w = window.innerWidth; h = window.innerHeight;
            canvas.width = w * dpr; canvas.height = h * dpr;
            canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        };

        const onMove = (e) => {
            mouse.x = e.clientX; mouse.y = e.clientY;
            if (Math.random() > 0.8) {
                particles.push({ x: e.clientX, y: e.clientY, vx: 0, vy: 0.4, life: 1, decay: 0.02, r: Math.random() * 2 + 1, color: '#f4f7fc' });
            }
        };

        const onDown = (e) => {
            mouse.down = true;
            for (let i = 0; i < 12; i++) {
                const angle = (Math.PI * 2 * i) / 12;
                const speed = Math.random() * 1.5 + 1.5;
                particles.push({
                    x: e.clientX, y: e.clientY,
                    vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed,
                    life: 1, decay: 0.03, r: Math.random() * 2.5 + 1.5,
                    color: BURST_COLORS[i % BURST_COLORS.length],
                });
            }
        };
        const onUp = () => { mouse.down = false; };

        const draw = () => {
            ctx.clearRect(0, 0, w, h);

            // Glow eases toward the pointer
            mouse.gx += (mouse.x - mouse.gx) * 0.35;
            mouse.gy += (mouse.y - mouse.gy) * 0.35;
            const radius = mouse.down ? 22 : 14;
            const g = ctx.createRadialGradient(mouse.gx, mouse.gy, 0, mouse.gx, mouse.gy, radius);
            g.addColorStop(0, mouse.down ? 'rgba(240,195,78,0.9)' : 'rgba(255,255,255,0.75)');
            g.addColorStop(1, 'rgba(169,212,245,0)');
            ctx.fillStyle = g;
            ctx.beginPath();
            ctx.arc(mouse.gx, mouse.gy, radius, 0, Math.PI * 2);
            ctx.fill();

            particles = particles.filter((p) => p.life > 0);
            if (particles.length > 80) particles.splice(0, particles.length - 80);
            for (const p of particles) {
                p.x += p.vx; p.y += p.vy;
                p.vx *= 0.95; p.vy *= 0.95;
                p.life -= p.decay;
                ctx.globalAlpha = Math.max(p.life, 0);
                ctx.fillStyle = p.color;
                ctx.shadowColor = p.color;
                ctx.shadowBlur = 8;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fill();
            }
            ctx.globalAlpha = 1;
            ctx.shadowBlur = 0;

            raf = requestAnimationFrame(draw);
        };

        resize();
        window.addEventListener('resize', resize);
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mousedown', onDown);
        window.addEventListener('mouseup', onUp);
        raf = requestAnimationFrame(draw);

        return () => {
            cancelAnimationFrame(raf);
            window.removeEventListener('resize', resize);
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mousedown', onDown);
            window.removeEventListener('mouseup', onUp);
        };
    }, [enabled]);

    if (!enabled) return null;
    return <canvas ref={canvasRef} className="fixed inset-0 pointer-events-none z-[300]" aria-hidden="true" />;
};

export default MouseFollower;
