import { useEffect, useRef, useState } from 'react';

const TableTennisGame = () => {
    const canvasRef = useRef(null);
    const [score, setScore] = useState({ player: 0, cpu: 0 });
    const [gameState, setGameState] = useState('menu'); // menu, playing, gameover

    // Game state in refs to persist across renders without closure stale issues
    const gameData = useRef({
        ball: { x: 225, y: 350, vx: 3, vy: 5, radius: 8 },
        player: { x: 225, width: 80, height: 12 },
        cpu: { x: 225, width: 80, height: 12 },
        particles: []
    });

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        let animationFrameId;
        const width = 450;
        const height = 700;

        // Physics Constants (Pong Style)
        const BALL_SPEED_START = 7;
        const MAX_X_SPEED = 7;
        const CPU_SPEED = 4.5;

        const update = () => {
            if (gameState !== 'playing') return;

            const { ball, player, cpu, particles } = gameData.current;

            // --- Ball Physics ---
            ball.x += ball.vx;
            ball.y += ball.vy;

            // --- Wall Collision (Left/Right) ---
            if (ball.x - ball.radius < 0 || ball.x + ball.radius > width) {
                ball.vx *= -1;
                // Clamp
                if (ball.x < ball.radius) ball.x = ball.radius;
                if (ball.x > width - ball.radius) ball.x = width - ball.radius;

                createParticles(ball.x, ball.y, '#ffffff');
            }

            // --- Scoring (Top/Bottom Walls) ---
            if (ball.y < 0) {
                // Top Wall passed -> Player scores (Ball went past CPU)
                setScore(s => ({ ...s, player: s.player + 1 }));
                resetBall(true);
            } else if (ball.y > height) {
                // Bottom Wall passed -> CPU scores (Ball went past Player)
                setScore(s => ({ ...s, cpu: s.cpu + 1 }));
                resetBall(false);
            }

            // --- Player Paddle Collision (Bottom) ---
            // Paddle Y position fixed near bottom
            const pY = height - 40;
            const pRect = { x: player.x - player.width / 2, y: pY, w: player.width, h: player.height };

            if (
                ball.y + ball.radius > pRect.y &&
                ball.y - ball.radius < pRect.y + pRect.h &&
                ball.x > pRect.x &&
                ball.x < pRect.x + pRect.w
            ) {
                // Ball moving down?
                if (ball.vy > 0) {
                    ball.vy = -Math.abs(ball.vy) * 1.05; // Bounce up & speed up
                    if (ball.vy < -16) ball.vy = -16; // Cap speed

                    // Angle based on hit position relative to paddle center
                    const hitOffset = (ball.x - player.x) / (player.width / 2);
                    ball.vx = hitOffset * MAX_X_SPEED;

                    ball.y = pRect.y - ball.radius - 1; // Push out
                    createParticles(ball.x, ball.y, '#34d399'); // Emerald glow
                }
            }

            // --- CPU AI Collision (Top) ---
            const cY = 40;
            const cRect = { x: cpu.x - cpu.width / 2, y: cY, w: cpu.width, h: cpu.height };

            if (
                ball.y - ball.radius < cRect.y + cRect.h &&
                ball.y + ball.radius > cRect.y &&
                ball.x > cRect.x &&
                ball.x < cRect.x + cRect.w
            ) {
                if (ball.vy < 0) {
                    ball.vy = Math.abs(ball.vy) * 1.05; // Bounce down
                    if (ball.vy > 16) ball.vy = 16;

                    const hitOffset = (ball.x - cpu.x) / (cpu.width / 2);
                    ball.vx = hitOffset * MAX_X_SPEED;

                    ball.y = cRect.y + cRect.h + ball.radius + 1;
                    createParticles(ball.x, ball.y, '#f87171'); // Red glow
                }
            }

            // --- CPU Movement ---
            // CPU tracks ball X
            if (ball.vy < 0) {
                // Ball coming towards CPU (Up)
                const targetX = ball.x;
                if (cpu.x < targetX - 10) cpu.x += CPU_SPEED;
                else if (cpu.x > targetX + 10) cpu.x -= CPU_SPEED;
            } else {
                // Return to center slowly
                if (cpu.x < width / 2) cpu.x += 2;
                else if (cpu.x > width / 2) cpu.x -= 2;
            }

            // Clamp CPU
            if (cpu.x < cpu.width / 2) cpu.x = cpu.width / 2;
            if (cpu.x > width - cpu.width / 2) cpu.x = width - cpu.width / 2;

            // --- Update Particles ---
            for (let i = particles.length - 1; i >= 0; i--) {
                const p = particles[i];
                p.x += p.vx;
                p.y += p.vy;
                p.life -= 0.04;
                if (p.life <= 0) particles.splice(i, 1);
            }
        };

        const resetBall = (playerWon) => {
            const { ball } = gameData.current;
            ball.x = width / 2;
            ball.y = height / 2;
            // If player won, serve towards CPU (up), else serve towards Player (down)
            const dirY = playerWon ? -1 : 1;
            ball.vy = dirY * BALL_SPEED_START;
            ball.vx = (Math.random() - 0.5) * 6;
        };

        const createParticles = (x, y, color) => {
            for (let i = 0; i < 12; i++) {
                gameData.current.particles.push({
                    x, y,
                    vx: (Math.random() - 0.5) * 8,
                    vy: (Math.random() - 0.5) * 8,
                    life: 1.0,
                    color
                });
            }
        };

        const draw = () => {
            const { ball, player, cpu, particles } = gameData.current;

            // Clear
            ctx.clearRect(0, 0, width, height);

            // Draw Court (Professional Blue Table)
            // Vertical Gradient
            const bgGradient = ctx.createLinearGradient(0, 0, 0, height);
            bgGradient.addColorStop(0, '#172554'); // Blue-950 (Top/CPU side)
            bgGradient.addColorStop(0.5, '#1e3a8a'); // Blue-900 (Middle)
            bgGradient.addColorStop(1, '#172554'); // Blue-950 (Bottom/Player side)
            ctx.fillStyle = bgGradient;
            ctx.fillRect(0, 0, width, height);

            // Table Lines
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
            ctx.lineWidth = 4;

            // Middle Net Line (Horizontal)
            ctx.setLineDash([15, 15]);
            ctx.beginPath();
            ctx.moveTo(0, height / 2);
            ctx.lineTo(width, height / 2);
            ctx.stroke();
            ctx.setLineDash([]);

            // Center Circle
            ctx.beginPath();
            ctx.arc(width / 2, height / 2, 50, 0, Math.PI * 2);
            ctx.stroke();

            // Borders
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
            ctx.lineWidth = 6;
            ctx.strokeRect(0, 0, width, height);

            // Glow Effect for elements
            ctx.shadowBlur = 15;

            // Player Paddle (Neon Green) - Bottom
            ctx.shadowColor = '#34d399';
            ctx.fillStyle = '#34d399';
            const pY = height - 40;
            const pX = player.x - player.width / 2;
            ctx.beginPath();
            ctx.roundRect(pX, pY, player.width, player.height, 6);
            ctx.fill();

            // CPU Paddle (Neon Red) - Top
            ctx.shadowColor = '#f87171';
            ctx.fillStyle = '#f87171';
            const cY = 40;
            const cX = cpu.x - cpu.width / 2;
            ctx.beginPath();
            ctx.roundRect(cX, cY, cpu.width, cpu.height, 6);
            ctx.fill();

            // Particles
            particles.forEach(p => {
                ctx.globalAlpha = p.life;
                ctx.shadowBlur = 10 * p.life;
                ctx.shadowColor = p.color;
                ctx.fillStyle = p.color;
                ctx.beginPath();
                ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
                ctx.fill();
            });
            ctx.globalAlpha = 1.0;

            // Ball (Bright White/Yellow)
            ctx.shadowBlur = 20;
            ctx.shadowColor = '#fbbf24';
            ctx.fillStyle = '#fbbf24';
            ctx.beginPath();
            ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
            ctx.fill();

            // Reset shadows
            ctx.shadowBlur = 0;
        };

        const loop = () => {
            update();
            draw();
            animationFrameId = requestAnimationFrame(loop);
        };
        loop();

        // Mouse Move (Horizontal control)
        const moveHandler = (e) => {
            const rect = canvas.getBoundingClientRect();
            // Scale X based on width
            const scaleX = width / rect.width;
            let x = (e.clientX - rect.left) * scaleX;

            const { player } = gameData.current;
            // Clamp
            if (x < player.width / 2) x = player.width / 2;
            if (x > width - player.width / 2) x = width - player.width / 2;
            gameData.current.player.x = x;
        };

        const touchHandler = (e) => {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const scaleX = width / rect.width;
            const touch = e.touches[0];
            let x = (touch.clientX - rect.left) * scaleX;

            const { player } = gameData.current;
            if (x < player.width / 2) x = player.width / 2;
            if (x > width - player.width / 2) x = width - player.width / 2;
            gameData.current.player.x = x;
        };

        window.addEventListener('mousemove', moveHandler);
        canvas.addEventListener('touchmove', touchHandler, { passive: false });

        return () => {
            cancelAnimationFrame(animationFrameId);
            window.removeEventListener('mousemove', moveHandler);
            canvas.removeEventListener('touchmove', touchHandler);
        };
    }, [gameState]);

    return (
        <div className="flex flex-col items-center gap-4 w-full h-full justify-center relative">

            {/* Canvas Container - Vertical aspect ratio */}
            <div className="relative w-full max-w-md aspect-[3/4.5] bg-[#0f172a] rounded-xl overflow-hidden cursor-none shadow-2xl ring-4 ring-slate-800 border-4 border-slate-600">
                <canvas
                    ref={canvasRef}
                    width={450}
                    height={700}
                    className="w-full h-full object-contain block"
                />

                {gameState === 'menu' && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/80 backdrop-blur-sm z-40 transition-opacity p-8 text-center">
                        <h2 className="text-5xl font-black text-white mb-2 tracking-tighter italic drop-shadow-lg">
                            <span className="text-emerald-400">NEON</span> PONG
                        </h2>
                        <div className="w-32 h-2 bg-gradient-to-r from-emerald-400 to-rose-500 rounded-full mb-8"></div>

                        <button
                            onClick={() => setGameState('playing')}
                            className="group relative px-10 py-4 bg-white text-slate-900 font-black text-xl rounded-full overflow-hidden transition-transform active:scale-95 hover:scale-105 shadow-[0_0_40px_rgba(255,255,255,0.3)]"
                        >
                            <span className="relative z-10">START MATCH</span>
                            <div className="absolute inset-0 bg-gradient-to-r from-emerald-300 to-emerald-100 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                        </button>

                        <p className="mt-8 text-slate-400 font-mono text-sm">FIRST TO 11 WINS</p>
                    </div>
                )}

                {/* Scoreboard Overlay inside canvas container for cleaner look */}
                <div className="absolute top-4 left-0 right-0 flex justify-between px-8 text-4xl font-black font-mono z-30 select-none pointer-events-none">
                    <div className="flex flex-col items-center">
                        <span className="text-rose-500 drop-shadow-[0_0_15px_rgba(251,113,133,0.8)]">{score.cpu}</span>
                        <span className="text-[10px] text-rose-500 font-bold tracking-widest uppercase opacity-70">CPU</span>
                    </div>

                    <div className="flex flex-col items-center mt-auto absolute bottom-4 right-8">
                        <span className="text-emerald-400 drop-shadow-[0_0_15px_rgba(52,211,153,0.8)]">{score.player}</span>
                        <span className="text-[10px] text-emerald-500 font-bold tracking-widest uppercase opacity-70">YOU</span>
                    </div>
                </div>
            </div>

            <p className="text-slate-400 text-sm flex items-center gap-2 bg-slate-800/80 px-4 py-2 rounded-full backdrop-blur-sm border border-white/5 mt-2">
                <span className="bg-emerald-500/20 text-emerald-400 px-2 py-1 rounded text-xs font-bold border border-emerald-500/30">MOUSE</span>
                Move Horizontally to Defend
            </p>
        </div>
    );
};

export default TableTennisGame;
