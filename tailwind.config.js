/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "rgb(var(--primary) / <alpha-value>)",
        secondary: "rgb(var(--secondary) / <alpha-value>)",
        background: "rgb(var(--bg-primary) / <alpha-value>)",
        "background-secondary": "rgb(var(--bg-secondary) / <alpha-value>)",
        "text-main": "rgb(var(--text-main) / <alpha-value>)",
        "text-muted": "rgb(var(--text-muted) / <alpha-value>)",
        // Christmas & New Year palette
        night: "#0a1324",
        "night-soft": "#111d36",
        "night-card": "#16233f",
        holly: "#d4243b",
        "holly-deep": "#9e1428",
        pine: "#1f8a55",
        "pine-deep": "#0f5132",
        gold: "#f0c34e",
        "gold-deep": "#b8891c",
        snow: "#f4f7fc",
        frost: "#a9d4f5",
        cream: "#fbf4e4",
        "cream-ink": "#2b1d14",
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.5s ease-out',
        'spin-slow': 'spin 40s linear infinite',
        'twinkle': 'twinkle 2.2s ease-in-out infinite',
        'sway': 'sway 4s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        twinkle: {
          '0%, 100%': { opacity: '1', filter: 'brightness(1.15)' },
          '50%': { opacity: '0.35', filter: 'brightness(0.7)' },
        },
        sway: {
          '0%, 100%': { transform: 'rotate(-4deg)' },
          '50%': { transform: 'rotate(4deg)' },
        },
      }
    },
    fontFamily: {
      sans: ['"Nunito"', 'sans-serif'],
      body: ['"Nunito"', 'sans-serif'],
      display: ['"Playfair Display"', 'serif'],
      serif: ['"Playfair Display"', 'serif'],
      script: ['"Great Vibes"', 'cursive'],
      mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
    },
  },
  plugins: [],
}
