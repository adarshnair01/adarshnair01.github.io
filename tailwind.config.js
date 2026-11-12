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
        // Ramayana palette
        saffron: "#f08a24",
        kumkum: "#b3261e",
        gold: "#e2b33c",
        "gold-deep": "#a87a12",
        ink: "#140c08",
        "ink-soft": "#21150e",
        "ink-card": "#2a1a10",
        palm: "#e9d6a8",
        "palm-deep": "#c9ad72",
        "palm-ink": "#3b2412",
        ocean: "#0f3b4a",
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.5s ease-out',
        'spin-slow': 'spin 40s linear infinite',
        'spin-slower': 'spin 90s linear infinite',
        'flicker': 'flicker 2.4s ease-in-out infinite',
        'wave': 'wave 6s ease-in-out infinite',
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
        flicker: {
          '0%, 100%': { transform: 'scaleY(1) rotate(-1deg)', opacity: '1' },
          '30%': { transform: 'scaleY(1.12) rotate(2deg)', opacity: '0.92' },
          '60%': { transform: 'scaleY(0.94) rotate(-2deg)', opacity: '1' },
        },
        wave: {
          '0%, 100%': { transform: 'translateX(0)' },
          '50%': { transform: 'translateX(-25px)' },
        },
      }
    },
    fontFamily: {
      sans: ['"Mukta"', 'sans-serif'],
      body: ['"Mukta"', 'sans-serif'],
      epic: ['"Yatra One"', 'serif'],
      serif: ['"Cormorant Garamond"', 'serif'],
      sanskrit: ['"Tiro Devanagari Sanskrit"', 'serif'],
      mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
    },
  },
  plugins: [],
}
