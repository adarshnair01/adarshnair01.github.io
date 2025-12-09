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
        background: "var(--bg-primary)",
        "background-secondary": "var(--bg-secondary)",
        "text-main": "var(--text-main)",
        "text-muted": "var(--text-muted)",
        dark: "#0f172a",
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.5s ease-out',
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
      }
    },
    fontFamily: {
      sans: ['"Josefin Sans"', 'sans-serif'],
      harry: ['"Cinzel Decorative"', 'cursive'],
      body: ['"Josefin Sans"', 'sans-serif'],
    },
  },
  plugins: [],
}
