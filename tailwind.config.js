/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      animation: {
        "fade-in": "fadeIn 0.5s ease-in-out",
        'slide-in-right': 'slideInRight 0.3s ease-out',
        "progress-bar": "progressBar 2s infinite",
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'orbit': 'orbit 3s linear infinite',
        'slide-up': 'slideUp 0.5s ease-out',
        'fade-out': 'fade-out 0.5s ease-out forwards',
  'fill-progress': 'fill-progress 2s ease-in-out forwards',
  'typing-effect': 'typing-effect 1s steps(20, end) forwards',
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideInRight: {
          '0%': { transform: 'translateX(100%)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        progressBar: {
          "0%": { width: "0%" },
          "50%": { width: "70%" },
          "100%": { width: "100%" },
        },
        orbit: {
          '0%': { transform: 'rotate(0deg) translateX(40px) rotate(0deg)' },
          '100%': { transform: 'rotate(360deg) translateX(40px) rotate(-360deg)' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        'fade-out': {
    '0%': { opacity: '1' },
    '100%': { opacity: '0', visibility: 'hidden' },
  },
  'fill-progress': {
    '0%': { strokeDasharray: '0 289' },
    '100%': { strokeDasharray: 'var(--progress, 289) 289' },
  },
  'typing-effect': {
    '0%': { width: '0', overflow: 'hidden' },
    '100%': { width: '100%', overflow: 'visible' },
  },
      },
    },
  },
  plugins: [],
};
