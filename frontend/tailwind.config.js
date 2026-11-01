/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        studio: {
          bg: "#090B10",
          surface: "#0F131E",
          panel: "#151A28",
          border: "#1E2638",
          hover: "#263047",
          text: "#F8FAFC",
          muted: "#94A3B8",
          dim: "#64748B"
        },
        violet: {
          primary: "#7C3AED",
          hover: "#6D28D9",
          glow: "#8B5CF6",
          dim: "#4C1D95"
        },
        cyan: {
          accent: "#06B6D4",
          glow: "#22D3EE",
          dim: "#0891B2"
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'violet-glow': '0 0 25px -5px rgba(124, 58, 237, 0.4)',
        'cyan-glow': '0 0 25px -5px rgba(6, 182, 212, 0.35)',
        'studio-card': '0 8px 30px rgba(0, 0, 0, 0.45)',
      }
    },
  },
  plugins: [],
}
