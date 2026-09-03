/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        // Electric-blue accent ramp, tuned against the supplied background art.
        accent: {
          50: '#eff9ff',
          100: '#def2ff',
          200: '#b6e7ff',
          300: '#75d6ff',
          400: '#2cc0ff',
          500: '#03a6f0',
          600: '#0084ce',
          700: '#0069a6',
          800: '#065989',
          900: '#0b4a71',
        },
        ink: {
          900: '#03060f',
          800: '#060b18',
          700: '#0a1122',
          600: '#0f1930',
        },
        safe: '#22d3a7',
        caution: '#f5b544',
        danger: '#ff5a6a',
      },
      fontFamily: {
        // Display: geometric extended sci-fi face, used for the wordmark,
        // headings, key figures and buttons. Never for running prose.
        display: ['Orbitron', 'Michroma', 'Eurostile', 'Arial Narrow', 'sans-serif'],
        // Body: everything a reader actually reads in sentences.
        sans: ['Manrope', 'Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      // Tracking scale, tuned per role. Orbitron is already an extended face,
      // so large display text needs LESS added spacing than small caps text:
      // spacing that reads as deliberate at 12px reads as broken at 60px.
      letterSpacing: {
        display: '0.03em',
        heading: '0.06em',
        button: '0.1em',
        nav: '0.14em',
        label: '0.16em',
        brand: '0.2em',
      },
      backdropBlur: {
        xs: '2px',
      },
      boxShadow: {
        glass: '0 8px 32px rgba(0, 0, 0, 0.45)',
        'glow-sm': '0 0 18px rgba(44, 192, 255, 0.28)',
        glow: '0 0 32px rgba(44, 192, 255, 0.32)',
        'glow-lg': '0 0 60px rgba(44, 192, 255, 0.35)',
        'glow-danger': '0 0 36px rgba(255, 90, 106, 0.35)',
        'glow-safe': '0 0 36px rgba(34, 211, 167, 0.32)',
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(14px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-9px)' },
        },
        scanline: {
          '0%': { transform: 'translateY(-120%)', opacity: '0' },
          '15%': { opacity: '1' },
          '85%': { opacity: '1' },
          '100%': { transform: 'translateY(120%)', opacity: '0' },
        },
        'pulse-ring': {
          '0%': { transform: 'scale(0.85)', opacity: '0.55' },
          '100%': { transform: 'scale(1.5)', opacity: '0' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-500px 0' },
          '100%': { backgroundPosition: '500px 0' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.55s cubic-bezier(0.22, 1, 0.36, 1) both',
        'fade-in': 'fade-in 0.5s ease-out both',
        float: 'float 5s ease-in-out infinite',
        scanline: 'scanline 1.9s cubic-bezier(0.45, 0, 0.55, 1) infinite',
        'pulse-ring': 'pulse-ring 2.4s cubic-bezier(0.22, 1, 0.36, 1) infinite',
        shimmer: 'shimmer 2.2s linear infinite',
      },
    },
  },
  plugins: [],
}
