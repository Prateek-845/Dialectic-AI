/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        'bg-primary': '#09090b',
        'card-bg': '#18181b',
        'border-color': '#27272a',
        'accent-blue': '#3b82f6',
        'text-primary': '#e4e4e7',
      }
    },
  },
  plugins: [],
}
