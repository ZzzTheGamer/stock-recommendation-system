/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
        './src/components/**/*.{js,ts,jsx,tsx,mdx}',
        './src/app/**/*.{js,ts,jsx,tsx,mdx}',
    ],
    theme: {
        extend: {
            colors: {
                primary: '#0070f3',
                secondary: '#1e293b',
                success: '#10b981',
                danger: '#ef4444',
                warning: '#f59e0b',
                info: '#3b82f6',
            },
        },
    },
    plugins: [],
}; 