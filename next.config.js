/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    images: {
        domains: ['localhost', '127.0.0.1'],
    },
    async rewrites() {
        return [
            {
                source: '/static/:path*',
                destination: 'http://localhost:5000/static/:path*',
            },
            {
                source: '/api/:path*',
                destination: 'http://localhost:5000/api/:path*',
            },
        ];
    },
    serverRuntimeConfig: {
        apiTimeout: 1800000,
    },
    staticPageGenerationTimeout: 1800,
    httpAgentOptions: {
        keepAlive: true,
        timeout: 1800000,
    },
    env: {
        NEXT_PUBLIC_ALPHA_VANTAGE_API_KEY: process.env.ALPHA_VANTAGE_API_KEY || 'YOUR_ALPHA_VANTAGE_API_KEY',
        MISTRAL_API_KEY: process.env.MISTRAL_API_KEY || 'YOUR_MISTRAL_API_KEY',
        OPENAI_API_KEY: process.env.OPENAI_API_KEY || 'YOUR_OPENAI_API_KEY',
    },
    output: 'standalone',
};

module.exports = nextConfig; 