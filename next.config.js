/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  images: {
    domains: ['localhost'],
  },
  // Configure API timeouts for audio processing
  experimental: {
    serverComponentsExternalPackages: ['tensorflow', 'faiss-node'],
    // Remove serverActions - now available by default
  },
  // Increase serverless function timeout for audio processing
  serverRuntimeConfig: {
    maxDuration: 60, // 60 seconds
  },
};

module.exports = nextConfig;
