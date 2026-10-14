/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL !== undefined ? process.env.NEXT_PUBLIC_API_URL : '',
  },
  // Dev server: accept all hosts (preview is proxied through a dynamic hostname)
  experimental: {
    allowedDevOrigins: ['*'],
  },
};

module.exports = nextConfig;
