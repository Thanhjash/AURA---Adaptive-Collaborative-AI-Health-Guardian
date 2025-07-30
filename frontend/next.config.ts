/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,   // ✅ skip ESLint on Vercel
  },
};

export default nextConfig;
