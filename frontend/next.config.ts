import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  webpack: (config) => {
    // Fix Fast Refresh loop in Docker/WSL
    config.watchOptions = {
      poll: 100000, // Check files every 1000ms
      aggregateTimeout: 300, // Wait 300ms after change before rebuild
    };
    return config;
  },
};

export default nextConfig;