import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // ✅  Bỏ qua lỗi ESLint khi build production trên Vercel
  eslint: {
    ignoreDuringBuilds: true,
  },

};

export default nextConfig;
