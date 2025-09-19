/** @type {import('next').NextConfig} */
module.exports = {
  images: {
    domains: ['localhost'],
  },
  webpack: (config, { isServer }) => {
    // Handle TensorFlow.js bundle splitting
    config.resolve.alias = {
      ...config.resolve.alias,
      '@tensorflow/tfjs': '@tensorflow/tfjs/dist/index.js',
    };

    // Handle TensorFlow.js Node.js polyfills for client-side builds
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
        os: false,
        util: false,
        stream: false,
        assert: false,
        http: false,
        https: false,
        net: false,
        tls: false,
        zlib: false,
        url: false,
        child_process: false,
        worker_threads: false,
      };
    }

    return config;
  },
};
