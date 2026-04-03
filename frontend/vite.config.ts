import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    proxy: {
      '/ws': { target: 'ws://localhost:8000', ws: true, changeOrigin: true },
      '/songs': { target: 'http://localhost:8000', changeOrigin: true },
      '/start': { target: 'http://localhost:8000', changeOrigin: true },
      '/stop': { target: 'http://localhost:8000', changeOrigin: true },
      '/upload': { target: 'http://localhost:8000', changeOrigin: true },
      '/song': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
