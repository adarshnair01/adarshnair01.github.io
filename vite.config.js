import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages skips files whose names start with "_" (a Jekyll rule), so build
// output must never use such names. public/.nojekyll turns the rule off as well;
// this keeps things working even if that file is ever lost.
const fileName = (chunk) => (chunk.name || 'chunk').replace(/^_+/, '') || 'chunk'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        chunkFileNames: (chunk) => `assets/${fileName(chunk)}-[hash].js`,
        assetFileNames: (asset) => {
          const base = (asset.names?.[0] || asset.name || 'asset').replace(/^_+/, '')
          return `assets/${base.replace(/\.[^.]+$/, '')}-[hash][extname]`
        },
      },
    },
  },
})
