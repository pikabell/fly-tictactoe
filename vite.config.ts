import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// GitHub Pages serves a project site from /<repo>/, so the base path must match.
// Set VITE_BASE in CI (the workflow derives it from the repository name). Locally the
// default './' keeps `npm run preview` and a custom domain working unchanged.
export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE ?? './',
});
