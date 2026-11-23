import { StrictMode, Suspense, lazy } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Tools live on their own paths and load their own code, so the portfolio stays light.
// GitHub Pages serves 404.html (a copy of index.html) for these paths.
const TravelEditorPage = lazy(() => import('./travel/TravelEditorPage.jsx'))

const path = window.location.pathname.replace(/\/+$/, '')

const page = path === '/travel_edit/route' ? (
  <Suspense fallback={<div className="min-h-screen bg-night" />}>
    <TravelEditorPage />
  </Suspense>
) : (
  <App />
)

createRoot(document.getElementById('root')).render(
  <StrictMode>
    {page}
  </StrictMode>,
)
