import { Component, StrictMode, Suspense, lazy } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// After a deploy, a cached page can ask for code files that no longer exist.
// Reload once to pick up the new page; if it still fails, let the error show.
const RELOAD_KEY = 'reloaded-for-new-version'
function reloadOnce(error) {
  try {
    if (!sessionStorage.getItem(RELOAD_KEY)) {
      sessionStorage.setItem(RELOAD_KEY, '1')
      window.location.reload()
      return new Promise(() => {})
    }
    sessionStorage.removeItem(RELOAD_KEY)
  } catch {
    // storage blocked: fall through to the error screen
  }
  throw error
}
// Tools live on their own paths and load their own code, so the portfolio stays light.
// GitHub Pages serves 404.html (a copy of index.html) for these paths.
const TravelEditorPage = lazy(() =>
  import('./travel/TravelEditorPage.jsx')
    .then((mod) => {
      try { sessionStorage.removeItem(RELOAD_KEY) } catch { /* ignore */ }
      return mod
    })
    .catch(reloadOnce),
)

// Show what went wrong instead of a blank page.
class ToolErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    return (
      <div className="min-h-screen bg-night text-snow flex items-center justify-center p-6">
        <div className="max-w-md space-y-4 text-center">
          <h1 className="font-display text-2xl font-bold">This tool couldn&apos;t start</h1>
          <p className="text-text-muted">Please reload the page. If it keeps happening, the message below says what went wrong.</p>
          <pre className="whitespace-pre-wrap break-words rounded-xl bg-night-card p-3 text-left text-xs text-frost">{String(error?.message || error)}</pre>
          <div className="flex justify-center gap-3">
            <button type="button" onClick={() => window.location.reload()} className="rounded-full bg-holly px-5 py-2 font-bold text-snow hover:bg-holly-deep">Reload</button>
            <a href="/" className="rounded-full border border-frost/30 px-5 py-2 font-bold text-snow hover:bg-frost/10">Back to the site</a>
          </div>
        </div>
      </div>
    )
  }
}

const path = window.location.pathname.replace(/\/+$/, '')

const page = path === '/travel_edit/route' ? (
  <ToolErrorBoundary>
    <Suspense fallback={<div className="min-h-screen bg-night" />}>
      <TravelEditorPage />
    </Suspense>
  </ToolErrorBoundary>
) : (
  <App />
)

createRoot(document.getElementById('root')).render(
  <StrictMode>
    {page}
  </StrictMode>,
)
