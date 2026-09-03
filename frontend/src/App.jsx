import { useCallback, useEffect, useState } from 'react'
import AppBackground from './components/AppBackground'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import ScannerPage from './pages/ScannerPage'
import InsightsPage from './pages/InsightsPage'
import ModelPage from './pages/ModelPage'
import AboutPage from './pages/AboutPage'
import useScanHistory from './hooks/useScanHistory'
import { api } from './lib/api'

const VIEWS = ['scanner', 'insights', 'model', 'about']

/**
 * Application shell.
 *
 * View state is kept in the URL hash so a view can be linked and the browser
 * back button behaves as users expect, without pulling in a router dependency
 * for four static views.
 */
function viewFromHash() {
  const hash = window.location.hash.replace('#', '')
  return VIEWS.includes(hash) ? hash : 'scanner'
}

export default function App() {
  const [view, setView] = useState(viewFromHash)
  const [health, setHealth] = useState(null)
  const [analysis, setAnalysis] = useState({ result: null, loading: false, error: null })

  const history = useScanHistory()

  // Probe the backend once on mount so the header can show real status and the
  // scanner can warn when no model is loaded.
  useEffect(() => {
    let cancelled = false
    api
      .health()
      .then((data) => !cancelled && setHealth(data))
      .catch(() => !cancelled && setHealth(null))
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    const onHashChange = () => setView(viewFromHash())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const navigate = useCallback((next) => {
    window.location.hash = next === 'scanner' ? '' : next
    setView(next)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }, [])

  const analyze = useCallback(
    async (url) => {
      setAnalysis({ result: null, loading: true, error: null })
      try {
        const result = await api.analyze(url)
        setAnalysis({ result, loading: false, error: null })
        history.record(result)
      } catch (error) {
        setAnalysis({ result: null, loading: false, error })
      }
    },
    [history],
  )

  const reset = useCallback(() => {
    setAnalysis({ result: null, loading: false, error: null })
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }, [])

  return (
    <AppBackground>
      <div className="flex min-h-screen flex-col">
        <Navbar view={view} onNavigate={navigate} health={health} />

        <main className="flex-1">
          {view === 'scanner' && (
            <ScannerPage
              health={health}
              analysis={analysis}
              onAnalyze={analyze}
              onReset={reset}
            />
          )}
          {view === 'insights' && <InsightsPage history={history} onClear={history.clear} />}
          {view === 'model' && <ModelPage />}
          {view === 'about' && <AboutPage />}
        </main>

        <Footer health={health} />
      </div>
    </AppBackground>
  )
}
