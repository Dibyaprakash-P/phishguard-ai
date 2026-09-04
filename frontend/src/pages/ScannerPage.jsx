import { AlertTriangle, RotateCcw } from 'lucide-react'
import Hero from '../components/Hero'
import UrlScanner from '../components/UrlScanner'
import ScanningAnimation from '../components/ScanningAnimation'
import ResultCard from '../components/ResultCard'
import FeatureGrid from '../components/FeatureGrid'
import IndicatorList from '../components/IndicatorList'
import AIExplanation from '../components/AIExplanation'

export default function ScannerPage({ health, analysis, onAnalyze, onReset }) {
  const { result, loading, error } = analysis
  const modelOffline = health && !health.model_loaded

  return (
    <>
      <Hero />

      <UrlScanner
        onAnalyze={onAnalyze}
        loading={loading}
        disabled={loading || modelOffline}
        disabledReason={
          modelOffline
            ? 'The detection model is not loaded on the backend. Run "python -m ml.train" and restart the API.'
            : null
        }
      />

      <div className="mx-auto max-w-5xl px-4 pb-20 sm:px-6 lg:px-8">
        {error && !loading && (
          <div
            role="alert"
            className="glass-strong mt-8 flex animate-fade-up items-start gap-3 border border-danger/25 p-5"
          >
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-danger" />
            <div className="min-w-0 flex-1">
              <h2 className="display-card-title">Analysis failed</h2>
              <p className="muted mt-1">{error.message}</p>
              {error.detail && (
                <p className="mt-1 font-mono text-[11px] text-slate-600">{error.detail}</p>
              )}
            </div>
          </div>
        )}

        {loading && <ScanningAnimation />}

        {result && !loading && (
          <>
            <ResultCard result={result} />
            <AIExplanation explanation={result.ai_explanation} />
            <IndicatorList indicators={result.suspicious_indicators} />
            <FeatureGrid highlights={result.feature_highlights} />

            <div className="mt-10 flex justify-center">
              <button type="button" onClick={onReset} className="btn-ghost">
                <RotateCcw className="h-4 w-4" />
                Analyze another URL
              </button>
            </div>
          </>
        )}
      </div>
    </>
  )
}
