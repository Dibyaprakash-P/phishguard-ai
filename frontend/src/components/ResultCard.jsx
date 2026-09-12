import { useEffect, useState } from 'react'
import {
  Clock,
  Fingerprint,
  Gauge,
  Globe,
  ShieldAlert,
  ShieldCheck,
  ShieldX,
  Target,
} from 'lucide-react'
import { percent, riskLevelLabel, verdictStyle } from '../lib/format'

const VERDICT_ICONS = { ShieldCheck, ShieldAlert, ShieldX }

/** Circular risk gauge that animates from 0 to the real score once on mount. */
function RiskGauge({ score, ringClass }) {
  const [displayed, setDisplayed] = useState(0)

  useEffect(() => {
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    if (reduceMotion) {
      setDisplayed(score)
      return undefined
    }
    // Ease-out sweep to the true value. Purely presentational — `score` is
    // always the number returned by the API.
    const duration = 850
    const start = performance.now()
    let frame
    const step = (now) => {
      const progress = Math.min((now - start) / duration, 1)
      setDisplayed(Math.round(score * (1 - Math.pow(1 - progress, 3))))
      if (progress < 1) frame = requestAnimationFrame(step)
    }
    frame = requestAnimationFrame(step)
    return () => cancelAnimationFrame(frame)
  }, [score])

  const radius = 52
  const circumference = 2 * Math.PI * radius
  const offset = circumference * (1 - displayed / 100)

  return (
    <div className="relative flex h-32 w-32 shrink-0 items-center justify-center">
      <svg className="h-full w-full -rotate-90" viewBox="0 0 120 120" aria-hidden="true">
        <circle
          cx="60"
          cy="60"
          r={radius}
          className="fill-none stroke-white/10"
          strokeWidth="8"
        />
        <circle
          cx="60"
          cy="60"
          r={radius}
          className={`fill-none ${ringClass} transition-[stroke-dashoffset] duration-100`}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="stat-number text-3xl">{displayed}</span>
        <span className="label-caps text-[10px]">
          Risk
        </span>
      </div>
    </div>
  )
}

function Metric({ icon: Icon, label, value, sub }) {
  return (
    <div className="glass glass-hover p-4">
      <div className="flex items-center gap-2 text-slate-400">
        <Icon className="h-3.5 w-3.5" />
        <span className="label-caps text-slate-400">{label}</span>
      </div>
      <p className="stat-number mt-2 truncate text-lg" title={value}>
        {value}
      </p>
      {sub && <p className="mt-0.5 truncate text-[11px] text-slate-500">{sub}</p>}
    </div>
  )
}

export default function ResultCard({ result }) {
  const style = verdictStyle(result.prediction)
  const VerdictIcon = VERDICT_ICONS[style.icon] ?? ShieldAlert

  return (
    <section className="mt-8 animate-fade-up space-y-4">
      {/* Verdict banner */}
      <div className={`glass-strong glass-edge overflow-hidden border ${style.border} ${style.glow}`}>
        <div className="flex flex-col gap-6 p-6 sm:flex-row sm:items-center sm:gap-8 sm:p-8">
          <div
            className={`flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl border ${style.border} ${style.bg}`}
          >
            <VerdictIcon className={`h-8 w-8 ${style.text}`} strokeWidth={1.9} />
          </div>

          <div className="min-w-0 flex-1">
            <p
              className={`flex items-center gap-2 font-display text-[0.625rem] font-semibold uppercase tracking-[0.18em] ${style.text}`}
            >
              <span className={`h-1.5 w-1.5 rounded-full ${style.dot}`} />
              {style.headline}
            </p>
            <h2
              className={`mt-2.5 font-display text-[clamp(1.5rem,5vw,2.25rem)] font-extrabold uppercase leading-none tracking-[0.06em] ${style.text}`}
            >
              {style.label.toUpperCase()}
            </h2>
            <p
              className="mt-2 truncate font-mono text-xs text-slate-400"
              title={result.normalized_url}
            >
              {result.normalized_url}
            </p>
          </div>

          <RiskGauge score={result.risk_score} ringClass={style.ring} />
        </div>

        {/* Risk-band scale — makes the score interpretable rather than opaque */}
        <div className="border-t border-white/10 px-6 pb-5 pt-4 sm:px-8">
          <div className="flex items-center justify-between font-display text-[0.5625rem] font-medium uppercase tracking-[0.12em] text-slate-500">
            <span>Low 0–30</span>
            <span>Medium 31–70</span>
            <span>High 71–100</span>
          </div>
          <div className="relative mt-2 h-1.5 overflow-hidden rounded-full bg-white/10">
            <div
              className="absolute inset-y-0 left-0 rounded-full transition-all duration-700"
              style={{
                width: `${result.risk_score}%`,
                background: 'linear-gradient(90deg, #22d3a7 0%, #f5b544 55%, #ff5a6a 100%)',
              }}
            />
          </div>
          <p className="mt-2.5 text-[11px] leading-relaxed text-slate-500">
            Model-derived risk indicator from URL structure only — not a universal
            standard, and not a statement that the site was visited or verified.
          </p>
        </div>
      </div>

      {/* Key metrics */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Metric
          icon={Gauge}
          label="Risk Score"
          value={`${result.risk_score}%`}
          sub={`${riskLevelLabel(result.risk_level)} risk band`}
        />
        <Metric
          icon={Target}
          label="ML Confidence"
          value={percent(result.confidence)}
          sub={`in the "${result.prediction}" verdict`}
        />
        <Metric
          icon={Fingerprint}
          label="P(phishing)"
          value={percent(result.phishing_probability, 2)}
          sub={
            // When a reputation rule capped the score, say so and show what the
            // classifier said alone — a displayed probability that quietly
            // differs from the model's would be the dishonest option.
            result.reputation_domain
              ? `capped · model alone ${percent(result.model_probability, 2)}`
              : `threshold ${result.decision_threshold?.toFixed(3) ?? '—'}`
          }
        />
        <Metric
          icon={Globe}
          label="Domain"
          value={result.components.registrable_domain || '—'}
          sub={result.components.scheme === 'https' ? 'HTTPS' : 'No HTTPS'}
        />
      </div>

      <p className="flex items-center justify-center gap-1.5 text-[11px] text-slate-500">
        <Clock className="h-3 w-3" />
        Analyzed in {result.analysis_ms} ms by {result.model_name} v{result.model_version} ·
        static analysis only
      </p>
    </section>
  )
}
