import { AlertTriangle, CheckCircle2, ShieldAlert } from 'lucide-react'
import { severityStyle } from '../lib/format'

/**
 * Observed red flags.
 *
 * Each entry is a fact about the URL text, with the literal evidence that
 * triggered it — deliberately distinct from the model's probabilistic verdict,
 * so a reader can see what was actually observed versus what was predicted.
 */
export default function IndicatorList({ indicators }) {
  const empty = !indicators?.length

  return (
    <section className="mt-8 animate-fade-up">
      <header className="mb-4 flex items-center gap-2.5">
        <ShieldAlert className="h-4 w-4 text-accent-400" />
        <h3 className="display-eyebrow text-slate-300">
          Observed Indicators
        </h3>
        {!empty && (
          <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[11px] font-semibold text-slate-400">
            {indicators.length}
          </span>
        )}
      </header>

      {empty ? (
        <div className="glass flex items-center gap-3 p-5">
          <CheckCircle2 className="h-5 w-5 shrink-0 text-safe" />
          <p className="text-sm text-slate-400">
            No red-flag patterns were found in the URL string. This is an absence of
            evidence, not a guarantee of safety.
          </p>
        </div>
      ) : (
        <ul className="space-y-2.5">
          {indicators.map((indicator, index) => {
            const tone = severityStyle(indicator.severity)
            return (
              <li
                key={indicator.code}
                className={`glass glass-hover border-l-2 p-4 ${tone.border}`}
                style={{ animationDelay: `${Math.min(index * 50, 350)}ms` }}
              >
                <div className="flex items-start gap-3">
                  <AlertTriangle className={`mt-0.5 h-4 w-4 shrink-0 ${tone.text}`} />
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <h4 className="display-item-title">{indicator.title}</h4>
                      <span
                        className={`rounded-full border px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider ${tone.border} ${tone.bg} ${tone.text}`}
                      >
                        {tone.label}
                      </span>
                    </div>
                    <p className="mt-1.5 text-[13px] leading-relaxed text-slate-400">
                      {indicator.description}
                    </p>
                    {indicator.evidence && (
                      <p className="mt-2 break-all font-mono text-[11px] text-slate-500">
                        <span className="text-slate-600">Evidence: </span>
                        {indicator.evidence}
                      </p>
                    )}
                  </div>
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
