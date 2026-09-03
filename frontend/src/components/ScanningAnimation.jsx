import { useEffect, useState } from 'react'
import { Brain, CheckCircle2, Cpu, Link2, Loader2, ShieldCheck } from 'lucide-react'

/**
 * Progress feedback shown while the API request is in flight.
 *
 * Important: this is *feedback*, not a simulation. The stage list mirrors the
 * real backend pipeline, and the component unmounts the moment the response
 * arrives — the displayed result is never derived from this animation, and the
 * animation never fabricates a verdict.
 */
const STAGES = [
  { icon: Link2, label: 'Validating URL structure' },
  { icon: Cpu, label: 'Extracting static features' },
  { icon: ShieldCheck, label: 'Running ML classifier' },
  { icon: Brain, label: 'Generating security analysis' },
]

export default function ScanningAnimation() {
  const [stage, setStage] = useState(0)

  useEffect(() => {
    // Advances through the stage labels while waiting. The final stage stays
    // lit until the real response replaces this component.
    const timer = setInterval(() => {
      setStage((current) => Math.min(current + 1, STAGES.length - 1))
    }, 900)
    return () => clearInterval(timer)
  }, [])

  return (
    <section
      className="glass-strong glass-edge mt-8 animate-fade-up overflow-hidden p-8 sm:p-12"
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <div className="flex flex-col items-center">
        {/* Shield with concentric pulse rings and a sweeping scan line */}
        <div className="relative flex h-32 w-32 items-center justify-center">
          <span className="absolute h-full w-full animate-pulse-ring rounded-full border border-accent-400/40" />
          <span
            className="absolute h-full w-full animate-pulse-ring rounded-full border border-accent-400/25"
            style={{ animationDelay: '0.8s' }}
          />
          <span
            className="absolute h-full w-full animate-pulse-ring rounded-full border border-accent-400/15"
            style={{ animationDelay: '1.6s' }}
          />

          <div className="relative flex h-20 w-20 items-center justify-center overflow-hidden rounded-2xl border border-accent-400/30 bg-accent-500/10 shadow-glow">
            <ShieldCheck className="h-9 w-9 animate-float text-accent-300" strokeWidth={1.8} />
            <span
              aria-hidden="true"
              className="absolute inset-x-0 h-10 animate-scanline"
              style={{
                background:
                  'linear-gradient(180deg, transparent, rgba(44,192,255,0.55), transparent)',
              }}
            />
          </div>
        </div>

        <h2 className="display-heading mt-7">
          Analyzing URL…
        </h2>
        <p className="mt-1.5 text-sm text-slate-400">
          Static analysis only — the site is not being visited.
        </p>

        {/* Pipeline stages */}
        <ol className="mt-8 w-full max-w-sm space-y-2.5">
          {STAGES.map(({ icon: Icon, label }, index) => {
            const done = index < stage
            const active = index === stage
            return (
              <li
                key={label}
                className={`flex items-center gap-3 rounded-xl border px-3.5 py-2.5 text-sm transition-all duration-500 ${
                  active
                    ? 'border-accent-400/30 bg-accent-500/10 text-white'
                    : done
                      ? 'border-white/10 bg-white/[0.03] text-slate-400'
                      : 'border-white/5 bg-transparent text-slate-600'
                }`}
              >
                {done ? (
                  <CheckCircle2 className="h-4 w-4 shrink-0 text-safe" />
                ) : active ? (
                  <Loader2 className="h-4 w-4 shrink-0 animate-spin text-accent-300" />
                ) : (
                  <Icon className="h-4 w-4 shrink-0" />
                )}
                <span className="truncate">{label}</span>
              </li>
            )
          })}
        </ol>
      </div>
    </section>
  )
}
