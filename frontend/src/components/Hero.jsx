import { Brain, Cpu, Lock, Sparkles } from 'lucide-react'

const TRUST_POINTS = [
  { icon: Cpu, label: 'Gradient-boosted classifier' },
  { icon: Brain, label: 'LLM security reasoning' },
  { icon: Lock, label: 'Static analysis — never visits the URL' },
]

/**
 * Landing headline. Deliberately restrained: one badge, one heading, one
 * subtitle, and three factual capability points — the scanner below is the
 * actual product, so the hero should hand off to it quickly.
 */
export default function Hero() {
  return (
    <section className="mx-auto max-w-4xl px-4 pt-14 text-center sm:px-6 sm:pt-20">
      <div className="animate-fade-in">
        <span className="badge border-accent-400/25 bg-accent-500/10 text-accent-200">
          <Sparkles className="h-3 w-3" />
          AI-Powered Cybersecurity
        </span>
      </div>

      <h1 className="display-hero mt-8 animate-fade-up" style={{ animationDelay: '60ms' }}>
        <span className="block">Detect Phishing.</span>
        <span className="mt-1 block text-gradient">Protect Every Click.</span>
      </h1>

      <p
        className="mx-auto mt-7 max-w-2xl animate-fade-up text-balance text-[15px] leading-relaxed text-slate-400 sm:text-base"
        style={{ animationDelay: '140ms' }}
      >
        Analyze suspicious URLs using machine learning and AI-powered security reasoning.
      </p>

      <ul
        className="mx-auto mt-8 flex animate-fade-up flex-wrap items-center justify-center gap-x-6 gap-y-3"
        style={{ animationDelay: '220ms' }}
      >
        {TRUST_POINTS.map(({ icon: Icon, label }) => (
          <li key={label} className="flex items-center gap-2 font-display text-[0.625rem] uppercase tracking-[0.12em] text-slate-400">
            <Icon className="h-3.5 w-3.5 text-accent-400/80" />
            {label}
          </li>
        ))}
      </ul>
    </section>
  )
}
