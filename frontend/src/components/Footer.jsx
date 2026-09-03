import { EyeOff, Github, Shield } from 'lucide-react'

export default function Footer({ health }) {
  return (
    <footer className="border-t border-white/10 bg-ink-900/50 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-8 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
        <div className="flex items-center gap-3">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-accent-400/25 bg-accent-500/10">
            <Shield className="h-4 w-4 text-accent-300" strokeWidth={2.2} />
          </span>
          <div>
            <p className="brand-mark text-[0.8125rem]">Phishguard AI</p>
            <p className="mt-0.5 text-[11px] text-slate-500">
              ML-driven phishing URL detection with explainable AI reasoning
            </p>
          </div>
        </div>

        <p className="flex items-center gap-2 text-[11px] text-slate-500">
          <EyeOff className="h-3.5 w-3.5 shrink-0 text-safe/70" />
          Static analysis only — submitted URLs are never visited or stored
        </p>

        <div className="flex items-center gap-4 text-[11px] text-slate-500">
          {health?.model_name && (
            <span className="font-mono">
              {health.model_name} · v{health.version}
            </span>
          )}
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer noopener"
            className="flex items-center gap-1.5 transition hover:text-slate-300"
          >
            <Github className="h-3.5 w-3.5" />
            Source
          </a>
        </div>
      </div>
    </footer>
  )
}
