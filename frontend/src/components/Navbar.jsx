import { useEffect, useState } from 'react'
import { Activity, BrainCircuit, Info, Menu, Radar, Shield, X } from 'lucide-react'

const NAV_ITEMS = [
  { id: 'scanner', label: 'Scanner', icon: Radar },
  { id: 'insights', label: 'Insights', icon: Activity },
  { id: 'model', label: 'Model', icon: BrainCircuit },
  { id: 'about', label: 'About', icon: Info },
]

/**
 * Sticky top navigation.
 *
 * Gains a stronger frosted backdrop once the page scrolls, so it stays legible
 * over the brighter middle of the background artwork without being an opaque
 * bar at rest.
 */
export default function Navbar({ view, onNavigate, health }) {
  const [scrolled, setScrolled] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const go = (id) => {
    onNavigate(id)
    setMenuOpen(false)
  }

  const modelOnline = health?.model_loaded
  const statusLabel = health
    ? modelOnline
      ? 'Model online'
      : 'Model offline'
    : 'Connecting…'

  return (
    <header
      className={`sticky top-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'border-b border-white/10 bg-ink-900/70 backdrop-blur-xl'
          : 'border-b border-transparent'
      }`}
    >
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
        {/* Brand */}
        <button
          type="button"
          onClick={() => go('scanner')}
          className="group flex items-center gap-3 rounded-xl"
          aria-label="PhishGuard AI home"
        >
          <span className="relative flex h-10 w-10 items-center justify-center rounded-xl border border-accent-400/30 bg-accent-500/15 shadow-glow-sm">
            <Shield className="h-5 w-5 text-accent-300" strokeWidth={2.2} />
          </span>
          <span className="flex flex-col items-start leading-none">
            <span className="brand-mark">Phishguard</span>
            <span className="brand-sub mt-1.5">AI</span>
          </span>
        </button>

        {/* Desktop links */}
        <div className="hidden items-center gap-1 md:flex">
          {NAV_ITEMS.map(({ id, label }) => (
            <button
              key={id}
              type="button"
              onClick={() => go(id)}
              className={`nav-link ${view === id ? 'nav-link-active' : ''}`}
              aria-current={view === id ? 'page' : undefined}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Status pill + mobile toggle */}
        <div className="flex items-center gap-3">
          <div
            className="hidden items-center gap-2 rounded-full border border-white/10 bg-white/[0.05] px-3 py-1.5 backdrop-blur-md sm:flex"
            title={
              health
                ? `API ${health.status} · LLM ${health.llm?.configured ? 'configured' : 'not configured'}`
                : 'Checking backend availability'
            }
          >
            <span className="relative flex h-2 w-2">
              {modelOnline && (
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-safe/60" />
              )}
              <span
                className={`relative inline-flex h-2 w-2 rounded-full ${
                  health ? (modelOnline ? 'bg-safe' : 'bg-danger') : 'bg-slate-500'
                }`}
              />
            </span>
            <span className="font-display text-[0.625rem] font-medium uppercase tracking-[0.14em] text-slate-300">
              {statusLabel}
            </span>
          </div>

          <button
            type="button"
            onClick={() => setMenuOpen((open) => !open)}
            className="rounded-lg border border-white/10 bg-white/[0.05] p-2 text-slate-300 backdrop-blur-md transition hover:text-white md:hidden"
            aria-label={menuOpen ? 'Close menu' : 'Open menu'}
            aria-expanded={menuOpen}
          >
            {menuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </nav>

      {/* Mobile drawer */}
      {menuOpen && (
        <div className="border-t border-white/10 bg-ink-900/85 backdrop-blur-xl md:hidden">
          <div className="mx-auto max-w-7xl space-y-1 px-4 py-3">
            {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                type="button"
                onClick={() => go(id)}
                className={`flex w-full items-center gap-3 rounded-xl px-3 py-3.5 text-left font-display text-[0.6875rem] font-medium uppercase tracking-[0.14em] transition ${
                  view === id
                    ? 'border border-accent-400/25 bg-accent-500/10 text-white'
                    : 'text-slate-400 hover:bg-white/5 hover:text-white'
                }`}
              >
                <Icon className="h-4 w-4" />
                {label}
              </button>
            ))}
          </div>
        </div>
      )}
    </header>
  )
}
