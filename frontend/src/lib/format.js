/** Presentation helpers shared across components. */

/** Verdict -> visual treatment. Keys match the backend `prediction` enum. */
export const VERDICT_STYLES = {
  legitimate: {
    label: 'Legitimate',
    headline: 'No Threat Detected',
    icon: 'ShieldCheck',
    text: 'text-safe',
    border: 'border-safe/35',
    bg: 'bg-safe/10',
    glow: 'shadow-glow-safe',
    ring: 'stroke-safe',
    dot: 'bg-safe',
  },
  suspicious: {
    label: 'Suspicious',
    headline: 'Potentially Unsafe',
    icon: 'ShieldAlert',
    text: 'text-caution',
    border: 'border-caution/35',
    bg: 'bg-caution/10',
    glow: 'shadow-[0_0_36px_rgba(245,181,68,0.3)]',
    ring: 'stroke-caution',
    dot: 'bg-caution',
  },
  phishing: {
    label: 'Phishing',
    headline: 'Security Threat Detected',
    icon: 'ShieldX',
    text: 'text-danger',
    border: 'border-danger/35',
    bg: 'bg-danger/10',
    glow: 'shadow-glow-danger',
    ring: 'stroke-danger',
    dot: 'bg-danger',
  },
}

export const SEVERITY_STYLES = {
  high: { text: 'text-danger', bg: 'bg-danger/10', border: 'border-danger/30', label: 'High' },
  medium: { text: 'text-caution', bg: 'bg-caution/10', border: 'border-caution/30', label: 'Medium' },
  low: { text: 'text-accent-300', bg: 'bg-accent-400/10', border: 'border-accent-400/25', label: 'Low' },
  info: { text: 'text-slate-400', bg: 'bg-white/5', border: 'border-white/10', label: 'Info' },
}

export const verdictStyle = (prediction) =>
  VERDICT_STYLES[prediction] ?? VERDICT_STYLES.suspicious

export const severityStyle = (severity) =>
  SEVERITY_STYLES[severity] ?? SEVERITY_STYLES.info

/** 0.9642 -> "96.4%" */
export const percent = (value, digits = 1) =>
  value === null || value === undefined || Number.isNaN(value)
    ? '—'
    : `${(value * 100).toFixed(digits)}%`

export const riskLevelLabel = (level) =>
  ({ low: 'Low', medium: 'Medium', high: 'High' })[level] ?? '—'

/** Shorten a URL for display without hiding the part that matters (the host). */
export function truncateUrl(url, max = 62) {
  if (!url || url.length <= max) return url
  return `${url.slice(0, max - 12)}…${url.slice(-11)}`
}

export function formatDate(iso) {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export const compactNumber = (value) =>
  typeof value === 'number' ? new Intl.NumberFormat(undefined, { notation: 'compact' }).format(value) : '—'

export const fullNumber = (value) =>
  typeof value === 'number' ? new Intl.NumberFormat().format(value) : '—'
