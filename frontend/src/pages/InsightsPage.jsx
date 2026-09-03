import {
  Activity,
  BarChart3,
  Database,
  Gauge,
  HardDriveDownload,
  ShieldAlert,
  ShieldCheck,
  Trash2,
} from 'lucide-react'
import { formatDate, truncateUrl, verdictStyle } from '../lib/format'

function StatCard({ icon: Icon, label, value, sub, accent = 'text-accent-300' }) {
  return (
    <article className="glass glass-hover p-5">
      <div className="flex items-center justify-between">
        <span className="label-caps">
          {label}
        </span>
        <Icon className={`h-4 w-4 ${accent}`} />
      </div>
      <p className="stat-number-lg mt-3">{value}</p>
      {sub && <p className="mt-1 text-[11px] text-slate-500">{sub}</p>}
    </article>
  )
}

/** Simple horizontal distribution bar — no chart library needed for 3 buckets. */
function VerdictBar({ stats }) {
  const segments = [
    { key: 'legitimate', count: stats.legitimate, color: '#22d3a7', label: 'Legitimate' },
    { key: 'suspicious', count: stats.suspicious, color: '#f5b544', label: 'Suspicious' },
    { key: 'phishing', count: stats.phishing, color: '#ff5a6a', label: 'Phishing' },
  ]

  return (
    <div className="glass p-5">
      <h3 className="display-eyebrow text-slate-300">Verdict distribution</h3>
      <div className="mt-4 flex h-3 overflow-hidden rounded-full bg-white/10">
        {segments.map((segment) =>
          segment.count > 0 ? (
            <div
              key={segment.key}
              className="transition-all duration-700"
              style={{
                width: `${(segment.count / stats.total) * 100}%`,
                background: segment.color,
              }}
              title={`${segment.label}: ${segment.count}`}
            />
          ) : null,
        )}
      </div>
      <ul className="mt-4 grid grid-cols-3 gap-3">
        {segments.map((segment) => (
          <li key={segment.key} className="flex items-center gap-2">
            <span
              className="h-2 w-2 shrink-0 rounded-full"
              style={{ background: segment.color }}
            />
            <span className="min-w-0">
              <span className="stat-number block text-base">
                {segment.count}
              </span>
              <span className="label-caps block truncate text-[10px]">
                {segment.label}
              </span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default function InsightsPage({ history, onClear }) {
  const { entries, stats } = history

  return (
    <div className="mx-auto max-w-6xl animate-fade-up px-4 py-12 sm:px-6 lg:px-8">
      <header className="mb-8">
        <span className="badge border-accent-400/25 bg-accent-500/10 text-accent-200">
          <HardDriveDownload className="h-3 w-3" />
          Local Session Insights
        </span>
        <h1 className="section-title mt-5">Security Insights</h1>
        <p className="muted mt-3 max-w-3xl">
          PhishGuard has no database and the backend never stores a submitted URL. Everything
          below is computed from scans held in{' '}
          <strong className="font-semibold text-slate-300">this browser&apos;s local storage</strong>{' '}
          only. These are not global product statistics, and they are not visible to anyone
          else or on any other device.
        </p>
      </header>

      {stats.total === 0 ? (
        <div className="glass-strong glass-edge flex flex-col items-center px-6 py-16 text-center">
          <span className="flex h-14 w-14 items-center justify-center rounded-2xl border border-white/10 bg-white/5">
            <Database className="h-6 w-6 text-slate-500" />
          </span>
          <h2 className="mt-5 font-display text-[0.875rem] font-bold uppercase tracking-[0.08em] text-white">No local scans yet</h2>
          <p className="muted mt-2 max-w-md">
            Analyze a URL from the Scanner tab and its result will appear here. Nothing is
            sent anywhere or stored on the server.
          </p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <StatCard
              icon={BarChart3}
              label="URLs Analyzed"
              value={stats.total}
              sub="in this browser"
            />
            <StatCard
              icon={ShieldAlert}
              label="Phishing Detected"
              value={stats.phishing}
              sub={`${stats.suspicious} flagged suspicious`}
              accent="text-danger"
            />
            <StatCard
              icon={ShieldCheck}
              label="Assessed Legitimate"
              value={stats.legitimate}
              accent="text-safe"
            />
            <StatCard
              icon={Gauge}
              label="Average Risk"
              value={`${stats.averageRisk}%`}
              sub={`peak ${stats.highestRisk}%`}
              accent="text-caution"
            />
          </div>

          <div className="mt-4">
            <VerdictBar stats={stats} />
          </div>

          <section className="mt-8">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="display-eyebrow flex items-center gap-2.5 text-slate-300">
                <Activity className="h-4 w-4 text-accent-400" />
                Recent scans
              </h2>
              <button type="button" onClick={onClear} className="btn-ghost !px-3 !py-1.5 text-xs">
                <Trash2 className="h-3.5 w-3.5" />
                Clear history
              </button>
            </div>

            <div className="glass overflow-hidden">
              <ul className="divide-y divide-white/5">
                {entries.map((entry) => {
                  const style = verdictStyle(entry.prediction)
                  return (
                    <li
                      key={entry.id}
                      className="flex items-center gap-4 px-4 py-3 transition-colors hover:bg-white/[0.03] sm:px-5"
                    >
                      <span className={`h-2 w-2 shrink-0 rounded-full ${style.dot}`} />
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-mono text-[13px] text-slate-300" title={entry.url}>
                          {truncateUrl(entry.url, 70)}
                        </p>
                        <p className="mt-0.5 text-[11px] text-slate-600">
                          {formatDate(entry.scannedAt)}
                        </p>
                      </div>
                      <span
                        className={`hidden shrink-0 rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider sm:inline-block ${style.border} ${style.bg} ${style.text}`}
                      >
                        {style.label}
                      </span>
                      <span className="stat-number w-12 shrink-0 text-right text-sm">
                        {entry.riskScore}
                      </span>
                    </li>
                  )
                })}
              </ul>
            </div>
          </section>
        </>
      )}
    </div>
  )
}
