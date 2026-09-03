import {
  Activity,
  Binary,
  FolderTree,
  Globe,
  Hash,
  KeyRound,
  Layers,
  ListFilter,
  Lock,
  Network,
  Percent,
  Ruler,
  Server,
} from 'lucide-react'
import { severityStyle } from '../lib/format'

/** Icon names arrive from the backend so the API stays the source of truth. */
const ICONS = {
  Ruler,
  Globe,
  Network,
  Hash,
  Binary,
  Lock,
  Server,
  KeyRound,
  Activity,
  FolderTree,
  ListFilter,
  Percent,
}

/**
 * Extracted URL characteristics, one glass card per measurement.
 *
 * Every value here is a measurement of the URL *string*. None of it comes from
 * contacting the site.
 */
export default function FeatureGrid({ highlights }) {
  if (!highlights?.length) return null

  return (
    <section className="mt-8 animate-fade-up">
      <header className="mb-4 flex items-center gap-2.5">
        <Layers className="h-4 w-4 text-accent-400" />
        <h3 className="display-eyebrow text-slate-300">
          Extracted Characteristics
        </h3>
      </header>

      <div className="grid-fade-in grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {highlights.map((feature, index) => {
          const Icon = ICONS[feature.icon] ?? Activity
          const tone = severityStyle(feature.tone)
          return (
            <article
              key={feature.key}
              className="glass glass-hover group p-4"
              style={{ animationDelay: `${Math.min(index * 45, 400)}ms` }}
            >
              <div className="flex items-start justify-between gap-2">
                <span
                  className={`flex h-8 w-8 items-center justify-center rounded-lg border ${tone.border} ${tone.bg} transition-transform duration-300 group-hover:scale-105`}
                >
                  <Icon className={`h-4 w-4 ${tone.text}`} strokeWidth={1.9} />
                </span>
                {feature.tone !== 'info' && (
                  <span
                    className={`rounded-full border px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider ${tone.border} ${tone.bg} ${tone.text}`}
                  >
                    {tone.label}
                  </span>
                )}
              </div>

              <p className="label-caps mt-3">
                {feature.label}
              </p>
              <p className="stat-number mt-1.5 text-xl">{feature.value}</p>
              <p className="mt-2 text-[11px] leading-snug text-slate-500">
                {feature.description}
              </p>
            </article>
          )
        })}
      </div>
    </section>
  )
}
