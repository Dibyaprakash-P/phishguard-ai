import { useEffect, useState } from 'react'
import {
  AlertTriangle,
  ClipboardCheck,
  SlidersHorizontal,
  BarChart3,
  Boxes,
  BrainCircuit,
  CheckCircle2,
  Cpu,
  Database,
  GitBranch,
  Loader2,
  Radar,
  Scale,
  Target,
} from 'lucide-react'
import { api } from '../lib/api'
import { formatDate, fullNumber, percent } from '../lib/format'

function MetricTile({ icon: Icon, label, value, hint }) {
  return (
    <article className="glass glass-hover p-5">
      <div className="flex items-center justify-between">
        <span className="label-caps">
          {label}
        </span>
        <Icon className="h-4 w-4 text-accent-400" />
      </div>
      <p className="stat-number-lg mt-3">{value}</p>
      {hint && <p className="mt-1.5 text-[11px] leading-snug text-slate-500">{hint}</p>}
    </article>
  )
}

function ComparisonTable({ comparison, selected }) {
  if (!comparison?.length) return null
  return (
    <div className="glass overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead>
            <tr className="label-caps border-b border-white/10 text-[10px]">
              <th className="px-5 py-3 font-semibold">Model</th>
              <th className="px-4 py-3 text-right font-semibold">Accuracy</th>
              <th className="px-4 py-3 text-right font-semibold">Precision</th>
              <th className="px-4 py-3 text-right font-semibold">Recall</th>
              <th className="px-4 py-3 text-right font-semibold">F1</th>
              <th className="px-4 py-3 text-right font-semibold">ROC-AUC</th>
              <th className="px-4 py-3 text-right font-semibold">PR-AUC</th>
              <th className="px-5 py-3 text-right font-semibold">Train</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {comparison.map((row) => {
              const isSelected = row.model === selected
              return (
                <tr
                  key={row.model}
                  className={isSelected ? 'bg-accent-500/[0.07]' : 'transition-colors hover:bg-white/[0.03]'}
                >
                  <td className="px-5 py-3.5">
                    <span className="flex items-center gap-2">
                      {isSelected && <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-accent-300" />}
                      <span className={isSelected ? 'font-semibold text-white' : 'text-slate-300'}>
                        {row.model.replace(/_/g, ' ')}
                      </span>
                    </span>
                  </td>
                  <td className="px-4 py-3.5 text-right tabular-nums text-slate-300">{percent(row.accuracy, 2)}</td>
                  <td className="px-4 py-3.5 text-right tabular-nums text-slate-300">{percent(row.precision, 2)}</td>
                  <td className="px-4 py-3.5 text-right tabular-nums text-slate-300">{percent(row.recall, 2)}</td>
                  <td className="px-4 py-3.5 text-right tabular-nums text-slate-300">{percent(row.f1, 2)}</td>
                  <td className="px-4 py-3.5 text-right tabular-nums text-slate-300">{row.roc_auc.toFixed(4)}</td>
                  <td className="px-4 py-3.5 text-right tabular-nums text-slate-300">{row.pr_auc.toFixed(4)}</td>
                  <td className="px-5 py-3.5 text-right tabular-nums text-slate-500">
                    {row.train_seconds ? `${row.train_seconds.toFixed(0)}s` : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="border-t border-white/10 px-5 py-3 text-[11px] text-slate-500">
        Comparison metrics are measured on the <strong className="text-slate-400">validation</strong>{' '}
        split, which is what model selection used. The headline figures above come from the
        test split, which was scored exactly once.
      </p>
    </div>
  )
}

function FeatureImportance({ features }) {
  if (!features?.length) return null
  const max = Math.max(...features.map((f) => f.importance)) || 1
  return (
    <div className="glass p-5">
      <h3 className="flex items-center gap-2 display-card-title">
        <BarChart3 className="h-4 w-4 text-accent-400" />
        Most influential features
      </h3>
      <ul className="mt-4 space-y-2.5">
        {features.slice(0, 12).map((feature) => (
          <li key={feature.feature}>
            <div className="flex items-baseline justify-between gap-3">
              <span className="truncate font-mono text-[11px] text-slate-400">
                {feature.feature}
              </span>
              <span className="shrink-0 text-[11px] tabular-nums text-slate-500">
                {(feature.importance * 100).toFixed(1)}%
              </span>
            </div>
            <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-white/[0.07]">
              <div
                className="h-full rounded-full bg-gradient-to-r from-accent-600 to-accent-300 transition-all duration-700"
                style={{ width: `${(feature.importance / max) * 100}%` }}
              />
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}


/**
 * The curated smoke-test result.
 *
 * Shown deliberately, failures included: an early build of this model scored
 * 92% test accuracy while calling google.com phishing, and a gate like this is
 * what catches that class of bug. It is labelled as a smoke test so it is never
 * read as model accuracy.
 */
const GROUP_LABELS = {
  well_known: 'Well-known sites',
  unlisted_legitimate: 'Legitimate, not on the list',
  phishing_shaped: 'Phishing-shaped',
  bypass_attempts: 'Bypass attempts',
}

function SanityGate({ sanity }) {
  if (!sanity) return null
  // The gate is judged on legitimate URLs the reputation list does NOT cover.
  // Judging it on the served number would let the list hide a model regression,
  // which is the one thing this gate exists to prevent.
  const unaided = sanity.unlisted_legitimate_accuracy
  const passed = (unaided ?? sanity.legitimate_accuracy) >= 0.8
  const groups = Object.entries(sanity.per_group ?? {})

  return (
    <div className={`glass border-l-2 p-5 ${passed ? 'border-safe/40' : 'border-caution/50'}`}>
      <div className="flex items-start justify-between gap-4">
        <h3 className="flex items-center gap-2 display-card-title">
          <ClipboardCheck className={`h-4 w-4 ${passed ? 'text-safe' : 'text-caution'}`} />
          Sanity gate
        </h3>
        <span className="stat-number shrink-0 text-base">
          {sanity.correct}/{sanity.total}
        </span>
      </div>

      <p className="muted mt-2">{sanity.note}</p>

      {groups.length > 0 && (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-[12px]">
            <thead>
              <tr className="text-[10px] uppercase tracking-wider text-slate-500">
                <th className="pb-1.5 font-medium">Group</th>
                <th className="pb-1.5 text-right font-medium">Model alone</th>
                <th className="pb-1.5 text-right font-medium">As served</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {groups.map(([name, g]) => (
                <tr key={name}>
                  <td className="py-1.5 text-slate-400">{GROUP_LABELS[name] ?? name}</td>
                  <td className="py-1.5 text-right tabular-nums text-slate-400">
                    {g.model_only_correct}/{g.total}
                  </td>
                  <td className="py-1.5 text-right font-semibold tabular-nums text-white">
                    {g.as_served_correct}/{g.total}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
        &ldquo;As served&rdquo; includes the known-good domain prior; &ldquo;model alone&rdquo;
        does not. Both are published so the curated list cannot mask a model
        regression behind a healthier served number.
      </p>

      {sanity.failures?.length > 0 && (
        <div className="mt-4 border-t border-white/10 pt-3">
          <p className="text-[11px] uppercase tracking-wider text-slate-500">
            Current failures ({sanity.failures.length})
          </p>
          <ul className="mt-2 space-y-1.5">
            {sanity.failures.map((failure) => (
              <li key={failure.url} className="flex items-baseline justify-between gap-3">
                <span className="truncate font-mono text-[11px] text-slate-400" title={failure.url}>
                  {failure.url}
                </span>
                <span className="shrink-0 text-[11px] tabular-nums text-caution">
                  {failure.model_probability.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
            Every one is a legitimate URL the curated list does not cover, which is why
            it is in the set: these measure the classifier with nothing shielding it.
            Reported rather than hidden.
          </p>
        </div>
      )}
    </div>
  )
}

/** Precision/recall trade-off, published instead of buried in a constant. */
function OperatingPoints({ points, shippedFloor }) {
  if (!points?.length) return null
  return (
    <div className="glass p-5">
      <h3 className="flex items-center gap-2 display-card-title">
        <SlidersHorizontal className="h-4 w-4 text-accent-400" />
        Operating points
      </h3>
      <p className="muted mt-2">
        Recall attainable at each precision floor, measured on the test split. The
        shipped operating point is marked.
      </p>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[360px] text-left text-[13px]">
          <thead>
            <tr className="label-caps border-b border-white/10 text-[10px]">
              <th className="pb-2 pr-3 font-semibold">Precision floor</th>
              <th className="pb-2 pr-3 text-right font-semibold">Threshold</th>
              <th className="pb-2 pr-3 text-right font-semibold">Precision</th>
              <th className="pb-2 text-right font-semibold">Recall</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {points.map((point) => {
              const shipped =
                shippedFloor != null && Math.abs(point.precision_floor - shippedFloor) < 1e-9
              return (
                <tr key={point.precision_floor} className={shipped ? 'bg-accent-500/[0.07]' : ''}>
                  <td className="py-2.5 pr-3 tabular-nums text-slate-300">
                    {point.precision_floor.toFixed(2)}
                    {shipped && (
                      <span className="ml-2 rounded-full border border-accent-400/25 bg-accent-500/10 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-accent-200">
                        shipped
                      </span>
                    )}
                  </td>
                  {point.attainable ? (
                    <>
                      <td className="py-2.5 pr-3 text-right tabular-nums text-slate-400">
                        {point.threshold.toFixed(4)}
                      </td>
                      <td className="py-2.5 pr-3 text-right tabular-nums text-slate-400">
                        {percent(point.precision)}
                      </td>
                      <td className="py-2.5 text-right tabular-nums font-semibold text-white">
                        {percent(point.recall)}
                      </td>
                    </>
                  ) : (
                    <td colSpan={3} className="py-2.5 text-right text-slate-600">
                      not attainable
                    </td>
                  )}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function ModelPage() {
  const [info, setInfo] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    api
      .modelInfo()
      .then((data) => !cancelled && setInfo(data))
      .catch((err) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [])

  if (loading) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-accent-400" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="mx-auto max-w-2xl px-4 py-20">
        <div className="glass-strong flex items-start gap-3 border border-danger/25 p-6">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-danger" />
          <div>
            <h2 className="display-card-title">Could not load model information</h2>
            <p className="muted mt-1">{error}</p>
          </div>
        </div>
      </div>
    )
  }

  // Honest empty state: no metrics are invented when the pipeline has not run.
  if (!info?.trained) {
    return (
      <div className="mx-auto max-w-2xl px-4 py-20">
        <div className="glass-strong glass-edge flex flex-col items-center border border-caution/25 px-6 py-14 text-center">
          <span className="flex h-14 w-14 items-center justify-center rounded-2xl border border-caution/25 bg-caution/10">
            <AlertTriangle className="h-6 w-6 text-caution" />
          </span>
          <h1 className="mt-5 display-panel-title">Model not trained</h1>
          <p className="muted mt-2 max-w-md">{info?.message}</p>
          <code className="mt-5 rounded-lg border border-white/10 bg-black/40 px-4 py-2.5 font-mono text-[13px] text-accent-200">
            python -m ml.train
          </code>
        </div>
      </div>
    )
  }

  const test = info.test_metrics
  const holdout = info.external_holdout
  const dataset = info.dataset ?? {}

  return (
    <div className="mx-auto max-w-6xl animate-fade-up px-4 py-12 sm:px-6 lg:px-8">
      <header className="mb-8">
        <span className="badge border-accent-400/25 bg-accent-500/10 text-accent-200">
          <BrainCircuit className="h-3 w-3" />
          Model Intelligence
        </span>
        <h1 className="section-title mt-5">
          {info.model_name?.replace(/_/g, ' ')}{' '}
          <span className="text-slate-500">v{info.model_version}</span>
        </h1>
        <p className="muted mt-3 max-w-3xl">
          Every figure on this page is read from the artifacts produced by the training
          pipeline — nothing is hardcoded. Trained {formatDate(info.trained_at)}.
        </p>
      </header>

      {/* Headline test metrics */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <MetricTile icon={Target} label="Accuracy" value={percent(test.accuracy, 2)} hint="Held-out test split" />
        <MetricTile
          icon={CheckCircle2}
          label="Precision"
          value={percent(test.precision, 2)}
          hint="Of URLs flagged phishing, how many were"
        />
        <MetricTile
          icon={Radar}
          label="Recall"
          value={percent(test.recall, 2)}
          hint="Of real phishing URLs, how many were caught"
        />
        <MetricTile icon={Scale} label="F1 Score" value={percent(test.f1, 2)} hint="Harmonic mean" />
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <MetricTile icon={BarChart3} label="ROC-AUC" value={test.roc_auc.toFixed(4)} />
        <MetricTile icon={BarChart3} label="PR-AUC" value={test.pr_auc.toFixed(4)} hint="Selection criterion" />
        <MetricTile
          icon={AlertTriangle}
          label="False Positive Rate"
          value={percent(test.false_positive_rate, 2)}
          hint="Legitimate URLs wrongly flagged"
        />
        <MetricTile icon={Boxes} label="Features" value={info.feature_count} hint="Static URL features" />
      </div>

      {/* External holdout — the strongest evidence of real generalisation */}
      {holdout && (
        <section className="mt-8">
          <div className="glass-strong glass-edge border border-safe/25 p-6">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <h2 className="display-eyebrow flex items-center gap-2 text-safe">
                  <GitBranch className="h-4 w-4" />
                  Independent holdout
                </h2>
                <p className="muted mt-2 max-w-2xl">{holdout.note}</p>
              </div>
              <div className="text-right">
                <p className="stat-number text-[clamp(1.75rem,4.5vw,2.5rem)] leading-none text-safe">
                  {percent(holdout.recall, 2)}
                </p>
                <p className="mt-1 text-[11px] text-slate-500">recall</p>
              </div>
            </div>
            <p className="mt-4 text-[13px] text-slate-400">
              Detected <strong className="text-white">{fullNumber(holdout.detected)}</strong> of{' '}
              <strong className="text-white">{fullNumber(holdout.rows)}</strong> live phishing URLs
              spanning {fullNumber(holdout.unique_domains)} domains that appear nowhere in the
              training corpus.
            </p>
          </div>
        </section>
      )}

      {/* Model comparison */}
      <section className="mt-8">
        <h2 className="display-eyebrow mb-4 flex items-center gap-2.5 text-slate-300">
          <Cpu className="h-4 w-4 text-accent-400" />
          Model comparison
        </h2>
        <ComparisonTable comparison={info.comparison} selected={info.model_name} />
      </section>

      {/* Decision policy + importances */}
      <section className="mt-8 grid gap-4 lg:grid-cols-2">
        <div className="glass p-5">
          <h3 className="flex items-center gap-2 display-card-title">
            <Scale className="h-4 w-4 text-accent-400" />
            Decision policy
          </h3>
          <dl className="mt-4 space-y-3 text-[13px]">
            <div className="flex items-start justify-between gap-4">
              <dt className="text-slate-500">Threshold</dt>
              <dd className="text-right font-mono text-slate-300">
                {info.decision_threshold?.toFixed(4)}
              </dd>
            </div>
            <div>
              <dt className="text-slate-500">Policy</dt>
              <dd className="mt-1 leading-relaxed text-slate-300">{info.threshold_policy}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Selection criterion</dt>
              <dd className="mt-1 leading-relaxed text-slate-300">{info.selection_criterion}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Split strategy</dt>
              <dd className="mt-1 leading-relaxed text-slate-300">{dataset.split_strategy}</dd>
            </div>
          </dl>
        </div>

        <FeatureImportance features={info.top_features} />
      </section>

      {/* Honesty section: the smoke-test gate and the precision/recall curve */}
      <section className="mt-4 grid gap-4 lg:grid-cols-2">
        <SanityGate sanity={info.sanity_check} />
        <OperatingPoints points={info.operating_points} shippedFloor={info.precision_floor} />
      </section>

      {/* Dataset provenance */}
      <section className="mt-4">
        <div className="glass p-5">
          <h3 className="flex items-center gap-2 display-card-title">
            <Database className="h-4 w-4 text-accent-400" />
            Training data
          </h3>
          <div className="mt-4 grid grid-cols-2 gap-4 text-[13px] sm:grid-cols-4">
            <div>
              <p className="text-slate-500">Rows used</p>
              <p className="mt-1 font-semibold tabular-nums text-white">
                {fullNumber(dataset.sampled_rows)}
              </p>
            </div>
            <div>
              <p className="text-slate-500">Unique domains</p>
              <p className="mt-1 font-semibold tabular-nums text-white">
                {fullNumber(dataset.unique_domains)}
              </p>
            </div>
            <div>
              <p className="text-slate-500">Train / Val / Test</p>
              <p className="mt-1 font-semibold tabular-nums text-white">
                {fullNumber(dataset.train_rows)} / {fullNumber(dataset.val_rows)} /{' '}
                {fullNumber(dataset.test_rows)}
              </p>
            </div>
            <div>
              <p className="text-slate-500">Phishing rate (test)</p>
              <p className="mt-1 font-semibold tabular-nums text-white">
                {percent(dataset.test_phishing_rate, 1)}
              </p>
            </div>
          </div>

          {dataset.rows_per_source && (
            <div className="mt-5 border-t border-white/10 pt-4">
              <p className="text-[11px] uppercase tracking-wider text-slate-500">Sources</p>
              <ul className="mt-2 space-y-1.5">
                {Object.entries(dataset.rows_per_source).map(([source, counts]) => (
                  <li key={source} className="flex items-center justify-between text-[12px]">
                    <span className="font-mono text-slate-400">{source}</span>
                    <span className="tabular-nums text-slate-500">
                      {fullNumber(Number(counts['0'] ?? 0))} legit ·{' '}
                      {fullNumber(Number(counts['1'] ?? 0))} phishing
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </section>
    </div>
  )
}
