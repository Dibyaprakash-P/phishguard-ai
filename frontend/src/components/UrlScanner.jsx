import { useCallback, useState } from 'react'
import { AlertCircle, Link as LinkIcon, Loader2, Search, X } from 'lucide-react'
import { validateUrlInput } from '../lib/validation'

/** Example URLs offered as one-click fill. */
const EXAMPLES = [
  'https://github.com',
  'https://secure-paypa1-login.example.com/verify/account',
  'http://192.168.1.10/login',
]

export default function UrlScanner({ onAnalyze, loading, disabled, disabledReason }) {
  const [url, setUrl] = useState('')
  const [error, setError] = useState(null)

  const submit = useCallback(
    (event) => {
      event?.preventDefault()
      const result = validateUrlInput(url)
      if (!result.valid) {
        setError(result.error)
        return
      }
      setError(null)
      onAnalyze(result.value)
    },
    [url, onAnalyze],
  )

  const onChange = (event) => {
    setUrl(event.target.value)
    if (error) setError(null)
  }

  const busy = loading || disabled

  return (
    <section className="mx-auto mt-12 max-w-3xl px-4 sm:px-6">
      <form
        onSubmit={submit}
        className="glass-strong glass-edge animate-fade-up p-5 sm:p-7"
        style={{ animationDelay: '300ms' }}
        noValidate
      >
        <label
          htmlFor="url-input"
          className="display-eyebrow flex items-center gap-2.5 text-slate-300"
        >
          <LinkIcon className="h-4 w-4 text-accent-400" />
          Enter a URL to analyze
        </label>

        <div className="relative mt-4">
          <input
            id="url-input"
            type="text"
            inputMode="url"
            autoComplete="off"
            autoCapitalize="none"
            autoCorrect="off"
            spellCheck="false"
            value={url}
            onChange={onChange}
            placeholder="https://example.com"
            disabled={busy}
            aria-invalid={Boolean(error)}
            aria-describedby={error ? 'url-error' : undefined}
            className={`input-glass pr-11 ${error ? 'border-danger/50' : ''}`}
          />
          {url && !busy && (
            <button
              type="button"
              onClick={() => {
                setUrl('')
                setError(null)
              }}
              className="absolute right-3 top-1/2 -translate-y-1/2 rounded-md p-1 text-slate-500 transition hover:text-white"
              aria-label="Clear input"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>

        {error && (
          <p
            id="url-error"
            role="alert"
            className="mt-3 flex items-start gap-2 text-sm text-danger"
          >
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            {error}
          </p>
        )}

        {disabled && disabledReason && (
          <p className="mt-3 flex items-start gap-2 rounded-lg border border-caution/25 bg-caution/10 px-3 py-2.5 text-sm text-caution">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            {disabledReason}
          </p>
        )}

        <div className="mt-5 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex flex-wrap items-center gap-2">
            <span className="label-caps">Try</span>
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => {
                  setUrl(example)
                  setError(null)
                }}
                disabled={busy}
                className="max-w-[190px] truncate rounded-lg border border-white/10 bg-white/[0.04] px-2.5 py-1.5 font-mono text-[11px] text-slate-400 transition hover:border-accent-400/30 hover:text-accent-200 disabled:opacity-40"
                title={example}
              >
                {example.replace(/^https?:\/\//, '')}
              </button>
            ))}
          </div>

          <button type="submit" disabled={busy} className="btn-primary w-full sm:w-auto">
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Analyzing…
              </>
            ) : (
              <>
                <Search className="h-4 w-4" />
                Analyze URL
              </>
            )}
          </button>
        </div>
      </form>
    </section>
  )
}
