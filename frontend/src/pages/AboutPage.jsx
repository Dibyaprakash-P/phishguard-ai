import {
  Brain,
  Cloud,
  Cpu,
  Database,
  EyeOff,
  GitBranch,
  Layers,
  Lock,
  Server,
  ShieldCheck,
  Workflow,
} from 'lucide-react'

const PIPELINE = [
  { icon: Lock, title: 'URL validation', body: 'Structure, scheme and hostname are checked. Only http and https are accepted; javascript:, data: and file: URLs are rejected outright.' },
  { icon: Layers, title: 'Static feature extraction', body: 'Dozens of lexical and structural features are computed from the URL string alone — length, entropy, subdomain depth, lure vocabulary, TLD reputation, character ratios and more.' },
  { icon: Cpu, title: 'ML classification', body: 'A gradient-boosted tree ensemble scores the feature vector and returns a calibrated phishing probability. The model is loaded once at startup, never retrained per request.' },
  { icon: ShieldCheck, title: 'Risk interpretation', body: 'The probability is mapped to a verdict and a 0–100 risk score using published, configurable thresholds returned with every response.' },
  { icon: Brain, title: 'AI explanation', body: 'LangChain passes the finished verdict and the extracted evidence to an LLM, which writes the narrative. The LLM never classifies — and if it is unavailable, a deterministic rule-based analysis is shown instead, clearly labelled.' },
]

const STACK = [
  { icon: Cpu, label: 'Python · scikit-learn · XGBoost' },
  { icon: Server, label: 'FastAPI · Pydantic · Uvicorn' },
  { icon: Brain, label: 'LangChain · LangGraph · OpenAI / Azure OpenAI' },
  { icon: GitBranch, label: 'MLflow · Docker · GitHub Actions' },
  { icon: Layers, label: 'React · Vite · Tailwind CSS' },
  { icon: Cloud, label: 'Azure-compatible deployment path' },
]

function Card({ icon: Icon, title, children, tone = 'accent' }) {
  const toneClass =
    tone === 'safe'
      ? 'border-safe/25 bg-safe/10 text-safe'
      : 'border-accent-400/25 bg-accent-500/10 text-accent-300'
  return (
    <article className="glass glass-hover p-6">
      <span className={`flex h-10 w-10 items-center justify-center rounded-xl border ${toneClass}`}>
        <Icon className="h-5 w-5" strokeWidth={1.9} />
      </span>
      <h3 className="mt-4 display-panel-title">{title}</h3>
      <div className="muted mt-2 space-y-2">{children}</div>
    </article>
  )
}

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-5xl animate-fade-up px-4 py-12 sm:px-6 lg:px-8">
      <header className="mb-10">
        <span className="badge border-accent-400/25 bg-accent-500/10 text-accent-200">
          <ShieldCheck className="h-3 w-3" />
          About
        </span>
        <h1 className="section-title mt-5">What PhishGuard AI does</h1>
        <p className="muted mt-3 max-w-3xl">
          PhishGuard AI classifies a URL as legitimate, suspicious or phishing using a
          supervised machine-learning model trained on public phishing corpora, then uses a
          large language model to explain that verdict in plain language. The machine-learning
          model makes the decision; the language model only describes it.
        </p>
      </header>

      {/* Privacy — the most important statement on the page */}
      <section className="glass-strong glass-edge border border-safe/25 p-6 sm:p-8">
        <div className="flex items-start gap-4">
          <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-safe/25 bg-safe/10">
            <EyeOff className="h-5 w-5 text-safe" strokeWidth={1.9} />
          </span>
          <div>
            <h2 className="display-panel-title">Privacy &amp; safety</h2>
            <p className="mt-2 text-[14px] font-medium leading-relaxed text-safe/90">
              The analyzer performs static URL analysis by default and does not automatically
              visit submitted websites.
            </p>
            <ul className="muted mt-4 space-y-2">
              <li>
                <strong className="text-slate-300">No outbound requests.</strong> The backend never
                fetches, resolves, crawls or renders the URL you submit. Every feature is computed
                from the string. This is what makes the service immune to SSRF and to
                executing hostile content.
              </li>
              <li>
                <strong className="text-slate-300">No database.</strong> The service is stateless.
                Submitted URLs are not stored, and there is no user account, session or scan table
                anywhere on the server.
              </li>
              <li>
                <strong className="text-slate-300">Redacted logging.</strong> When a URL appears in a
                server log for debugging, sensitive query parameters (tokens, credentials,
                email addresses, session ids) are replaced before the line is written.
              </li>
              <li>
                <strong className="text-slate-300">Local-only history.</strong> Scan history lives in
                your browser&apos;s localStorage and never leaves your device.
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* How detection works */}
      <section className="mt-10">
        <h2 className="display-eyebrow flex items-center gap-2.5 text-slate-300">
          <Workflow className="h-4 w-4 text-accent-400" />
          How detection works
        </h2>
        <ol className="mt-5 space-y-3">
          {PIPELINE.map(({ icon: Icon, title, body }, index) => (
            <li key={title} className="glass glass-hover flex gap-4 p-5">
              <div className="flex flex-col items-center">
                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-accent-400/25 bg-accent-500/10">
                  <Icon className="h-4 w-4 text-accent-300" strokeWidth={1.9} />
                </span>
                {index < PIPELINE.length - 1 && (
                  <span className="mt-2 w-px flex-1 bg-gradient-to-b from-white/15 to-transparent" />
                )}
              </div>
              <div className="min-w-0">
                <h3 className="display-card-title">
                  <span className="mr-2 font-mono text-[11px] text-accent-400/70">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  {title}
                </h3>
                <p className="muted mt-1.5">{body}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      {/* Design notes */}
      <section className="mt-10 grid gap-4 sm:grid-cols-2">
        <Card icon={Database} title="Feature engineering">
          <p>
            Features come from a single shared module used by both the training pipeline and
            the live API, so the vector the model sees in production is identical to the one it
            was trained on.
          </p>
          <p>
            URLs are canonicalised before extraction — the scheme and a leading{' '}
            <code className="font-mono text-[11px] text-accent-200">www.</code> are stripped —
            because in the public corpora those markers encode which dataset a URL came from
            rather than whether it is a phishing page.
          </p>
        </Card>

        <Card icon={Cpu} title="Why not use the LLM as the classifier?">
          <p>
            A gradient-boosted model over engineered features is faster, far cheaper, fully
            deterministic, and can be measured against a held-out test set with precision and
            recall. An LLM verdict cannot be evaluated that way, would cost a call per scan and
            could not be audited.
          </p>
          <p>The LLM is used for what it is genuinely good at: explaining the result.</p>
        </Card>

        <Card icon={ShieldCheck} title="Guarding against fabrication" tone="safe">
          <p>
            The prompt hands the model only the finished verdict and the mechanically extracted
            evidence, and forbids it from claiming the site was visited or that WHOIS,
            DNS, certificate, blocklist or malware data was consulted — because none of it was.
          </p>
        </Card>

        <Card icon={Brain} title="When the LLM is unavailable">
          <p>
            No API key, a provider outage or a timeout never breaks a scan. The ML verdict is
            served as normal and a deterministic rule-based explanation, built from the same
            evidence, is shown in its place — explicitly labelled so it is never mistaken for
            an AI analysis.
          </p>
        </Card>
      </section>

      {/* Stack */}
      <section className="mt-10">
        <h2 className="display-eyebrow flex items-center gap-2.5 text-slate-300">
          <Layers className="h-4 w-4 text-accent-400" />
          Technology
        </h2>
        <ul className="mt-5 grid gap-3 sm:grid-cols-2">
          {STACK.map(({ icon: Icon, label }) => (
            <li key={label} className="glass flex items-center gap-3 px-4 py-3">
              <Icon className="h-4 w-4 shrink-0 text-accent-400" />
              <span className="text-[13px] text-slate-300">{label}</span>
            </li>
          ))}
        </ul>
      </section>

      {/* Limitations — stated plainly */}
      <section className="mt-10">
        <div className="glass border-l-2 border-caution/40 p-6">
          <h2 className="display-panel-title">Limitations</h2>
          <ul className="muted mt-3 list-disc space-y-2 pl-5">
            <li>
              A URL-only classifier cannot see page content. A phishing page hosted on a
              compromised but ordinary-looking domain has few lexical tells and may be missed.
            </li>
            <li>
              Conversely, unusual but legitimate URLs — long tracking links, generated
              subdomains, newer TLDs — can be scored as risky. The false-positive rate is
              published on the Model page.
            </li>
            <li>
              Training data comes from public corpora collected at a point in time. Phishing
              vocabulary and hosting patterns drift, so periodic retraining is required.
            </li>
            <li>
              The risk score is a model-derived indicator, not an industry-standard measure, and
              should inform judgement rather than replace it.
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}
