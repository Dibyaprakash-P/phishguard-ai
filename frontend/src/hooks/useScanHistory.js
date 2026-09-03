import { useCallback, useEffect, useMemo, useState } from 'react'

const STORAGE_KEY = 'phishguard.scan-history.v1'
const MAX_ENTRIES = 50

/**
 * Scan history kept **only in this browser**.
 *
 * PhishGuard has no database and the backend is stateless — it never stores a
 * submitted URL. This hook is therefore the only place history exists, and it
 * is scoped to one browser profile. The UI labels anything derived from it as
 * "Local Session Insights" so it is never mistaken for global product
 * analytics.
 *
 * Every localStorage access is guarded: private-mode browsers and blocked
 * site-data settings make these calls throw, and the app must still work.
 */
function readStorage() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function writeStorage(entries) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(entries))
  } catch {
    // Storage unavailable or full — history simply does not persist.
  }
}

export default function useScanHistory() {
  const [entries, setEntries] = useState([])

  useEffect(() => {
    setEntries(readStorage())
  }, [])

  const record = useCallback((result) => {
    setEntries((current) => {
      const entry = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        url: result.normalized_url,
        domain: result.components?.registrable_domain ?? '',
        prediction: result.prediction,
        riskLevel: result.risk_level,
        riskScore: result.risk_score,
        confidence: result.confidence,
        scannedAt: new Date().toISOString(),
      }
      const next = [entry, ...current].slice(0, MAX_ENTRIES)
      writeStorage(next)
      return next
    })
  }, [])

  const clear = useCallback(() => {
    setEntries([])
    writeStorage([])
  }, [])

  const stats = useMemo(() => {
    const total = entries.length
    if (!total) {
      return { total: 0, phishing: 0, suspicious: 0, legitimate: 0, averageRisk: 0, highestRisk: 0 }
    }
    const count = (prediction) => entries.filter((e) => e.prediction === prediction).length
    return {
      total,
      phishing: count('phishing'),
      suspicious: count('suspicious'),
      legitimate: count('legitimate'),
      averageRisk: Math.round(entries.reduce((sum, e) => sum + e.riskScore, 0) / total),
      highestRisk: Math.max(...entries.map((e) => e.riskScore)),
    }
  }, [entries])

  return { entries, stats, record, clear }
}
