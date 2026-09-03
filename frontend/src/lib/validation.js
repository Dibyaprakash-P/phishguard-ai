/**
 * Client-side pre-flight validation.
 *
 * The goal is to catch obvious mistakes (empty input, whitespace, an
 * unsupported scheme) without ever pre-judging risk. A URL that *looks*
 * suspicious is exactly what this product exists to analyze, so nothing is
 * blocked for looking dangerous — only for being unanalyzable.
 */
export function validateUrlInput(raw) {
  const value = (raw ?? '').trim()

  if (!value) return { valid: false, error: 'Enter a URL to analyze.' }
  if (/\s/.test(value)) return { valid: false, error: 'A URL cannot contain spaces.' }
  if (value.length > 2048) {
    return { valid: false, error: 'That URL is too long (maximum 2048 characters).' }
  }

  const scheme = value.includes(':') ? value.split(':')[0].toLowerCase() : ''
  if (scheme && !['http', 'https'].includes(scheme)) {
    if (value.includes('://') || ['javascript', 'data', 'file', 'vbscript'].includes(scheme)) {
      return { valid: false, error: `Only http and https URLs can be analyzed (got "${scheme}:").` }
    }
  }

  // Prepend a scheme when missing so the URL parser can do the real work.
  const candidate = /^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(value) ? value : `http://${value}`

  let parsed
  try {
    parsed = new URL(candidate)
  } catch {
    return { valid: false, error: 'That does not look like a valid URL.' }
  }

  const host = parsed.hostname
  if (!host) return { valid: false, error: 'The URL is missing a hostname.' }

  const isIpLiteral = /^\d{1,3}(\.\d{1,3}){3}$/.test(host) || host.startsWith('[')
  if (!isIpLiteral && !host.includes('.')) {
    return { valid: false, error: 'Include a domain suffix, for example example.com.' }
  }
  if (!isIpLiteral && (host.startsWith('.') || host.endsWith('.') || host.includes('..'))) {
    return { valid: false, error: 'The hostname contains empty domain labels.' }
  }

  return { valid: true, error: null, value }
}
