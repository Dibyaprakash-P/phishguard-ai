import { describe, expect, it } from 'vitest'
import { validateUrlInput } from '../lib/validation'
import { percent, riskLevelLabel, truncateUrl, verdictStyle } from '../lib/format'

describe('validateUrlInput', () => {
  it('accepts ordinary URLs', () => {
    for (const url of [
      'https://google.com',
      'http://example.com/path',
      'example.com',
      'www.example.co.uk/a/b?c=1',
      'http://192.168.1.10/login',
    ]) {
      expect(validateUrlInput(url).valid, url).toBe(true)
    }
  })

  it('accepts suspicious-looking URLs — classifying them is the point', () => {
    for (const url of [
      'http://paypal.com.secure-login-verify.tk/webscr',
      'https://secure-paypa1-login.example.com/verify/account',
      'http://192.3.44.201:8080/wp-content/paypal/login.php',
    ]) {
      expect(validateUrlInput(url).valid, url).toBe(true)
    }
  })

  it('trims surrounding whitespace', () => {
    const result = validateUrlInput('   https://example.com   ')
    expect(result.valid).toBe(true)
    expect(result.value).toBe('https://example.com')
  })

  it('rejects empty input', () => {
    for (const value of ['', '   ', null, undefined]) {
      const result = validateUrlInput(value)
      expect(result.valid).toBe(false)
      expect(result.error).toMatch(/enter a url/i)
    }
  })

  it('rejects internal whitespace', () => {
    expect(validateUrlInput('not a url').error).toMatch(/spaces/i)
  })

  it('rejects unsupported and dangerous schemes', () => {
    for (const url of ['javascript:alert(1)', 'data:text/html,<script>', 'file:///etc/passwd']) {
      const result = validateUrlInput(url)
      expect(result.valid, url).toBe(false)
      expect(result.error).toMatch(/http and https/i)
    }
  })

  it('rejects hostnames with no domain suffix', () => {
    expect(validateUrlInput('localhost').error).toMatch(/domain suffix/i)
  })

  it('rejects empty domain labels', () => {
    expect(validateUrlInput('http://a..b.com').valid).toBe(false)
  })

  it('rejects over-length URLs', () => {
    expect(validateUrlInput(`https://example.com/${'a'.repeat(3000)}`).error).toMatch(/too long/i)
  })
})

describe('format helpers', () => {
  it('formats percentages', () => {
    expect(percent(0.9642)).toBe('96.4%')
    expect(percent(0.9642, 2)).toBe('96.42%')
    expect(percent(null)).toBe('—')
    expect(percent(undefined)).toBe('—')
  })

  it('maps every verdict to a distinct style', () => {
    const styles = ['legitimate', 'suspicious', 'phishing'].map(verdictStyle)
    expect(new Set(styles.map((s) => s.label)).size).toBe(3)
    styles.forEach((style) => {
      expect(style.icon).toBeTruthy()
      expect(style.text).toBeTruthy()
    })
  })

  it('falls back for an unknown verdict rather than crashing', () => {
    expect(verdictStyle('nonsense')).toBe(verdictStyle('suspicious'))
  })

  it('labels risk levels', () => {
    expect(riskLevelLabel('high')).toBe('High')
    expect(riskLevelLabel('nope')).toBe('—')
  })

  it('truncates only long URLs, keeping both ends visible', () => {
    expect(truncateUrl('https://a.com')).toBe('https://a.com')
    const long = `https://example.com/${'x'.repeat(200)}/end`
    const short = truncateUrl(long, 40)
    expect(short.length).toBeLessThanOrEqual(41)
    expect(short).toContain('…')
    expect(short.startsWith('https://example.com')).toBe(true)
  })
})
