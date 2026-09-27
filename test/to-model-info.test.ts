import { describe, expect, it } from 'vitest'
import { toModelInfo } from '../src/plugin/index'
import type { LiteLLMModel } from '../src/types'

function model(id: string, extra: Partial<LiteLLMModel> = {}): LiteLLMModel {
  return { id, object: 'model', ...extra }
}

describe('toModelInfo naming (formatModelNames)', () => {
  const chat = model('anthropic/claude-3-5-sonnet')

  it('prettifies the id by default', () => {
    expect(toModelInfo(chat, 'litellm')?.name).toBe('Claude 3.5 Sonnet')
    expect(toModelInfo(chat, 'litellm', undefined, true)?.name).toBe(
      'Claude 3.5 Sonnet',
    )
  })

  it('keeps the raw /v1/models id when formatting is disabled', () => {
    expect(toModelInfo(chat, 'litellm', undefined, false)?.name).toBe(
      'anthropic/claude-3-5-sonnet',
    )
  })

  it('leaves provider prefixes and version suffixes intact when raw', () => {
    const versioned = model('claude-opus-4-5@20251101')
    expect(toModelInfo(versioned, 'litellm', undefined, false)?.name).toBe(
      'claude-opus-4-5@20251101',
    )
    // ...while the formatted view would collapse them.
    expect(toModelInfo(versioned, 'litellm')?.name).toBe('Claude Opus 4.5')
  })

  it('still hides non-chat models regardless of the naming choice', () => {
    const embedding = model('text-embedding-3-large', { mode: 'embedding' })
    expect(toModelInfo(embedding, 'litellm', undefined, false)).toBeNull()
    expect(toModelInfo(embedding, 'litellm', undefined, true)).toBeNull()
  })
})
