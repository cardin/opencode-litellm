import { describe, expect, it } from 'vitest'
import plugin, { LiteLLMPlugin, LiteLLMResponsesPlugin } from '../src/index'

describe('OpenCode V2 entrypoint', () => {
  it('default-exports the LiteLLM Plugin.define object', () => {
    expect(plugin).toBe(LiteLLMPlugin)
    expect(plugin.id).toBe('litellm')
    expect(plugin.setup).toBeTypeOf('function')
  })

  it('keeps the legacy responses symbol as a valid V2 plugin', () => {
    expect(LiteLLMResponsesPlugin.id).toBe('litellm.responses')
    expect(LiteLLMResponsesPlugin.setup).toBeTypeOf('function')
  })
})
