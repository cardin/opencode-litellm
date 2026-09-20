import { Model, Plugin, Provider } from '@opencode/plugin'
import {
  autoDetectLiteLLM,
  checkLiteLLMHealth,
  discoverLiteLLMModelInfo,
  discoverLiteLLMModels,
  getRequestTimeoutMs,
  normalizeBaseURL,
} from '../utils/litellm-api'
import {
  formatModelName,
  categorizeModel,
} from '../utils/format-model-name'
import type { LiteLLMModel, LiteLLMModelInfo } from '../types'
import {
  buildCacheKey,
  readModelCache,
  writeModelCache,
  readModelCacheSavedAt,
} from '../utils/model-cache'
import { passesModelFilter } from '../utils/model-filter'
import type { ModelFilters } from '../utils/model-filter'
import { applyCapabilityOverrides, parseModelCapabilities } from '../utils/model-capabilities'
import type { ModelCapabilities } from '../utils/model-capabilities'

const CHAT_PROVIDER_ID = 'litellm'
// Covers the 3 s health check plus the parallel models/model-info fetch
// phase, with headroom. Scales with LITELLM_REQUEST_TIMEOUT_MS so slow
// proxies aren't cut off by the overall cap either (issue #20).
const DISCOVERY_TIMEOUT_MS = Math.max(20000, getRequestTimeoutMs() + 5000)
// Don't revalidate a baseURL's cache more often than this, so a burst
// of `session.created` events can't generate repeated discovery traffic.
const REFRESH_MIN_INTERVAL_MS = 5 * 60 * 1000 // 5 minutes

type LogLevel = 'info' | 'warn' | 'error' | 'debug'

/**
 * V2 plugins run in the OpenCode service, so console output is captured by
 * the service logger rather than being written into the terminal UI.
 */
function log(level: LogLevel, message: string): void {
  if (level === 'error') console.error(message)
  else if (level === 'warn') console.warn(message)
  else console.log(message)
}

/** Per-provider discovery state captured by the V2 provider transform. */
interface ProviderState {
  baseURL: string
  apiKey?: string
  customHeaders?: Record<string, string>
  filters: ModelFilters
  capabilities: ModelCapabilities
  providerId: string
  name: string
  package: string
  models: Record<string, Model.Info>
}

interface ProviderCandidate {
  id: string
  provider?: Provider.Info
  options: Record<string, unknown>
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

/**
 * Race a promise against a timeout, resolving to `null` if the timeout
 * wins. Clears the timer either way so a resolved discovery can't keep
 * a short-lived process alive waiting on a pending `setTimeout`.
 */
function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
): Promise<T | null> {
  let timer: ReturnType<typeof setTimeout>
  const timeout = new Promise<null>((resolve) => {
    timer = setTimeout(() => resolve(null), timeoutMs)
  })
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer))
}

/**
 * Helper to determine if a provider ID or its configured options indicate
 * compatibility with LiteLLM.
 */
function isLiteLLMProvider(
  providerId: string,
  options: Record<string, unknown>,
): boolean {
  if (providerId === CHAT_PROVIDER_ID) return true
  if (providerId.startsWith('litellm-') || providerId.startsWith('litellm_')) return true
  if (options.litellm === true) return true
  if (options.litellmCompatible === true) return true
  if (options['litellm-compatible'] === true) return true
  if (options.litellm_compatible === true) return true
  return false
}

/**
 * Read `customHeaders` from a provider options block.
 */
function readCustomHeaders(
  options: Record<string, unknown>,
): Record<string, string> | undefined {
  const raw = options.customHeaders
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
    const out: Record<string, string> = {}
    for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
      if (typeof v === 'string') out[k] = v
    }
    return Object.keys(out).length > 0 ? out : undefined
  }
  return undefined
}

/**
 * Read the `includeModels`/`excludeModels` glob filters from a provider
 * options block (issue #21's feature: split one proxy's catalog across
 * several OpenCode providers). Non-string entries are dropped; an empty
 * result means "don't filter".
 */
function readModelFilters(options: Record<string, unknown>): ModelFilters {
  const readPatterns = (raw: unknown): string[] | undefined => {
    if (!Array.isArray(raw)) return undefined
    const out = raw.filter((v): v is string => typeof v === 'string')
    return out.length > 0 ? out : undefined
  }
  return {
    includeModels: readPatterns(options.includeModels),
    excludeModels: readPatterns(options.excludeModels),
  }
}

/**
 * Overlay metadata onto a `/v1/models` entry in three tiers: the entry's
 * own fields win, `/v1/model/info` fills gaps (notably `mode`, which
 * `/v1/models` omits for database-defined models), and finally
 * user-configured `modelCapabilities` overrides apply — explicit
 * `false` included — because the user knows their deployment better
 * than either endpoint (issue #25).
 */
function enrichModel(
  model: LiteLLMModel,
  info: LiteLLMModelInfo | undefined,
  overrides?: Record<string, boolean>,
): LiteLLMModel {
  return applyCapabilityOverrides(
    {
      ...model,
      mode: model.mode ?? info?.mode,
      max_tokens: model.max_tokens ?? info?.max_tokens,
      max_input_tokens: model.max_input_tokens ?? info?.max_input_tokens,
      max_output_tokens: model.max_output_tokens ?? info?.max_output_tokens,
      supports_function_calling:
        model.supports_function_calling ?? info?.supports_function_calling,
      supports_vision: model.supports_vision ?? info?.supports_vision,
      supports_reasoning: model.supports_reasoning ?? info?.supports_reasoning,
      supports_pdf_input: model.supports_pdf_input ?? info?.supports_pdf_input,
      supports_audio_input: model.supports_audio_input ?? info?.supports_audio_input,
      input_cost_per_token: model.input_cost_per_token ?? info?.input_cost_per_token,
      output_cost_per_token: model.output_cost_per_token ?? info?.output_cost_per_token,
      cache_read_input_token_cost:
        model.cache_read_input_token_cost ?? info?.cache_read_input_token_cost,
      cache_creation_input_token_cost:
        model.cache_creation_input_token_cost ?? info?.cache_creation_input_token_cost,
    },
    overrides,
  )
}

/**
 * OpenCode's `cost` config field is USD per **million** tokens (the
 * models.dev convention); LiteLLM's `/v1/model/info` reports USD per
 * single token. `1e6` bridges the two — verified against a live
 * `x-litellm-response-cost` header, not just the unit names.
 */
const USD_PER_TOKEN_TO_PER_MILLION = 1_000_000

/**
 * Convert a discovered LiteLLM model into an OpenCode V2 model definition.
 * Returns `null` for non-chat models (embedding,
 * image, audio) — they can't be used as primary chat models and would
 * clutter the picker.
 */
function toModelInfo(
  model: LiteLLMModel,
  providerId: string,
  info?: LiteLLMModelInfo,
): Model.Info | null {
  const type = categorizeModel(model)
  if (type === 'embedding' || type === 'image' || type === 'audio') {
    return null
  }
  const providerID = Provider.ID.make(providerId)
  const id = Model.ID.make(model.id)
  const defaults = Model.Info.default(providerID, id)
  // Some deployments only report the OpenAI-style `max_tokens` total;
  // use it as the context limit when `max_input_tokens` is absent.
  const contextLimit = model.max_input_tokens ?? model.max_tokens
  const limit = {
    ...defaults.limit,
    context: contextLimit ?? defaults.limit.context,
    output: model.max_output_tokens ?? defaults.limit.output,
  }
  // Only emit `cost` when LiteLLM actually reported a price. Omitting
  // it (rather than defaulting to 0) lets OpenCode/models.dev fall back
  // to their own default instead of us asserting "this model is free"
  // for something LiteLLM simply has no price anchor for (e.g. rerank).
  const cost: Model.Cost[] = []
  if (model.input_cost_per_token != null || model.output_cost_per_token != null) {
    cost.push({
      input: (model.input_cost_per_token ?? 0) * USD_PER_TOKEN_TO_PER_MILLION,
      output: (model.output_cost_per_token ?? 0) * USD_PER_TOKEN_TO_PER_MILLION,
      cache: {
        read: (model.cache_read_input_token_cost ?? 0) * USD_PER_TOKEN_TO_PER_MILLION,
        write: (model.cache_creation_input_token_cost ?? 0) * USD_PER_TOKEN_TO_PER_MILLION,
      },
    } as Model.Cost)
  }
  const input: Array<'text' | 'image' | 'pdf' | 'audio'> = ['text']
  if (model.supports_vision) input.push('image')
  if (model.supports_pdf_input) input.push('pdf')
  if (model.supports_audio_input) input.push('audio')
  const variants: Model.Variant[] = (info?.supports_reasoning_efforts ?? []).map(
    (effort) => ({
      id: Model.VariantID.make(effort),
      settings: { reasoningEffort: effort },
    }),
  )
  return {
    ...defaults,
    name: formatModelName(model),
    limit,
    capabilities: {
      tools: model.supports_function_calling === true,
      input,
      output: ['text'],
    },
    cost,
    variants,
  }
}

/**
 * Fetch and build OpenCode model entries from a LiteLLM proxy.
 *
 * Pure with respect to plugin config: it performs the network calls,
 * classifies + formats each model, and returns a `{ id -> entry }` map.
 * The provider's `includeModels`/`excludeModels` filters and
 * `modelCapabilities` overrides are applied here (not at merge time) so
 * every path that persists or serves a cache — cold discovery and
 * background refresh — writes the same adjusted view.
 *
 * Returns `null` when the proxy is unreachable/unauthorized or exposes
 * no models, so callers can distinguish "no data" from "empty result".
 */
async function discoverModels(
  baseURL: string,
  apiKey: string | undefined,
  customHeaders: Record<string, string> | undefined,
  providerId: string,
  filters: ModelFilters = {},
  capabilities: ModelCapabilities = {},
): Promise<Record<string, Model.Info> | null> {
  if (!(await checkLiteLLMHealth(baseURL, apiKey, customHeaders))) {
    log(
      'warn',
      `[opencode-litellm] LiteLLM appears offline or unauthorized for provider "${providerId}" at ${baseURL}`,
    )
    return null
  }

  // `/v1/models` omits `mode` and capability metadata for
  // database-defined models, so fetch `/v1/model/info` alongside
  // it. The info call is best-effort: without it, classification
  // falls back to id heuristics.
  const [modelsResult, infoResult] = await Promise.allSettled([
    discoverLiteLLMModels(baseURL, apiKey, customHeaders),
    discoverLiteLLMModelInfo(baseURL, apiKey, customHeaders),
  ])

  if (modelsResult.status === 'rejected') {
    const error = modelsResult.reason
    log(
      'warn',
      `[opencode-litellm] Model discovery failed for provider "${providerId}": ` +
        (error instanceof Error ? error.message : String(error)),
    )
    return null
  }

  const discovered = modelsResult.value
  let infoByName: Map<string, LiteLLMModelInfo> | null = null
  if (infoResult.status === 'fulfilled') {
    infoByName = infoResult.value
  } else {
    const reason = infoResult.reason
    log(
      'warn',
      `[opencode-litellm] /v1/model/info unavailable for provider "${providerId}"; non-chat model filtering will use id heuristics only: ` +
        (reason instanceof Error ? reason.message : String(reason)),
    )
  }

  if (discovered.length === 0) {
    log(
      'warn',
      `[opencode-litellm] LiteLLM responded for provider "${providerId}" but exposed zero models.`,
    )
    return null
  }

  const built: Record<string, Model.Info> = {}
  let skipped = 0
  let wildcards = 0
  let filtered = 0
  const unmatched: string[] = []
  for (const model of discovered) {
    // `deepseek/*` is an access rule, not a callable model. But a
    // trailing `*` (`claude-sonnet-4-6*`) is a model-group alias,
    // so only skip the `provider/*` form.
    if (model.id.includes('/*')) {
      wildcards++
      continue
    }
    // `includeModels`/`excludeModels` let one LiteLLM proxy be split
    // across several OpenCode providers (e.g. by upstream naming
    // prefix) without hand-maintaining a model list.
    if (!passesModelFilter(model.id, filters.includeModels, filters.excludeModels)) {
      filtered++
      continue
    }
    const info = infoByName?.get(model.id)
    if (infoByName && !info) unmatched.push(model.id)
    const entry = toModelInfo(
      enrichModel(model, info, capabilities[model.id]),
      providerId,
      info,
    )
    if (!entry) {
      skipped++
      continue
    }
    built[model.id] = entry
  }

  if (unmatched.length > 0) {
    log(
      'warn',
      `[opencode-litellm] /v1/model/info has no entry for ${unmatched.length} model(s) on provider "${providerId}"; ` +
        `classification uses id heuristics for: ${unmatched.slice(0, 5).join(', ')}` +
        (unmatched.length > 5 ? `, +${unmatched.length - 5} more` : ''),
    )
  }

  // Only blame the filters when every non-wildcard model was rejected by
  // them — if some hit `skipped` (non-chat) instead, `built` being empty
  // has an unrelated cause and this warning would misdirect the user.
  if (filtered > 0 && filtered + wildcards === discovered.length) {
    log(
      'warn',
      `[opencode-litellm] includeModels/excludeModels filtered out all ${filtered} model(s) discovered for provider "${providerId}" — check the glob patterns in options.includeModels/options.excludeModels.`,
    )
  }

  log(
    'info',
    `[opencode-litellm] Discovered ${discovered.length} models for provider "${providerId}" from ${baseURL} ` +
      `(${Object.keys(built).length} built` +
      (skipped > 0 ? `, ${skipped} non-chat hidden` : '') +
      (wildcards > 0 ? `, ${wildcards} wildcard ignored` : '') +
      (filtered > 0 ? `, ${filtered} filtered by includeModels/excludeModels` : '') +
      ')',
  )

  return built
}

/**
 * Revalidate a provider's model cache off the critical path (SWR), then
 * replay the V2 provider transform so the live model picker sees the refresh.
 */
async function backgroundRefresh(
  cacheKey: string,
  state: ProviderState,
  refreshInFlight: Set<string>,
  reload: () => Promise<void>,
): Promise<void> {
  if (refreshInFlight.has(cacheKey)) return
  // Skip if the cache was refreshed recently — a burst of new sessions
  // shouldn't hammer the proxy with health checks and discovery calls.
  const savedAt = readModelCacheSavedAt(cacheKey)
  if (savedAt !== null && Date.now() - savedAt < REFRESH_MIN_INTERVAL_MS) {
    return
  }
  refreshInFlight.add(cacheKey)
  try {
    const built = await withTimeout(
      discoverModels(
        state.baseURL,
        state.apiKey,
        state.customHeaders,
        state.providerId,
        state.filters,
        state.capabilities,
      ),
      DISCOVERY_TIMEOUT_MS,
    )
    if (built && Object.keys(built).length > 0) {
      state.models = built
      writeModelCache(cacheKey, built)
      await reload()
      log(
        'info',
        `[opencode-litellm] Background-refreshed models for ${state.baseURL} (${Object.keys(built).length} models)`,
      )
    }
  } catch {
    // Best-effort — a failed refresh just leaves the stale cache in place.
  } finally {
    refreshInFlight.delete(cacheKey)
  }
}

/**
 * LiteLLM Plugin for OpenCode.
 *
 * Uses the OpenCode V2 provider transform API to discover models from a
 * LiteLLM proxy and add them to the provider catalog.
 *
 * Configure the plugin in your `opencode.json`:
 *
 * {
 *   "plugins": [
 *     {
 *       "package": "opencode-plugin-litellm@latest",
 *       "options": {
 *         "baseURL": "http://localhost:4000/v1",
 *         "apiKey": "{env:LITELLM_API_KEY}"
 *       }
 *     }
 *   ]
 * }
 */
export const LiteLLMPlugin = Plugin.define({
  id: 'litellm',
  async setup(ctx) {
    log('info', `[opencode-litellm] Loading for ${ctx.location.directory}`)
    // A short-lived transform gives setup access to configured providers,
    // including inactive providers that the public list endpoint omits.
    let configuredProviders: Provider.Info[] = []
    const inspection = await ctx.provider.transform((editor) => {
      configuredProviders = editor.list().map((record) => record.provider)
    })
    await inspection.dispose()

    const matches = configuredProviders.filter((provider) =>
      isLiteLLMProvider(provider.id, provider.settings ?? {}),
    )
    const candidates: ProviderCandidate[] = []
    const optionProviders = isRecord(ctx.options.providers)
      ? ctx.options.providers
      : undefined

    if (optionProviders) {
      for (const [id, options] of Object.entries(optionProviders)) {
        if (!isRecord(options)) continue
        candidates.push({
          id,
          provider: matches.find((provider) => provider.id === id),
          options,
        })
      }
    } else if (
      ['baseURL', 'apiKey', 'includeModels', 'excludeModels', 'modelCapabilities'].some(
        (key) => key in ctx.options,
      )
    ) {
      const id =
        typeof ctx.options.providerId === 'string'
          ? ctx.options.providerId
          : CHAT_PROVIDER_ID
      candidates.push({
        id,
        provider: matches.find((provider) => provider.id === id),
        options: { ...ctx.options },
      })
    }

    for (const provider of matches) {
      if (candidates.some((candidate) => candidate.id === provider.id)) continue
      candidates.push({ id: provider.id, provider, options: {} })
    }
    if (candidates.length === 0) {
      candidates.push({ id: CHAT_PROVIDER_ID, options: {} })
    }
    const states = new Map<string, ProviderState>()

    for (const candidate of candidates) {
      const provider = candidate.provider
      const providerId = candidate.id
      const settings = {
        ...(provider?.settings ?? {}),
        ...candidate.options,
      } as Record<string, unknown>
      const configuredBase =
        typeof settings.baseURL === 'string' ? settings.baseURL : undefined
      const configuredKey =
        typeof settings.apiKey === 'string' && settings.apiKey
          ? settings.apiKey
          : undefined
      const envKey =
        process.env.LITELLM_API_KEY || process.env.LITELLM_MASTER_KEY || undefined

      let connectedKey: string | undefined
      try {
        const connection = await ctx.integration.connection.active(
          provider?.integrationID ?? providerId,
        )
        const credential = connection
          ? await ctx.integration.connection.resolve(connection)
          : undefined
        if (credential?.type === 'key') connectedKey = credential.key
      } catch {
        // A custom provider may not have an integration. Config/env auth
        // remains fully supported in that case.
      }

      const apiKey = configuredKey ?? envKey ?? connectedKey
      const legacyHeaders = readCustomHeaders(settings)
      const customHeaders = {
        ...(provider?.headers ?? {}),
        ...(isRecord(settings.headers) ? readCustomHeaders({ customHeaders: settings.headers }) : {}),
        ...(legacyHeaders ?? {}),
      }
      const discoveryHeaders =
        Object.keys(customHeaders).length > 0 ? customHeaders : undefined
      const filters = readModelFilters(settings)
      const capabilities = parseModelCapabilities(settings.modelCapabilities)
      const baseURL = configuredBase
        ? normalizeBaseURL(configuredBase)
        : await autoDetectLiteLLM(apiKey, discoveryHeaders)

      if (!baseURL) {
        log(
          'warn',
          `[opencode-litellm] No LiteLLM proxy found for provider "${providerId}". Configure plugin options.baseURL or start LiteLLM on port 4000/8000/8080.`,
        )
        continue
      }

      const cacheKey = buildCacheKey(providerId, baseURL, filters, capabilities)
      const cached = readModelCache(cacheKey) as Record<string, Model.Info> | null
      let models = cached && Object.keys(cached).length > 0 ? cached : null

      if (models) {
        log(
          'info',
          `[opencode-litellm] Loaded ${Object.keys(models).length} models from cache for provider "${providerId}" (${baseURL}).`,
        )
      } else {
        models = await withTimeout(
          discoverModels(
            baseURL,
            apiKey,
            discoveryHeaders,
            providerId,
            filters,
            capabilities,
          ),
          DISCOVERY_TIMEOUT_MS,
        )
        if (models && Object.keys(models).length > 0) {
          writeModelCache(cacheKey, models)
        }
      }

      if (!models || Object.keys(models).length === 0) continue
      states.set(cacheKey, {
        baseURL,
        apiKey,
        customHeaders: discoveryHeaders,
        filters,
        capabilities,
        providerId,
        name:
          typeof settings.name === 'string'
            ? settings.name
            : provider?.name ?? 'LiteLLM (proxy)',
        package:
          typeof settings.package === 'string'
            ? settings.package
            : provider?.package || '@opencode/ai/providers/openai-compatible',
        models,
      })
    }

    await ctx.provider.transform((editor) => {
      for (const state of states.values()) {
        const current = editor.get(state.providerId)
        if (!current) {
          const providerID = Provider.ID.make(state.providerId)
          const info: Provider.Info = {
            ...Provider.Info.empty(providerID),
            name: state.name,
            activation: 'enabled',
            package: state.package,
            settings: {
              baseURL: `${state.baseURL}/v1`,
              ...(state.apiKey ? { apiKey: state.apiKey } : {}),
            },
            ...(state.customHeaders ? { headers: state.customHeaders } : {}),
          }
          editor.add({ info, models: Object.values(state.models) })
          continue
        }

        editor.update(state.providerId, (info) => {
          if (!info.package) {
            info.package = state.package
          }
          info.settings = {
            ...info.settings,
            baseURL: info.settings?.baseURL ?? `${state.baseURL}/v1`,
            ...(info.settings?.apiKey || !state.apiKey
              ? {}
              : { apiKey: state.apiKey }),
          }
          if (state.customHeaders) {
            info.headers = { ...info.headers, ...state.customHeaders }
          }
        })

        // Preserve hand-curated config models. Discovered definitions only
        // fill IDs that no earlier source or transform already supplied.
        const existing = [...current.models.values()]
        const existingIds = new Set(existing.map((model) => model.id))
        const discovered = Object.values(state.models).filter(
          (model) => !existingIds.has(model.id),
        )
        editor.models.set(state.providerId, [...existing, ...discovered])
      }
    })

    const controller = new AbortController()
    const refreshInFlight = new Set<string>()
    void (async () => {
      try {
        for await (const event of ctx.event.subscribe({ signal: controller.signal })) {
          if (event.type !== 'session.created') continue
          for (const [cacheKey, state] of states) {
            void backgroundRefresh(
              cacheKey,
              state,
              refreshInFlight,
              () => ctx.provider.reload(),
            )
          }
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          log(
            'warn',
            `[opencode-litellm] Event subscription stopped: ${error instanceof Error ? error.message : String(error)}`,
          )
        }
      }
    })()

    return () => controller.abort()
  },
})

// Kept as a named export for users that imported the old symbol directly.
export const LiteLLMResponsesPlugin = Plugin.define({
  id: 'litellm.responses',
  setup() {},
})

export default LiteLLMPlugin
