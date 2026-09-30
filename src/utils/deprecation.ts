/**
 * Sunset notice for the deprecated `@cardinal4/opencode-plugin-litellm` fork.
 *
 * Upstream (`yuseferi/opencode-litellm`) now ships official OpenCode V2
 * support, so this package is no longer maintained. Logged on every plugin
 * load through both the V1 and V2 entrypoints.
 */
export const DEPRECATION_MESSAGE =
  '@cardinal4/opencode-plugin-litellm is deprecated: upstream opencode-plugin-litellm now supports OpenCode V2 natively. ' +
  'Migrate by replacing "@cardinal4/opencode-plugin-litellm" with "opencode-plugin-litellm" in your opencode.json plugins array. ' +
  'See https://www.npmjs.com/package/opencode-plugin-litellm'

let warned = false

/** Print the deprecation notice once per process. */
export function logDeprecation(): void {
  if (warned) return
  warned = true
  console.warn(`[opencode-litellm] DEPRECATED — ${DEPRECATION_MESSAGE}`)
}
