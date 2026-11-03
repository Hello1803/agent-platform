import { useEffect, useState } from "react";
import type { AgentApiClient, AiProvidersResponse } from "./apiClient";
import type { AgentSpecification, AgentSpecMutation } from "@agent-platform/shared-types";
import { parseModelString } from "./specToFlow";

export interface AiProviderSelectorProps {
  client: AgentApiClient;
  spec: AgentSpecification;
  onMutate: (mutations: AgentSpecMutation[]) => void;
}

/**
 * Lets a person pick which AI provider an agent's AI nodes run through —
 * without ever seeing or setting a credential. This component only ever
 * reads/writes two non-secret things:
 *
 *   1. GET /ai/providers — which providers exist, which are actually
 *      configured server-side right now (a plain boolean, never a key —
 *      see backend/api/app.py's route docstring), and the server's
 *      current default.
 *   2. spec.runtimeConfig.model — a "<provider>:<model>" string
 *      (validated server-side by RuntimeConfig's field_validator in
 *      backend/models/agent_spec.py), sent via the existing
 *      update_runtime_config mutation. This is a provider *preference*,
 *      not a credential — the actual API key resolution happens entirely
 *      server-side (backend/ai_providers/registry.py) regardless of what
 *      this field says.
 *
 * A provider whose credentials aren't configured on the server is shown
 * but disabled, with an explanatory title — better than a confusing
 * failure after the fact.
 */
export function AiProviderSelector({ client, spec, onMutate }: AiProviderSelectorProps) {
  const [status, setStatus] = useState<AiProvidersResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    client
      .getAiProviders()
      .then((result) => {
        if (!cancelled) setStatus(result);
      })
      .catch(() => {
        if (!cancelled) setLoadError("Could not load provider status");
      });
    return () => {
      cancelled = true;
    };
  }, [client]);

  if (loadError) return null; // fails soft — this is a convenience control, not core functionality
  if (!status) return <span style={{ fontSize: 11, color: "var(--ac-ink-muted, #6b7280)" }}>Loading providers…</span>;

  const [specProvider] = parseModelString(spec.runtimeConfig?.model);
  const activeProvider = specProvider ?? status.current;

  const DEFAULT_MODEL_BY_PROVIDER: Record<string, string> = { anthropic: "claude-sonnet-5", openai: "gpt-4o", lmstudio: "local-model" };

  const handleChange = (providerId: string) => {
    // Preserve whatever model name is already set, if any; otherwise use
    // a sensible per-provider default so the value sent is always a real
    // "provider:model" string — a bare provider name with no model
    // suffix would parse server-side as having NO provider prefix at all
    // (see RuntimeConfig.model's validator / parse_model_string), which
    // would silently fail to select anything.
    const [, existingModel] = parseModelString(spec.runtimeConfig?.model);
    const modelName = existingModel || DEFAULT_MODEL_BY_PROVIDER[providerId] || "default";
    onMutate([{ kind: "update_runtime_config", patch: { model: `${providerId}:${modelName}` } }]);
  };

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
      <span style={{ color: "var(--ac-ink-muted, #6b7280)" }}>AI provider</span>
      <select
        value={activeProvider}
        onChange={(e) => handleChange(e.target.value)}
        style={{ fontSize: 12, padding: "3px 6px", borderRadius: 5, border: "1px solid var(--ac-border, #d8dad5)" }}
      >
        {status.providers.map((provider) => (
          <option
            key={provider.id}
            value={provider.id}
            disabled={provider.status === "coming_soon" || !provider.hasCredentials}
          >
            {provider.id.charAt(0).toUpperCase() + provider.id.slice(1)}
            {provider.status === "coming_soon"
              ? " (coming soon)"
              : !provider.hasCredentials
                ? " (no API key configured)"
                : provider.id === status.current
                  ? " (server default)"
                  : ""}
          </option>
        ))}
      </select>
    </div>
  );
}
