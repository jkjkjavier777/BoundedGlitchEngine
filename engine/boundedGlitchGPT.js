// engine/boundedGlitchGPT.js
// Bridge from BoundedGlitchEngine to The-BoundedGlitchGPT over HTTP.

const BGE_URL = process.env.BOUNDED_GLITCHGPT_URL || 'http://127.0.0.1:8000';
const GENERATE_ENDPOINT = `${BGE_URL}/generate`;
const DEFAULT_MAX_TOKENS = Number(process.env.BOUNDED_GLITCHGPT_MAX_TOKENS || 150);
const REQUEST_TIMEOUT = Number(process.env.BOUNDED_GLITCHGPT_TIMEOUT || 30000);

async function generate(prompt, options = {}) {
  if (typeof prompt !== 'string' || prompt.trim().length === 0) {
    throw new Error('BoundedGlitchGPT requires a non-empty prompt.');
  }

  const body = {
    prompt,
    max_tokens: Number(options.maxTokens) || DEFAULT_MAX_TOKENS,
  };
  if (options.temperature !== undefined) body.temperature = options.temperature;
  if (options.topK !== undefined) body.top_k = options.topK;
  if (options.topP !== undefined) body.top_p = options.topP;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

  try {
    const response = await fetch(GENERATE_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`BoundedGlitchGPT server returned ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    if (!data || typeof data.text !== 'string') {
      throw new Error('BoundedGlitchGPT returned an invalid response.');
    }
    return data.text.trim();
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error(`BoundedGlitchGPT request timed out after ${REQUEST_TIMEOUT}ms.`);
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function isAvailable() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch(`${BGE_URL}/health`, { signal: controller.signal });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

function getConfig() {
  return {
    url: BGE_URL,
    endpoint: GENERATE_ENDPOINT,
    maxTokens: DEFAULT_MAX_TOKENS,
    timeout: REQUEST_TIMEOUT,
  };
}

module.exports = { generate, isAvailable, getConfig };
