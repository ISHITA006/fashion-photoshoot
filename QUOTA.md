# Fixing "Insufficient quota" (429) errors

When you see **429** or "insufficient quota", the API key you’re using has no remaining free quota (or no billing). Fix it by enabling billing or adding credits for the provider you use.

---

## Google Gemini (current default)

- **Error:** `generate_content_free_tier_requests, limit: 0` or "You exceeded your current quota"
- **Cause:** Free tier is exhausted or not available for this project.
- **Fix:**
  1. Open [Google AI Studio](https://aistudio.google.com/) or [Google Cloud Console](https://console.cloud.google.com/).
  2. Enable **billing** for the project that owns your API key (pay-as-you-go gives higher quota).
  3. Or try another model in `agents.py`, e.g. `GEMINI_LLM = "gemini-1.5-flash-8b"` (sometimes has separate free limits).

---

## OpenAI

- **Error:** `insufficient_quota` or "You exceeded your current quota"
- **Fix:**
  1. Go to [OpenAI Platform → Billing](https://platform.openai.com/account/billing).
  2. Add a payment method and purchase credits (or upgrade plan).
  3. To use OpenAI for agents, set in `agents.py`:  
     `GEMINI_LLM = "openai/gpt-4o-mini"` (and use that for all agents).

---

## Anthropic (Claude)

- **Error:** "Your credit balance is too low"
- **Fix:**
  1. Go to [Anthropic Console → Plans & Billing](https://console.anthropic.com/).
  2. Add credits or upgrade your plan.
  3. To use Claude for agents, set in `agents.py`:  
     `GEMINI_LLM = "anthropic/claude-sonnet-4-20250514"` (and use that for all agents).

---

## Summary

- **Agents** use the LLM set in `agents.py` (e.g. `GEMINI_LLM`). They use **GOOGLE_API_KEY** when the model is Gemini.
- **Vision and QC tools** still call the Anthropic API inside their code; if those fail, you’d need to switch those tools to another vision API (e.g. Gemini) or add Anthropic credits.
