# Run the workflow for free (no API credits)

Everything except **image generation** can run locally with **Ollama**—no API keys or billing.

## 1. Install Ollama

- **macOS:** [ollama.com](https://ollama.com) → Download, then open the app.
- **Linux:** `curl -fsSL https://ollama.com/install.sh | sh`

## 2. Pull the models

In a terminal:

```bash
# For agent reasoning (text)
ollama pull llama3.2

# For clothing/model analysis and QC (vision)
ollama pull llava
```

`llama3.2` is ~2GB; `llava` is ~4.7GB. Keep Ollama running (it runs in the background after install).

## 3. Install Python dependencies

In your project venv:

```bash
pip install -r requirements.txt
# or: uv pip install -r requirements.txt
```

## 4. Run the app

No `.env` needed for the free path. From the project root:

```bash
python main.py
```

- **Agents** use local Ollama (`llama3.2`) via LiteLLM.
- **Vision and QC tools** use local Ollama (`llava`) for image analysis.

## 5. Image generation (still needs Google)

The **Nano Banana** step that creates the final photos uses Google’s Gemini image API and still requires:

- `GOOGLE_API_KEY` in `.env`, and  
- Quota/billing on that project.

If you don’t set `GOOGLE_API_KEY` or quota is exceeded, that step will fail; the rest of the pipeline (clothing analysis, model analysis, prompt engineering) will run locally for free.

To use **cloud APIs** again (OpenAI, Anthropic, Gemini) when you have credits:

- In `agents.py` change `OLLAMA_LLM` to `GEMINI_LLM` (or set `OPENAI_API_KEY` and use an OpenAI model).
- For vision/QC to use Claude: set `USE_ANTHROPIC_VISION=1` and `ANTHROPIC_API_KEY` in `.env`.
