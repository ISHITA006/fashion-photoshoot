import os
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel

class QCInput(BaseModel):
    image_path: str
    shot_number: int
    garment_profile: str
    model_profile: str

def _run_qc_ollama(image_path: str, shot_number: int, garment_profile: str, model_profile: str) -> str:
    """Use local Ollama + LLaVA (free). Requires: ollama pull llava"""
    import ollama
    prompt = f"""QC check this fashion photo. Garment profile: {garment_profile}. Model profile: {model_profile}.
Return JSON only: {{ "pass": true/false, "garment_visible": true/false, "model_face_clear": true/false,
"hands_ok": true/false, "clean_background": true/false, "issues": ["list any problems"] }}"""
    try:
        r = ollama.chat(
            model="llava",
            messages=[{"role": "user", "content": prompt, "images": [image_path]}],
        )
        return r["message"]["content"] or "{}"
    except Exception as e:
        return f'{{"error": "Ollama vision failed. Run: ollama pull llava. Details: {e!s}"}}'

def _run_qc_anthropic(image_path: str, shot_number: int, garment_profile: str, model_profile: str) -> str:
    """Use Anthropic Claude (requires ANTHROPIC_API_KEY and credits)."""
    import anthropic
    import base64
    client = anthropic.Anthropic()
    with open(image_path, "rb") as f:
        img_data = base64.standard_b64encode(f.read()).decode("utf-8")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}},
            {"type": "text", "text": f"""
QC check this fashion photo. Garment profile: {garment_profile}. Model profile: {model_profile}.
Return JSON: {{ "pass": true/false, "garment_visible": true/false, "model_face_clear": true/false,
"hands_ok": true/false, "clean_background": true/false, "issues": ["list any problems"] }}"""}
        ]}]
    )
    return response.content[0].text

class QualityCheckTool(BaseTool):
    name: str = "Quality Check Tool"
    description: str = "Checks generated image quality against reference profiles"
    args_schema: Type[BaseModel] = QCInput

    def _run(self, image_path: str, shot_number: int,
             garment_profile: str, model_profile: str) -> str:
        if os.environ.get("USE_ANTHROPIC_VISION") and os.environ.get("ANTHROPIC_API_KEY"):
            return _run_qc_anthropic(image_path, shot_number, garment_profile, model_profile)
        return _run_qc_ollama(image_path, shot_number, garment_profile, model_profile)
