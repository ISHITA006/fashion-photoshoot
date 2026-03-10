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
    """Use local Ollama + Gemma 3 (vision). Requires: ollama pull gemma3:4b"""
    import ollama
    prompt = (
        f"QC check this fashion photo. Garment profile: {garment_profile}. "
        f"Model profile: {model_profile}.\n"
        'Return JSON only: { "pass": true/false, "garment_visible": true/false, '
        '"model_face_clear": true/false, "hands_ok": true/false, '
        '"clean_background": true/false, "issues": ["list any problems"] }'
    )
    try:
        r = ollama.chat(
            model="gemma3:4b",  # was "llava"
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path],
                }
            ],
        )
        return r["message"]["content"] or "{}"
    except Exception as e:
        return (
            '{"error": '
            '"Ollama Gemma 3 vision failed. '
            'Run: ollama pull gemma3:4b. '
            f'Details: {e!s}"}}'
        )

class QualityCheckTool(BaseTool):
    name: str = "Quality Check Tool"
    description: str = (
        "Checks generated image quality against reference profiles. "
        "Requires a JSON arguments object with exactly these fields: "
        '{"image_path": string, "shot_number": integer, '
        '"garment_profile": string, "model_profile": string}. '
        "Do NOT pass tool schemas or objects with 'properties' / 'required' keys — "
        "always pass concrete values for those four fields."
    )
    args_schema: Type[BaseModel] = QCInput

    def _run(self, image_path: str, shot_number: int,
             garment_profile: str, model_profile: str) -> str:
        if os.environ.get("USE_ANTHROPIC_VISION") and os.environ.get("ANTHROPIC_API_KEY"):
            return _run_qc_anthropic(image_path, shot_number, garment_profile, model_profile)
        return _run_qc_ollama(image_path, shot_number, garment_profile, model_profile)
