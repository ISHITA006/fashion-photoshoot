import os
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel

class VisionInput(BaseModel):
    image_path: str
    analysis_type: str   # "clothing" or "model"
    product_brief: str = ""

def _run_vision_ollama(image_path: str, analysis_type: str, product_brief: str) -> str:
    """Use local Ollama + LLaVA (free). Requires: ollama pull llava"""
    import ollama
    if analysis_type == "clothing":
        prompt = f"""Analyze this flat lay clothing image. Product brief: {product_brief}
Return JSON with: garment_type, primary_color, secondary_colors, fabric_texture,
fit_silhouette, key_details (list), styling_notes, brand_aesthetic"""
    else:
        prompt = """Analyze this model photo. Return JSON with: age_range, skin_tone,
hair_color, hair_length, hair_style, body_type, distinctive_features, pose_style"""
    try:
        r = ollama.chat(
            model="llava",
            messages=[{"role": "user", "content": prompt, "images": [image_path]}],
        )
        return r["message"]["content"] or "{}"
    except Exception as e:
        return f'{{"error": "Ollama vision failed. Run: ollama pull llava. Details: {e!s}"}}'

def _run_vision_anthropic(image_path: str, analysis_type: str, product_brief: str) -> str:
    """Use Anthropic Claude (requires ANTHROPIC_API_KEY and credits)."""
    import anthropic
    import base64
    client = anthropic.Anthropic()
    with open(image_path, "rb") as f:
        img_data = base64.standard_b64encode(f.read()).decode("utf-8")
    if analysis_type == "clothing":
        prompt = f"""Analyze this flat lay clothing image. Product brief: {product_brief}
Return JSON with: garment_type, primary_color, secondary_colors, fabric_texture,
fit_silhouette, key_details (list), styling_notes, brand_aesthetic"""
    else:
        prompt = """Analyze this model photo. Return JSON with: age_range, skin_tone,
hair_color, hair_length, hair_style, body_type, distinctive_features, pose_style"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}},
            {"type": "text", "text": prompt}
        ]}]
    )
    return response.content[0].text

class VisionAnalysisTool(BaseTool):
    name: str = "Vision Analysis Tool"
    description: str = "Analyzes clothing or model images using vision AI"
    args_schema: Type[BaseModel] = VisionInput

    def _run(self, image_path: str, analysis_type: str, product_brief: str = "") -> str:
        # Free path: Ollama + llava. Cloud path: Anthropic when key is set.
        if os.environ.get("USE_ANTHROPIC_VISION") and os.environ.get("ANTHROPIC_API_KEY"):
            return _run_vision_anthropic(image_path, analysis_type, product_brief)
        return _run_vision_ollama(image_path, analysis_type, product_brief)
