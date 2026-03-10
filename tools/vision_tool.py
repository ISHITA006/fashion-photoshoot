import os
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel

class VisionInput(BaseModel):
    image_path: str
    analysis_type: str  # "clothing" or "model"
    product_brief: str = ""


CLOTHING_PROMPT_TEMPLATE = """Analyze this flat lay clothing image. Product brief: {product_brief}

First, identify the garment type (e.g. shirt, top, dress, kurta, skirt, pants, shorts, co-ord, jacket, etc.).

Then return ONLY a valid JSON object with these universal fields:
  garment_type, primary_color, secondary_colors, fabric_texture, fit_silhouette,
  key_details (array of 5+ specific observations covering the hero design features —
    e.g. prints, embroidery, cutwork, pleats, buttons, zippers, pockets, lace, ruffles,
    drawstrings, waistbands, leg openings, slits, etc. — whatever is most prominent),
  styling_notes, brand_aesthetic, target_demographic

Then add ALL garment-type-specific fields that apply:
  - If top / shirt / blouse / kurta / jacket: sleeve_style, neckline, hem_style, closure_type
  - If dress / tunic / co-ord top: sleeve_style, neckline, hem_style, waist_definition, length
  - If skirt: waist_style, silhouette, length, hem_style
  - If pants / trousers / palazzos: waist_style, leg_style, rise, length, hem_style
  - If shorts: waist_style, leg_style, length, hem_style

Do NOT force fields that don't apply (e.g. do not add neckline for pants).
Focus key_details on what makes this specific product visually distinctive."""

MODEL_PROMPT = """Analyze this model reference photo. Your output will be used for AI identity \
preservation — the goal is to reproduce this exact person's appearance in generated images.

Return ONLY a valid JSON object with these exact keys:
  age_range, ethnicity,
  skin_tone (use Fitzpatrick scale + undertone, e.g. 'warm medium-brown, Fitzpatrick IV, golden undertone'),
  eye_color, eye_shape, nose_description, lip_fullness, face_shape,
  hair_color, hair_length, hair_style, hair_texture,
  body_type, height_impression,
  distinctive_features (array of 3+ specific traits), expression_style, pose_style

Use precise, specific descriptors — no vague terms like 'medium skin' or 'average build'.
The more specific your description, the better the identity match will be."""

def _run_vision_ollama(image_path: str, analysis_type: str, product_brief: str) -> str:
    """Use local Ollama + Gemma 3 (vision). Requires: ollama pull gemma3:4b"""
    import ollama

    prompt = (
        CLOTHING_PROMPT_TEMPLATE.format(product_brief=product_brief)
        if analysis_type == "clothing"
        else MODEL_PROMPT
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
    
class VisionAnalysisTool(BaseTool):
    name: str = "Vision Analysis Tool"
    description: str = (
        "Analyzes clothing flat lay images or model reference photos using vision AI. "
        "For clothing: extracts garment attributes for generation prompts. "
        "For model: extracts precise physical descriptors for identity preservation."
    )
    args_schema: Type[BaseModel] = VisionInput

    def _run(self, image_path: str, analysis_type: str, product_brief: str = "") -> str:
        if os.environ.get("USE_ANTHROPIC_VISION") and os.environ.get("ANTHROPIC_API_KEY"):
            return _run_vision_anthropic(image_path, analysis_type, product_brief)
        return _run_vision_ollama(image_path, analysis_type, product_brief)