import os
import json
from typing import Type

from crewai.tools import BaseTool
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel


class GenerationInput(BaseModel):
    prompt: str
    shot_number: int
    clothing_image_path: str
    model_image_path: str
    negative_prompt: str = (
        "bad anatomy, extra fingers, deformed hands, fused fingers, blurry, low resolution, "
        "overexposed, underexposed, watermark, text overlay, logo, cluttered background, "
        "poor lighting, color cast, grain, noise, jpeg artifacts, cropped head, out of frame"
    )
    aspect_ratio: str = "2:3"   # portrait — ideal for fashion
    resolution: str = "2K"


class NanaBananaGenerationTool(BaseTool):
    name: str = "Nano Banana Generation Tool"
    description: str = (
        "Generates photorealistic fashion try-on images using Nano Banana (Gemini image generation). "
        "Requires BOTH clothing_image_path (flat lay reference) AND model_image_path (identity reference). "
        "Both images are passed to the model on every call — never omit either one."
    )
    args_schema: Type[BaseModel] = GenerationInput

    def _run(
        self,
        prompt: str,
        shot_number: int,
        clothing_image_path: str,
        model_image_path: str,
        negative_prompt: str = (
            "bad anatomy, extra fingers, deformed hands, fused fingers, blurry, low resolution, "
            "overexposed, underexposed, watermark, text overlay, logo, cluttered background, "
            "poor lighting, color cast, grain, noise, jpeg artifacts, cropped head, out of frame"
        ),
        aspect_ratio: str = "2:3",
        resolution: str = "2K",
    ) -> str:

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

        clothing_image = Image.open(clothing_image_path)
        model_image = Image.open(model_image_path)

        # Build a structured generation prompt that explicitly instructs the model to:
        # 1. Use the provided person's exact appearance (identity preservation)
        # 2. Dress them in the provided garment
        # 3. Avoid the negative prompts
        generation_prompt = (
            f"{prompt}\n\n"
            f"REFERENCE IMAGE 1 (clothing): Use the exact garment shown — preserve all colors, "
            f"design details, fabric texture, and silhouette.\n"
            f"REFERENCE IMAGE 2 (model): Use this exact person — preserve their face, skin tone, "
            f"hair, and body type faithfully. Do not change their appearance.\n"
            f"Negative: {negative_prompt}"
        )

        response = client.models.generate_content(
            # model="gemini-2.5-flash-image",
            model="gemini-3.1-flash-image-preview",
            contents=[
                generation_prompt,
                clothing_image,  # Reference 1: the garment flat lay
                model_image,     # Reference 2: the model identity
            ],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=resolution,
                ),
            ),
        )

        os.makedirs("outputs", exist_ok=True)
        output_path = f"outputs/shot_{shot_number}.png"

        saved = False
        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                image.save(output_path)
                saved = True
                break

        if not saved:
            # Surface any text response for debugging
            text_parts = [p.text for p in response.parts if hasattr(p, "text") and p.text]
            debug_msg = " | ".join(text_parts) if text_parts else "No image or text returned."
            return json.dumps({
                "status": "error",
                "shot": shot_number,
                "message": f"No image generated. Model response: {debug_msg}",
            })

        return json.dumps({
            "status": "success",
            "path": output_path,
            "shot": shot_number,
        })