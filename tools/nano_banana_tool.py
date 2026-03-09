from typing import Type
from crewai.tools import BaseTool
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel
import os, json

class GenerationInput(BaseModel):
    prompt: str
    shot_number: int
    clothing_image_path: str
    model_image_path: str
    aspect_ratio: str = "2:3"    # portrait, good for fashion
    resolution: str = "2K"

class NanaBananaGenerationTool(BaseTool):
    name: str = "Nano Banana Generation Tool"
    description: str = "Generates photorealistic fashion images using Nano Banana (Gemini image generation)"
    args_schema: Type[BaseModel] = GenerationInput

    def _run(self, prompt: str, shot_number: int,
             clothing_image_path: str, model_image_path: str,
             aspect_ratio: str = "2:3", resolution: str = "2K") -> str:

        client = genai.Client()

        clothing_image = Image.open(clothing_image_path)
        model_image = Image.open(model_image_path)

        # Nano Banana accepts images directly alongside the text prompt.
        # The model natively understands to dress the model in the clothing.
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            # model="gemini-3.1-flash-image-preview",   # Nano Banana 2
            contents=[
                prompt,
                clothing_image,   # reference 1: the garment
                model_image,      # reference 2: the model
            ],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=resolution,
                ),
                # thinking_config=types.ThinkingConfig(
                #     thinking_level="High"   # better composition reasoning
                # )
            ),
        )

        os.makedirs("outputs", exist_ok=True)
        output_path = f"outputs/shot_{shot_number}.png"

        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                image.save(output_path)
                break

        return json.dumps({
            "status": "success",
            "path": output_path,
            "shot": shot_number
        })