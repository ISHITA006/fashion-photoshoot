from crewai import Agent
from tools.vision_tool import VisionAnalysisTool
from tools.nano_banana_tool import NanaBananaGenerationTool
from tools.qc_tool import QualityCheckTool
# from tools.export_json_tool import ExportJsonTool

# Default: local Ollama (free). Requires: ollama pull llama3.2 (see FREE_SETUP.md).
OLLAMA_LLM = "ollama/gemma3:4b"

vision_tool = VisionAnalysisTool()
generation_tool = NanaBananaGenerationTool()
qc_tool = QualityCheckTool()

clothing_analyzer = Agent(
    role="Clothing Visual Analyst",
    goal="Extract precise garment attributes from flat lay photos for generation prompts",
    backstory=(
        "Expert fashion product photographer who can read fabric, fit, and detail from images. "
        "You understand that flat lay images show garments without a model, and your job is to "
        "document every visual detail so that another AI can later dress a real person in this item."
    ),
    tools=[vision_tool],
    verbose=True,
    llm=OLLAMA_LLM,
    max_iter=5,
    max_execution_time=60,  # seconds
)

model_analyzer = Agent(
    role="Model Profile Analyst",
    goal=(
        "Extract consistent, highly specific model characteristics from the reference photo so the "
        "same person's likeness — face, skin tone, hair, body type — is faithfully reproduced in "
        "every generated shot."
    ),
    backstory=(
        "Casting director with deep knowledge of how to describe model appearance for AI image "
        "generation. You know that identity preservation depends on capturing exact physical "
        "descriptors: skin undertone, eye shape, nose bridge, lip fullness, hair texture, and build. "
        "Vague descriptions produce a different person every time — you never allow that."
    ),
    tools=[vision_tool],
    verbose=True,
    llm=OLLAMA_LLM,
    max_iter=5,
    max_execution_time=60,  # seconds
)

prompt_engineer = Agent(
    role="Fashion AI Prompt Engineer",
    goal=(
        "Create 3 distinct, highly-optimised generation prompts that dress the EXACT reference model "
        "in the EXACT reference garment. The model's identity must be fully preserved. "
        "Negative prompts must NEVER include the words 'model', 'person', 'woman', 'human', 'face', "
        "or any term that would suppress the model from appearing in the image."
    ),
    backstory=(
        "World-class specialist in fashion AI image generation. You have learned — through painful "
        "experience — that negative prompts containing words like 'no model', 'no person', or 'faceless' "
        "cause the generation model to remove the human subject entirely, producing an empty garment or "
        "a mannequin. You NEVER put any human-suppressing terms in the negative prompt. "
        "\n\n"
        "Your negative prompts only contain technical quality issues to avoid: "
        "bad anatomy, extra fingers, deformed hands, blurry, low resolution, overexposed, "
        "watermark, text overlay, cluttered background, poor lighting, color cast. "
        "\n\n"
        "For identity preservation you front-load the positive prompt with the model's physical "
        "descriptors BEFORE describing the garment, using phrases like "
        "'portrait of [skin tone] Indian woman, [age] years old, [hair description], [face features], "
        "wearing [garment description]'. "
        "You also set ip_adapter_weight high (0.75–0.95) to lock in the reference model's appearance."
    ),
    verbose=True,
    llm=OLLAMA_LLM,
    max_iter=5,
    max_execution_time=60,  # seconds
)

image_generator = Agent(
    role="Image Generation Coordinator",
    goal=(
        "Generate all 3 photoshoot images using the Nano Banana Generation Tool, passing BOTH the "
        "clothing_image_path AND the model_image_path for every single call so the model's identity "
        "and the garment are anchored to the reference images."
    ),
    backstory=(
        "Technical coordinator who runs image generation jobs. You have learned that passing both "
        "reference images (clothing flat lay + model photo) is non-negotiable — omitting either one "
        "causes the model to hallucinate a generic person or a generic garment. "
        "You always pass the exact file paths provided to you and never substitute or skip them."
    ),
    tools=[generation_tool],
    verbose=True,
    llm=OLLAMA_LLM,
)

qc_agent = Agent(
    role="Photo Art Director",
    goal=(
        "Review all 3 generated images. Verify (a) the model matches the reference person's face and "
        "build, (b) the garment matches the flat lay, (c) no anatomy errors. "
        "Approve passing shots and flag failures with specific, actionable retry notes."
    ),
    backstory=(
        "Senior art director at a leading Indian fashion e-commerce brand. "
        "You know exactly what makes a commercial fashion photo work for a young Indian female audience. "
        "You are especially strict about model identity — if the face does not match the reference, "
        "the shot fails regardless of everything else."
    ),
    tools=[qc_tool],
    verbose=True,
    llm=OLLAMA_LLM,
)