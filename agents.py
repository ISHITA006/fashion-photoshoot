from crewai import Agent
from tools.vision_tool import VisionAnalysisTool
from tools.nano_banana_tool import NanaBananaGenerationTool
from tools.qc_tool import QualityCheckTool
# from tools.export_json_tool import ExportJsonTool

# Default: local Ollama (free). Requires: ollama pull llama3.2 (see FREE_SETUP.md).
# To use cloud instead: set OPENAI_API_KEY or GOOGLE_API_KEY and use e.g. "gemini-2.0-flash-001".
OLLAMA_LLM = "ollama/llama3.2"
GEMINI_LLM = "gemini-2.0-flash-001"

vision_tool = VisionAnalysisTool()
generation_tool = NanaBananaGenerationTool()
qc_tool = QualityCheckTool()
# export_json_tool = ExportJsonTool()

clothing_analyzer = Agent(
    role="Clothing Visual Analyst",
    goal="Extract precise garment attributes from flat lay photos for generation prompts",
    backstory="Expert fashion product photographer who can read fabric, fit, and detail from images",
    tools=[vision_tool],
    verbose=True,
    llm=OLLAMA_LLM,
)

model_analyzer = Agent(
    role="Model Profile Analyst",
    goal="Extract consistent model characteristics to preserve identity across all shots",
    backstory="Casting director with deep knowledge of how to describe model appearance for AI",
    tools=[vision_tool],
    verbose=True,
    llm=OLLAMA_LLM,
)

prompt_engineer = Agent(
    role="Fashion AI Prompt Engineer",
    goal="Create 3 distinct, optimized generation prompts for Nano Banana models",
    backstory="""Specialist in Nano Banana diffusion models and fashion photography composition.
    Knows exactly how to translate garment and model profiles into prompts that produce
    studio-quality, commercially viable photoshoot images.""",
    # tools=[export_json_tool],
    verbose=True,
    llm=OLLAMA_LLM,
)

image_generator = Agent(
    role="Image Generation Coordinator",
    goal="Generate all 3 photoshoot images using Nano Banana API",
    backstory="Technical coordinator who runs parallel image generation jobs",
    tools=[generation_tool],
    verbose=True,
    llm=OLLAMA_LLM,
)

qc_agent = Agent(
    role="Photo Art Director",
    goal="Review all generated images, approve passing shots, flag failures for retry",
    backstory="Senior art director who knows exactly what makes a commercial fashion photo work",
    tools=[qc_tool],
    verbose=True,
    llm=OLLAMA_LLM,
)