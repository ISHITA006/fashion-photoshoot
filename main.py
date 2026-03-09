from dotenv import load_dotenv

load_dotenv()  # Load before CrewAI so MODEL / API keys are set when agents are created

from crew import build_crew

if __name__ == "__main__":
    result = build_crew(
        clothing_path="inputs/clothing_flatlay.jpeg",
        model_path="inputs/model_photo.png",
        product_brief=open("inputs/product_brief.txt", "r").read()
    ).kickoff()

    print("\n✅ Final Output:\n", result)