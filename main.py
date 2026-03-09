from dotenv import load_dotenv

load_dotenv()  # Load before CrewAI so MODEL / API keys are set when agents are created

from crew import build_crew

if __name__ == "__main__":
    result = build_crew(
        clothing_path="inputs/clothing_flatlay.jpeg",
        model_path="inputs/model_photo.jpg",
        product_brief="""
            All over embroidery shirt on a beige Cambric Base ,, Target is a Young Indian Woman,, between 18-27
        """
    ).kickoff()

    print("\n✅ Final Output:\n", result)