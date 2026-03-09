from crewai import Crew, Process
from agents import clothing_analyzer, model_analyzer, prompt_engineer, image_generator, qc_agent
from tasks import create_tasks

def build_crew(clothing_path, model_path, product_brief):
    tasks = create_tasks(clothing_path, model_path, product_brief)
    return Crew(
        agents=[clothing_analyzer, model_analyzer, prompt_engineer, image_generator, qc_agent],
        tasks=tasks,
        process=Process.sequential,   # switch to hierarchical for auto retry logic
        verbose=True
    )