from crewai import Crew, Process
from agents import stylist_prompt_planner, image_generator, qc_agent
from tasks import create_tasks


def build_crew(clothing_path, model_path, product_brief):
    tasks = create_tasks(clothing_path, model_path, product_brief)
    return Crew(
        agents=[stylist_prompt_planner, image_generator, qc_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )
