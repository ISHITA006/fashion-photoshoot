from crewai import Task
from agents import clothing_analyzer, model_analyzer, prompt_engineer, image_generator, qc_agent

def create_tasks(clothing_path, model_path, product_brief):

    analyze_clothing = Task(
        description=f"""Analyze the clothing flat lay at: {clothing_path}
        Product brief: {product_brief}
        Use the Vision Analysis Tool with analysis_type='clothing'.
        Return the full garment profile JSON.""",
        expected_output="Detailed garment profile JSON",
        agent=clothing_analyzer
    )

    analyze_model = Task(
        description=f"""Analyze the model photo at: {model_path}
        Use the Vision Analysis Tool with analysis_type='model'.
        Return the full model profile JSON.""",
        expected_output="Detailed model profile JSON",
        agent=model_analyzer
    )

    create_prompts = Task(
        description="""Using the garment profile and model profile from previous tasks,
        create exactly 3 distinct photoshoot prompts for Nano Banana.

        Each prompt must target a different shot:
        - Shot 1: Hero front-facing full body
        - Shot 2: Three-quarter turn showing garment detail
        - Shot 3: Dynamic/movement shot showing drape

        For each, produce:
        - prompt (detailed, fashion-photography-specific)
        - negative_prompt
        - ip_adapter_weight (0.0-1.0)
        - style_strength (0.0-1.0)

        Return as JSON array of 3 objects.""",
        expected_output="JSON array of 3 prompt objects",
        agent=prompt_engineer,
        context=[analyze_clothing, analyze_model]
    )

    generate_images = Task(
        description=f"""Using the 3 prompts from the previous task, generate all 3 images
        using the Nano Banana Generation Tool.
        Clothing image path: {clothing_path}
        Model image path: {model_path}
        Run generation for shot_number 1, 2, and 3.
        Return paths to all 3 generated images.""",
        expected_output="Paths to 3 generated image files",
        agent=image_generator,
        context=[create_prompts]
    )

    quality_check = Task(
        description="""Review all 3 generated images using the Quality Check Tool.
        Use the garment and model profiles from earlier tasks as reference.
        For any image that fails QC, note the specific issues.
        Return a final QC report with pass/fail status for each shot.""",
        expected_output="QC report with pass/fail and issues for each of the 3 shots",
        agent=qc_agent,
        context=[analyze_clothing, analyze_model, generate_images]
    )

    return [analyze_clothing, analyze_model, create_prompts, generate_images, quality_check]