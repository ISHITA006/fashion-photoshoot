from crewai import Task
from agents import stylist_prompt_planner, image_generator, qc_agent


def create_tasks(clothing_path, model_path, product_brief):

    plan_prompts = Task(
        description=f"""Analyze both references and create exactly 3 prompts for virtual try-on.

          Inputs:
          - Clothing flatlay image: {clothing_path}
          - Model reference image: {model_path}
          - Product brief: {product_brief}

          First, inspect clothing with Vision Analysis Tool:
            - image_path = "{clothing_path}"
            - analysis_type = "clothing"
            - product_brief = "{product_brief}"

          Then inspect model with Vision Analysis Tool:
            - image_path = "{model_path}"
            - analysis_type = "model"

          Use those observations to choose 3 pose concepts that best showcase the product.
          IMPORTANT REQUIREMENTS FOR ALL SHOTS:
          1. SAME MODEL: The generated person must be the exact same identity as the reference model.
          2. SAME GARMENT: The clothing item must match the flatlay exactly (shape, print, trim, color, texture).
          3. SAME BACKGROUND: all 3 outputs must use a clean white seamless studio background.
          4. Human realism: natural anatomy, hands, face proportions, and believable pose transitions.

          Shot plan (all on white studio background):
          - Shot 1: Hero full-body front pose to show overall silhouette.
          - Shot 2: Pose selected specifically to highlight the garment's key design feature.
          - Shot 3: Gentle dynamic pose showing drape/movement while keeping product visibility.

          Return ONLY a valid JSON array of exactly 3 objects, each with:
            shot_number (int), shot_type (str), pose_rationale (str), prompt (str),
            negative_prompt (str), ip_adapter_weight (float), style_strength (float)

          Rules for negative_prompt:
            - Never include terms that suppress humans (e.g., no model/person/human/woman/man/face/body).
            - Only include quality defect terms such as bad anatomy, extra fingers, deformed hands,
              blurry, low resolution, watermark, text overlay, cluttered background, color cast.

          No markdown, no explanation — raw JSON array only.""",
        expected_output="JSON array of exactly 3 prompt objects with pose rationale and generation parameters",
        agent=stylist_prompt_planner,
    )

    generate_images = Task(
        description=f"""Using the 3 prompts JSON from the previous task, generate all 3 images.

          IMPORTANT: For EVERY generation call you MUST pass:
            - clothing_image_path = "{clothing_path}"
            - model_image_path = "{model_path}"
            Both paths are required every single time.

          For each shot, call the Nano Banana Generation Tool with:
            - prompt
            - shot_number
            - clothing_image_path
            - model_image_path
            - negative_prompt
            - aspect_ratio: "2:3"
            - resolution: "2K"

          After all 3 calls, return a summary JSON with paths:
            [{{"shot": 1, "path": "..."}}, {{"shot": 2, "path": "..."}}, {{"shot": 3, "path": "..."}}]""",
        expected_output="JSON list of 3 objects with shot number and output file path for each generated image",
        agent=image_generator,
        context=[plan_prompts],
    )

    quality_check = Task(
        description="""Review all 3 generated images using the Quality Check Tool.

          For each image evaluate:
          1. MODEL IDENTITY MATCH — face/skin tone/hair/build match the reference model.
          2. GARMENT ACCURACY — garment matches reference flatlay exactly.
          3. BACKGROUND CONSISTENCY — clean white studio background in all shots.
          4. ANATOMY — natural hands, limbs, facial proportions.
          5. COMMERCIAL READINESS — suitable for e-commerce listing.

          Return QC report JSON with pass/fail for each shot and retry notes if failed.
          Use concrete values when calling the Quality Check Tool.""",
        expected_output="Detailed QC report JSON with pass/fail and issues per shot",
        agent=qc_agent,
        context=[plan_prompts, generate_images],
    )

    return [plan_prompts, generate_images, quality_check]
