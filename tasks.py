from crewai import Task
from agents import clothing_analyzer, model_analyzer, prompt_engineer, image_generator, qc_agent

def create_tasks(clothing_path, model_path, product_brief):

    analyze_clothing = Task(
        description=f"""Analyze the clothing flat lay image at: {clothing_path}
          Product brief: {product_brief}

          Use the Vision Analysis Tool with:
            - image_path = "{clothing_path}"
            - analysis_type = "clothing"
            - product_brief = "{product_brief}"

          Extract and return a detailed garment profile JSON containing ALL of these fields:
            garment_type, primary_color, secondary_colors, fabric_texture,
            fit_silhouette, key_details (list of 5+ specifics — include any surface details such as
            embroidery, prints, cutwork, pleats, drape, waistband, pockets, closures, etc. as relevant
            to the specific garment), styling_notes, brand_aesthetic, target_demographic

          Also include any garment-type-specific fields that apply, for example:
            - Tops/shirts/dresses: sleeve_style, neckline, hem_style
            - Bottoms (pants/shorts/skirts): waist_style, leg_style, length, hem_style
            - Dresses/co-ords: silhouette, length, waist_definition

          Focus your key_details on the hero design features that make this product unique.""",
        expected_output="Detailed garment profile JSON with all fields above",
        agent=clothing_analyzer,
    )

    analyze_model = Task(
        description=f"""Analyze the model reference photo at: {model_path}

          Use the Vision Analysis Tool with:
            - image_path = "{model_path}"
            - analysis_type = "model"

          Your goal is identity preservation — extract enough detail so the SAME person can be
          reproduced faithfully in AI-generated images. Return a model profile JSON with ALL of:
            age_range, ethnicity, skin_tone (use specific Fitzpatrick scale terms + warm/cool undertone),
            eye_color, eye_shape, nose_description, lip_fullness, face_shape,
            hair_color, hair_length, hair_style, hair_texture,
            body_type, height_impression, distinctive_features (list),
            expression_style, pose_style

          Do NOT use vague terms like "medium skin" — use precise descriptors like
          "warm medium-brown skin, Fitzpatrick IV, golden undertone".""",
        expected_output="Detailed model profile JSON with all identity fields above",
        agent=model_analyzer,
    )

    create_prompts = Task(
        description="""Using the garment profile (from clothing analysis task) and model profile
          (from model analysis task), create exactly 3 distinct photoshoot prompts.

          CRITICAL RULES YOU MUST FOLLOW:
          1. NEGATIVE PROMPTS: NEVER include words like: model, person, woman, man, human, face, body,
            figure, subject, people, individual, portrait, silhouette — these words suppress the human
            subject and will produce an empty garment or mannequin. This is the #1 failure mode.
            Negative prompts may ONLY contain technical defect terms:
            "bad anatomy, extra fingers, deformed hands, fused fingers, blurry, low resolution,
            overexposed, underexposed, watermark, text overlay, logo, cluttered background,
            poor lighting, color cast, grain, noise, jpeg artifacts, cropped head, out of frame"

          2. POSITIVE PROMPTS: Start every prompt by anchoring the model's identity using the
            model profile descriptors BEFORE describing what she's wearing. Structure:
            "photorealistic fashion editorial, [skin tone] Indian woman [age] years old,
            [hair description], [face features], wearing [garment description], [shot type],
            [lighting], [background], shot on Phase One, f/2.8, commercial fashion photography"

          3. IP ADAPTER WEIGHT: Set to 0.80–0.95 to strongly lock in the reference model's appearance.

          4. STYLE STRENGTH: Set to 0.65–0.80 to preserve garment detail while allowing pose variation.

          Shot specifications:
          - Shot 1 (hero): Full body, front-facing, white studio background
          - Shot 2 (detail): Three-quarter turn, framing chosen to best showcase the garment's hero design feature (e.g. waist-up for tops, full-length for bottoms/dresses), white studio background
          - Shot 3 (dynamic): Slight movement/walking pose showing garment drape and fabric behavior, white studio background

          Return ONLY a valid JSON array of exactly 3 objects, each with:
            shot_number (int), shot_type (str), prompt (str), negative_prompt (str),
            ip_adapter_weight (float), style_strength (float)

          No markdown, no explanation — raw JSON array only.""",
        expected_output="JSON array of exactly 3 prompt objects with shot_number, shot_type, prompt, negative_prompt, ip_adapter_weight, style_strength",
        agent=prompt_engineer,
        context=[analyze_clothing, analyze_model],
    )

    generate_images = Task(
        description=f"""Using the 3 prompts JSON from the previous task, generate all 3 images.

          IMPORTANT: For EVERY generation call you MUST pass:
            - clothing_image_path = "{clothing_path}"   ← the flat lay garment reference
            - model_image_path = "{model_path}"         ← the model identity reference
            Both paths are required every single time. Never omit either one.

          For each of the 3 shots, call the Nano Banana Generation Tool with:
            - prompt: the prompt string from the JSON
            - shot_number: the shot_number from the JSON (1, 2, or 3)
            - clothing_image_path: "{clothing_path}"
            - model_image_path: "{model_path}"
            - aspect_ratio: "2:3"
            - resolution: "2K"

          You MUST generate exactly one image per shot_number (1, 2, 3). For each shot, you may
          call the Nano Banana Generation Tool AT MOST ONCE. As soon as you have 3 results 
          (one for each of shot_number 1, 2, and 3), STOP using this tool.

          After all 3 shots are generated, return a summary JSON with the output paths:
            [{{"shot": 1, "path": "..."}}, {{"shot": 2, "path": "..."}}, {{"shot": 3, "path": "..."}}]""",
        expected_output="JSON list of 3 objects with shot number and output file path for each generated image",
        agent=image_generator,
        context=[create_prompts],
    )

    quality_check = Task(
        description="""Review all 3 generated images using the Quality Check Tool.
        Use the garment profile and model profile from the earlier analysis tasks as reference.

        For each image evaluate:
        1. MODEL IDENTITY MATCH — Does the face/skin tone/hair match the reference model photo?
          This is the most critical check. Fail immediately if the person looks different.
        2. GARMENT ACCURACY — Is the garment clearly visible with correct colors, silhouette, and hero design details (prints, embroidery, drape, cut, etc.) matching the flat lay reference?
        3. ANATOMY — Are hands, fingers, and limbs natural and well-formed?
        4. BACKGROUND — Is the background clean and appropriate for the shot type?
        5. OVERALL COMMERCIAL VIABILITY — Would this pass e-commerce listing standards?

        For each failed image, provide specific retry instructions that address the root cause.

        Return a QC report JSON:
          {{
            "shots": [
              {{
                "shot_number": 1,
                "pass": true/false,
                "model_identity_match": true/false,
                "garment_accurate": true/false,
                "anatomy_ok": true/false,
                "background_clean": true/false,
                "issues": ["..."],
                "retry_notes": "..."
              }},
              ...
            ],
            "overall_pass_rate": "X/3"
          }}""",
        expected_output="Detailed QC report JSON with pass/fail and issues per shot",
        agent=qc_agent,
        context=[analyze_clothing, analyze_model, generate_images],
    )

    return [analyze_clothing, analyze_model, create_prompts, generate_images, quality_check]