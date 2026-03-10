# fashion-photoshoot

## Workflow design (simplified)

This app now uses a **lean 3-step CrewAI pipeline** for virtual try-on:

1. **Stylist + Prompt Planner (single agent)**
   - Analyzes clothing flatlay + model reference directly with vision tool.
   - Selects pose ideas based on the garment's design details.
   - Produces 3 generation prompts (with pose rationale) in one pass.

2. **Image Generator**
   - Calls Nano Banana for each shot.
   - Always passes both references (`clothing_image_path` and `model_image_path`) to preserve fidelity.

3. **Quality Check**
   - Verifies model identity match, garment match, white studio background consistency, and anatomy.

## Why this structure

You do **not** need a complicated multi-JSON orchestration to get good results. A combined planning step is usually enough when:
- prompts explicitly enforce **same model** and **same garment**,
- reference images are always attached to generation calls,
- QC loop checks fidelity and retries failures.

## Non-negotiable constraints in prompts

- Same model identity in every shot.
- Same garment as the flatlay (shape, print, color, trims).
- White seamless studio background in all outputs.
- Humanized, realistic anatomy and poses.
