"""Tool for agents to submit their final JSON result. Prevents hallucinated tool calls (e.g. model_profile)."""
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel


class SubmitResultInput(BaseModel):
    result_json: str


class SubmitResultTool(BaseTool):
    name: str = "Submit Result Tool"
    description: str = (
        "Call this with the exact JSON string from the Vision Analysis Tool to submit it as your final answer. "
        "Use this after you get the vision tool output. Do not call any other tool."
    )
    args_schema: Type[BaseModel] = SubmitResultInput

    def _run(self, result_json: str) -> str:
        return result_json.strip()
