from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel


class ExportJsonInput(BaseModel):
    x: str


class ExportJsonTool(BaseTool):
    name: str = "export_json"
    description: str = "Utility tool that simply returns the provided JSON/text string."
    args_schema: Type[BaseModel] = ExportJsonInput

    def _run(self, x: str) -> str:
        # Pass-through: lets the agent \"call a tool\" to finalize JSON
        return x

