"""Lightweight tool abstraction"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence, Type

from pydantic import BaseModel


class Tool(ABC):
    """One named tool the LLM can call."""

    name: str
    description: str
    args_schema: Type[BaseModel]

    @abstractmethod
    def _run(self, **kwargs: Any) -> str: ...

    def invoke(self, arguments: Dict[str, Any]) -> str:
        try:
            data = self.args_schema.model_validate(arguments)
        except Exception as e:
            return f"Error: invalid tool arguments: {e}"
        try:
            return self._run(**data.model_dump())
        except Exception as e:
            return f"Error calling tool: {e}"

    def openai_function_dict(self) -> dict[str, Any]:
        schema = self.args_schema.model_json_schema()
        schema.pop("definitions", None) # pydantic 1
        schema.pop("$defs", None) # pydantic 2
        schema.pop("description", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._rm_titles(schema),
            },
        }
    
    def _rm_titles(self, kv: dict, prev_key: str = "") -> dict:
        new_kv = {}
        for k, v in kv.items():
            if k == "title":
                if isinstance(v, dict) and prev_key == "properties":
                    new_kv[k] = self._rm_titles(v, k)
                else:
                    continue
            elif isinstance(v, dict):
                new_kv[k] = self._rm_titles(v, k)
            else:
                new_kv[k] = v
        return new_kv

