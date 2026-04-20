from typing import Any
from MatcherBase import MatcherBase
from RDFGraphWrapper import RDFGraphWrapper
from Alignment import Alignment


class MatcherFileLoader(MatcherBase):
    """A matcher that loads an alignment from a file."""

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def match(
        self,
        kg_source: RDFGraphWrapper,
        kg_target: RDFGraphWrapper,
        input_alignment: Alignment,
        parameters: dict[str, Any] = None,
    ) -> Alignment:
        return Alignment(self.filepath)

    def __str__(self):
        return f"MatcherFileLoader({self.filepath})"
