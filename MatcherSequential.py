from typing import Any
from MatcherBase import MatcherBase
from RDFGraphWrapper import RDFGraphWrapper
from Alignment import Alignment


class MatcherSequential(MatcherBase):
    """A matcher that applies a sequence of matchers."""

    def __init__(self, matchers: list[MatcherBase]):
        super().__init__()
        self.matchers = matchers

    def match(
        self,
        kg_source: RDFGraphWrapper,
        kg_target: RDFGraphWrapper,
        input_alignment: Alignment,
        parameters: dict[str, Any] = None,
    ) -> Alignment:
        current_alignment = input_alignment
        for matcher in self.matchers:
            current_alignment = matcher.match(kg_source, kg_target, current_alignment, parameters)
        return current_alignment

    def __str__(self):
        return "+".join(str(matcher) for matcher in self.matchers)
