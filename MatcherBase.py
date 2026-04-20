from abc import ABC, abstractmethod
from typing import Any
from RDFGraphWrapper import RDFGraphWrapper
from Alignment import Alignment


class MatcherBase(ABC):
    """Abstract base for ontology matchers."""

    @abstractmethod
    def match(
        self,
        kg_source: RDFGraphWrapper,
        kg_target: RDFGraphWrapper,
        input_alignment: Alignment,
        parameters: dict[str, Any] = None,
    ) -> Alignment:
        """Return an Alignment object."""
        raise NotImplementedError