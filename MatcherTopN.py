from collections import Counter
from typing import Any
from MatcherBase import MatcherBase
from RDFGraphWrapper import RDFGraphWrapper
from Alignment import Alignment
import logging

class MatcherTopN(MatcherBase):

    def __init__(self, n:int = 1):
        super().__init__()
        self.n = n     


    def match(self, kg_source: RDFGraphWrapper, kg_target: RDFGraphWrapper, input_alignment: Alignment, parameters: dict[str, Any] = None) -> Alignment:
        
        counts = Counter()
        alignment = Alignment()
        for correspondence in input_alignment.sort_by_confidence():
            counts[correspondence.source] += 1
            counts[correspondence.target] += 1
            if counts[correspondence.source] <= self.n and counts[correspondence.target] <= self.n:
                alignment.add(correspondence)
        return alignment
    
    def __str__(self):
        return f"MatcherTopN(n={self.n})"

    
def main():
    from MatcherSequential import MatcherSequential
    from MatcherFileLoader import MatcherFileLoader
    from Evaluation import run_oaei_tracks
    run_oaei_tracks(
        systems=[
            MatcherFileLoader("system_anatomy.rdf"),
            MatcherSequential([MatcherFileLoader("system_anatomy.rdf"), MatcherTopN(0)]),
            MatcherSequential([MatcherFileLoader("system_anatomy.rdf"), MatcherTopN(1)]),
            MatcherSequential([MatcherFileLoader("system_anatomy.rdf"), MatcherTopN(3)]),
            MatcherSequential([MatcherFileLoader("system_anatomy.rdf"), MatcherTopN(5)]),
            MatcherSequential([MatcherFileLoader("system_anatomy.rdf"), MatcherTopN(10)]),
            MatcherSequential([MatcherFileLoader("system_anatomy.rdf"), MatcherTopN(15)]),
            MatcherSequential([MatcherFileLoader("system_anatomy.rdf"), MatcherTopN(20)]),
            MatcherSequential([MatcherFileLoader("system_anatomy.rdf"), MatcherTopN(25)]),
            MatcherSequential([MatcherFileLoader("system_anatomy.rdf"), MatcherTopN(30)]),
            MatcherSequential([MatcherFileLoader("system_anatomy.rdf"), MatcherTopN(35)]),
            MatcherSequential([MatcherFileLoader("system_anatomy.rdf"), MatcherTopN(40)]),
            MatcherSequential([MatcherFileLoader("system_anatomy.rdf"), MatcherTopN(45)]),
            MatcherSequential([MatcherFileLoader("system_anatomy.rdf"), MatcherTopN(50)]),
        ],
        tracks = [ 
            ("anatomy_track", "anatomy_track-default"),
        ],
        timestamp_replacement = "topn"
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    main()
