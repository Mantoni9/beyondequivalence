from collections import defaultdict
from typing import Any
from MatcherBase import MatcherBase
from Evaluation import run_oaei_tracks
from rdflib.term import URIRef
import re
from RDFGraphWrapper import RDFGraphWrapper
import logging
from Alignment import Alignment
from Correspondence import Correspondence

import re
import urllib.parse



class MatcherSimple(MatcherBase):

    # Precompiled regex patterns
    CAMEL_CASE = re.compile(r'(?<!^)(?<!\s|")(?=[A-Z][a-z])')
    NON_ALPHA = re.compile(r'[^a-z0-9\s:_]')  # matches non-alphanumeric (except space, colon, underscore)
    ENGLISH_GENITIVE_S = re.compile(r"'s")
    MULTIPLE_UNDERSCORES = re.compile(r'_+')

    STOPWORDS = {
        "a", "about", "after", "afterwards", "again", "all", "almost", "alone",
        "along", "already", "also", "although", "always", "am", "among", "amongst",
        "amoungst", "an", "and", "another", "any", "anyhow", "anyone", "anything",
        "anyway", "anywhere", "are", "as", "at", "be", "became", "because", "become",
        "becomes", "becoming", "been", "before", "beforehand", "being", "but", "by",
        "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de",
        "describe", "do", "done", "during", "each", "eg", "either", "else",
        "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fify", "fill", "find", "fire",
        "for", "found", "further", "get", "give", "go", "had", "has", "hasnt",
        "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein",
        "hereupon", "hers", "herself", "him", "himself", "his", "how", "however",
        "i", "ie", "if", "in", "inc", "indeed", "into", "is", "it", "its", "itself",
        "keep", "latterly", "less", "ltd", "made", "may", "me", "meanwhile", "might",
        "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must",
        "my", "myself", "namely", "neither", "never", "nevertheless", "next", "no",
        "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of",
        "off", "often", "on", "once", "only", "onto", "or", "otherwise", "our",
        "ours", "ourselves", "out", "own", "per", "perhaps", "please", "put",
        "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems",
        "serious", "several", "she", "should", "since", "sincere", "so", "some",
        "somehow", "someone", "something", "sometime", "sometimes", "somewhere",
        "still", "such", "take", "than", "that", "the", "their", "them",
        "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "this", "those",
        "though", "through", "throughout", "thru", "thus", "to", "together", "too",
        "toward", "towards", "un", "until", "up", "upon", "us", "very", "was", "we",
        "well", "were", "what", "whatever", "when", "whence", "whenever", "where",
        "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever",
        "whether", "which", "while", "whither", "who", "whoever", "whole", "whom",
        "whose", "why", "will", "with", "within", "without", "would", "yet", "you",
        "your", "yours", "yourself", "yourselves", ""
    }

    def __normalize(self, string_to_be_normalized: str) -> list[str]:
        if string_to_be_normalized is None:
            return []
        string_to_be_normalized = string_to_be_normalized.strip()
        string_to_be_normalized = self.CAMEL_CASE.sub("_", string_to_be_normalized)
        string_to_be_normalized = string_to_be_normalized.replace(" ", "_")
        string_to_be_normalized = string_to_be_normalized.lower()
        try:
            string_to_be_normalized = urllib.parse.unquote(string_to_be_normalized, encoding="utf-8")
        except Exception:
            pass
        string_to_be_normalized = self.NON_ALPHA.sub("_", string_to_be_normalized)
        string_to_be_normalized = self.ENGLISH_GENITIVE_S.sub("", string_to_be_normalized)
        string_to_be_normalized = self.MULTIPLE_UNDERSCORES.sub("_", string_to_be_normalized)
        tokenized = set(string_to_be_normalized.split("_"))
        tokenized.difference_update(self.STOPWORDS)
        return frozenset(tokenized)



    def match(self, kg_source: RDFGraphWrapper, kg_target: RDFGraphWrapper, input_alignment: Alignment, parameters: dict[str, Any] = None) -> Alignment:

        label_to_uri = defaultdict(set)

        for source_element in kg_source.get_classes():
            for label in kg_source.get_labels(source_element):
                normalized_label = self.__normalize(label)
                if normalized_label:
                    label_to_uri[normalized_label].add(source_element)

        alignment = Alignment()

        for target_element in kg_target.get_classes():
            for label in kg_target.get_labels(target_element):
                normalized_label = self.__normalize(label)
                if normalized_label:
                    uris = label_to_uri[normalized_label]
                    if uris:
                        for source_uri in uris:
                            alignment.add(Correspondence(str(source_uri), str(target_element), "=", 1.0))
        return alignment
    
    def __str__(self):
        return "MatcherSimple"


def main():
    run_oaei_tracks(
        systems=[MatcherSimple()],

        tracks = [ 
            ("anatomy_track", "anatomy_track-default"),
            ("conference", "conference-v1"),
            #("biodiv", "2023")
            #("commonkg", "yago-wikidata-v1"),
            #("commonkg", "nell-dbpedia-v1")
        ]
        #tracks=[

            #("conference", "cmt"),
            #("conference", "edas"),
            #("conference", "confOf"),
            #()"lifeScience", "anatomy"),
            #("lifeScience", "biomed"),
            #("lifeScience", "chemistry"),
            #("library", "bibs"),
            #("library", "dbpedia2wikidata"),
            #("library", "gnd2wikidata"),
            #("multimedia", "imdb2wikidata"),
            #("multimedia", "lastfm2dbpedia"),
            #("multimedia", "musicbrainz2wikidata"),
            #("socialNetworks", "dblp_acm"),
            #("socialNetworks", "dblp_scholar"),
            #("socialNetworks", "freebase_dbpedia")
        #],
        #testcases=None
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    main()

    