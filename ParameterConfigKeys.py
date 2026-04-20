"""Known parameter keys for OAEI matching configuration.
Matchers can also define their own keys (any string).
"""

MATCHING_CLASSES = "http://oaei.ontologymatching.org/matchingClasses"
"""Boolean (true/false) — whether matching classes is required."""

MATCHING_DATA_PROPERTIES = "http://oaei.ontologymatching.org/matchingDataProperties"
"""Boolean (true/false) — whether matching data properties is required."""

MATCHING_OBJECT_PROPERTIES = "http://oaei.ontologymatching.org/matchingObjectProperties"
"""Boolean (true/false) — whether matching object properties is required."""

MATCHING_RDF_PROPERTIES = "http://oaei.ontologymatching.org/matchingRDFProperties"
"""Boolean (true/false) — whether matching RDF properties is required."""

MATCHING_INSTANCES = "http://oaei.ontologymatching.org/matchingInstances"
"""Boolean (true/false) — whether matching instances is required."""

MATCHING_INSTANCE_TYPES = "http://oaei.ontologymatching.org/matchingInstanceTypes"
"""List of class URIs whose instances should be matched (allowlist)."""

NON_MATCHING_INSTANCE_TYPES = "http://oaei.ontologymatching.org/nonMatchingInstanceTypes"
"""List of class URIs whose instances should not be matched (blocklist)."""

SOURCE_LANGUAGE = "http://oaei.ontologymatching.org/sourceLanguage"
"""ISO 639-1 (alpha-2) or ISO 639-2 (alpha-3) language code of the source ontology."""

TARGET_LANGUAGE = "http://oaei.ontologymatching.org/targetLanguage"
"""ISO 639-1 (alpha-2) or ISO 639-2 (alpha-3) language code of the target ontology."""

DEFAULT_PARAMETERS_SERIALIZATION_FORMAT = "http://oaei.ontologymatching.org/defaultParametersSerializationFormat"
"""Serialization format for parameters ('json' or 'yaml', case-insensitive)."""

DEFAULT_ONTOLOGY_SERIALIZATION_FORMAT = "http://oaei.ontologymatching.org/defaultOntologySerializationFormat"
"""RDF serialization format for ontologies (e.g. RDF/XML, Turtle, N-Triples)."""

SERIALIZATION_FOLDER = "http://oaei.ontologymatching.org/serializationFolder"
"""Path to the folder where alignments and properties files are stored."""

USE_ONTOLOGY_CACHE = "http://oaei.ontologymatching.org/useOntologyCache"
"""Boolean — whether to keep ontologies in memory (defaults to true)."""

JENA_ONTMODEL_SPEC = "http://oaei.ontologymatching.org/jenaOntModelSpec"
"""Jena OntModelSpec constant name (e.g. OWL_MEM, OWL_DL_MEM_RDFS_INF)."""

ALLOW_ALIGNMENT_REPAIR = "http://oaei.ontologymatching.org/allowAlignmentRepair"
"""Boolean — whether unparsable alignment files should be auto-repaired (defaults to true)."""

FORMAT = "http://oaei.ontologymatching.org/format"
"""String describing the format of the input files."""

HINT_LANG = "http://oaei.ontologymatching.org/hintLang"
"""RDF serialization format hint (e.g. RDFXML, TTL, NTriple, NQuad)."""

TOPIC_DISTANCE = "http://oaei.ontologymatching.org/topicDistance"
"""Non-normalized topical distance between KGs (float, 0 = same topic)."""

TOPIC_DISTANCE_NORMALIZED = "http://oaei.ontologymatching.org/topicDistanceNormalized"
"""Normalized topical distance between KGs (float in [0, 1])."""

ADDITIONAL_OUTPUT_FOLDER = "http://oaei.ontologymatching.org/additionalOutputFolder"
"""Path (string) to a folder where matchers can store additional output files."""
