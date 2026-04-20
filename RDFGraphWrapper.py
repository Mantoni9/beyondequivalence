from typing import Union
from rdflib import RDF, SKOS, BNode, Graph, RDFS, OWL, FOAF
from rdflib.term import URIRef
from rdflib.query import Result
import networkx as nx
import logging

logger = logging.getLogger(__name__)

owl_prefix = "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
rdfs_prefix = "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
xsd_prefix = "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"

LABEL_PREDICATES = [
    RDFS.label,
    SKOS.prefLabel,
    URIRef("http://schema.org/name"),
    FOAF.name,
    SKOS.altLabel,
    URIRef("http://schema.org/alternateName"),
    URIRef("http://schema.org/additionalName"),
    SKOS.hiddenLabel
]

DESCRIPTION_PREDICATES = [
    URIRef("http://www.w3.org/2000/01/rdf-schema#comment"),
    URIRef("http://purl.org/dc/terms/description"),
    URIRef("http://schema.org/description"),    
]

class RDFGraphWrapper:
    
    def __init__(self, graph: Union[str, None] = None):
        if graph is None:
            self.graph = Graph()
        elif isinstance(graph, str):
            g = Graph()
            g.parse(graph)
            self.graph = g
        else:
            raise ValueError("Input must be a file path or None")
        
    def add_graph(self, graph_to_be_added: Graph):
        self.graph += graph_to_be_added

    def query_sparql(self, query: str) -> str:
        """Execute a SPARQL query and return results as formatted text.

        Handles SELECT (text table), ASK (boolean), and CONSTRUCT/DESCRIBE
        (serialized triples). Parse/execution errors are returned as strings
        so the caller (e.g. an LLM agent) can retry with a corrected query.
        """
        try:
            results: Result = self.graph.query(query)
        except Exception as e:
            logger.warning(f"SPARQL query error: {e}")
            return f"Error executing SPARQL query: {e}"

        if results.type == "ASK":
            return str(bool(results.askAnswer))

        if results.type == "SELECT":
            if not results.vars:
                return "No results."
            header = [str(v) for v in results.vars]
            rows = []
            for row in results:
                rows.append([str(val) if val is not None else "" for val in row])
            if not rows:
                return "No results."
            col_widths = [len(h) for h in header]
            for row in rows:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(cell))
            fmt_row = lambda cells: " | ".join(c.ljust(w) for c, w in zip(cells, col_widths))
            lines = [fmt_row(header), "-+-".join("-" * w for w in col_widths)]
            for row in rows:
                lines.append(fmt_row(row))
            return "\n".join(lines)

        if results.type in ("CONSTRUCT", "DESCRIBE"):
            g = Graph()
            for triple in results:
                g.add(triple)
            return g.serialize(format="turtle")

        return "Unsupported query type."

    def get_ontology_context(self)->str:
        subgraph = Graph()
        for s, _, _ in self.graph.triples((None, RDF.type, OWL.Ontology)):
            for p2, o2 in self.graph.predicate_objects(s):
                subgraph.add((s, p2, o2))
        
        for root_class in self.get_root_classes(top_n=20):
            for p2, o2 in self.graph.predicate_objects(root_class):
                subgraph.add((root_class, p2, o2))

            # for each root class, get sample of instances
            instances = self.get_instances_by_class(root_class, sample=2)
            for instance in instances:
                for p3, o3 in self.graph.predicate_objects(instance):
                    subgraph.add((instance, p3, o3))
                    
        all_properties = list(self.get_properties())
        all_properties.sort() # to have a deterministic order
        for prop in all_properties[:20]:
            for p4, o4 in self.graph.predicate_objects(prop):
                subgraph.add((prop, p4, o4))

        # get ontology information -> type owl:Ontology
        # get sample of classes: e.g. root classes
        # get sample of instances e.g. instances of root classes
        # get sample of properties e.g.

        # then ask
        # what is this ontology about:
        return subgraph.serialize(format="turtle")

    def get_root_classes(self, top_n:int=5) -> set[URIRef]:

        DG = nx.DiGraph()
        
        for s, _, _ in self.graph.triples((None, RDF.type, RDFS.Class)):
            DG.add_node(s)
        for s, _, _ in self.graph.triples((None, RDF.type, OWL.Class)):
            DG.add_node(s)

        for s, _, o in self.graph.triples((None, RDFS.subClassOf, None)):
            if o == OWL.Thing or o == RDFS.Class or o == OWL.Class:
                DG.add_node(s)
            else:
                DG.add_edge(s, o)
        
        for s, _, _ in self.graph.triples((None, RDF.type, OWL.Restriction)):
            try:
                DG.remove_node(s)
            except nx.NetworkXError:
                pass


        graph_roots = list[tuple[URIRef, int]]()
        #roots_in_taxonomy = set()
        #isolated_roots = set()
        for n in DG.nodes:
            if DG.out_degree(n) == 0:
                len_ancestors = len(nx.ancestors(DG, n))
                if len_ancestors > 0:
                    graph_roots.append((n, len_ancestors))
            #if DG.out_degree(n) == 0:
            #    if DG.in_degree(n) > 0:
            #        roots_in_taxonomy.add(n)
            #    else:
            #        isolated_roots.add(n)
        # sort by number of ancestors highest first
        graph_roots.sort(key=lambda x: x[1], reverse=True)
        #expand if top_n is larger than available roots
        if top_n > len(graph_roots):
            for one_root in graph_roots:
                for ancestor in nx.ancestors(DG, one_root[0]):
                    len_ancestors = len(nx.ancestors(DG, ancestor))
                    if len_ancestors == 0:
                        continue
                    if (ancestor, len_ancestors) not in graph_roots:
                        graph_roots.append((ancestor, len_ancestors))
            graph_roots.sort(key=lambda x: x[1], reverse=True)
        
        final_roots = graph_roots[:top_n]
        return {root for root, _ in final_roots}


        #                    else:
        #        root_classes.add(o)
        #        no_root_classes.add(s)
        #final_roots = root_classes - no_root_classes
        #final_roots.discard(OWL.Thing)
        #final_roots.discard(RDFS.Class)
        #final_roots.discard(OWL.Class)
        #return final_roots
        

    def get_classes(self) -> set[URIRef]:
        all_classes = set()
        for s, _, _ in self.graph.triples((None, RDF.type, RDFS.Class)):
            if isinstance(s, URIRef):
                all_classes.add(s)
        for s, _, _ in self.graph.triples((None, RDF.type, OWL.Class)):
            if isinstance(s, URIRef):
                all_classes.add(s)
        for s, _, o in self.graph.triples((None, RDFS.subClassOf, None)):
            if isinstance(s, URIRef):
                all_classes.add(s)
            if isinstance(o, URIRef):
                all_classes.add(o)
        all_classes.discard(OWL.Thing)
        return all_classes
    
    def get_object_properties(self) -> set[URIRef]:
        return set(self.graph.subjects(RDF.type, OWL.ObjectProperty))
    
    def get_datatype_properties(self) -> set[URIRef]:
        return set(self.graph.subjects(RDF.type, OWL.DatatypeProperty))

    def get_rdf_properties(self) -> set[URIRef]:
        return set(self.graph.subjects(RDF.type, RDF.Property))
    
    def get_properties(self) -> set[URIRef]:
        properties = set()
        properties.update(self.get_object_properties())
        properties.update(self.get_datatype_properties())
        properties.update(self.get_rdf_properties())
        return properties
    
    #def get_instances(self) -> set[Node]:
    #    return set(self.graph.subjects(RDF.type, None)) - self.get_classes()

    def get_instances_by_class(self, clazz : URIRef, sample : Union[int, None] = None) -> set[URIRef]:
        """Get instances of a given class. If sample is provided, limit to that many instances."""
        instances = set()
        for instance in self.graph.subjects(RDF.type, clazz):
            instances.add(instance)
            if sample is not None and len(instances) >= sample:
                break
        return instances

    @staticmethod
    def get_uri_fragment(uri: str) -> str:
        last_index = uri.rfind("#")
        if last_index >= 0:
            return uri[last_index + 1:]

        last_index = uri.rfind("/")
        if last_index >= 0:
            return uri[last_index + 1:]
        return uri
    
    @staticmethod
    def contains_mostly_numbers(term: str) -> bool:
        digits = sum(c.isdigit() for c in term)
        non_ws = sum(not c.isspace() for c in term)
        return digits >= non_ws / 2

    def get_labels(self, resource: URIRef):
        # Get rdfs:label if present
        labels = set()
        for predicate in LABEL_PREDICATES:
            for label in self.graph.objects(resource, predicate):
                labels.add(str(label))
        if not labels:
            fragment = RDFGraphWrapper.get_uri_fragment(str(resource))
            if not RDFGraphWrapper.contains_mostly_numbers(fragment):
                labels.add(fragment)
        return labels





    # TODO: add whole path until root node
    def description_three_outgoing(self, entity: URIRef) -> Union[Graph, str]:
        graph_outgoing = Graph()
        for s, p, o in self.graph.triples((entity, None, None)):
            graph_outgoing.add((s, p, o))            
            for s2, p2, o2 in self.graph.triples((o, None, None)):
                graph_outgoing.add((s2, p2, o2))
                for s3, p3, o3 in self.graph.triples((o2, None, None)):
                    graph_outgoing.add((s3, p3, o3))
        return graph_outgoing



    def description_two_outgoing_blank(self, entity: URIRef) -> Union[Graph, str]:
        graph_outgoing = Graph()
        for s, p, o in self.graph.triples((entity, None, None)):
            graph_outgoing.add((s, p, o))
            if isinstance(o, BNode):
                for s2, p2, o2 in self.graph.triples((o, None, None)):
                    graph_outgoing.add((s2, p2, o2))
                    for s3, p3, o3 in self.graph.triples((o2, None, None)):
                        graph_outgoing.add((s3, p3, o3))
            else:
                for s2, p2, o2 in self.graph.triples((o, None, None)):
                    graph_outgoing.add((s2, p2, o2))
        return graph_outgoing

    def description_two_outgoing(self, entity: URIRef) -> Union[Graph, str]:
        graph_outgoing = Graph()
        for s, p, o in self.graph.triples((entity, None, None)):
            graph_outgoing.add((s, p, o))
            for s2, p2, o2 in self.graph.triples((o, None, None)):
                graph_outgoing.add((s2, p2, o2))
        return graph_outgoing
    
        
    def _description_depth_restricted(self, entity :URIRef, depth=1) -> Union[Graph, str]:
        """
        Extract a focused RDF subgraph describing an entity.
        
        Parameters
        ----------
        graph : rdflib.Graph
            The source RDF graph
        entity : rdflib.term.Identifier
            The resource of interest (URIRef or BNode)
        depth : int
            Depth of expansion for blank nodes and linked resources.
            - 0: only the entity's triples (no expansion)
            - 1: expand blank nodes fully, URIs minimally (label/type)
            - >1: recursively expand further
            
        Returns
        -------
        rdflib.Graph
            Subgraph containing the relevant triples
        """
        subgraph = Graph()
        visited = set()

        def recurse(node, current_depth):
            if (node, current_depth) in visited:
                return
            visited.add((node, current_depth))

            for p, o in self.graph.predicate_objects(node):
                subgraph.add((node, p, o))

                if isinstance(o, BNode):
                    if current_depth < depth:
                        recurse(o, current_depth + 1)

                elif isinstance(o, URIRef):
                    if current_depth < depth:
                        # add minimal context (label + type)
                        for lp in (RDFS.label, RDF.type):
                            for l in self.graph.objects(o, lp):
                                subgraph.add((o, lp, l))
        
        recurse(entity, 0)
        return subgraph

    def description_one_gen(self, entity: URIRef) -> Union[Graph, str]:
        return self._description_depth_restricted(entity, depth=1)

    def description_two_gen(self, entity: URIRef) -> Union[Graph, str]:
        return self._description_depth_restricted(entity, depth=2)

    def description_three_gen(self, entity: URIRef) -> Union[Graph, str]:
        return self._description_depth_restricted(entity, depth=3)

    def description_basic(self, entity: URIRef) -> Union[Graph, str]:
        subgraph = Graph()
    
        # labels
        for p in LABEL_PREDICATES:
            for o in self.graph.objects(entity, p):
                subgraph.add((entity, p, o))
        
        # descriptions
        for p in DESCRIPTION_PREDICATES:
            for o in self.graph.objects(entity, p):
                subgraph.add((entity, p, o))
        
        # types
        for t in self.graph.objects(entity, RDF.type):
            subgraph.add((entity, RDF.type, t))
            for label in self.graph.objects(t, RDFS.label):
                subgraph.add((t, RDFS.label, label))

        return subgraph

    def description_text(self, entity: URIRef) -> Union[Graph, str]:
        text = ""
    
        # labels
        for p in LABEL_PREDICATES:
            for o in self.graph.objects(entity, p):
                text += f"Label: {o}\n"

        # descriptions
        for p in DESCRIPTION_PREDICATES:
            for o in self.graph.objects(entity, p):
                text += f"Description: {o}\n"

        # types
        for t in self.graph.objects(entity, RDF.type):
            for label in self.graph.objects(t, RDFS.label):
                text += f"Type: {label}\n"

        return text

    @staticmethod
    def serialize(graph, format: str = "turtle", remove_common_prefix: bool = True) -> str:
        if isinstance(graph, str):
            return graph
        text = graph.serialize(format=format)
        if remove_common_prefix:
            text = text.replace(owl_prefix, "").replace(rdfs_prefix, "").replace(xsd_prefix, "")
        text = text.strip("\n")
        return text

if __name__ == "__main__":
    from Evaluation import get_test_cases
    testcases = get_test_cases([
        ("anatomy_track", "anatomy_track-default"),
        #("conference_track", "conference_track-default"),
    ])
    all_files = [s for s,t in testcases] + [t for s, t in testcases]
    for f in set(all_files):
        print(f"Processing {f}")
        g = Graph()
        g.parse(f)
        kg = RDFGraphWrapper(g)
        roots = kg.get_root_classes(top_n=5)
        print(f"  Found {len(roots)} root classes:")
        for r in roots:
            print(kg.description_text(r))
            print("===")
        print()
        print()
        print()
            
