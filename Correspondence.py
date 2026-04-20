
class Correspondence:
    def __init__(self, source:str, target:str, relation:str, confidence:float):
        self.source = str(source)
        self.target = str(target)
        self.relation = str(relation)
        self.confidence = float(confidence)
        self.extensions = {}

    def __eq__(self, other):
        return (self.source, self.target, self.relation) == (other.source, other.target, other.relation)

    def __hash__(self):
        return hash((self.source, self.target, self.relation))

    def __repr__(self):
        return f"({self.source}, {self.target}, {self.relation}, {self.confidence})"

    def key(self):
        return (self.source, self.target, self.relation)

    def __iter__(self):
        yield self.source
        yield self.target
        yield self.relation
        yield self.confidence
        yield self.extensions