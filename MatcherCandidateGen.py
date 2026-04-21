from typing import Any
from MatcherBase import MatcherBase
from rdflib.term import URIRef
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.models import Pooling, Normalize
import logging
import torch


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
from MatcherSimple import MatcherSimple

from RDFGraphWrapper import RDFGraphWrapper
from Alignment import Alignment
from Correspondence import Correspondence

from Evaluation import run_oaei_tracks
from prompt import get_embedding_prompt

class MatcherCandidateGen(MatcherBase):

    def __init__(self, model:str="all-MiniLM-L6-v2", description:str="description_two_outgoing", query_prompt_id:str = "", document_prompt_id:str = "", 
                 both_directions:bool=True, format:str = "ttl", top_k:int=5, method="sentence", include_simple:bool=True):
        self.model = model
        self.description = description
        self.query_prompt_id = query_prompt_id
        self.document_prompt_id = document_prompt_id
        self.both_directions = both_directions
        self.format = format
        self.top_k = top_k
        self.method = method.lower()
        self.include_simple = include_simple


    def _find_sublist(self, full, sub, find_all=False, token_ids_to_include=None):
        """Finds the first occurrence of the sublist `sub` in the list `full`."""
        indices = []
        if token_ids_to_include:
            #z = torch.isin(full, token_ids_to_include)
            #a = torch.nonzero(z, as_tuple=False)
            #indices.extend(a)
            for i, val in enumerate(full):
                if val.item() in token_ids_to_include:
                    indices.append(i)

        #for i, val in enumerate(full):
        #    if val in token_ids_to_include:
        #        indices.append(i)
        for i in range(len(full) - len(sub) + 1):
            if torch.equal(full[i:i+len(sub)], sub):
                indices.extend(list(range(i, i+len(sub))))
                if find_all is False:
                    break
        indices = list(set(indices))
        indices.sort()
        return indices

    def _extract_embedding(self, embedder: SentenceTransformer, text:list[str], elements:list[URIRef], pooling_mode="mean", keep_attention=False, find_all=False) -> torch.Tensor:
        embeddings = embedder.encode(text, convert_to_tensor=True, output_value=None)

        ex_token = embeddings[0]["token_embeddings"]
        hidden_dim = ex_token.shape[-1]
        max_len = max(e["token_embeddings"].shape[0] for e in embeddings)

        customized_token_embeddings = torch.zeros((len(embeddings), max_len, hidden_dim), dtype=ex_token.dtype, device=ex_token.device)
        customized_attention_mask = torch.zeros((len(embeddings), max_len), dtype=torch.int64, device=ex_token.device)
        if keep_attention:
            for i, (one_embedding, uri) in enumerate(zip(embeddings, elements)):
                seq_len = one_embedding["token_embeddings"].shape[0]
                customized_token_embeddings[i, :seq_len, :] = one_embedding["token_embeddings"]
                customized_attention_mask[i, :seq_len] = one_embedding["attention_mask"]
        else:
            tokenizer = embedder.tokenizer
            for i, (one_embedding, uri) in enumerate(zip(embeddings, elements)):
                seq_len = one_embedding["token_embeddings"].shape[0]
                customized_token_embeddings[i, :seq_len, :] = one_embedding["token_embeddings"]
                target_ids = tokenizer(uri, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(one_embedding["input_ids"].device)
                indices = self._find_sublist(one_embedding["input_ids"], target_ids, find_all=find_all)
                if indices:
                    customized_attention_mask[i, indices] = 1.0
                else:
                    logging.warning(f"Could not find URL for entitiy {uri}. Using mean pooling over all tokens.")
                    customized_attention_mask[i, :seq_len] = one_embedding["attention_mask"]

        pooling = Pooling(embedder.get_sentence_embedding_dimension(), pooling_mode=pooling_mode)
        norm = Normalize()
        result = pooling({
            "token_embeddings": customized_token_embeddings,
            "attention_mask": customized_attention_mask
        })
        result = norm(result)
        return result["sentence_embedding"]


    def _extract_embedding_less_memory(self, embedder: SentenceTransformer, text:list[str], elements:list[URIRef], pooling_mode="mean", 
                                       find_all:bool=False, pooling_special_tokens="none") -> torch.Tensor:
        embeddings = embedder.encode(text, convert_to_tensor=True, output_value=None)
        pooling = Pooling(embedder.get_sentence_embedding_dimension(), pooling_mode=pooling_mode)
        norm = Normalize()

        new_embeddings = []
        for emb, uri in zip(embeddings, elements):
            if pooling_mode == "cls":
                mask = emb["attention_mask"].clone()
            else:                
                tokenizer = embedder.tokenizer
                device = emb["input_ids"].device
                if pooling_special_tokens == "special":
                    token_ids_to_include = set(tokenizer.all_special_ids)
                elif pooling_special_tokens == "cls":
                    token_ids_to_include = {tokenizer.cls_token_id}
                elif pooling_special_tokens == "clssep":
                    token_ids_to_include = {tokenizer.cls_token_id, tokenizer.sep_token_id}
                else:
                    token_ids_to_include = set()
                target_ids = tokenizer(uri, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
                indices = self._find_sublist(emb["input_ids"], target_ids, find_all=find_all, token_ids_to_include=token_ids_to_include)
                if indices:
                    mask = torch.zeros_like(emb["attention_mask"])
                    mask[indices] = 1.0
                else:
                    logging.warning(f"Could not find URL for entitiy {uri}. Using mean pooling over all tokens.")
                    mask = emb["attention_mask"].clone()

            result = pooling({
                "token_embeddings": emb["token_embeddings"].clone().unsqueeze(0),
                "attention_mask": mask.unsqueeze(0),
            })
            result = norm(result)

            new_embeddings.append(result["sentence_embedding"].squeeze(0))

        x = torch.stack(new_embeddings, dim=0)
        return x

    def _get_embeddings_with_prompt(self, embedder : SentenceTransformer, text:list[str], elements:list[URIRef], prompt_id:str) -> torch.Tensor:
        if prompt_id:
            prompt = get_embedding_prompt(prompt_id)
            processed_text = [prompt.format(uri=uri, text=t).to_text() for t, uri in zip(text, elements)]
        else:
            processed_text = text

        if self.method == "sentence":
            embeddings = embedder.encode(processed_text, convert_to_tensor=True)
        elif self.method == "cls":
            embeddings = self._extract_embedding_less_memory(embedder, processed_text, elements, pooling_mode="cls")
        elif self.method == "firstmean":
            embeddings = self._extract_embedding_less_memory(embedder, processed_text, elements, pooling_mode="mean", find_all=False)
        elif self.method == "allmean":
            embeddings = self._extract_embedding_less_memory(embedder, processed_text, elements, pooling_mode="mean", find_all=True)
        elif self.method == "firstmax":
            embeddings = self._extract_embedding_less_memory(embedder, processed_text, elements, pooling_mode="max", find_all=False)
        elif self.method == "allmax":
            embeddings = self._extract_embedding_less_memory(embedder, processed_text, elements, pooling_mode="max", find_all=True)        
        elif self.method == "firstmeanspecial":
            embeddings = self._extract_embedding_less_memory(embedder, processed_text, elements, pooling_mode="mean", find_all=False, pooling_special_tokens="special")
        elif self.method == "firstmeancls":
            embeddings = self._extract_embedding_less_memory(embedder, processed_text, elements, pooling_mode="mean", find_all=False, pooling_special_tokens="cls")
        elif self.method == "firstmeanclssep":
            embeddings = self._extract_embedding_less_memory(embedder, processed_text, elements, pooling_mode="mean", find_all=False, pooling_special_tokens="clssep")
        elif self.method == "allmeanspecial":
            embeddings = self._extract_embedding_less_memory(embedder, processed_text, elements, pooling_mode="mean", find_all=True, pooling_special_tokens="special")
        elif self.method == "allmeancls":
            embeddings = self._extract_embedding_less_memory(embedder, processed_text, elements, pooling_mode="mean", find_all=True, pooling_special_tokens="cls")
        elif self.method == "allmeanclssep":
            embeddings = self._extract_embedding_less_memory(embedder, processed_text, elements, pooling_mode="mean", find_all=True, pooling_special_tokens="clssep")

        embeddings = embeddings.to(_best_device())
        embeddings = util.normalize_embeddings(embeddings)

        return embeddings

    def match(self, kg_source:RDFGraphWrapper, kg_target:RDFGraphWrapper, input_alignment: Alignment, parameters: dict[str, Any] = None) -> Alignment:

        embedder = SentenceTransformer(self.model, trust_remote_code=True)

        source_elements = list(kg_source.get_classes())
        target_elements = list(kg_target.get_classes())
        logging.info(f"Source Elements: {len(source_elements)}")
        logging.info(f"Target Elements: {len(target_elements)}")

        logging.info("Generate text")

        source_method = getattr(kg_source, self.description)
        target_method = getattr(kg_target, self.description)
        source_text = [RDFGraphWrapper.serialize(source_method(cls), format=self.format) for cls in source_elements]
        target_text = [RDFGraphWrapper.serialize(target_method(cls), format=self.format) for cls in target_elements]

        
        logging.info("Embedding source")
        source_embeddings = self._get_embeddings_with_prompt(embedder, source_text, source_elements, self.query_prompt_id)
        logging.info("Embedding target")
        target_embeddings = self._get_embeddings_with_prompt(embedder, target_text, target_elements, self.document_prompt_id)

        logging.info("Compute semantic search")
        hits = util.semantic_search(source_embeddings, target_embeddings, top_k=self.top_k, score_function=util.dot_score)

        alignment = Alignment()
        for hit, source_cls in zip(hits, source_elements):
            alignment.update([Correspondence(source_cls, target_elements[h['corpus_id']], '=', h['score']) for h in hit])

        if self.both_directions:
            if self.query_prompt_id == self.document_prompt_id:
                logging.info("Compute semantic search in the other direction without reencoding")
                hits = util.semantic_search(target_embeddings, source_embeddings, top_k=self.top_k, score_function=util.dot_score)
                for hit, target_cls in zip(hits, target_elements):
                    alignment.update([Correspondence(source_elements[h['corpus_id']], target_cls, '=', h['score']) for h in hit])
            else:
                logging.info("Compute semantic search in the other direction with reencoding")
                logging.info("Embedding source")
                source_embeddings = self._get_embeddings_with_prompt(embedder, source_text, source_elements, self.document_prompt_id)
                logging.info("Embedding target")
                target_embeddings = self._get_embeddings_with_prompt(embedder, target_text, target_elements, self.query_prompt_id)

                logging.info("Compute semantic search in the other direction")
                hits = util.semantic_search(target_embeddings, source_embeddings, top_k=self.top_k, score_function=util.dot_score)
                for hit, target_cls in zip(hits, target_elements):
                    alignment.update([Correspondence(source_elements[h['corpus_id']], target_cls, '=', h['score']) for h in hit])

        if self.include_simple:
            # finally add all simple matches
            simple_alignment = MatcherSimple().match(kg_source, kg_target, Alignment(), parameters)
            alignment.update(simple_alignment)

        alignment.remove_trivial()
        return alignment
    
    def __str__(self):
        #system_name = f"olala_only_source_{model.split('/')[-1]}_{description}_p{i}"
        #return f"MatcherCandidateGen(model={self.model}, description={self.description}, query_prompt={self.query_prompt}, document_prompt={self.document_prompt}, both_directions={self.both_directions}, format={self.format}, top_k={self.top_k}, method={self.method})"
        return f"MatcherCandidateGen#{self.model.split('/')[-1]}#{self.description}#q{self.query_prompt_id}#d{self.document_prompt_id}#b{self.both_directions}#f{self.format}#k{self.top_k}#m{self.method}#s{self.include_simple}"

def main():
    systems = []
    #for desc in ["description_text", "description_basic", "description_one_gen", "description_two_gen", "description_three_gen",
    #             "description_two_outgoing", "description_two_outgoing_blank", "description_three_outgoing"]:
    
    #for desc in ["description_one_gen", "description_two_gen", "description_three_gen",
    #             "description_two_outgoing", "description_three_outgoing"]:
    #    for q in ["one", "two", "three", "four", "five"]:
    #        for method in ["sentence", "cls", "firstmean", "allmean", "firstmax", "allmax"]:
    #            systems.append(MatcherCandidateGen(model = "Qwen/Qwen3-Embedding-4B", description=desc, method=method))
    #            systems.append(MatcherCandidateGen(model = "Qwen/Qwen3-Embedding-4B", description=desc, method=method, query_prompt_id=q))
    #            systems.append(MatcherCandidateGen(model = "Qwen/Qwen3-Embedding-4B", description=desc, method=method, query_prompt_id=q, document_prompt_id=q))

    #for method in ["sentence", "cls", "firstmean", "allmean", "firstmax", "allmax"]:
    #    systems.append(MatcherCandidateGen(model = "all-MiniLM-L6-v2", description="description_one_gen", method=method))
    #    systems.append(MatcherCandidateGen(model = "all-MiniLM-L6-v2", description="description_text", method=method))

    #systems.append(MatcherCandidateGen(model = "all-MiniLM-L6-v2", description="description_one_gen", method="allmeanclssep"))

    #for desc in ["description_one_gen", "description_two_gen", "description_three_gen"]:
    #    for q in ["", "one"]:
    #        for method in ["sentence", "cls", "firstmean", "allmean", "firstmax", "allmax"]:
    #            systems.append(MatcherCandidateGen(model = "Qwen/Qwen3-Embedding-4B", description=desc, method=method, query_prompt_id=q))

    #for desc in ["description_one_gen", "description_two_gen", "description_three_gen"]:
    #    for method in ["firstmeanspecial", "firstmeancls", "firstmeanclssep", 
    #                   "allmeanspecial", "allmeancls", "allmeanclssep"]:
    #        systems.append(MatcherCandidateGen(model = "Qwen/Qwen3-Embedding-4B", description=desc, method=method)) # no prompt
    #        for q in ["one", "three", "four", "five"]:
    #            systems.append(MatcherCandidateGen(model = "Qwen/Qwen3-Embedding-4B", description=desc, method=method, query_prompt_id=q))
    #            systems.append(MatcherCandidateGen(model = "Qwen/Qwen3-Embedding-4B", description=desc, method=method, query_prompt_id=q, document_prompt_id=q))

    systems.append(MatcherCandidateGen(model = "Qwen/Qwen3-Embedding-4B", description="description_one_gen", query_prompt_id="four", document_prompt_id="four"))

    run_oaei_tracks(
        systems=systems,#[

            #MatcherCandidateGen(model = "all-MiniLM-L6-v2", description="description_text", include_simple=False),
            #MatcherCandidateGen(model = "all-MiniLM-L6-v2", description="description_text"),
            #MatcherCandidateGen(model = "all-MiniLM-L6-v2", description="description_text", query_prompt_id="one"),
            #MatcherCandidateGen(model = "Qwen/Qwen3-Embedding-4B"),
            #MatcherSimple()
        #],

        tracks = [ 
            ("anatomy_track", "anatomy_track-default"),
            #("conference", "conference-v1"),
            #("biodiv", "2023")
            #("commonkg", "yago-wikidata-v1"),
            #("commonkg", "nell-dbpedia-v1")
        ],
        timestamp_replacement="myrun"
    )
    logging.info(f"Experiment completed.")

def experiment(method):
    #systems = []
    #for desc in ["description_one_gen", "description_two_gen", "description_three_gen"]:
    #    systems.append(MatcherCandidateGen(model = "Qwen/Qwen3-Embedding-4B", description=desc, method=method)) # no prompt
    #    for q in ["one", "three", "four", "five"]:
    #        systems.append(MatcherCandidateGen(model = "Qwen/Qwen3-Embedding-4B", description=desc, method=method, query_prompt_id=q))
    #        systems.append(MatcherCandidateGen(model = "Qwen/Qwen3-Embedding-4B", description=desc, method=method, query_prompt_id=q, document_prompt_id=q))

    #run_oaei_tracks(
    #    systems=systems,
    #    tracks = [ ("anatomy_track", "anatomy_track-default"),],
    #    timestamp_replacement="methodexperiment"
    #)
    #logging.info(f"Experiment with method {method} completed.")


    systems = []
    for desc in ["description_one_gen", "description_two_gen", "description_three_gen"]:
        for q in ["one", "four", "five"]:
            systems.append(MatcherCandidateGen(model = "Qwen/Qwen3-Embedding-4B", description=desc, method=method, query_prompt_id=q, document_prompt_id=q))

    run_oaei_tracks(
        systems=systems,
        tracks = [ ("anatomy_track", "anatomy_track-default"),],
        timestamp_replacement="methodexperimenttwo"
    )

    logging.info(f"Experiment with method {method} completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    # if experiment_a provided as argument, run experiment a
    import sys
    if len(sys.argv) > 1:
        experiment(sys.argv[1])
    else:
        main()

    #else:
    #    main()