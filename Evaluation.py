import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
from typing import Any
from urllib.parse import quote
import requests
from Alignment import serialize_mapping_to_file
from Alignment import Alignment
from RDFGraphWrapper import RDFGraphWrapper
import csv
import json
from MatcherBase import MatcherBase
from ParameterConfigKeys import ADDITIONAL_OUTPUT_FOLDER
from rdflib import Graph
import torch


@dataclass
class TestCase:
    source: RDFGraphWrapper
    target: RDFGraphWrapper
    alignment: Alignment
    reference: Alignment
    testcase_name: str
    track_name: str
    source_path: str = None
    target_path: str = None
    reference_path: str = None
    alignment_path: str = None
    system_name: str = None
    runtime_ns: int = 0
    parameters_path: str = None




def load_parameters(file_path: str) -> dict[str, Any]:
    """Load parameters from a JSON file and return as a dict.

    Returns an empty dict when the file does not exist or cannot be parsed.
    """
    path = Path(file_path)
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if isinstance(data, dict):
        return data
    return {"value": data}


def evaluate(reference: Alignment, system: Alignment):
    """
    Evaluates the system output against the reference using prec, recall, f1 metric.
    """
    reference_set = set([(c.source, c.target, c.relation) for c in reference])
    system_set = set([(c.source, c.target, c.relation) for c in system])

    true_positives = len(reference_set & system_set)
    false_positives = len(system_set - reference_set)
    false_negatives = len(reference_set - system_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1, true_positives, false_positives, false_negatives


def load_testcase_from_json(json_path: str) -> TestCase:
    with open(json_path, "r") as f:
        data = json.load(f)

    source = RDFGraphWrapper(data["source_path"])
    target = RDFGraphWrapper(data["target_path"])
    alignment = Alignment(data["alignment_path"])
    reference = Alignment(data["reference_path"])

    return TestCase(
        source=source, target=target,
        alignment=alignment, reference=reference,
        testcase_name=data["testcase_name"],
        track_name=data["track_name"],
        source_path=data["source_path"],
        target_path=data["target_path"],
        reference_path=data["reference_path"],
        alignment_path=data["alignment_path"],
        system_name=data.get("system_name"),
        runtime_ns=data.get("runtime_ns", 0),
    )


def get_test_cases(tracks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    oaei_track_cache = Path.home() / "oaei_track_cache" / "oaei.webdatacommons.org"
    testcases = []
    for track in tracks:
        track_path = oaei_track_cache / track[0] / track[1]
        _ensure_track_downloaded(track_path, track[0], track[1])
        testcase_names = [f.name for f in track_path.iterdir() if f.is_dir()]
        for testcase_name in testcase_names:
            testcases.append((str(track_path / testcase_name / "source.rdf"), 
                              str(track_path / testcase_name / "target.rdf")))
    return testcases

def _run_and_evaluate(system: MatcherBase, prev: TestCase, timestamp: str) -> TestCase:
    system_name = f"{prev.system_name}+{system}" if prev.system_name else str(system)

    result_dir = Path("results") / f"{timestamp}_results"
    if prev.track_name:
        result_dir = result_dir / prev.track_name

    logging.info(f"Running matcher {system_name} on test case {prev.testcase_name}")

    result_testcase_system = result_dir / prev.testcase_name / system_name
    result_testcase_system.mkdir(parents=True, exist_ok=True)
    additional_output = result_testcase_system / "additional_output"
    additional_output.mkdir(exist_ok=True)

    parameters = load_parameters(prev.parameters_path) if prev.parameters_path else {}
    parameters[ADDITIONAL_OUTPUT_FOLDER] = str(additional_output.resolve())

    start_time = time.perf_counter_ns()
    alignment = system.match(prev.source, prev.target, prev.alignment, parameters)
    end_time = time.perf_counter_ns()
    elapsed_time = end_time - start_time
    total_runtime_ns = prev.runtime_ns + elapsed_time

    logging.info(f"Matcher {system_name} on test case {prev.testcase_name} finished in {timedelta(seconds=elapsed_time // 1_000_000_000)}")

    torch.cuda.empty_cache()

    alignment_path = str((result_testcase_system / "systemAlignment.rdf").resolve())
    serialize_mapping_to_file(alignment_path, alignment)
    with open(result_testcase_system / "performance.csv", "w", newline="") as perf_file:
        perf_file.write(f"Time\n")
        perf_file.write(f"{total_runtime_ns}\n")

    with open(result_testcase_system / "run_info.json", "w") as json_file:
        json.dump({
            "source_path": prev.source_path,
            "target_path": prev.target_path,
            "alignment_path": alignment_path,
            "reference_path": prev.reference_path,
            "testcase_name": prev.testcase_name,
            "track_name": prev.track_name,
            "runtime_ns": total_runtime_ns,
            "system_name": system_name,
        }, json_file, indent=2)

    precision, recall, f1, tp, fp, fn = evaluate(prev.reference, alignment)

    evaluator_csv = result_dir / "resultsEvaluatorBasic.csv"
    file_exists = evaluator_csv.exists()
    with evaluator_csv.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists or evaluator_csv.stat().st_size == 0:
            writer.writerow(["Track", "TestCase", "Matcher", "Precision", "Recall", "F1", "TP", "FP", "FN", "Time (ns)", "Time HH:MM:SS"])
        writer.writerow([prev.track_name or "", prev.testcase_name, system_name, precision, recall, f1, tp, fp, fn, total_runtime_ns, str(timedelta(seconds=total_runtime_ns // 1_000_000_000))])

    logging.info(f"Wrote results of matcher {system_name} on test case {prev.testcase_name}")
    return TestCase(
        source=prev.source, target=prev.target,
        alignment=alignment, reference=prev.reference,
        testcase_name=prev.testcase_name, track_name=prev.track_name,
        source_path=prev.source_path, target_path=prev.target_path,
        reference_path=prev.reference_path, alignment_path=alignment_path,
        system_name=system_name, runtime_ns=total_runtime_ns,
        parameters_path=prev.parameters_path,
    )


def run_single_testcase(system: MatcherBase, source_path: str, target_path: str, reference_path: str,
                        testcase_name: str = None, track_name: str = None, timestamp_replacement: str = None,
                        input_alignment: Alignment = None, parameters_path: str = None):
    """
    Runs a single matcher on one test case and writes results (alignment, performance, evaluation CSV).

    Results are stored under ``results/{timestamp}_results/{track_name}/{testcase_name}/{system}/``.

    :param system: The matcher to run.
    :param source_path: Path to the source ontology (RDF file).
    :param target_path: Path to the target ontology (RDF file).
    :param reference_path: Path to the reference alignment (RDF file).
    :param testcase_name: Human-readable name for the test case (defaults to parent directory name of source_path).
    :param track_name: Human-readable name for the track. Omitted from the path when not provided.
    :param timestamp_replacement: Optional fixed timestamp string for the result folder name.
    :param input_alignment: Optional pre-existing alignment to pass to the matcher. Defaults to an empty alignment.
    :param parameters_path: Optional path to a JSON parameters file.
    :return: TestCase containing loaded ontologies, alignment, and reference.
    """
    if input_alignment is None:
        input_alignment = Alignment()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if not timestamp_replacement else timestamp_replacement
    if testcase_name is None:
        testcase_name = Path(source_path).parent.name

    source = RDFGraphWrapper(str(source_path))
    target = RDFGraphWrapper(str(target_path))
    reference = Alignment(str(reference_path))

    prev = TestCase(
        source=source, target=target,
        alignment=input_alignment, reference=reference,
        testcase_name=testcase_name, track_name=track_name,
        source_path=str(source_path), target_path=str(target_path),
        reference_path=str(reference_path),
        parameters_path=parameters_path,
    )
    return _run_and_evaluate(system=system, prev=prev, timestamp=timestamp)

def _url_exists(session: requests.Session, url: str) -> bool:
    try:
        resp = session.head(url)
        return resp.status_code == 200
    except Exception as e:
        logging.error(f"Error checking URL {url}: {e}")
        return False

def _ensure_track_downloaded(track_path: Path, collection: str, version: str):
    """Download an OAEI track from the TDRS repository if not cached locally."""
    if track_path.is_dir() and any(track_path.iterdir()):
        return
    suite_url = f"http://oaei.webdatacommons.org/tdrs/testdata/persistent/{quote(collection)}/{quote(version)}/suite/"
    logging.info("Downloading track suite from %s", suite_url)

    session = requests.Session()

    g = Graph()
    resp = session.get(suite_url)
    resp.raise_for_status()
    g.parse(data=resp.content, format="xml")

    query = (
        "SELECT ?name WHERE { "
        "?x <http://www.seals-project.eu/ontologies/SEALSMetadata.owl#hasSuiteItem> ?item . "
        "?item <http://purl.org/dc/terms/identifier> ?name . "
        "} ORDER BY ?name"
    )
    testcase_names = [str(row.name) for row in g.query(query)]

    for tc_name in testcase_names:
        logging.info("  downloading test case %s", tc_name)
        base = f"{suite_url}{quote(tc_name)}/component/"

        if not _url_exists(session, base + "source/") or not _url_exists(session, base + "target/"):
            logging.warning("  skipping %s — source or target not available", tc_name)
            continue

        tc_dir = track_path / tc_name
        tc_dir.mkdir(parents=True, exist_ok=True)

        for comp_type in ['source', 'target', 'reference', 'input', 'parameters', 'evaluationexclusion']:
            comp_url = base + comp_type + "/"
            if comp_type not in ("source", "target") and not _url_exists(session, comp_url):
                continue
            resp = session.get(comp_url)
            resp.raise_for_status()
            (tc_dir / (comp_type + ".rdf")).write_bytes(resp.content)

    logging.info("Finished downloading track %s/%s", collection, version)


def run_oaei_tracks(systems:list[MatcherBase], tracks:list[tuple[str, str]], testcases:set=None, timestamp_replacement=None) -> list[TestCase]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if not timestamp_replacement else timestamp_replacement

    results: list[TestCase] = []
    oaei_track_cache = Path.home() / "oaei_track_cache" / "oaei.webdatacommons.org"
    for system in systems:
        for track in tracks:
            track_path = oaei_track_cache / track[0] / track[1]
            _ensure_track_downloaded(track_path, track[0], track[1])
            available_testcases = [f.name for f in track_path.iterdir() if f.is_dir()]

            for testcase in available_testcases:
                if testcases is not None and testcase not in testcases:
                    continue
                if Path(track_path / testcase / "reference.rdf").exists() == False:
                    continue
                parameters_file = track_path / testcase / "parameters.rdf"
                result = run_single_testcase(
                    system=system,
                    source_path=str(track_path / testcase / "source.rdf"),
                    target_path=str(track_path / testcase / "target.rdf"),
                    reference_path=str(track_path / testcase / "reference.rdf"),
                    testcase_name=testcase,
                    track_name=f"{track[0]}_{track[1]}",
                    timestamp_replacement=timestamp,
                    parameters_path=str(parameters_file) if parameters_file.exists() else None,
                )
                results.append(result)
    return results


def run_matcher_on_top(systems: list[MatcherBase], results: list[TestCase], timestamp_replacement: str = None) -> list[TestCase]:
    """
    Runs one or more matchers sequentially on top of previously produced results.
    Each matcher uses the output alignment of the previous one as its input.

    :param systems: List of matchers to run on top, applied in order.
    :param results: List of TestCase from a previous run.
    :param timestamp_replacement: Optional fixed timestamp string for the result folder name.
    :return: List of new TestCase with updated alignments and evaluation scores.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if not timestamp_replacement else timestamp_replacement

    new_results: list[TestCase] = []
    for system in systems:
        for prev in results:
            result = _run_and_evaluate(system=system, prev=prev, timestamp=timestamp)
            new_results.append(result)

    return new_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(get_test_cases([
        ("anatomy_track", "anatomy_track-default"),
        ("conference", "conference-v1"),
    ]))