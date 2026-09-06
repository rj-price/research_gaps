"""Runs the eval suites and scores them against the regression thresholds.

    python -m evals.runner --suite all
    python -m evals.runner --suite relevance --model google/gemini-2.5-flash
    python -m evals.runner --suite all --check      # exit 1 on any threshold breach

Every run writes a JSON record to `evals/results/`, so two models — or the same model
before and after a prompt change — can be compared after the fact.
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from aiolimiter import AsyncLimiter
from dotenv import load_dotenv

from modules.llm import DEFAULT_MODEL, get_client

from evals import judge as judge_module
from evals.suites import SUITES, SuiteResult

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
THRESHOLDS_PATH = Path(__file__).parent / "thresholds.json"


def load_thresholds(path: Path = THRESHOLDS_PATH) -> Dict[str, dict]:
    return json.loads(path.read_text())


def check_thresholds(result: SuiteResult, thresholds: Dict[str, dict]) -> List[str]:
    """Returns one message per breached threshold; an empty list means the suite passed."""
    breaches = []
    for metric, bound in thresholds.get(result.suite, {}).items():
        value = result.metrics.get(metric)
        if value is None:
            breaches.append(f"{result.suite}.{metric}: not reported by the suite")
            continue
        if "min" in bound and value < bound["min"]:
            breaches.append(f"{result.suite}.{metric} = {value} (below minimum {bound['min']})")
        if "max" in bound and value > bound["max"]:
            breaches.append(f"{result.suite}.{metric} = {value} (above maximum {bound['max']})")
    return breaches


def format_report(results: List[SuiteResult], breaches: Dict[str, List[str]]) -> str:
    lines = ["# Eval run", ""]
    for result in results:
        header = f"## {result.suite} ({result.dataset}) — {result.model}"
        if result.judge_model:
            header += f", judged by {result.judge_model}"
        lines += [header, "", f"{len(result.cases)} cases in {result.duration_s}s", "",
                  "| metric | value |", "| --- | --- |"]
        for key, value in result.metrics.items():
            lines.append(f"| {key} | {value} |")
        suite_breaches = breaches.get(result.suite, [])
        lines += ["", "**FAIL**" if suite_breaches else "**PASS**"]
        lines += [f"- {breach}" for breach in suite_breaches]
        lines.append("")
    return "\n".join(lines)


def format_failures(result: SuiteResult) -> str:
    """The cases worth reading after a run: the ones the pipeline got wrong."""
    lines = []
    if result.suite == "relevance":
        for row in result.cases:
            if not row["correct"]:
                verdict = "kept, should have been rejected" if row["predicted"] else "rejected, should have been kept"
                lines.append(f"  {row['paper_id']}: {verdict} (score {row['score']}) — {row['reason']}")
    elif result.suite == "gap_matching":
        for row in result.cases:
            missed = set(row["expected_gaps"]) - set(row["predicted_gaps"])
            spurious = set(row["predicted_gaps"]) - set(row["expected_gaps"]) - set(row["allowed_gaps"])
            wrong = [r for r in row["relationships"] if not r["correct"]]
            if missed:
                lines.append(f"  {row['paper_id']}: missed {', '.join(sorted(missed))}")
            if spurious:
                lines.append(f"  {row['paper_id']}: spurious {', '.join(sorted(spurious))}")
            for rel in wrong:
                lines.append(
                    f"  {row['paper_id']}: {rel['gap_id']} labelled '{rel['predicted']}', "
                    f"expected one of {rel['expected']}"
                )
    elif result.suite.startswith("gap_analysis"):
        for row in result.cases:
            for issue in row["structural_issues"]:
                lines.append(f"  {row['case_id']}: {issue}")
            for theme in row["themes"]:
                if not theme["covered"]:
                    lines.append(f"  {row['case_id']}: missed theme — {theme['theme']}")
            for fabrication in row["fabrications"]:
                lines.append(f"  {row['case_id']}: fabricated — {fabrication['claim']} (\"{fabrication['quote']}\")")
            for gap in row["gaps"]:
                if gap["grounded"] is not None and gap["grounded"] <= 2:
                    lines.append(f"  {row['case_id']}: poorly grounded gap '{gap['title']}' — {gap['judge_comment']}")
    return "\n".join(lines)


def write_result(result: SuiteResult, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.started_at.replace(":", "").replace("-", "")
    path = results_dir / f"{result.suite}-{stamp}.json"
    path.write_text(result.model_dump_json(indent=2))
    return path


async def run_suites(
    suite_names: List[str], model: str, rate_limit: int, limit: int | None,
    judge_model_id: str | None, write: bool, results_dir: Path,
) -> Tuple[List[SuiteResult], Dict[str, List[str]]]:
    client = get_client()
    limiter = AsyncLimiter(rate_limit, 60)
    thresholds = load_thresholds()
    results: List[SuiteResult] = []
    breaches: Dict[str, List[str]] = {}

    try:
        for name in suite_names:
            logger.info(f"Running the {name} suite against {model}...")
            kwargs = {"limit": limit}
            if name == "gap_analysis":
                kwargs["judge_model_id"] = judge_model_id
            result = await SUITES[name](client, model, limiter, **kwargs)
            results.append(result)
            breaches[name] = check_thresholds(result, thresholds)
            if write:
                logger.info(f"Wrote {write_result(result, results_dir)}")
    finally:
        await client.aclose()

    return results, breaches


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the research_gaps eval suites")
    parser.add_argument(
        "--suite", action="append", choices=list(SUITES) + ["all"], default=None,
        help="Suite to run; repeatable. Defaults to all.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model ID under test")
    parser.add_argument(
        "--judge-model", default=None,
        help=f"Judge model for the gap_analysis suite (default {judge_module.DEFAULT_JUDGE_MODEL}, "
             "or $EVAL_JUDGE_MODEL)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N cases per suite")
    parser.add_argument("--rate-limit", type=int, default=20, help="Max LLM requests per minute")
    parser.add_argument("--check", action="store_true", help="Exit non-zero if any threshold is breached")
    parser.add_argument("--no-write", action="store_true", help="Do not save a JSON record of the run")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="Where to write run records")
    args = parser.parse_args()

    names = args.suite or ["all"]
    suite_names = list(SUITES) if "all" in names else list(dict.fromkeys(names))

    results, breaches = asyncio.run(run_suites(
        suite_names, args.model, args.rate_limit, args.limit,
        args.judge_model, not args.no_write, Path(args.results_dir),
    ))

    print()
    print(format_report(results, breaches))
    for result in results:
        failures = format_failures(result)
        if failures:
            print(f"Cases to look at ({result.suite}):")
            print(failures)
            print()

    total_breaches = sum(len(v) for v in breaches.values())
    if args.check and total_breaches:
        print(f"{total_breaches} threshold breach(es).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
