"""The eval suites as a regression test.

These call OpenRouter and therefore cost money, so they are skipped unless RUN_LLM_EVALS=1:

    RUN_LLM_EVALS=1 pytest evals/test_suites.py -v

The thresholds in `evals/thresholds.json` are the contract. A failure here means the
model, a prompt or a schema change has moved behaviour, not that a test is flaky —
check the reported metrics before relaxing a bound.
"""
import os

import pytest
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv

from modules.llm import DEFAULT_MODEL, get_client

from evals.runner import check_thresholds, format_failures, load_thresholds, write_result
from evals.suites import SUITES

load_dotenv()

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        os.getenv("RUN_LLM_EVALS") != "1",
        reason="Set RUN_LLM_EVALS=1 to run the paid eval suites.",
    ),
]

MODEL = os.getenv("EVAL_MODEL", DEFAULT_MODEL)


@pytest.mark.asyncio
@pytest.mark.parametrize("suite_name", list(SUITES))
async def test_suite_meets_thresholds(suite_name):
    client = get_client()
    limiter = AsyncLimiter(int(os.getenv("EVAL_RATE_LIMIT", "20")), 60)
    try:
        result = await SUITES[suite_name](client, MODEL, limiter)
    finally:
        await client.aclose()

    write_result(result)
    breaches = check_thresholds(result, load_thresholds())

    assert not breaches, (
        f"{suite_name} regressed:\n  " + "\n  ".join(breaches)
        + f"\n\nMetrics: {result.metrics}\n\n{format_failures(result)}"
    )
