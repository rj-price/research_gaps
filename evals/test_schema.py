"""The schema contract the suites and the watcher both depend on.

`IdentifiedGap.category` is stored in SQLite, shown to the gap matcher and printed in the
digest, so it has to be one of the three known values. The eval suite caught models
returning the plural section headings instead; these tests pin the fix.
"""
import pytest
from pydantic import ValidationError

from modules.llm import build_response_format
from modules.models import CriticResult, IdentifiedGap


@pytest.mark.parametrize("supplied, expected", [
    ("unexplored_territories", "unexplored_territory"),
    ("Methodological Limitations", "methodological_limitation"),
    ("contradictions", "contradiction"),
    ("Contradictions and Tensions", "contradiction"),
    ("unexplored_territory", "unexplored_territory"),
])
def test_plural_section_headings_are_accepted_as_the_singular_category(supplied, expected):
    assert IdentifiedGap(title="t", category=supplied, description="d").category == expected


def test_an_unrecognised_category_is_rejected_rather_than_stored():
    with pytest.raises(ValidationError):
        IdentifiedGap(title="t", category="something_else", description="d")


def test_the_category_reaches_the_model_as_a_schema_enum():
    schema = build_response_format(CriticResult)["json_schema"]["schema"]
    category = schema["$defs"]["IdentifiedGap"]["properties"]["category"]
    assert set(category["enum"]) == {"unexplored_territory", "methodological_limitation", "contradiction"}


def test_strict_mode_survives_the_enum_field():
    # _strictify walks $defs; an enum property has no sub-properties to rewrite
    schema = build_response_format(CriticResult)["json_schema"]
    assert schema["strict"] is True
    gap = schema["schema"]["$defs"]["IdentifiedGap"]
    assert gap["additionalProperties"] is False
    assert set(gap["required"]) == {"title", "category", "description"}
