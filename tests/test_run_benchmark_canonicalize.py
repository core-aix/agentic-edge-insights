import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_benchmark import canonicalize_final_answer


def test_canonicalize_strips_long_narrative_answer():
    raw = (
        "Based on the gathered evidence, the answer is that the deleted word was "
        "inference. This follows from the amendment notes and cross-references."
    )
    out = canonicalize_final_answer(raw)
    assert len(out.split()) <= 24
    assert "Based on" not in out


def test_canonicalize_strips_markdown_emphasis():
    assert canonicalize_final_answer("**Honolulu, Milton**") == "Honolulu, Milton"


def test_canonicalize_fraction_list_dedupes_and_normalizes():
    raw = "3/4, 1/4, 3/4, 2/4, 1/2"
    out = canonicalize_final_answer(raw)
    assert out == "3/4,1/4,2/4,1/2"
