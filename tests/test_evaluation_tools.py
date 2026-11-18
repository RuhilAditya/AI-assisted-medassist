"""Phase 5 tests: evaluation tooling must be correct, or every reported number is wrong."""
from __future__ import annotations

import random

import pytest

from scripts.evaluate.answers import claims, score
from scripts.evaluate.heldout_text import perturb, typo
from scripts.data.generate_heldout_notes import make_note
from scripts.evaluate.user_study import adjective, describe, sus_score


# ---------- SUS ----------
def _sus_row(values):
    return {"participant": "P", **{f"S{i}": str(v) for i, v in enumerate(values, 1)}}


def test_sus_extremes_and_neutral():
    assert sus_score(_sus_row([5, 1] * 5)) == 100.0  # best answer on every item
    assert sus_score(_sus_row([1, 5] * 5)) == 0.0    # worst
    assert sus_score(_sus_row([3] * 10)) == 50.0     # neutral


def test_sus_rejects_out_of_range():
    with pytest.raises(ValueError):
        sus_score(_sus_row([6] + [3] * 9))


def test_sus_adjectives_and_stats():
    assert adjective(90) == "Excellent" and adjective(75) == "Good" and adjective(60) == "OK"
    d = describe([60.0, 70.0, 80.0])
    assert d["mean"] == 70.0 and d["n"] == 3 and d["ci95"][0] < 70 < d["ci95"][1]


# ---------- held-out generator & typos ----------
def test_generator_is_deterministic_and_gold_is_present_in_text():
    a, b = make_note(0, random.Random(5)), make_note(0, random.Random(5))
    assert a == b
    for item in a["phi"]:
        assert item["value"] in a["text"], item
    for k in a["keep"]:
        assert k in a["text"], k


def test_typo_rate_zero_is_identity_and_gold_follows_changes():
    text = "Patient presents with breathlessness and productive cough."
    assert typo(text, 0.0, random.Random(1))[0] == text
    noisy, changed = typo(text, 1.0, random.Random(1))
    assert noisy != text and changed
    assert perturb("productive cough", changed) in noisy


# ---------- answer scoring ----------
def _partial(level="URGENT", conflicts=()):
    return {
        "evidence": {"notes": [{"note_id": "note_1"}], "image_analysis": {"similar_cases": [{}]}, "passages": []},
        "triage": {"level": level},
        "conflicts": list(conflicts),
    }


def test_claims_skip_headings_and_short_fragments():
    text = "Clinical picture:\n- Fever for three days with productive cough [note_1].\n- Short one."
    assert claims(text) == ["Fever for three days with productive cough [note_1]."]


def test_score_counts_citations_validity_and_ungrounded_numbers():
    raw = ("SUMMARY: The patient is URGENT with SpO2 92% and effusion likely 0.95 [image].\n"
           "DETAILS:\n- Heart rate is 140 beats per minute recorded [note_1].\n- Similar case suggests effusion here [case-4].")
    s = score(_partial(), "evidence: spo2=92 effusion 0.95 heart rate 120", raw, "How urgent?")
    assert s["format_ok"] and s["claims"] == 3 and s["cited_claims"] == 3
    assert s["citations"] == 3 and s["valid_citations"] == 2       # case-4 does not exist
    assert s["ungrounded_numbers"] == ["140"]                      # 92, 0.95 are in the evidence
    assert s["states_triage_level"] and s["other_level_in_summary"] == []


def test_score_conflict_disclosure_and_unsafe_language():
    conflict = {"severity": "high", "finding": "pleural effusion"}
    raw = "SUMMARY: Effusion is definitely ruled out on this film [image].\nDETAILS:\n- nothing more to add here today."
    s = score(_partial(conflicts=[conflict]), "evidence", raw, "q")
    assert s["has_severe_conflict"] and s["discloses_conflict"] is False
    assert s["unsafe_phrases"]
    ok = score(_partial(conflicts=[conflict]), "evidence",
               "SUMMARY: The note and image disagree about pleural effusion [conflict].\nDETAILS:\n- x", "q")
    assert ok["discloses_conflict"] is True
