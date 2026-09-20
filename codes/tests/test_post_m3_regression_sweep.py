"""Focused QA tests for the post-M3 regression sweep (QA-1, Issue #8).

These tests do not re-verify M1/M2/M3 semantics directly — the milestone test suites already
do that.  They verify the **sweep harness** itself:

* its independent evaluators and the pinned ``DOMerge`` simulator are correct on
  hand-computed fixtures, including one whose fallback branch really fires;
* the harness has teeth — a deliberately corrupted observation makes the relevant checker
  fail, so the sweep is not a vacuous PASS;
* ``--quick`` passes with non-degenerate counters (every subsystem exercised, at least one
  genuine pinned-fallback case, at least one mutated case, at least one instrumented case);
* a fixed master seed is deterministic in-process *and* across processes;
* the harness stays outside the production package and off the frozen manifest.
"""
from __future__ import annotations

import importlib.util
import io
import random
import subprocess
import sys
from pathlib import Path

import pytest

from enhanced_letindex.block import Block
from enhanced_letindex.identifiers import BlockId, RecordKey
from enhanced_letindex.record import Record
from swat_m_block import (
    LogicalBlockRunView,
    allocate_block_bins,
    plan_swat_block_merge_schedule,
    swat_reference_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CODES_DIR = REPO_ROOT / "codes"
SRC_DIR = CODES_DIR / "src"
TOOL_PATH = CODES_DIR / "tools" / "post_m3_regression_sweep.py"
MANIFEST_PATH = REPO_ROOT / "provenance" / "common-substrate.sha256"
SWAT_PACKAGE = SRC_DIR / "swat_m_block"

SOURCE = "source"
TARGET = "target"


def _load_harness():
    spec = importlib.util.spec_from_file_location("post_m3_regression_sweep", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sweep = _load_harness()


# ---------------------------------------------------------------------------
# A. the harness exists where the issue says it does
# ---------------------------------------------------------------------------


def test_a_the_harness_lives_outside_the_production_package():
    assert TOOL_PATH.is_file()
    assert TOOL_PATH.parent.name == "tools"
    assert TOOL_PATH.parent.parent == CODES_DIR
    assert SWAT_PACKAGE not in TOOL_PATH.parents


def test_a_the_focused_qa_test_lives_in_the_test_tree():
    assert Path(__file__).parent.name == "tests"


def test_a_no_production_module_references_the_harness():
    for path in sorted(SWAT_PACKAGE.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "post_m3_regression_sweep" not in text, path.name
        assert "codes.tools" not in text, path.name


def test_a_the_harness_is_not_covered_by_the_frozen_manifest():
    frozen = {
        line.split()[1].strip()
        for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert frozen, "the manifest must not be empty"
    for relative in ("codes/tools/post_m3_regression_sweep.py",
                     "codes/tests/test_post_m3_regression_sweep.py"):
        assert relative not in frozen, relative


def test_a_the_static_separation_check_passes():
    sweep.check_static_separation()


# ---------------------------------------------------------------------------
# B. independent evaluators, on hand-computed fixtures
# ---------------------------------------------------------------------------


def test_b_the_independent_newer_wins_evaluator_matches_hand_computation():
    assert sweep.independent_newer_wins((), ()) == ()
    assert sweep.independent_newer_wins(((1, "s1"), (3, "s3")), ()) == ((1, "s1"), (3, "s3"))
    assert sweep.independent_newer_wins((), ((2, "t2"),)) == ((2, "t2"),)
    # the newer source wins every shared key, and each key appears once
    assert sweep.independent_newer_wins(
        ((1, "s1"), (2, "s2")), ((2, "t2"), (3, "t3"))
    ) == ((1, "s1"), (2, "s2"), (3, "t3"))
    # negative and zero keys sort numerically
    assert sweep.independent_newer_wins(
        ((-5, "s-5"), (0, "s0")), ((-3, "t-3"), (0, "t0"), (7, "t7"))
    ) == ((-5, "s-5"), (-3, "t-3"), (0, "s0"), (7, "t7"))


def test_b_the_independent_packing_evaluator_matches_the_canonical_rule():
    assert sweep.independent_expected_block_sizes(0, 8) == ()
    assert sweep.independent_expected_block_sizes(1, 8) == (1,)
    assert sweep.independent_expected_block_sizes(8, 8) == (8,)
    assert sweep.independent_expected_block_sizes(9, 8) == (8, 1)
    assert sweep.independent_expected_block_sizes(18, 8) == (8, 8, 2)
    assert sweep.independent_expected_block_sizes(7, 1) == (1,) * 7


def test_b_the_pinned_simulator_reproduces_a_hand_computed_fallback_case():
    """3 source bins, 4 target bins, tagged order t, s, s, s, t, t, t.

    The third source tag arrives once the source is exhausted (both source fetches are
    already spent) while the target still has unread bins, so the pinned ``else if`` fires and
    fetches ``(target, 2)``.  Hand-computed projection:

    preloads (source, 0), (target, 0); then (target, 1) on the t tag, (source, 1) and
    (source, 2) on the first two s tags, (target, 2) on the third s tag (the fallback),
    (target, 3) on the second t tag, and nothing for the last two tags.
    """
    tagged = [(TARGET, 1), (SOURCE, 1000), (SOURCE, 1128), (SOURCE, 1256),
              (TARGET, 2256), (TARGET, 2384), (TARGET, 2516)]
    fetches, fallbacks = sweep.pinned_projected_fetches(3, 4, tagged)
    assert fetches == ((SOURCE, 0), (TARGET, 0), (TARGET, 1), (SOURCE, 1), (SOURCE, 2),
                       (TARGET, 2), (TARGET, 3))
    assert fallbacks == ((SOURCE, 2),)
    # every planned bin is still fetched exactly once
    assert sorted(fetches) == sorted(
        [(SOURCE, index) for index in range(3)] + [(TARGET, index) for index in range(4)]
    )


def test_b_the_pinned_simulator_does_not_invent_a_fallback():
    """One bin per side: no fallback is possible and none is reported."""
    fetches, fallbacks = sweep.pinned_projected_fetches(1, 1, [(SOURCE, 1), (TARGET, 2)])
    assert fetches == ((SOURCE, 0), (TARGET, 0))
    assert fallbacks == ()


def test_b_the_pinned_simulator_handles_one_sided_and_empty_sides():
    assert sweep.pinned_projected_fetches(0, 0, []) == ((), ())
    fetches, fallbacks = sweep.pinned_projected_fetches(2, 0, [(SOURCE, 1), (SOURCE, 2)])
    assert fetches == ((SOURCE, 0), (SOURCE, 1))
    assert fallbacks == ()


def test_b_the_pinned_simulator_never_imports_the_m3_schedule_builder():
    source = TOOL_PATH.read_text(encoding="utf-8")
    body = source.split("def pinned_projected_fetches(", 1)[1].split("\ndef ", 1)[0]
    assert "build_abstract_merge_schedule" not in body
    assert "bind_block_allocation" not in body


def test_b_the_schedule_structure_helper_ignores_record_values():
    def build(values, level):
        blocks = tuple(
            Block.sorted_block(
                BlockId(level * 100 + index),
                [Record(RecordKey(key), f"{values}-{key}") for key in chunk],
                2,
            )
            for index, chunk in enumerate(([1, 2], [3, 4]))
        )
        return LogicalBlockRunView(level, blocks, 2)

    plan = allocate_block_bins(2, swat_reference_config(5))
    first = plan_swat_block_merge_schedule(build("a", 4), build("b", 5), source_plan=plan,
                                           target_plan=plan)
    second = plan_swat_block_merge_schedule(build("x", 4), build("y", 5), source_plan=plan,
                                            target_plan=plan)
    assert sweep.schedule_structure(first) == sweep.schedule_structure(second)
    assert first != second, "the full dataclass must still see the different values"


# ---------------------------------------------------------------------------
# C. the harness has teeth (mutating harness-local oracles must fail the sweep)
# ---------------------------------------------------------------------------


def test_c_a_corrupted_pinned_projection_makes_the_m3_checker_fail(monkeypatch):
    monkeypatch.setattr(
        sweep, "pinned_projected_fetches",
        lambda source_bins, target_bins, tagged: (((SOURCE, 0),), ()),
    )
    budget = sweep.QUICK_BUDGET
    rng = random.Random(sweep.DEFAULT_MASTER_SEED)
    sweep.generate_m1_cases(rng, budget)
    sweep.generate_m2_cases(rng, budget)
    cases = sweep.generate_m3_cases(rng, budget)
    with pytest.raises(sweep.SweepFailure) as error:
        for case in cases:
            sweep.check_m3_case(case)
    assert error.value.subsystem == "M3"
    assert "pinned projected fetches" in error.value.invariant


def test_c_a_corrupted_m1_evaluator_makes_the_m1_checker_fail(monkeypatch):
    monkeypatch.setattr(sweep, "independent_newer_wins", lambda source, target: ())
    budget = sweep.QUICK_BUDGET
    rng = random.Random(sweep.DEFAULT_MASTER_SEED)
    cases = [case for case in sweep.generate_m1_cases(rng, budget)
             if case.source_items or case.target_items]
    with pytest.raises(sweep.SweepFailure) as error:
        for case in cases:
            sweep.check_m1_case(case)
    assert error.value.subsystem == "M1"
    assert "newer-wins" in error.value.invariant


def test_c_a_failure_report_carries_the_reproduction_minimum():
    failure = sweep.SweepFailure(subsystem="M3", case_index=17, invariant="an invariant",
                                 detail="some detail", context="family=x block_count=3")
    report = failure.report(4242)
    for expected in ("FAILURE", "4242", "17", "M3", "an invariant", "some detail",
                     "family=x block_count=3", "--quick --seed 4242"):
        assert expected in report


# ---------------------------------------------------------------------------
# D. the quick sweep passes with non-degenerate counters
# ---------------------------------------------------------------------------


def test_d_the_quick_sweep_passes_with_every_subsystem_exercised():
    report = sweep.run_sweep("quick", sweep.DEFAULT_MASTER_SEED,
                             stream=io.StringIO())
    assert report.passed is True
    assert report.m1_cases > 0
    assert report.m2_plans > 0
    assert report.m2_failures > 0
    assert report.m3_schedules > 0
    assert report.m3_two_sided > 0
    assert report.pinned_comparisons == report.m3_two_sided
    assert report.instrumented_cases > 0
    assert report.mutation_checked_cases > 0
    assert report.total_cases == (report.m1_cases + report.m2_plans + report.m2_failures
                                 + report.m3_schedules)


def test_d_the_quick_sweep_genuinely_triggers_the_pinned_fallback():
    """Acceptance requires at least one generated case to exercise the fallback branch."""
    report = sweep.run_sweep("quick", sweep.DEFAULT_MASTER_SEED,
                             stream=io.StringIO())
    assert report.fallback_triggered_cases >= 1


def test_d_every_subsystem_counter_survives_a_seed_change():
    for seed in (1, 7, 999):
        report = sweep.run_sweep("quick", seed, stream=io.StringIO())
        assert report.passed is True
        assert report.m1_cases > 0 and report.m3_schedules > 0
        assert report.fallback_triggered_cases >= 1


def test_d_the_summary_is_compact_and_labelled():
    report = sweep.run_sweep("quick", sweep.DEFAULT_MASTER_SEED,
                             stream=io.StringIO())
    text = "\n".join(sweep.summary_lines(report))
    for expected in ("total cases", "M1 cases", "M2 successful plans",
                     "M2 explicit failures", "M3 schedules",
                     "pinned merge-loop comparisons",
                     "pinned fallback-triggered cases", "elapsed", "PASS"):
        assert expected in text
    assert "performance claim" in text, "elapsed time must be labelled as operational only"
    assert len(sweep.summary_lines(report)) <= 16


# ---------------------------------------------------------------------------
# E. determinism
# ---------------------------------------------------------------------------


def test_e_a_fixed_seed_is_deterministic_in_process():
    first = sweep.run_sweep("quick", 4242, stream=io.StringIO())
    second = sweep.run_sweep("quick", 4242, stream=io.StringIO())
    first.elapsed_seconds = second.elapsed_seconds = 0.0
    assert first == second


def test_e_a_fixed_seed_is_deterministic_across_processes():
    command = [sys.executable, str(TOOL_PATH), "--quick", "--seed", "31337"]
    first = subprocess.run(command, capture_output=True, text=True, cwd=CODES_DIR, check=True)
    second = subprocess.run(command, capture_output=True, text=True, cwd=CODES_DIR, check=True)

    def without_elapsed(text):
        return "\n".join(line for line in text.splitlines() if "elapsed" not in line)

    assert without_elapsed(first.stdout) == without_elapsed(second.stdout)
    assert first.stdout.count("PASS") == 1
    assert first.returncode == 0 and second.returncode == 0


def test_e_generated_cases_are_a_pure_function_of_the_master_seed():
    first = sweep.generate_m3_cases(random.Random(5), sweep.QUICK_BUDGET)
    second = sweep.generate_m3_cases(random.Random(5), sweep.QUICK_BUDGET)
    assert first == second
    other = sweep.generate_m3_cases(random.Random(6), sweep.QUICK_BUDGET)
    assert other != first


# ---------------------------------------------------------------------------
# F. CLI surface
# ---------------------------------------------------------------------------


def test_f_the_cli_exposes_quick_full_and_seed():
    result = subprocess.run([sys.executable, str(TOOL_PATH), "--help"],
                            capture_output=True, text=True, cwd=CODES_DIR, check=True)
    assert "--quick" in result.stdout
    assert "--full" in result.stdout
    assert "--seed" in result.stdout


def test_f_the_quick_cli_run_passes_and_prints_pass():
    result = subprocess.run([sys.executable, str(TOOL_PATH), "--quick"],
                            capture_output=True, text=True, cwd=CODES_DIR)
    assert result.returncode == 0
    assert "PASS" in result.stdout
    assert "FAIL" not in result.stdout


def test_f_the_full_budget_is_several_thousand_cases_and_bounded():
    """The full mode is not run in the test suite; its budget is asserted instead."""
    budget = sweep.FULL_BUDGET
    assert budget.m1_cases + budget.m2_cases + budget.m3_cases >= 3000
    generated = budget.m1_cases + budget.m2_cases + budget.m3_cases
    assert generated == 640 + 2400 + 2000
    assert sweep.QUICK_BUDGET.m1_cases + sweep.QUICK_BUDGET.m2_cases \
        + sweep.QUICK_BUDGET.m3_cases < 500


def test_f_an_unknown_mode_is_refused():
    with pytest.raises(KeyError):
        sweep.run_sweep("enormous", 1)
