"""M0 gates: provenance freeze, defence-module exclusion, repository independence.

These tests are **written for this repository** (they are not imported from
EnhancedLETIndex) and they enforce the M0 acceptance criteria of
``decisions/0001-common-letindex-substrate.md`` and ``PROVENANCE.md``.

They are deliberately self-contained: they never read, fetch or require the
EnhancedLETIndex source repository.
"""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CODES_DIR = REPO_ROOT / "codes"
TESTS_DIR = CODES_DIR / "tests"
SRC_DIR = CODES_DIR / "src"
FROZEN_PACKAGE = SRC_DIR / "enhanced_letindex"
SWAT_PACKAGE = SRC_DIR / "swat_m_block"
MANIFEST_PATH = REPO_ROOT / "provenance" / "common-substrate.sha256"
VERIFY_SCRIPT = REPO_ROOT / "provenance" / "verify_manifest.py"

#: The source repository and commit this snapshot was taken from (PROVENANCE.md).
FROZEN_SOURCE_REPO = "Russell-H-0/EnhancedLETIndex"
FROZEN_SOURCE_COMMIT = "4e68be75b0ac45e09ce6da8f6d587490fea4f35e"
SWAT_REFERENCE_COMMIT = "b33646061ec1899ccf75c9ba8fe43b2653c44a6b"

#: Defence implementations that must never be imported into the SWAT baseline.
EXCLUDED_DEFENSE_MODULES = (
    "block_prp",
    "defense_state",
    "query_defense",
    "query_stash",
    "protected_merge",
    "deamortized_merge",
    "defended_index",
)

#: The frozen package, module by module.
FROZEN_MODULES = (
    "__init__", "identifiers", "record", "block", "dataset", "config", "level",
    "mapping", "storage", "trace", "pgm", "builder", "engine", "merge",
    "incremental_merge", "leakage", "leakage_metrics", "letindex_ref",
    "ref_workload", "ref_truth", "ref_adversary", "workload",
)

#: Files claimed byte-identical in PROVENANCE.md and therefore in the manifest.
BYTE_IDENTICAL_TEST_FILES = (
    "test_record.py", "test_block.py", "test_dataset.py", "test_mapping.py",
    "test_storage.py", "test_level_construction.py", "test_lookup.py",
    "test_multilevel_lookup.py", "test_engine.py", "test_pgm.py",
    "test_blocking_merge.py", "test_leakage.py", "test_letindex_ref.py",
)

#: Reduced copies; each must carry the derivation marker.
DERIVED_TEST_FILES = ("test_incremental_merge.py", "test_pgm_rank_certificate.py")
DERIVATION_MARKER = "DERIVED TEST, NOT BYTE-IDENTICAL IMPORT"

#: Tests written for this repository (M0 gates, the M1 / M2 / M3 focused suites and the
#: QA-1 stabilization sweep suite added by Issue #8).
REPO_TEST_FILES = (
    "test_m0_provenance_gate.py",
    "test_m1_functional_oracle.py",
    "test_m2_block_bin_allocation.py",
    "test_m3_content_bound_bin_schedule.py",
    "test_post_m3_regression_sweep.py",
)

#: The QA harness directory (QA-1, Issue #8).  Like these gate tests themselves, the harness
#: is a separation guard: it names the physical surface in order to instrument it and to
#: search for it, so its prose and that token table are prose, not an implementation.
QA_HARNESS_DIR = Path(__file__).resolve().parents[1] / "tools"

#: Algorithmic mechanism tokens that stay refused in *every* file, the QA harness included.
QA_HARNESS_FORBIDDEN = (
    "DOAllocate", "DOMerge", "DOMerger", "differential_oblivious", "do_allocate",
    "do_merge", "output_shuffle", "oblivious_shuffle", "bitonic", "sorting_network",
    "deamortiz", "de_amortiz", "epsilon_delta", "prp_writeback", "safe_output",
    "bounded_buffer", "ciphertext", "sgx_",
)


def _load_verify_module():
    spec = importlib.util.spec_from_file_location("m0_verify_manifest", VERIFY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iter_repo_files(*suffixes: str):
    """Every file in the working tree that is not inside ``.git``."""
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
        for name in filenames:
            path = Path(dirpath) / name
            if not suffixes or path.suffix in suffixes:
                yield path


def _names_prohibited_tokens_by_design(path: Path) -> bool:
    """True for text that legitimately *names* the prohibited constructs.

    That is the frozen snapshot itself (its vocabulary is not ours to police, and it
    is hash-gated anyway) and these M0 gate tests, whose lists spell the prohibited
    tokens out on purpose.
    """
    if path.name in REPO_TEST_FILES:
        return True
    try:
        path.relative_to(FROZEN_PACKAGE)
    except ValueError:
        return path.name in BYTE_IDENTICAL_TEST_FILES or path.name in DERIVED_TEST_FILES
    return True


def _manifest_entries() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, relative = line.split()
        entries[relative] = digest
    return entries


def _normalised(path: Path) -> str:
    """Document text with markdown decoration stripped and whitespace collapsed."""
    text = path.read_text(encoding="utf-8")
    for decoration in "*`>#":
        text = text.replace(decoration, " ")
    return " ".join(text.split())


def _code_without_docstrings(path: Path) -> str:
    """The executable part of a module: prose in docstrings is not an implementation."""
    source = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                source = source.replace(doc, "")
    return source


def _is_qa_harness(path: Path) -> bool:
    """True for the QA-1 harness under ``codes/tools``."""
    try:
        path.relative_to(QA_HARNESS_DIR)
    except ValueError:
        return False
    return path.suffix == ".py"


def _tokens_without_prose(path: Path) -> str:
    """Executable tokens of a module, with docstrings *and* comments dropped.

    The comment-stripping form is used for the QA harness: a comment naming a mechanism is
    documentation of what the harness deliberately does not do, exactly like a docstring,
    while identifiers, attribute names, imports and string literals stay in scope.
    """
    import io
    import tokenize

    pieces: list[str] = []
    stream = io.StringIO(_code_without_docstrings(path)).readline
    for token in tokenize.generate_tokens(stream):
        if token.type in (
            tokenize.COMMENT,
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENDMARKER,
        ):
            continue
        pieces.append(token.string)
    return " ".join(pieces)


# ---------------------------------------------------------------------------
# provenance / hash gate
# ---------------------------------------------------------------------------


def test_the_manifest_verifies_against_the_frozen_snapshot():
    verify = _load_verify_module()
    assert verify.main(["--quiet", "--root", str(REPO_ROOT)]) == 0


def test_the_manifest_covers_exactly_the_claimed_byte_identical_files():
    expected = {f"codes/src/enhanced_letindex/{m}.py" for m in FROZEN_MODULES}
    expected |= {f"codes/tests/{name}" for name in BYTE_IDENTICAL_TEST_FILES}
    assert set(_manifest_entries()) == expected
    assert len(expected) == 35  # 22 substrate modules + 13 tests


def test_there_is_no_unlisted_module_in_the_frozen_package():
    on_disk = {path.stem for path in FROZEN_PACKAGE.glob("*.py")}
    assert on_disk == set(FROZEN_MODULES)


def test_the_frozen_source_commit_is_recorded_in_provenance():
    text = (REPO_ROOT / "PROVENANCE.md").read_text(encoding="utf-8")
    assert FROZEN_SOURCE_REPO in text
    assert FROZEN_SOURCE_COMMIT in text
    assert SWAT_REFERENCE_COMMIT in text
    assert "include/enclave/DOMerger.hpp" in text
    assert "2026-09-20" in text
    for rule in (
        "one-time source snapshot",
        "no live synchronization",
        "no submodule/subtree/package dependency",
        "future source import requires explicit provenance revision",
        "EnhancedLETIndex defense implementations are excluded",
    ):
        assert rule in text, rule


# ---------------------------------------------------------------------------
# separation audit
# ---------------------------------------------------------------------------


def test_no_excluded_defense_module_exists_anywhere_in_the_repository():
    present = [
        str(path.relative_to(REPO_ROOT))
        for path in _iter_repo_files(".py")
        if path.stem in EXCLUDED_DEFENSE_MODULES
    ]
    assert present == []


def test_no_imported_module_reaches_an_excluded_defense_module():
    """Static import closure of the frozen package must stay free of the defences."""
    reached = set()
    for path in FROZEN_PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                reached.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                reached.add(node.module.rsplit(".", 1)[-1])
                reached.update(alias.name for alias in node.names)
    assert reached & set(EXCLUDED_DEFENSE_MODULES) == set()


def test_the_frozen_package_does_not_reach_a_defense_module_at_import_time():
    """Import the whole frozen package and assert no defence module appears."""
    import subprocess
    import sys

    program = (
        "import importlib, sys\n"
        "import enhanced_letindex\n"
        "for name in " + repr(list(FROZEN_MODULES)) + ":\n"
        "    importlib.import_module('enhanced_letindex.' + name)\n"
        "bad = [m for m in sys.modules if m.rsplit('.', 1)[-1] in "
        + repr(list(EXCLUDED_DEFENSE_MODULES)) + "]\n"
        "print(bad)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(CODES_DIR), env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_no_submodule_subtree_or_live_dependency_on_the_source_repository():
    assert not (REPO_ROOT / ".gitmodules").exists()
    assert not (CODES_DIR / ".gitmodules").exists()

    # no git / git-submodule mechanism anywhere in the sources
    for path in _iter_repo_files(".py", ".toml", ".cfg", ".in"):
        if _names_prohibited_tokens_by_design(path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in ("git+", "submodule", "subtree"):
            assert token not in text, (str(path.relative_to(REPO_ROOT)), token)

    # no dependency declaration may name the source repository.  (The frozen
    # substrate's own module docstrings do name EnhancedLETIndex - that is the
    # recorded provenance, not a dependency - so only manifests are scanned.)
    dependency_files = list(_iter_repo_files(".toml", ".cfg", ".in"))
    dependency_files += list(REPO_ROOT.glob("requirements*.txt"))
    for path in dependency_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in ("EnhancedLETIndex", "enhanced-letindex", "enhanced_letindex@"):
            assert token not in text, (str(path.relative_to(REPO_ROOT)), token)


def test_no_source_code_reads_the_enhancedletindex_working_tree():
    for path in _iter_repo_files(".py"):
        if _names_prohibited_tokens_by_design(path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "F:\\HHC\\Projects\\EnhancedLETIndex" not in text
        assert "/f/HHC/Projects/EnhancedLETIndex" not in text


# ---------------------------------------------------------------------------
# test-tree composition
# ---------------------------------------------------------------------------


def test_the_test_tree_holds_exactly_the_imported_and_m0_tests():
    on_disk = {path.name for path in TESTS_DIR.glob("test_*.py")}
    expected = set(BYTE_IDENTICAL_TEST_FILES) | set(DERIVED_TEST_FILES) | set(REPO_TEST_FILES)
    assert on_disk == expected


def test_derived_tests_declare_that_they_are_not_byte_identical_imports():
    for name in DERIVED_TEST_FILES:
        text = (TESTS_DIR / name).read_text(encoding="utf-8")
        assert DERIVATION_MARKER in text, name


def test_imported_tests_do_not_import_an_excluded_defense_module():
    for name in BYTE_IDENTICAL_TEST_FILES + DERIVED_TEST_FILES:
        tree = ast.parse((TESTS_DIR / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
                if node.module.rsplit(".", 1)[-1] == "enhanced_letindex":
                    names += [alias.name for alias in node.names]
            for dotted in names:
                assert dotted.rsplit(".", 1)[-1] not in EXCLUDED_DEFENSE_MODULES, (
                    name, dotted)


# ---------------------------------------------------------------------------
# no SWAT algorithm in M0
# ---------------------------------------------------------------------------


def test_swat_m_block_holds_only_the_authorised_milestone_modules():
    """Only modules an authorised milestone has added may exist in this package.

    M0 shipped a marker; M1 (Issue #2) added the functional oracle; M2 (Issue #4) added the
    distribution kernel and the block-bin allocation planner; M3 (Issue #6) added the
    content-bound bin schedule.  A further module needs its own milestone and decision
    record.
    """
    assert {item.name for item in SWAT_PACKAGE.glob("*.py")} == {
        "__init__.py", "functional_oracle.py", "distribution.py", "bin_allocator.py",
        "content_schedule.py",
    }
    path = SWAT_PACKAGE / "__init__.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert defined == [], "the package initialiser must not define behaviour"
    assert "functional_merge_oracle" in source
    assert "allocate_block_bins" in source
    assert "plan_swat_block_merge_schedule" in source
    assert "__all__" in source


def test_no_excluded_swat_mechanism_is_implemented():
    """Repo-wide guard on the mechanisms still excluded after M2.

    Each milestone narrows this guard by removing the tokens naming the mechanism it
    authorises: M1 (Issue #2) the logical oracle, M2 (Issue #4) the stochastic block-bin
    allocation planner, M3 (Issue #6) content binding and interior points.  Everything still
    excluded stays refused repo-wide: the DO data path and the DO merge, the physical
    surface (storage, trace, slot), the output permutation, the PRP writeback, the
    de-amortisation, the record-unit safe-output frontier and any (epsilon, delta) claim.

    QA-1 (Issue #8) adds one branch, not an exemption: the QA harness under ``codes/tools``
    proves zero physical I/O by *naming and patching* the physical surface, so that surface
    is allowed there while every algorithmic mechanism token in
    :data:`QA_HARNESS_FORBIDDEN` stays refused, and only prose (docstrings and comments) is
    ignored.
    """
    forbidden = (
        "DOAllocate", "DOMerge", "DOMerger", "differential_oblivious",
        "do_allocate", "do_merge",
        "output_shuffle", "oblivious_shuffle", "bitonic",
        "sorting_network", "deamortiz", "de_amortiz", "epsilon_delta",
        "UntrustedStorage", "TraceEvent", "TraceOperation", "SlotId",
        "prp_writeback", "physical_slot", "safe_output", "bounded_buffer",
    )
    owned = [
        path for path in _iter_repo_files(".py")
        if not _names_prohibited_tokens_by_design(path)
    ]
    assert owned, "expected at least the M0 package marker and the verifier script"
    for path in owned:
        # docstrings may name the mechanism (that is how it is documented as absent);
        # only executable code is scanned.
        if _is_qa_harness(path):
            code = _tokens_without_prose(path)
            for token in QA_HARNESS_FORBIDDEN:
                assert token not in code, (str(path.relative_to(REPO_ROOT)), token)
            continue
        code = _code_without_docstrings(path)
        for token in forbidden:
            assert token not in code, (str(path.relative_to(REPO_ROOT)), token)

    harnesses = [path for path in _iter_repo_files(".py") if _is_qa_harness(path)]
    for path in harnesses:
        assert not list(path.parent.glob("*.hpp")), str(path)
        assert "swat_m_block" not in path.name


def test_no_upstream_swat_source_file_is_present():
    for path in _iter_repo_files(".cc", ".cpp", ".hpp", ".h"):
        pytest.fail(f"upstream SWAT source file present: {path}")


# ---------------------------------------------------------------------------
# contract documents
# ---------------------------------------------------------------------------


def test_decision_0001_is_accepted_and_carries_the_frozen_contract():
    text = _normalised(REPO_ROOT / "decisions" / "0001-common-letindex-substrate.md")
    assert "Status: ACCEPTED" in text
    assert FROZEN_SOURCE_COMMIT in text
    for phrase in (
        "one-time snapshot of the LETIndex common substrate",
        "BlockId \u2260 SlotId namespace separation",
        "same newer-wins merge semantics",
        "same canonical PGM implementation",
        "no Enhanced defence mechanism",
        "shared experiment / artifact contract",
    ):
        assert phrase in text, phrase


def test_decision_0002_freezes_the_contract_without_claiming_differential_obliviousness():
    text = _normalised(REPO_ROOT / "decisions" / "0002-swat-m-block-contract.md")
    assert "Status: ACCEPTED FOR M1+ IMPLEMENTATION" in text
    assert "M0 IMPLEMENTS NO SWAT ALGORITHM" in text
    assert "block-observable adaptation of SWAT's differential-oblivious merge" in text
    assert "adapted at block-I/O granularity" in text
    assert "NOT automatically claimed" in text
    assert "one atomic sortable record" in text
    assert "oblivious output shuffle" in text
    assert "secret random tags" in text
    assert "bitonic" in text
    assert SWAT_REFERENCE_COMMIT in text


def test_project_state_records_m3_as_current_and_m4_as_not_started():
    text = _normalised(REPO_ROOT / "PROJECT-STATE.md")
    assert "M0 — repository / bootstrap freeze" in text
    assert "M1 — functional SWAT-M-Block oracle" in text
    assert "M2 — SWAT-style block-bin allocation planner" in text
    assert "M3 — Content-bound SWAT-M-Block bin schedule" in text
    assert ("M4 — Physically stage/fetch padded bins and implement the trusted bounded "
            "merge executor") in text
    assert "NOT AUTHORIZED by M3 and NOT STARTED" in text
    for item in (
        "physical SlotId scheduling", "temporary padded-bin physical materialization",
        "physical dummy / cover block I/O", "the DOAllocate data path", "DOMerge",
        "the record-unit safe-output frontier", "the bounded trusted merge buffer",
        "output block construction / output oblivious shuffle",
        "PRP publication / level retirement", "de-amortisation", "attacks",
        "benchmarks / performance experiments", "privacy theorem claim",
    ):
        assert item in text, item


def test_decision_0005_freezes_the_fourteen_m3_points():
    text = _normalised(REPO_ROOT / "decisions" / "0005-m3-content-bound-bin-schedule.md")
    assert "Status: ACCEPTED" in text
    for phrase in (
        "block is the allocation / I/O unit, record is the trusted comparison unit",
        "binds exactly the contiguous logical blocks it covers",
        "No block representative key is introduced",
        "sampled from the bin's actual record keys",
        "DUMMY_POS_INF",
        "min(i, load - i) + 1",   # the pinned weight exponent
        "reproduced verbatim",
        "deterministic and domain-separated",
        "Signed-tag tie ordering matches the pinned DOMerge pair ordering",
        "abstract bin-read order only",
        "Zero physical, slot and trace behaviour",
        "not reinterpreted as record-unit safe-output prefixes",
        "M1 remains the only exact newer-wins output oracle",
        "No privacy theorem claim",
        "M4 is not authorized by M3",
    ):
        assert phrase in text, phrase


def test_decision_0004_freezes_the_twelve_m2_points():
    text = _normalised(REPO_ROOT / "decisions" / "0004-m2-block-bin-allocation.md")
    assert "Status: ACCEPTED" in text
    for phrase in (
        "One allocation item is one logical LETIndex block",
        "Atomic bucket capacity is one block",
        "measured in blocks",
        "The pinned SWAT source is the provenance basis",
        "not claimed bit-identical to C++",
        "No AES/datum byte alignment in the abstract block planner",
        "explicit failure, never silently repaired",
        "M2 is not full DOAllocate",
        "The DP interior point is deferred to M3",
        "No physical I/O and no observable surface",
        "No formal privacy theorem claim",
        "M3 is not authorised by M2",
    ):
        assert phrase in text, phrase
