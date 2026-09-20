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

#: Tests written for this repository.
REPO_TEST_FILES = ("test_m0_provenance_gate.py",)


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


def test_swat_m_block_package_is_a_marker_only():
    path = SWAT_PACKAGE / "__init__.py"
    assert path.is_file()
    assert {item.name for item in SWAT_PACKAGE.glob("*.py")} == {"__init__.py"}
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert defined == []
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)
    ), "the M0 package marker must not import anything"
    assert "__all__: list[str] = []" in source


def test_no_swat_m_block_algorithm_is_implemented():
    """No M0-authored module may implement the SWAT-M-Block mechanism."""
    forbidden = (
        "DOAllocate", "DOMerge", "DOMerger", "differential_oblivious",
        "do_allocate", "do_merge", "bin_allocator", "BinAllocator",
        "noisy_allocat", "output_shuffle", "oblivious_shuffle", "bitonic",
        "sorting_network", "deamortiz", "de_amortiz", "epsilon_delta",
    )
    owned = [
        path for path in _iter_repo_files(".py")
        if not _names_prohibited_tokens_by_design(path)
    ]
    assert owned, "expected at least the M0 package marker and the verifier script"
    for path in owned:
        # docstrings may name the mechanism (that is how it is documented as absent);
        # only executable code is scanned.
        code = _code_without_docstrings(path)
        for token in forbidden:
            assert token not in code, (str(path.relative_to(REPO_ROOT)), token)


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


def test_project_state_says_m1_is_not_started():
    text = _normalised(REPO_ROOT / "PROJECT-STATE.md")
    assert "M0 — Repository / bootstrap freeze" in text
    assert "M1 — Functional SWAT-M-Block oracle" in text
    assert "NOT STARTED" in text
    for item in (
        "noisy bin allocation", "DO merge", "dummy / cover block I/O",
        "output oblivious shuffle", "de-amortisation", "attacks",
        "performance experiments",
    ):
        assert item in text, item
