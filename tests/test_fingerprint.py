from pathlib import Path

import pytest

from ragwatcher.config import Settings
from ragwatcher.errors import SchemaMismatch
from ragwatcher.manifest import Manifest, build_fingerprint, save


def _write_manifest(tmp: Path, fp: dict[str, object]) -> None:
    persist = tmp / ".rag_index"
    persist.mkdir(parents=True, exist_ok=True)
    m = Manifest.new(fp)
    save(persist / "manifest.json", m)


def test_fingerprint_stable_across_settings_reload():
    s1 = Settings()
    s2 = Settings()
    assert build_fingerprint(s1) == build_fingerprint(s2)


PINNED_MUTATIONS = [
    ("embed_model", {"embed": {"model": "different/model"}}),
    ("chunk_size", {"chunk": {"size": 999}}),
    ("chunk_overlap", {"chunk": {"overlap": 999}}),
    ("chunk_strategy", {"chunk": {"strategy": "semantic"}}),
    ("store_backend", {"store": {"backend": "simple"}}),
]


@pytest.mark.parametrize("field, mutation", PINNED_MUTATIONS)
def test_pinned_field_mutation_changes_fp(field: str, mutation: dict) -> None:
    base = Settings()
    mutated_dump = base.model_dump()
    for k, v in mutation.items():
        mutated_dump[k].update(v)
    mutated = Settings(**mutated_dump)
    assert build_fingerprint(base) != build_fingerprint(mutated), field


NON_PINNED_MUTATIONS = [
    {"rerank": {"model": "any/other"}},
    {"retrieve": {"top_k": 42}},
    {"log": {"level": "debug"}},
    {"watch": {"debounce_sec": 5.5}},
]


@pytest.mark.parametrize("mutation", NON_PINNED_MUTATIONS)
def test_non_pinned_field_does_not_change_fp(mutation: dict) -> None:
    base = Settings()
    mutated_dump = base.model_dump()
    for k, v in mutation.items():
        mutated_dump[k].update(v)
    mutated = Settings(**mutated_dump)
    assert build_fingerprint(base) == build_fingerprint(mutated)


def test_schema_mismatch_raised_on_drift(tmp_path: Path):
    # Simulate manifest saved with different fingerprint, then check_fingerprint
    fp1 = build_fingerprint(Settings())
    m = Manifest.new(fp1)
    fp2 = dict(fp1) | {"embed_model": "changed"}
    with pytest.raises(SchemaMismatch):
        m.check_fingerprint(fp2)
