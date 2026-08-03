"""Debug retrieval shape — locks A.15a schema."""
from ragwatcher.index import RetrievalHit, RetrievalResult


def test_retrieval_result_shape():
    hit = RetrievalHit(
        rank=1,
        doc_id="d1",
        source="/x/a.md",
        snippet="hello",
        score=0.9,
        scores={"dense": 0.8, "bm25": 4.0, "rrf": 0.02, "rerank": 0.9},
        stage_survived="reranked",
        neighbors=["prev", "next"],
    )
    r = RetrievalResult(
        query="q",
        returned=[hit],
        dropped=[{"doc_id": "d2", "reason": "min_score"}],
        stages={
            "dense_candidates": 20,
            "bm25_candidates": 20,
            "fused": 30,
            "after_filters": 25,
            "reranked": 5,
            "returned": 1,
        },
        timings_ms={"retrieve": 42, "rerank": 118, "total": 165},
    )
    assert r.returned[0].scores.keys() == {"dense", "bm25", "rrf", "rerank"}
    assert r.stages.keys() >= {"dense_candidates", "fused", "reranked", "returned"}
    assert r.timings_ms.keys() >= {"retrieve", "rerank", "total"}
    assert r.dropped[0]["reason"] == "min_score"
