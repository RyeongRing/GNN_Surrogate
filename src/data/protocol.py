"""Input validation for the locked, event-level Seocho evaluation protocol."""

from collections import Counter


SPLITS = {"calib", "train_noobs", "val_modelsel", "val_conformal", "test_temporal"}


def validate_catalog(catalog, expected_counts=None):
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("Expected a nonempty list of locked event records")
    ids = []
    for event in catalog:
        missing = {"event_id", "start", "end", "split", "regime"} - event.keys()
        if missing:
            raise ValueError(f"Event record lacks {sorted(missing)}")
        if event["split"] not in SPLITS:
            raise ValueError(f"{event['event_id']}: unsupported split {event['split']!r}; "
                             "use the locked final catalog, not select_events.py")
        if event["regime"] not in {"A", "B", "C"}:
            raise ValueError(f"{event['event_id']}: unknown rainfall regime")
        ids.append(event["event_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate event IDs in catalog")
    counts = Counter(event["split"] for event in catalog)
    if expected_counts is not None and dict(counts) != dict(expected_counts):
        raise ValueError(f"Split counts {dict(counts)} differ from {expected_counts}")
    return dict(counts)


def validate_completed_runs(index, catalog, n_sets):
    if not catalog or not isinstance(n_sets, int) or n_sets < 1:
        raise ValueError("A nonempty catalog and positive integer n_sets are required")
    expected = {(event["event_id"], sid) for event in catalog for sid in range(n_sets)}
    actual = set(index)
    if actual != expected or len(actual) != len(index):
        missing = sorted(expected - actual)
        raise ValueError(f"Incomplete/duplicate ensemble archive: expected {len(expected)} "
                         f"runs, found {len(index)}; missing examples: {missing[:5]}")
