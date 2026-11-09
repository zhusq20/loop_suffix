"""Check denominator semantics and reject incomplete source attribution."""
from collections import Counter
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from loop_token_sources import counts, metrics


def record(n, suffix=0, fill=0, sibling=0, deep=0):
    return dict(tokens=list(range(n + 1)), n_tokens=n + 1,
                stats=dict(tokens=n, verified_parents=n, covered_tokens=suffix + fill + sibling + deep,
                           from_suffix=suffix, from_fill=fill, from_sibling=sibling, from_deep=deep))


def test_pooled_token_weights_and_three_distinct_denominators():
    total = Counter()
    for row in [record(1, suffix=1), record(9, fill=6, deep=1)]:
        total.update(counts(row))
    m = metrics(total)
    assert m['suffix_decode_pct'] == 10  # Not the per-request mean of 50%.
    assert m['suffix_covered_pct'] == 12.5
    assert m['suffix_output_pct'] == pytest.approx(100 / 12)
    assert m['model_decode_pct'] == 70 and m['uncovered_decode_pct'] == 20
    assert m['prefill_output_pct'] == pytest.approx(200 / 12)


def test_prefill_only_is_retained_and_decode_percentages_are_undefined():
    m = metrics(counts(record(0)))
    assert m['prefill_only_requests'] == 1 and m['output_tokens'] == 1
    assert m['suffix_decode_pct'] is None and m['suffix_covered_pct'] is None
    assert m['prefill_output_pct'] == 100


@pytest.mark.parametrize('field,value', [('covered_tokens', 0), ('tokens', 5),
                                       ('verified_parents', 0), ('from_fill', -1)])
def test_inconsistent_counters_fail_instead_of_silently_renormalizing(field, value):
    row = record(2, suffix=1)
    row['stats'][field] = value
    with pytest.raises(ValueError):
        counts(row)
