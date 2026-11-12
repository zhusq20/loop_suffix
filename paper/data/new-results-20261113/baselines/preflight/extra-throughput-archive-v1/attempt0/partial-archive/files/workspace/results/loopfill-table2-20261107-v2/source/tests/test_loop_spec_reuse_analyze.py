from scripts.loop_spec_reuse_analyze import lookup_lengths, recycle_lengths, suffix_stats


def _record():
    return {
        "prefix_len": 3,
        "prompt_ids": [10, 11, 12],
        "tokens": [5, 6, 7, 8, 9, 10, 11],
        "drafts": [
            {"tick": 9, "pos": 6, "src": 70, "tok": 77, "top": [77, 8]},
            {"tick": 11, "pos": 6, "src": 7, "tok": 8, "top": [8, 1]},
            {"tick": 12, "pos": 7, "src": 8, "tok": 9, "top": [9, 2]},
            {"tick": 13, "pos": 8, "src": 9, "tok": 10, "top": [10, 3]},
            {"tick": 14, "pos": 9, "src": 10, "tok": 11, "top": [11, 4]},
        ],
        "rejects": [
            {
                "tick": 10,
                "pos": 5,
                "draft": 70,
                "gold": 7,
                "suffix": [
                    {"pos": 6, "tok": 8, "step": 2},
                    {"pos": 7, "tok": 9, "step": 1},
                    {"pos": 8, "tok": 99, "step": 0},
                ],
            }
        ],
    }


def test_suffix_survival_and_cross_checked_saving():
    events, saved = suffix_stats(_record(), d=1)
    (e,) = events
    assert e["match"] == [True, True, False]
    assert e["L"] == 2
    assert e["K"] == 2  # third reused token differs from WFD's later draft (10)
    assert e["gain"] == 2
    assert saved == 2
    assert e["wasted_rows"] == 1


def test_saving_is_capped_by_next_rejection():
    rec = _record()
    rec["rejects"].append({"tick": 15, "pos": 7, "draft": 9, "gold": 9, "suffix": []})
    events, saved = suffix_stats(rec, d=4)
    assert events[0]["gain"] == 1  # only position 6 lies strictly between the two rejections
    assert saved == 4


def test_prompt_lookup_lengths():
    rec = {"prefix_len": 3, "prompt_ids": [1, 2, 3], "tokens": [4, 1, 2, 3, 4]}
    assert lookup_lengths(rec, k_max=3) == [0, 3, 2, 1]


def test_recycle_uses_only_drafts_up_to_position():
    chain, tree = recycle_lengths(_record(), k_max=3)
    assert len(chain) == len(tree) == 6
    # At position 6 (token 8) the draft 8 -> 9 is created for position 7, so it is not yet visible.
    assert chain[3] == 0
