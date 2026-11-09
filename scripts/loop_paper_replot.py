"""Regenerate mechanism/headroom/tree/source figures from committed aggregates only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def records(path):
    def scalar(value):
        if value=='':
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    with path.open() as f:
        return [{k:scalar(v) for k,v in row.items()} for row in csv.DictReader(f)]


def main(out):
    out.mkdir(parents=True,exist_ok=True)
    results=ROOT/'results'
    from loop_mechanism_analyze import plot as mechanisms
    source=results/'loopfill-mechanisms-20261109/existing'
    target=out/'mechanisms'
    target.mkdir(exist_ok=True)
    mechanisms(target,records(source/'load-budget.csv'),records(source/'readiness.csv'),
               records(source/'readiness-cdf.csv'),records(source/'ablation-efficiency.csv'))
    from loop_score_headroom_audit import plot as headroom
    source=results/'loopfill-score-headroom-t0-20261109'
    target=out/'headroom'
    target.mkdir(exist_ok=True)
    shutil.copy2(source/'protocol.json',target/'protocol.json')
    headroom(target,json.loads((source/'analysis.json').read_text()))
    from loop_tree_sampling_report import plot as sampling
    source=results/'loopfill-tree-sampling-b16-seed0-20261109'
    target=out/'tree-sampling'
    target.mkdir(exist_ok=True)
    sampling(target,json.loads((source/'analysis.json').read_text()))
    from loop_token_sources import plot as sources
    source=results/'loopfill-token-sources-20261109'
    target=out/'token-sources'
    target.mkdir(exist_ok=True)
    sources(records(source/'overall.csv'),records(source/'by-task.csv'),target)
    print(f'Regenerated paper figures from committed numerical data: {out}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    main(args.out.resolve())
