import csv
from pathlib import Path

import pytest

import plot

ROOT=Path(__file__).resolve().parent


def test_context_scale_uses_the_same_source_questions_at_each_length():
    grouped=plot.scale_rows(plot.load())
    for model,expected in (('Qwen3-4B',103),('GPT-OSS-20B',44)):
        per_length=grouped[model]
        assert [len(per_length[length]) for length in plot.SIZES]==[expected]*4
        first={row['source_question'] for row in per_length[plot.SIZES[0]]}
        assert all({row['source_question'] for row in per_length[length]}==first for length in plot.SIZES)


def test_coverage_bins_are_disjoint_and_use_every_matched_coverage_record():
    rows=plot.load();grouped=plot.coverage_rows(rows)
    assert [len(grouped['Qwen3-4B'][group]) for group in ('0%','1–99%','100%')]==[129,168,495]
    assert [len(grouped['GPT-OSS-20B'][group]) for group in ('0%','1–99%','100%')]==[102,122,390]
    assert sum(len(group) for groups in grouped.values() for group in groups.values())==len(rows)


def test_exported_summary_agrees_with_saved_binary_observations():
    rows=plot.load();scale=plot.scale_rows(rows);coverage=plot.coverage_rows(rows)
    table=list(csv.DictReader((ROOT/'plot_summary.csv').open(newline='')))
    assert len(table)==42
    for item in table:
        model=item['model'];method=item['method'];group=item['group']
        if item['figure']=='context_scale':sample=scale[model][int(group[:-1])*1024]
        elif item['figure']=='evidence_coverage':sample=coverage[model][group]
        else:raise AssertionError('Unknown figure')
        assert int(item['n'])==len(sample)
        assert int(item['correct'])==sum(row[method] for row in sample)
        assert float(item['accuracy_percent'])==pytest.approx(plot.rate(sample,method))
