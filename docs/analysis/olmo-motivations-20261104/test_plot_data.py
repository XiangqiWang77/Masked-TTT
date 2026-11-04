import csv
from pathlib import Path

import pytest

import plot

ROOT=Path(__file__).resolve().parent


def test_original_olmo_totals_and_length_groups():
    rows=plot.load();groups=plot.length_groups(rows)
    assert len(rows)==1000
    assert [sum(row[key] for row in rows) for key in plot.METHODS]==[361,412,446]
    assert [len(groups[size]) for size in plot.SIZES]==[250,250,250,250]
    assert sum(row['working_document']=='compacted' for row in groups[65536])==162


def test_coverage_excludes_compacted_documents():
    groups=plot.coverage_groups(plot.load())
    assert [len(groups[key]) for key in ('0%','1–99%','100%')]==[148,166,524]
    assert all(row['working_document']=='complete' for rows in groups.values() for row in rows)
    assert sum(row['cartridge'] for row in groups['100%'])==290
    assert sum(row['teacher'] for row in groups['100%'])==268


def test_summary_is_derived_from_the_saved_binary_rows():
    rows=plot.load();length=plot.length_groups(rows);coverage=plot.coverage_groups(rows)
    table=list(csv.DictReader((ROOT/'plot_summary.csv').open(newline='')))
    assert len(table)==21
    for item in table:
        if item['figure']=='context_scale':group=length[int(item['group'][:-1])*1024]
        elif item['figure']=='evidence_coverage':group=coverage[item['group']]
        else:raise AssertionError('Unexpected figure')
        method=item['method']
        assert int(item['n'])==len(group)
        assert int(item['correct'])==sum(row[method] for row in group)
        assert float(item['accuracy_percent'])==pytest.approx(plot.rate(group,method))
