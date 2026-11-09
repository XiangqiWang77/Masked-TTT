Recreate the released figures without private source files:

```bash
python -m pip install numpy matplotlib
python code/plot_matched_synthetic_comparison.py --statistics comparison.json --output recreated
```

`comparison.json` contains group means and task-stratified source-episode bootstrap intervals. `tables/per_case_metrics.csv` contains all 1,000 case-level diagnostics and their source cluster identifiers. `report_answer_guided_remaining.py` archives the original source-position analysis; it requires the source annotation/data environment.

Recreate the styled tables and PDF:

```bash
python code/style_synthetic_paper_tables.py --statistics comparison.json --output recreated
```

The LaTeX sources use the supplied style (array/booktabs/graphicx/xcolor/ulem). 