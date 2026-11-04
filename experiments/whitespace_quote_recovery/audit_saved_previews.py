#!/usr/bin/env python3
"""Replay saved GPT practice previews with source-whitespace quote recovery."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.truncated_study_json.audit_saved_previews import collect
from experiments.whitespace_quote_recovery.parser import parse_with_quote_recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.workspace.resolve(), candidate_parser=parse_with_quote_recovery)
    result["candidate"] = "whitespace_quote_recovery"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
