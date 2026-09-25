#!/usr/bin/env python3
"""Materialize CSV files alongside compressed results for the original analyzers."""

import argparse
import gzip
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage', type=Path, help='results/v49, results/v50 or results/v51')
args = parser.parse_args()
paths = sorted(args.stage.glob('*.csv.gz'))
if not paths:
    raise SystemExit(f'no compressed root CSVs found: {args.stage}')
for src in paths:
    dest = src.with_suffix('')
    if dest.exists():
        raise SystemExit(f'refusing to overwrite existing file: {dest}')
    with gzip.open(src, 'rb') as inp, dest.open('xb') as out:
        shutil.copyfileobj(inp, out)
    print(dest)
