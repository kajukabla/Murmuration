#!/usr/bin/env python3
"""Validate all results files and dashboard data routing."""
import csv, os, json

DIR = os.path.dirname(os.path.abspath(__file__))
FILES = {
    'classic_perf': 'results_classic_perf.tsv',
    'classic_compute': 'results_classic_compute.tsv',
    'classic_quality': 'results_classic_quality.tsv',
    'topo_perf': 'results_topo_perf.tsv',
    'topo_compute': 'results_topo_compute.tsv',
    'topo_quality': 'results_topo_quality.tsv',
}

ok = True
for key, fname in FILES.items():
    path = os.path.join(DIR, fname)
    if not os.path.exists(path):
        print(f"MISSING: {fname}")
        ok = False
        continue
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
    kept = [r for r in rows if r.get('result') == 'kept']
    vals = []
    for r in kept:
        try:
            v = r.get('max_boids', '0')
            vals.append(float(v) if '.' in v else int(v))
        except:
            pass
    best = max(vals) if vals else 0
    is_score = 'quality' in key
    fmt = f"{best:.4f}" if is_score else f"{best:,}"
    status = "OK" if len(rows) > 0 or 'quality' not in key else "EMPTY"
    print(f"  {key:20s}: {len(rows):4d} rows, {len(kept):3d} kept, best={fmt:>12s}  [{status}]")

# Check schedule_config.json references correct files
cfg_path = os.path.join(DIR, 'schedule_config.json')
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    print(f"\nSchedule: {len(cfg['phases'])} phases, loop={cfg.get('loop')}")
    for p in cfg['phases']:
        rf = p.get('results_file', '?')
        exists = os.path.exists(os.path.join(DIR, rf))
        print(f"  {p['name']:20s} -> {rf:30s} {'OK' if exists else 'MISSING!'}")
else:
    print("\nNo schedule_config.json found")

if ok:
    print("\nAll validation passed.")
else:
    print("\nWARNING: Some issues found.")
