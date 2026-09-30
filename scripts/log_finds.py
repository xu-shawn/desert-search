#!/usr/bin/env python3
"""Append MC-Measure-verified desert finds (>= MIN M blocks^2) from *.verified files to results/desert_finds.jsonl.
usage: log_finds.py BOXLABEL FILE... [--min 75]"""
import json, os, re, sys
args = sys.argv[1:]; mn = 75e6
if '--min' in args: i = args.index('--min'); mn = float(args[i + 1]) * 1e6; del args[i:i + 2]
box, files = args[0], args[1:]
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'desert_finds.jsonl')
have = {json.loads(l)['seed'] for l in open(path) if l.strip()}
with open(path, 'a') as out:
    for f in files:
        for l in open(f):
            m = re.search(r'^(\S+) (\S+) CANDIDATE step4=([\d.]+)M seed=(-?\d+) x=(-?\d+) z=(-?\d+).*1:4 Size: (\d+) \| Blocks\^2 Size: (\d+)', l)
            if not m: continue
            ts, run, s4, seed, x, z, cells, b2 = m.groups()
            if seed in have or int(b2) < mn: continue
            have.add(seed)
            out.write(json.dumps({"date": ts[:10], "biome": "Desert", "seed": seed, "x": int(x), "z": int(z), "y": 256, "cells": int(cells),
                                  "blocks2": int(b2), "verified_by": [f"MC-Measure ({box})"], "note": f"{run}, step4 {s4}M"}) + '\n')
            print(f'{int(b2):>11,} {seed} x={x} z={z} {run}')
