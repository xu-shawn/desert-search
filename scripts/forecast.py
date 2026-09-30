#!/usr/bin/env python3
"""Forecast from collected fleet archives (deploy/collect.sh -> runs/fleet/*.tgz).

Uses every MC-Measure-verified desert and the number of gated seeds searched. Verification covers every site
whose step-4 flood reaches 80M (step 4 reads up to ~5% under exact), so the exact tail is complete above ~85M.
It fits an exponential tail P(per-seed best >= x) = (n/N) * exp(-(x - U)/beta) above U by maximum likelihood.
usage: forecast.py [ARCHIVES...] [--u 85] [--target 100 105] [--rate SEEDS_PER_S] [--plot out.png]"""
import argparse, glob, math, os, re, tarfile

ap = argparse.ArgumentParser()
ap.add_argument('archives', nargs='*')
ap.add_argument('--u', type=float, default=85.0)
ap.add_argument('--target', type=float, nargs='+', default=[100.0, 105.0])
ap.add_argument('--rate', type=float, default=0.0, help='fleet gated seeds/s, to print hours')
ap.add_argument('--plot')
a = ap.parse_args()
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
files = a.archives or sorted(glob.glob(os.path.join(root, 'runs', 'fleet', '*.tgz')))

seeds, best = 0, {}
for path in files:
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            f = tf.extractfile(m) if m.isfile() else None
            if f is None: continue
            name, text = os.path.basename(m.name), f.read().decode(errors='replace')
            if name == 'seed_counts.txt':
                seeds += sum(int(l.split()[1]) for l in text.splitlines() if len(l.split()) == 2)
            elif name.endswith('.verified'):
                for l in text.splitlines():
                    s = re.search(r'seed=(-?\d+)', l); v = re.search(r'Blocks\^2 Size: (\d+)', l)
                    if s and v: best[s.group(1)] = max(best.get(s.group(1), 0), int(v.group(1)) / 1e6)
vals = sorted(best.values(), reverse=True)
tail = [v - a.u for v in vals if v >= a.u]
n = len(tail)
print(f'{len(files)} archives, {seeds:,} gated seeds, {len(vals)} verified deserts, best {vals[0] if vals else 0:.2f}M')
if n < 5 or not seeds:
    raise SystemExit(f'only {n} verified deserts >= {a.u}M: not enough for a fit')
beta = sum(tail) / n; se = beta / math.sqrt(n)
print(f'tail above {a.u}M: n = {n}, e-fold beta = {beta:.2f} +- {se:.2f}M')
for t in a.target:
    per = lambda b: seeds / (n * math.exp(-(t - a.u) / b))
    c, lo, hi = per(beta), per(beta + 2 * se), per(max(beta - 2 * se, 0.3))
    line = f'  >= {t:g}M: {c:.3g} seeds per find (2 sigma {lo:.2g} .. {hi:.2g})'
    if a.rate: line += f' = {c / a.rate / 3600:.1f} h at {a.rate:g}/s ({lo / a.rate / 3600:.1f} .. {hi / a.rate / 3600:.0f} h)'
    print(line)
if a.plot:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    xs = [v for v in vals if v >= a.u - 10]
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor='#fcfcfb')
    ax.step(xs, [(i + 1) / seeds for i in range(len(xs))], where='post', color='#2a78d6', lw=2, label='verified deserts')
    g = [a.u + 0.1 * i for i in range(int((max(a.target) + 3 - a.u) * 10))]
    ax.plot(g, [n / seeds * math.exp(-(x - a.u) / beta) for x in g], color='#eb6834', lw=2, label=f'exponential fit, beta={beta:.2f}M')
    ax.set_yscale('log'); ax.set_xlabel('desert size, M blocks²'); ax.set_ylabel('finds per gated seed (>= size)')
    ax.grid(True, color='#e4e3df'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(a.plot, dpi=150)
    print('wrote', a.plot)
