#!/usr/bin/env python3
"""Rank vast.ai GPU offers by estimated searcher throughput per dollar.

Model (measured on the 2026-09-30 fleet): the pipeline is GPU-bound at ~9.6 gated seeds/s per SM·GHz
(shared-memory-bound kernels) and needs ~0.035 CPU core-seconds per seed; old Xeon hosts (E5 v3/v4)
ran ~6x slower per seed and are excluded unless --any-cpu.
usage: offers.py [--top 20] [--min-cores 16] [--any-cpu]"""
import argparse, json, subprocess
SM = {  # name: (SMs, boost GHz)
    'RTX 5090': (170, 2.41), 'RTX 5080': (84, 2.62), 'RTX 5070 Ti': (70, 2.45), 'RTX 5070': (48, 2.51), 'RTX 5060 Ti': (36, 2.57),
    'RTX 4090': (128, 2.52), 'RTX 4080S': (80, 2.55), 'RTX 4080': (76, 2.5), 'RTX 4070S Ti': (66, 2.61), 'RTX 4070 Ti': (60, 2.61),
    'RTX 4070S': (56, 2.48), 'RTX 4070': (46, 2.48), 'RTX 4060 Ti': (34, 2.54), 'RTX 4060': (24, 2.46),
    'RTX 3090 Ti': (84, 1.86), 'RTX 3090': (82, 1.7), 'RTX 3080 Ti': (80, 1.67), 'RTX 3080': (68, 1.71), 'RTX 3070 Ti': (48, 1.77),
    'RTX 3070': (46, 1.73), 'RTX 3060 Ti': (38, 1.67), 'RTX 3060': (28, 1.78), 'RTX 2080 Ti': (68, 1.545), 'RTX 2080S': (48, 1.815),
    'RTX A4000': (48, 1.56), 'RTX A4500': (56, 1.65), 'RTX A5000': (64, 1.695), 'RTX A6000': (84, 1.8), 'A40': (84, 1.74),
    'L4': (58, 2.04), 'L40S': (142, 2.52), 'L40': (142, 2.49), 'RTX 6000Ada': (142, 2.5), 'RTX 4000Ada': (48, 2.18),
    'A100 PCIE': (108, 1.41), 'A100 SXM4': (108, 1.41), 'H100 PCIE': (114, 1.76), 'H100 SXM': (132, 1.98), 'Tesla V100': (80, 1.53)}
ap = argparse.ArgumentParser(); ap.add_argument('--top', type=int, default=20); ap.add_argument('--min-cores', type=int, default=16)
ap.add_argument('--any-cpu', action='store_true'); a = ap.parse_args()
q = f'num_gpus>=1 cpu_cores_effective>={a.min_cores} reliability>0.97 disk_space>=30 rentable=true cuda_max_good>=12.4'
offers = json.loads(subprocess.run(['vastai', 'search', 'offers', q, '-o', 'dph', '--raw', '--limit', '3000'], capture_output=True, text=True, check=True).stdout)
rows = []
for x in offers:
    g, cpu = x['gpu_name'], x.get('cpu_name') or ''
    if g not in SM or (not a.any_cpu and not any(k in cpu for k in ('EPYC', 'Ryzen', 'Threadripper', 'Xeon® Gold', 'Xeon® Platinum'))): continue
    sm, clk = SM[g]; gpu = 9.6 * sm * clk * x['num_gpus']; cpu_cap = (x.get('cpu_cores_effective') or 0) / 0.035
    cap = min(gpu, cpu_cap); rows.append((cap / x['dph_total'], cap, x['dph_total'], x['num_gpus'], g, x.get('cpu_cores_effective'), cpu[:26], x['id'], x.get('geolocation')))
rows.sort(reverse=True)
print(f'{"seeds/s/$":>9} {"seeds/s":>7} {"$/h":>6}  GPU                cores  CPU                         offer      location')
for r in rows[:a.top]:
    print(f'{r[0]:9.0f} {r[1]:7.0f} {r[2]:6.3f}  {r[3]}x {r[4]:14s} {r[5]:5.0f}  {r[6]:26s}  {r[7]:<10} {r[8]}')
