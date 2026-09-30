#!/usr/bin/env python3
"""Fleet status: per pipeline, seeds done and current rate (from the last two progress lines), GPU load, best finds."""
import os, re, subprocess
D = os.path.dirname(os.path.abspath(__file__))
boxes = [l.split()[0] for l in open(f'{D}/boxes.txt') if l.strip() and not l.startswith('#')] if os.path.exists(f'{D}/boxes.txt') else []
REMOTE = r'''cd /root/desert-search/runs 2>/dev/null || exit 0
echo "GPU $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ',')"
for L in $(find . -maxdepth 1 -name '*g[0-9].log' -mmin -2); do echo "LOG $L"; grep scanned $L | tail -2; done
cat *.verified 2>/dev/null | grep -o 'Blocks^2 Size: [0-9]*' | awk '{print "BEST", $3}' | sort -k2 -n | tail -3'''
tot = 0; best = []
for b in boxes:
    try: out = subprocess.run([f'{D}/box.sh', b, REMOTE], capture_output=True, text=True, timeout=60).stdout
    except subprocess.TimeoutExpired: print(f'{b}: ssh timeout'); continue
    gpu, log, lines = '', None, {}
    for l in out.splitlines():
        if l.startswith('GPU '): gpu = l[4:]
        elif l.startswith('LOG '): log = l[4:].lstrip('./'); lines[log] = []
        elif l.startswith('BEST '): best.append((int(l.split()[1]), b))
        elif log: lines[log].append(l)
    if not lines: print(f'{b:10s} no active pipeline   gpu%[{gpu}]')
    for lg, ls in sorted(lines.items()):
        pts = [(int(m.group(1)), float(m.group(2))) for m in (re.match(r'(\d+) seeds.* ([\d.]+) passed/s', x) for x in ls) if m]
        (n1, r1), (n2, r2) = (pts + pts)[-2:] if pts else ((0, 0), (0, 0))
        rate = (n2 - n1) / (n2 / r2 - n1 / r1) if n2 > n1 else r2
        tot += rate
        print(f'{b:10s} {lg:14s} seeds {n2:>9}  {rate:6.0f}/s  gpu%[{gpu}]')
print(f'fleet ~{tot:.0f} gated seeds/s')
for v, b in sorted(best, reverse=True)[:5]: print(f'  verified {v:>12,} on {b}')
