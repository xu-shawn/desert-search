#!/usr/bin/env python3
"""Render bin/biomemap output. usage: bin/biomemap SEED X Z [HALF STEP] > map.bin && scripts/plot_map.py map.bin out.png [HALF]"""
import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
raw = open(sys.argv[1], 'rb').read(); n = int.from_bytes(raw[:4], 'little')
a = np.frombuffer(raw[4:], dtype=np.uint8).reshape(n, n)
half = float(sys.argv[3]) / 1000 if len(sys.argv) > 3 else 16.0
cats = [('measured desert', '#eda100', [255]), ('other desert', '#f3dca8', [2]), ('badlands', '#c2410c', [37, 38, 165]),
        ('ocean', '#2a78d6', [0, 10, 24, 44, 45, 46, 47, 48, 49, 50]), ('river', '#8fbbef', [7, 11]),
        ('savanna, jungle, mangrove', '#1baf7a', [35, 36, 163, 21, 23, 168, 184]), ('other land', '#d4d2cb', None)]
rgb = lambda c: [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]
img = np.tile(rgb(cats[-1][1]), (n, n, 1))
for _, col, ids in cats[:-1]: img[np.isin(a, ids)] = rgb(col)
fig, ax = plt.subplots(figsize=(8.2, 7.2))
ax.imshow(img, extent=(-half, half, half, -half), interpolation='nearest')
ax.set_xlabel('X offset, thousand blocks'); ax.set_ylabel('Z offset, thousand blocks')
ax.legend(handles=[Patch(color=c, label=l) for l, c, _ in cats], loc='upper left', bbox_to_anchor=(1.01, 1))
fig.savefig(sys.argv[2], dpi=150, bbox_inches='tight'); print('wrote', sys.argv[2])
