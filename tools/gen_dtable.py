#!/usr/bin/env python3
"""Turn bin/dfunc output into src/desert_dtable.hpp: P(desert or bridgeable river | hot point, low-octave C/E).
usage: bin/dfunc SEEDS POINTS THREADS > dfunc.txt && python3 tools/gen_dtable.py dfunc.txt > src/desert_dtable.hpp
Rows of dfunc.txt: set ci ei total desert passable (set 1 = C octaves 0-2, E octaves 0-1). Cells with < 20 samples -> 0."""
import sys
N = 40
tot = [[0] * N for _ in range(N)]; ok = [[0] * N for _ in range(N)]
for line in open(sys.argv[1]):
    s, i, j, t, d, p = map(int, line.split())
    if s == 1: tot[i][j] += t; ok[i][j] += p
npts = sum(map(sum, tot))
print(f'// P(desert or bridgeable river | T>=0.55, C octaves 0-2, E octaves 0-1), measured on {npts / 1e6:.1f}M hot points')
print('// (tools/dfunc.cpp, tools/gen_dtable.py). Bins of 0.05: C_lo in [-0.8,1.2) (rows), E_lo in [-1,1) (cols); <20 samples -> 0.')
print('#pragma once\nnamespace standalone {\ninline constexpr float desert_pass_prob[40][40] = {')
for i in range(N):
    print('{' + ','.join(f'{(ok[i][j] / tot[i][j] if tot[i][j] >= 20 else 0):.3f}f' for j in range(N)) + '},')
print('};\n}')
