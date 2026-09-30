# desert-search

Whole-world GPU search for giant **Deserts** in Minecraft Java 26.3 (default world type), measured the
way the Large Biomes spreadsheet does: MC-Measure's 4-connected area on the 1:4 biome grid at Y=256,
rivers bridged, times 16. An 8-box vast.ai fleet (~9,000 gated seeds/s, ~$1.26/h) running this code
found the first known desert over **100,000,000 blocks²**. The previous record was 70,927,696.

How the search got there, step by step, with the evidence behind each idea: [docs/PROGRESSION.pdf](docs/PROGRESSION.pdf) (source: [PROGRESSION.md](docs/PROGRESSION.md)).

## Results

Verified with `bin/mcmeasure` (MC-Measure) at Y=256; the 100M find also matches the standalone flood
cell for cell (`make check`).

| Blocks² | Seed | X | Z |
|---:|---:|---:|---:|
| **100,674,560** | -8125596009223775875 | -22910974 | 2295170 |
| 96,296,976 | -2302310700424482066 | 15072770 | 20790914 |
| 94,382,208 | 4700091708396341869 | 8167682 | 29811202 |
| 94,266,208 | 4734564070272446950 | -3356414 | -6515454 |
| 93,248,992 | 5070202994980095446 | 12667010 | 8442882 |
| 93,184,944 | 2716081104032222096 | 3236098 | 26627970 |

All 222 verified deserts ≥ 75M (15 of them ≥ 90M) are in [`results/desert_finds.jsonl`](results/desert_finds.jsonl).

## Quick start

```sh
git clone --recursive <this repo> && cd desert-search
make              # CPU tools: bin/{dsearch_desert,mcmeasure,validate,dflood,dfunc}
make check        # 1M points vs Cubiomes at Y=256; old record and the 100M find measured two ways
make gpu          # needs the CUDA toolkit: bin/dsearch_desert_gpu, bin/tgate
bin/mcmeasure SEED X Z                                     # exact area of the biome at block (X, 256, Z)
bin/biomemap SEED X Z > m.bin && scripts/plot_map.py m.bin map.png   # map of the find and what bounds it
bin/dsearch_desert 0 3000000 local $(nproc)                # CPU-only search of seed indices [0, 3e6)
bin/dsearch_desert_gpu 0 0 gpu 16 gate=1 gpu=1 gscan=1     # GPU search, unbounded
```

`dsearch_desert START COUNT PREFIX THREADS [key=value ...]` searches seeds `splitmix64(START + k)` and writes
`PREFIX.csv` (sites with proxy ≥ 50M or an exact flood), `PREFIX.seeds.csv`, `PREFIX.json` and a progress log
on stderr. `CANDIDATE` lines (step-4 flood ≥ 80M) should be measured with `bin/mcmeasure`; `deploy/gpipe.sh`
does that automatically. The defaults are the production settings; see `struct Params` in `src/dsearch_desert.cpp`.
Validation switches: `gpucheck=N` (GPU proxy vs CPU), `gscancheck=N` (GPU tables and scan vs CPU, bit-exact),
`bin/tgate START COUNT check=1` (GPU gate vs CPU).

## Running a fleet on vast.ai

Needs the `vastai` CLI with an API key and an SSH key registered on vast.ai. Every script only touches
instances whose label starts with `$FLEET_PREFIX` (default `desert-`), so other instances on the account are safe.

```sh
export VAST_SSH_KEY=~/.ssh/vastai                  # private key registered with vast.ai
deploy/offers.py                                   # offers ranked by estimated seeds/s per dollar
deploy/rent.sh OFFER_ID v100a OFFER_ID 4080s       # rent (label desert-NAME), wait for SSH, add to deploy/boxes.txt
deploy/bringup.sh v100a run1 10000000000000        # upload repo, `make gpu` on the box, one pipeline per GPU
deploy/bringup.sh 4080s run2 20000000000000        # give every box its own seed range (1e13 apart)
deploy/status.py                                   # per-pipeline rate, GPU load, best verified finds
deploy/watch.sh 100                                # returns when a verified >= 100M (or a >= 99M candidate) appears
nohup deploy/credit_watchdog.sh 8 &                # stop the fleet if account credit < $8
deploy/collect.sh && scripts/forecast.py --rate 9000   # pull compact results; tail fit and time to target
deploy/teardown.sh --yes                           # destroy every desert-* instance
scripts/log_finds.py BOX runs/fleet/*.verified     # append verified finds to results/desert_finds.jsonl
```

On each box, `deploy/prun.sh` starts one `deploy/gpipe.sh` per GPU. It reads the container's cgroup CPU
quota (often half of `nproc`) and splits threads across GPUs. Each pipeline measures every `CANDIDATE` with
MC-Measure, one line per candidate, into `runs/RUN.verified`. If a box has no CUDA toolkit, build portable
fat binaries once with `make bundle` (sm_70–sm_90 plus PTX) and ship `dist.tgz`.

What to rent: the kernels are shared-memory bound, so throughput tracks SM count × clock
(~9.6 seeds/s per SM·GHz). Each seed also costs ~0.035 CPU core-seconds. Measured on the record fleet:

| GPU | CPU quota | gated seeds/s | $/h |
|---|---|---:|---:|
| RTX 4080 SUPER | Threadripper, 61 cores | 2,200 | 0.277 |
| Tesla V100 | EPYC, 46–61 cores | 1,050–1,480 | 0.16–0.21 |
| RTX 4060 Ti | EPYC 7502, 30 cores | 1,000 | 0.122 |
| RTX 2080 Ti / 3070 Ti | Ryzen / Threadripper, 30–61 cores | 840–850 | 0.115–0.118 |
| RTX A4000 | EPYC, 30 cores | 760 | 0.134 |

Avoid old dual-Xeon (E5 v3/v4) hosts. They took ~6× the CPU time per seed and left the GPUs idle.

## How it works

**What wins the lookup at Y=256.** Depth is always negative there, so every surface biome pays the same
depth penalty. The winner is simply the depth-0 surface box containing (T, H, C, E, W). `src/surface.hpp`
indexes those boxes on the grid of all box boundaries, with a tree fallback on cut lines (0 errors on 1M points).
Desert needs T ≥ 0.55 and C ≥ −0.19 (ocean). E must sit in a middle band, roughly 0.05–0.55 inland, or
badlands (below) or mangrove/other hot biomes (above) take over. Rivers (|W| < 0.05) are crossed but not
counted when the biome without the river would be desert, as in MC-Measure.

**Pipeline** (`src/dsearch_desert.cpp`, GPU parts in `src/gpu/`), per seed:

1. **Gate** (GPU, `gate.cuh`). First the temperature octave −10 y-phase: both sides' fractional y-offsets
   near 0.5 (RNG only, phase ≤ 0.06). Then an anti-aliased 256² amplitude pre-gate: the four half-lattice
   parities are averaged per cell and blurred, with 99.5% recall. Finally the full 512² smoothed octave-0
   temperature amplitude, max(A) + max(B) ≥ 0.70. About 1 in 13,000 seeds passes.
2. **Tables and world scan** (GPU). Octave-0 T, C and E period tables for both noise sides. Side A of every
   noise repeats every 262,144 quarts; side B (x·337/331) does not. So each A cell meets ~57² B phases across
   the ±30M world. Score = T + 0.25·C + 0.41·E, keeping the best candidate per 5,000-quart bucket.
   Bit-exact with the CPU path.
3. **Hot zones** (CPU). Flood T ≥ 0.55 around each site, at step 256 then 128. Only zones ≥ 180M blocks²
   continue. No desert fills more than ~41% of its hot zone (so 100M needs ≳ 240M of T ≥ 0.55), and hot-zone area is the strongest
   predictor of size.
4. **Proxy** (GPU). A table of P(desert | low-octave C, E) is evaluated on every step-32 cell of each big hot
   zone's bounding box. The CPU labels connected components and ranks them.
5. **Exact** (CPU). The top two components with proxy ≥ 75M get bridged desert floods: step 8, then step 4
   from interior points. Step 4 ≥ 80M prints a `CANDIDATE`, which MC-Measure verifies.

One process per GPU. The gate runs on a low-priority stream; the proxy, tables and scan run on high-priority
streams. Requests from all CPU workers are batched into single kernel launches.

**Exactness.** `src/standalone.hpp` is a dependency-free re-implementation of 26.3 climate sampling and the
biome tree. It is bit-exact with Cubiomes (`bin/validate`; needs `-ffp-contract=off`). The standalone
step-1 flood (`bin/dflood SEED X Z 1`) reproduces MC-Measure's cell count on both the old record and the
100M find. Everything upstream of the exact floods is a heuristic ranking. Its recall is measured, not assumed:
the pre-gate calibration, `hot_min` against the ≥ 80M finds, and proxy-gated exact floods.

**Forecasting.** `scripts/forecast.py` fits an exponential tail to verified sizes per gated seed. On the
record fleet's 15.5M gated seeds, the e-fold above 85M is ~3.0M, and 100M needs ~4.6×10⁷ gated seeds per find
(2σ: 1.5×10⁷ to 3.3×10⁸).

## TODO

- [ ] **Gate cost** (~65% of GPU time). Skip side B when side A's pre-gate maximum is already too low; try a
      coarser anti-aliased pre-pre-gate.
- [ ] **Hot-zone floods on GPU** (~40% of remaining CPU time). This would let 4090-class cards run on hosts
      with fewer cores.
- [ ] **Multi-GPU per process**, instead of one process per GPU sharing the CPU quota.
- [ ] **Better site/zone score.** Learn from `runs/*.csv` which hot zones fill (regional mean E and C drive
      the badlands and ocean losses).
- [ ] **Other land biomes** on the spreadsheet: swap the predicate, the proxy table (`tools/dfunc.cpp`) and the
      hot-zone variable.
- [ ] Large Biomes world type.

## Layout

```
src/          standalone sampler, surface grid, floods, proxy table, searcher, validation
src/gpu/      gate.cuh (seed gate), gproxy.cu (proxy, tables, scan engine), tgate.cu (standalone gate tool)
tools/        dflood (exact standalone flood), biomemap (biome map of a find), dfunc + gen_dtable.py (proxy table)
scripts/      forecast.py (tail fit), log_finds.py, plot_map.py
deploy/       offers, rent, bringup, prun/gpipe (on box), status, watch, collect, teardown, credit watchdog
results/      desert_finds.jsonl
docs/         PROGRESSION.pdf/.md (build_pdf.sh: pandoc + chromium) and figures
MC-Measure/   submodule (ImAchievers/MC-Measure @ a58650e): exact measurement + Cubiomes copy
```

## License

MIT. Third-party code and data are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
