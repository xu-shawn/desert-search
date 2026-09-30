// Whole-world desert searcher (Minecraft 26.3, surface Y=256, MC-Measure river bridging).
//   dsearch_desert START COUNT OUT_PREFIX [THREADS] [key=value ...]     seeds = splitmix64(START + k)
// Defaults are the production settings (see struct Params); the GPU build (make gpu) adds gate=1 gpu=1 gscan=1.
//  S0  seed gate: temperature octave-0 (octave -10) y-phase of both noise sides (RNG only), then the
//      smoothed octave-0 temperature amplitude max(TA) + max(TB) (GPU: gate.cuh, CPU: lowfield.hpp).
//  S1  tables: octave-0 T (lattice 1024 quarts) and octave-0 C, E (lattice 512 quarts) for each
//      side over its 256-lattice period, Gaussian-smoothed.
//  S2  world scan: every side-A table repeats within 262144 quarts (T's period; C/E divide it), so
//      each A cell meets ~57^2 B phases across the world. Score = T + wc*C + we*E; best per NMS bucket.
//  S3  hot zones: flood T >= 0.55 (step 256, then 128) around each site; zones >= hot_min get a
//      low-octave desert-probability proxy flood (step 32, GPU or CPU) whose components are ranked.
//  S4  exact: the top components get step-8 then step-4 bridged desert floods from interior points;
//      step-4 >= verify prints a CANDIDATE line for MC-Measure (deploy/gpipe.sh verifies automatically).
#include "dense_flood.hpp"
#include "lowfield.hpp"
#include "surface.hpp"
#include "desert_proxy.hpp"
#ifdef DRIP_GPU
#include "gpu/gproxy.h"
#endif
#include <cstring>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <cstdlib>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace standalone;
using Clock = std::chrono::steady_clock;

static volatile sig_atomic_t interrupted = 0;
static void on_signal(int) { interrupted = 1; }

constexpr double BS = 337.0 / 331.0;
constexpr int WORLD = 7499996;
constexpr int PERIOD = 262144;             // side-A period of temperature octave -10, in quarts
constexpr int NT = 60;                     // translations per axis covering the world
constexpr int KMIN = -30;
constexpr int CELL = 512;                  // quarts per T-table cell (0.5 T lattice units)

struct Params {
    double max_phase = 0.10;   // sum over sides of |fy-.5| for temperature octave 0 (CPU gate)
    double amp = 0.70;         // min max(TA)+max(TB) (smoothed octave-0 temperature) to scan (CPU gate)
    double amp_margin = 0.01;  // cheap sub-grid estimate must reach amp - margin before full tables
    double sigma = 2.0;        // T blur in T-table cells (512 quarts)
    double sigmac = 2.0;       // C/E blur in their table cells (256 quarts)
    double wc = 0.25, we = 0.41; // score weights for smoothed C0 and E0 (logistic fit, AUC 0.87)
    double t = 0.55;           // score threshold for candidates
    double ta = 0.25;          // side-A T prune
    int max_sites = 300;       // candidate sites per seed
    double pcut = 0.35;        // proxy passability cut
    double hot_min = 180;      // >0: hot-zone mode; only zones with T>=0.55 area >= hot_min (M blocks^2) get proxy floods
    int multi = 1;             // proxy starts per site: 1 = best point only, else grid of (2*multi+1)^2 starts at 512-quart spacing
    int exact_n = 2;           // exact floods per seed (best by proxy)
    double exact_min = 75;     // ... and only if the proxy size is at least this (M blocks^2)
    double hot_pre = 0.8;      // >0: step-256 T flood first; step-128 flood only if that area >= hot_pre * hot_min
    double csv_min = 50;       // PREFIX.csv rows only for sites with proxy >= this or an exact flood
    int nms = 5000;            // quarts
    double refine_mb = 70;     // step-8 size that triggers a step-4 measurement
    double verify_mb = 80;     // step-4 size flagged as CANDIDATE (step 4 reads up to ~5% under exact)
    int study = 0;             // 1: no exact floods; write per-zone features to PREFIX.zones.csv
    int gpu = 0;               // 1: proxy floods of big hot zones on the GPU (build with -DDRIP_GPU)
    int gpucheck = 0;          // compare GPU proxy bitmaps with the CPU proxy for the first zones
    int gscan = 0;             // 1 (with gpu=1): period tables + translation scan on the GPU
    int gscancheck = 0;        // compare GPU tables/scan with the CPU for the first N seeds
    int gate = 0;              // 1 (with gpu=1): run the GPU seed gate in-process over [START, START+COUNT) (COUNT 0 = unbounded)
    double gate_phase = 0.06, gate_amp = 0.70, gate_pthr = 0.645;   // GPU gate: phase cut, amplitude, anti-aliased pre-gate
    int from_stdin = 0;        // 1: read pre-gated seeds ("seed [amp]" lines, e.g. from the GPU gate) instead of enumerating
};

static double phase_t(uint64_t seed, double *fy) {
    Random world(seed);
    uint64_t a = world.next(), b = world.next();
    Random c(a ^ configs[Temperature].a, b ^ configs[Temperature].b);
    double s = 0;
    for (int side = 0; side < 2; side++) {
        uint64_t fa = c.next(), fb = c.next();
        Random r(fa ^ octave_salt[0][0], fb ^ octave_salt[0][1]);   // octave -10
        r.next();
        double y = r.uniform() * 256.;
        fy[side] = y - std::floor(y);
        s += std::abs(fy[side] - .5);
    }
    return s;
}

// A cell i (i*CELL quarts within [0,PERIOD)) translated by t periods -> B table cell of an octave
// whose side-A lattice unit is `lat` quarts.
struct TranslationMap {
    int16_t ib[TN][NT];
    int8_t lo[TN], hi[TN];
    explicit TranslationMap(double lat) {
        for (int i = 0; i < TN; i++) {
            lo[i] = NT; hi[i] = -1;
            for (int t = 0; t < NT; t++) {
                long x = long(i) * CELL + long(t + KMIN) * PERIOD;
                bool ok = x >= -WORLD + 6000 && x <= WORLD - 6000;
                double u = x * BS / lat; u -= std::floor(u / 256.0) * 256.0;
                ib[i][t] = int16_t(int(u * (TN / 256.0) + 0.5) & (TN - 1));
                if (ok) { lo[i] = std::min<int>(lo[i], t); hi[i] = std::max<int>(hi[i], t); }
            }
        }
    }
};

struct Site { float s, t, c, e; int x, z; };

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "dsearch_desert START COUNT OUT_PREFIX [threads] [key=value...]\n");
        return 2;
    }
    uint64_t start = std::stoull(argv[1]), count = std::stoull(argv[2]);
    std::string prefix = argv[3];
    int threads = argc > 4 ? std::stoi(argv[4]) : 11;
    Params P;
    for (int i = 5; i < argc; i++) {
        std::string kv = argv[i]; auto eq = kv.find('=');
        std::string k = kv.substr(0, eq); double v = std::stod(kv.substr(eq + 1));
        if (k == "gpu") P.gpu = int(v); else if (k == "gpucheck") P.gpucheck = int(v); else if (k == "stdin") P.from_stdin = int(v); else if (k == "study") P.study = int(v); else if (k == "phase") P.max_phase = v; else if (k == "amp") P.amp = v; else if (k == "amp_margin") P.amp_margin = v; else if (k == "sigma") P.sigma = v; else if (k == "sigmac") P.sigmac = v;
        else if (k == "wc") P.wc = v; else if (k == "we") P.we = v; else if (k == "t") P.t = v; else if (k == "ta") P.ta = v;
        else if (k == "sites") P.max_sites = int(v); else if (k == "nms") P.nms = int(v);
        else if (k == "pcut") P.pcut = v; else if (k == "multi") P.multi = int(v); else if (k == "hot_min") P.hot_min = v; else if (k == "exact_n") P.exact_n = int(v); else if (k == "exact_min") P.exact_min = v; else if (k == "csv_min") P.csv_min = v; else if (k == "hot_pre") P.hot_pre = v; else if (k == "gate") P.gate = int(v); else if (k == "gscan") P.gscan = int(v); else if (k == "gscancheck") P.gscancheck = int(v); else if (k == "gate_phase") P.gate_phase = v; else if (k == "gate_amp") P.gate_amp = v; else if (k == "gate_pthr") P.gate_pthr = v;
        else if (k == "refine") P.refine_mb = v; else if (k == "verify") P.verify_mb = v;
        else { fprintf(stderr, "unknown key %s\n", k.c_str()); return 2; }
    }
    if ((P.gscan || P.gscancheck) && !P.gpu) { fprintf(stderr, "gscan=1 needs gpu=1\n"); return 2; }
    if (P.gate) {
        if (!P.gpu) { fprintf(stderr, "gate=1 needs gpu=1\n"); return 2; }
        P.from_stdin = 1;                                  // workers take seeds from the gate's queue
    }
    FILE *out = fopen((prefix + ".csv").c_str(), "wx");
    FILE *sout = fopen((prefix + ".seeds.csv").c_str(), "wx");
    if (!out || !sout) { fprintf(stderr, "cannot create %s outputs (exist?)\n", prefix.c_str()); return 2; }
    fprintf(out, "seed,fyA,fyB,maxta,maxtb,rank,x,z,score,t0,c0,e0,proxy,s8,s4,edge,hot\n");
    fprintf(sout, "seed,fyA,fyB,maxta,maxtb,candidates,sites,best\n");
    FILE *zout = nullptr;
    if (P.study) {
        zout = fopen((prefix + ".zones.csv").c_str(), "wx");
        fprintf(zout, "seed,phT,amp,phE0,phE1,phC0,phC1,rank,x,z,t0,c0,e0,hot,proxy,f_ocean,f_badl,f_mang,f_good,mean_c,mean_e,sd_e\n");
    }
    signal(SIGINT, on_signal); signal(SIGTERM, on_signal);

    static TranslationMap TMT(1024.0), TMC(512.0);
#ifdef DRIP_GPU
    if (P.gpu) gproxy::init(0, float(P.pcut));
#else
    if (P.gpu) { fprintf(stderr, "gpu=1 needs a -DDRIP_GPU build\n"); return 2; }
#endif
    static SurfaceGrid G;
    std::atomic<uint64_t> next(0), passed(0), scanned(0), done(0), nsites(0), nhot(0), nbig(0);
    std::atomic<double> t_tab(0), t_scan(0), t_site(0), t_proxy(0), t_hot(0), t_gpu(0);
    std::mutex mu;
    std::mutex qmu; std::condition_variable qcv; std::deque<uint64_t> q; bool qdone = false;
    double best = 0;
    auto begin = Clock::now(), last = begin;
    Blur blur(P.sigma), blurc(P.sigmac);
#ifdef DRIP_GPU
    if (P.gscan || P.gscancheck)
        gproxy::scan_init(&TMT.ib[0][0], TMT.lo, TMT.hi, &TMC.ib[0][0], NT, blur.k.data(), blur.r, blurc.k.data(), blurc.r, CELL, PERIOD, KMIN);
    std::atomic<int> gscan_checks(P.gscancheck);
#endif

    auto worker = [&]() {
        std::vector<float> TA, TB, CA, CB, EA, EB, tmp;
        std::vector<Site> cand;
        std::unordered_map<uint64_t, uint32_t> bucket;
        std::unordered_set<uint64_t> covered, hotcovered;
#ifdef DRIP_GPU
        std::vector<gproxy::Zone> gzones; std::vector<std::vector<uint8_t>> gbits;
#else
        struct GZ { int x0, z0, nx, nz; }; std::vector<GZ> gzones;
#endif
        std::vector<std::pair<size_t, double>> gmeta;
        auto hotkey = [](int x, int z) { return (uint64_t(uint32_t((x + 8000000) >> 7)) << 32) | uint32_t((z + 8000000) >> 7); };
        // key of the step-32 lattice cell containing (x,z); floods from different starts use different
        // lattices, so snap to a global 32-quart grid
        auto cellkey = [](int x, int z) { return (uint64_t(uint32_t((x + 8000000) >> 5)) << 32) | uint32_t((z + 8000000) >> 5); };
        std::vector<uint8_t> state, state2; std::vector<int> stack, stack2;
        while (!interrupted) {
            uint64_t seed;
            double fy[2];
            if (P.from_stdin) {
                std::unique_lock<std::mutex> l(qmu);
                while (q.empty() && !qdone && !interrupted) qcv.wait_for(l, std::chrono::milliseconds(200));
                if (q.empty()) break;
                seed = q.front(); q.pop_front();
                qcv.notify_all();
                l.unlock();
                phase_t(seed, fy);
                done++; passed++;
            } else {
                uint64_t k = next++;
                if (k >= count) break;
                seed = mix(start + k);
                done++;
                if (phase_t(seed, fy) > P.max_phase) continue;
                passed++;
            }
            auto t0 = Clock::now();
            SurfaceSampler S(seed, G);
            const auto &T = S.s.get(Temperature), &C = S.s.get(Continentalness), &E = S.s.get(Erosion);
            auto table = [&](const NormalNoise &nn, int side, std::vector<float> &t, const Blur &b) {
                tabulate_octave(nn.octave[side][0], nn.factor * nn.octave[side][0].weight, t);
                b(t, tmp);
            };
            if (!P.from_stdin) {
                float est = amplitude_sub(T.octave[0][0], T.factor * T.octave[0][0].weight, TA, tmp)
                          + amplitude_sub(T.octave[1][0], T.factor * T.octave[1][0].weight, TA, tmp);
                if (est < P.amp - P.amp_margin) {
                    t_tab = t_tab + std::chrono::duration<double>(Clock::now() - t0).count();
                    continue;
                }
            }
            float maxta = 0, maxtb = 0;
            bool on_gpu = false;
            const float tt = P.t, ta = P.ta, wc = P.wc, we = P.we;
            auto bucket_key = [&](int x, int z) { return (uint64_t(uint32_t((x + 8000000) / P.nms)) << 32) | uint32_t((z + 8000000) / P.nms); };
#ifdef DRIP_GPU
            std::vector<gproxy::RawCand> raw;
            bool check = P.gscancheck && gscan_checks.fetch_sub(1) > 0;
            if ((P.gscan || check) && P.from_stdin) {
                gproxy::TabIn tin[6];
                const NormalNoise *nz[3] = {&T, &C, &E};
                for (int k = 0; k < 6; k++) {
                    const NormalNoise &nn = *nz[k / 2]; const Perlin &pp = nn.octave[k & 1][0];
                    std::memcpy(tin[k].perm, pp.permutation.data(), 256);
                    tab_setup(pp, nn.factor * pp.weight, tin[k].col, tin[k].i0, tin[k].k0, tin[k].T);
                }
                on_gpu = gproxy::scan(tin, {ta, tt, wc, we}, maxta, maxtb, raw);
                if (!on_gpu) fprintf(stderr, "gscan: candidate overflow for seed %lld, CPU fallback\n", (long long)seed);
            }
            if (check) on_gpu = false;                          // run the CPU path too and compare below
            const float gmaxta = maxta, gmaxtb = maxtb;
#endif
            if (!on_gpu) {
                table(T, 0, TA, blur); table(T, 1, TB, blur);
                maxta = *std::max_element(TA.begin(), TA.end()); maxtb = *std::max_element(TB.begin(), TB.end());
                if (!P.from_stdin && maxta + maxtb < P.amp) {
                    t_tab = t_tab + std::chrono::duration<double>(Clock::now() - t0).count();
                    std::lock_guard<std::mutex> g(mu);
                    fprintf(sout, "%lld,%.4f,%.4f,%.4f,%.4f,0,0,0\n", (long long)seed, fy[0], fy[1], maxta, maxtb);
                    continue;
                }
                table(C, 0, CA, blurc); table(C, 1, CB, blurc);
                table(E, 0, EA, blurc); table(E, 1, EB, blurc);
                cand.clear(); bucket.clear();
                for (int j = 0; j < TN; j++) for (int i = 0; i < TN; i++) {
                    float sa = TA[j * TN + i];
                    if (sa < ta) continue;
                    // C/E side-A cells at this quart position: x/256 mod 512 = 2i mod 512
                    int ca = ((2 * j) & (TN - 1)) * TN + ((2 * i) & (TN - 1));
                    float cA = CA[ca], eA = EA[ca];
                    for (int tz = TMT.lo[j]; tz <= TMT.hi[j]; tz++) {
                        const float *rT = &TB[TMT.ib[j][tz] * TN];
                        const float *rC = &CB[TMC.ib[j][tz] * TN], *rE = &EB[TMC.ib[j][tz] * TN];
                        for (int tx = TMT.lo[i]; tx <= TMT.hi[i]; tx++) {
                            float tv = sa + rT[TMT.ib[i][tx]];
                            if (tv < tt - 0.3f) continue;
                            int cb = TMC.ib[i][tx];
                            float cv = cA + rC[cb], ev = eA + rE[cb];
                            float sc = tv + wc * cv + we * ev;
                            if (sc < tt) continue;
                            int x = i * CELL + (tx + KMIN) * PERIOD, z = j * CELL + (tz + KMIN) * PERIOD;
                            uint64_t key = bucket_key(x, z);
                            auto [it, fresh] = bucket.try_emplace(key, uint32_t(cand.size()));
                            if (fresh) cand.push_back({sc, tv, cv, ev, x, z});
                            else if (sc > cand[it->second].s) cand[it->second] = {sc, tv, cv, ev, x, z};
                        }
                    }
                }
            }
            scanned++;
            auto t1 = Clock::now();
#ifdef DRIP_GPU
            if (on_gpu || check) {
                std::vector<Site> gc; std::unordered_map<uint64_t, uint32_t> gb;
                for (auto &r : raw) {
                    auto [it, fresh] = gb.try_emplace(bucket_key(r.x, r.z), uint32_t(gc.size()));
                    if (fresh) gc.push_back({r.s, r.t, r.c, r.e, r.x, r.z});
                    else if (r.s > gc[it->second].s) gc[it->second] = {r.s, r.t, r.c, r.e, r.x, r.z};
                }
                if (check) {
                    long miss = 0;
                    for (auto &c : cand) { auto it = gb.find(bucket_key(c.x, c.z)); if (it == gb.end() || gc[it->second].s != c.s) miss++; }
                    fprintf(stderr, "gscancheck seed %lld: maxta %.6f/%.6f maxtb %.6f/%.6f, buckets cpu %zu gpu %zu, mismatched %ld, raw %zu\n",
                            (long long)seed, maxta, gmaxta, maxtb, gmaxtb, cand.size(), gc.size(), miss, raw.size());
                } else cand.swap(gc);
            }
#endif
            auto t2 = Clock::now();
            std::sort(cand.begin(), cand.end(), [](auto &a, auto &b) { return a.s > b.s; });
            std::vector<Site> sites;
            for (auto &c : cand) {
                bool near = false;
                for (auto &q : sites) if (std::abs(q.x - c.x) < P.nms && std::abs(q.z - c.z) < P.nms) { near = true; break; }
                if (!near) sites.push_back(c);
                if ((int)sites.size() >= P.max_sites) break;
            }
            auto pred = [&](int x, int z) -> int { return S.desert_state(x, z); };
            DesertProxy PX(S.s, float(P.pcut));
            auto phase_of = [](const NormalNoise &N, int oct) { return std::abs(N.octave[0][oct].fy - .5) + std::abs(N.octave[1][oct].fy - .5); };
            double phE0 = phase_of(E, 0), phE1 = phase_of(E, 1), phC0 = phase_of(C, 0), phC1 = phase_of(C, 1);
            struct Eval { size_t rank; int x, z; double proxy, s8 = -1, s4 = 0; bool edge = false; double hot = 0;
                          std::vector<std::pair<int, int>> alt; };   // further interior starts (GPU path)
            std::vector<Eval> ev;
            std::vector<std::pair<int, int>> seen;              // proxy component signature (min x, min z)
            covered.clear();                                    // step-32 cells inside proxy components flooded so far
            gzones.clear(); gmeta.clear();
            hotcovered.clear();
            auto tp0 = Clock::now();
            for (size_t rank = 0; rank < sites.size(); rank++) {
                const auto &st = sites[rank];
                nsites++;
                std::vector<std::pair<int, int>> starts;
                double hot_area = 0;
                if (P.hot_min > 0) {
                    // hot zone of this site: flood T >= 0.55 at step 128 from the hottest nearby point
                    int hx = st.x, hz = st.z; double bt = -9;
                    for (int dz = -3; dz <= 3; dz++) for (int dx = -3; dx <= 3; dx++) {
                        double tv = T.sample(st.x + dx * 256, 0, st.z + dz * 256);
                        if (tv > bt) { bt = tv; hx = st.x + dx * 256; hz = st.z + dz * 256; }
                    }
                    if (bt < 0.55 || hotcovered.count(hotkey(hx, hz))) continue;
                    auto hotp = [&](int x, int z) -> int { return T.sample(x, 0, z) >= 0.55 ? 2 : 0; };
                    auto th0 = Clock::now();
                    if (P.hot_pre > 0) {        // coarse pre-flood: most hot zones are far below hot_min
                        auto h2 = dense_flood_bridged(hotp, hx, hz, 256, 32000 / 256, state2, stack2, false);
                        if (h2.cells * 256.0 * 256 * 16 / 1e6 < P.hot_pre * P.hot_min) {
                            const int half = 32000 / 256, n = 2 * half + 1;
                            int i0 = (h2.minx - hx) / 256 + half, i1 = (h2.maxx - hx) / 256 + half;
                            int j0 = (h2.minz - hz) / 256 + half, j1 = (h2.maxz - hz) / 256 + half;
                            for (int j = j0; j <= j1; j++) for (int i = i0; i <= i1; i++) {
                                if (state2[size_t(j) * n + i] != 3) continue;
                                int x = hx + (i - half) * 256, z = hz + (j - half) * 256;
                                for (int d = 0; d < 4; d++) hotcovered.insert(hotkey(x + (d & 1) * 128, z + (d >> 1) * 128));
                            }
                            nhot++;
                            t_hot = t_hot + std::chrono::duration<double>(Clock::now() - th0).count();
                            continue;
                        }
                    }
                    auto hzn = dense_flood_bridged(hotp, hx, hz, 128, 32000 / 128, state2, stack2, false);
                    t_hot = t_hot + std::chrono::duration<double>(Clock::now() - th0).count();
                    hot_area = hzn.cells * 128.0 * 128 * 16 / 1e6;
                    const int half = 32000 / 128, n = 2 * half + 1;
                    int i0 = (hzn.minx - hx) / 128 + half, i1 = (hzn.maxx - hx) / 128 + half;
                    int j0 = (hzn.minz - hz) / 128 + half, j1 = (hzn.maxz - hz) / 128 + half;
                    for (int j = j0; j <= j1; j++) for (int i = i0; i <= i1; i++) {
                        if (state2[size_t(j) * n + i] != 3) continue;
                        int x = hx + (i - half) * 128, z = hz + (j - half) * 128;
                        hotcovered.insert(hotkey(x, z));
                        if (hot_area >= P.hot_min && ((i - half) & 3) == 0 && ((j - half) & 3) == 0) starts.push_back({x, z});
                    }
                    nhot++;
                    if (hot_area < P.hot_min) continue;
                    nbig++;
                    if (P.gpu) {
                        int x0 = int(std::floor((hzn.minx - 128) / 32.0)) * 32, z0 = int(std::floor((hzn.minz - 128) / 32.0)) * 32;
                        gzones.push_back({x0, z0, (hzn.maxx + 128 - x0) / 32 + 1, (hzn.maxz + 128 - z0) / 32 + 1});
                        gmeta.push_back({rank, hot_area});
                        continue;
                    }
                } else if (P.multi <= 1) {
                    int bx = st.x, bz = st.z; float bp = -1;
                    for (int dz = -4; dz <= 4; dz++) for (int dx = -4; dx <= 4; dx++) {
                        float pr = PX.prob(st.x + dx * 64, st.z + dz * 64);
                        if (pr > bp) { bp = pr; bx = st.x + dx * 64; bz = st.z + dz * 64; }
                    }
                    if (bp >= P.pcut) starts.push_back({bx, bz});
                } else {
                    for (int dz = -P.multi; dz <= P.multi; dz++) for (int dx = -P.multi; dx <= P.multi; dx++)
                        starts.push_back({st.x + dx * 512, st.z + dz * 512});
                }
                for (auto [bx, bz] : starts) {
                    if (PX(bx, bz) != 2) continue;
                    if (covered.count(cellkey(bx, bz))) continue;   // start already inside a flooded component
                    auto a = dense_flood_bridged(PX, bx, bz, 32, 16000 / 32, state, stack, true);
                    {   // remember this component's cells (state 3 on the flood lattice centred at bx,bz); scan
                        // only its bounding box, not the whole window
                        const int half = 16000 / 32, n = 2 * half + 1;
                        int i0 = (a.minx - bx) / 32 + half, i1 = (a.maxx - bx) / 32 + half;
                        int j0 = (a.minz - bz) / 32 + half, j1 = (a.maxz - bz) / 32 + half;
                        for (int j = j0; j <= j1; j++) for (int i = i0; i <= i1; i++)
                            if (state[size_t(j) * n + i] == 3) covered.insert(cellkey(bx + (i - half) * 32, bz + (j - half) * 32));
                    }
                    std::pair<int, int> sig(a.minx, a.minz);
                    if (std::find(seen.begin(), seen.end(), sig) != seen.end()) continue;
                    seen.push_back(sig);
                    ev.push_back({rank, bx, bz, a.cells * 32.0 * 32 * 16 / 1e6});
                    ev.back().hot = hot_area;
                    if (P.study) {
                        // hot zone (T >= 0.55) at step 64 from the same start, with low-octave C/E census
                        double sc = 0, se = 0, se2 = 0; long nh = 0, no = 0, nb = 0, nm = 0, ng = 0;
                        auto hotstat = [&](int x, int z) -> int {
                            if (T.sample(x, 0, z) < 0.55) return 0;
                            double c = DesertProxy::octaves(C, x, z, 0, 3), e = DesertProxy::octaves(E, x, z, 0, 2);
                            nh++; sc += c; se += e; se2 += e * e;
                            no += c < -0.19; nb += e < 0.05; nm += e > 0.55;
                            int i = std::clamp(int(std::floor((c + 0.8) / 0.05)), 0, 39), j = std::clamp(int(std::floor((e + 1.0) / 0.05)), 0, 39);
                            ng += desert_pass_prob[i][j] >= P.pcut;
                            return 2;
                        };
                        auto hz = dense_flood_bridged(hotstat, bx, bz, 64, 24000 / 64, state2, stack2, false);
                        double me = se / std::max(1L, nh);
                        std::lock_guard<std::mutex> g(mu);
                        fprintf(zout, "%lld,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%zu,%d,%d,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                                (long long)seed, std::abs(fy[0] - .5) + std::abs(fy[1] - .5), maxta + maxtb, phE0, phE1, phC0, phC1, rank, bx * 4, bz * 4,
                                st.t, st.c, st.e, hz.cells * 64.0 * 64 * 16 / 1e6, a.cells * 32.0 * 32 * 16 / 1e6,
                                double(no) / std::max(1L, nh), double(nb) / std::max(1L, nh), double(nm) / std::max(1L, nh), double(ng) / std::max(1L, nh),
                                sc / std::max(1L, nh), me, std::sqrt(std::max(0.0, se2 / std::max(1L, nh) - me * me)));
                    }
                }
            }
#ifdef DRIP_GPU
            if (P.gpu && !gzones.empty()) {
                gproxy::SeedNoise sn;
                auto fill = [](const NormalNoise &N, int oct, int side, gproxy::Octave &o) {
                    const Perlin &p = N.octave[side][oct];
                    std::memcpy(o.perm, p.permutation.data(), 256);
                    o.k = int(std::lround(-std::log2(p.frequency))); o.side = side;
                    o.ox_i = int(std::floor(p.ox)); o.ox_f = float(p.ox - std::floor(p.ox));
                    o.oz_i = int(std::floor(p.oz)); o.oz_f = float(p.oz - std::floor(p.oz));
                    o.iy = p.iy; o.fy = float(p.fy); o.vy = float(p.vy); o.w = float(N.factor * p.weight);
                };
                for (int side = 0; side < 2; side++) {
                    for (int o = 0; o < 2; o++) fill(T, o, side, sn.t[side * 2 + o]);
                    for (int o = 0; o < 3; o++) fill(C, o, side, sn.c[side * 3 + o]);
                    for (int o = 0; o < 2; o++) fill(E, o, side, sn.e[side * 2 + o]);
                }
                auto tg0 = Clock::now();
                gproxy::evaluate(sn, gzones, gbits);
                t_gpu = t_gpu + std::chrono::duration<double>(Clock::now() - tg0).count();
                for (size_t zi = 0; zi < gzones.size(); zi++) {
                    const auto &zn = gzones[zi]; auto &bits = gbits[zi];
                    if (P.gpucheck > 0) {                          // spot-check against the CPU proxy
                        long diff = 0, tot = 0;
                        for (int c = 0; c < zn.nx * zn.nz; c += 7) { tot++; diff += (PX(zn.x0 + (c % zn.nx) * 32, zn.z0 + (c / zn.nx) * 32) == 2) != (bits[c] != 0); }
                        fprintf(stderr, "gpucheck zone %dx%d: %ld/%ld cells differ\n", zn.nx, zn.nz, diff, tot);
                        P.gpucheck--;
                    }
                    // 8-connected components of the bitmap
                    for (int c0 = 0; c0 < zn.nx * zn.nz; c0++) {
                        if (bits[c0] != 1) continue;
                        long cells = 0; int mx = zn.nx, mz = zn.nz;
                        std::vector<int> stk{c0}, comp; bits[c0] = 2;
                        while (!stk.empty()) {
                            int c = stk.back(); stk.pop_back(); cells++; comp.push_back(c);
                            int cx = c % zn.nx, cz = c / zn.nx;
                            mx = std::min(mx, cx); mz = std::min(mz, cz);
                            for (int dz = -1; dz <= 1; dz++) for (int dx = -1; dx <= 1; dx++) {
                                int nx2 = cx + dx, nz2 = cz + dz;
                                if (nx2 < 0 || nz2 < 0 || nx2 >= zn.nx || nz2 >= zn.nz) continue;
                                int k2 = nz2 * zn.nx + nx2;
                                if (bits[k2] == 1) { bits[k2] = 2; stk.push_back(k2); }
                            }
                        }
                        double area = cells * 32.0 * 32 * 16 / 1e6;
                        if (area < 1) continue;
                        std::pair<int, int> sig(zn.x0 + mx * 32, zn.z0 + mz * 32);
                        if (std::find(seen.begin(), seen.end(), sig) != seen.end()) continue;
                        seen.push_back(sig);
                        // interior cells (all 8 neighbours in the component), ordered by distance to the centroid;
                        // an edge cell can sit next to a small detached exact pocket
                        double cx = 0, cz = 0;
                        for (int c : comp) { cx += c % zn.nx; cz += c / zn.nx; }
                        cx /= comp.size(); cz /= comp.size();
                        std::vector<std::pair<double, int>> inner;
                        for (int c : comp) {
                            int x = c % zn.nx, z = c / zn.nx; bool ok = x > 0 && z > 0 && x < zn.nx - 1 && z < zn.nz - 1;
                            for (int dz = -1; ok && dz <= 1; dz++) for (int dx = -1; dx <= 1; dx++) if (bits[(z + dz) * zn.nx + x + dx] != 2) { ok = false; break; }
                            if (ok) inner.push_back({(x - cx) * (x - cx) + (z - cz) * (z - cz), c});
                        }
                        std::sort(inner.begin(), inner.end());
                        auto cell_xy = [&](int c) { return std::pair<int, int>(zn.x0 + (c % zn.nx) * 32, zn.z0 + (c / zn.nx) * 32); };
                        auto first = inner.empty() ? cell_xy(c0) : cell_xy(inner[0].second);
                        ev.push_back({gmeta[zi].first, first.first, first.second, area});
                        ev.back().hot = gmeta[zi].second;
                        for (int t = 1; t < 3 && size_t(t) * inner.size() / 3 < inner.size() && !inner.empty(); t++)
                            ev.back().alt.push_back(cell_xy(inner[size_t(t) * inner.size() / 3].second));
                    }
                }
            }
#endif
            std::sort(ev.begin(), ev.end(), [](auto &a, auto &b) { return a.proxy > b.proxy; });
            auto tp1 = Clock::now();
            t_proxy = t_proxy + std::chrono::duration<double>(tp1 - tp0).count();
            double seed_best = 0;
            for (size_t k2 = 0; !P.study && k2 < ev.size() && int(k2) < P.exact_n && ev[k2].proxy >= P.exact_min; k2++) {
                auto &e = ev[k2];
                std::vector<std::pair<int, int>> tries{{e.x, e.z}};
                tries.insert(tries.end(), e.alt.begin(), e.alt.end());
                double best8 = -1; int best_x = e.x, best_z = e.z; bool best_edge = false;
                for (auto [sx, sz] : tries) {
                    // nearest exact desert point to this proxy start
                    int bx = sx, bz = sz; bool found = false;
                    for (int r = 0; r <= 8 && !found; r++)
                        for (int dz = -r; dz <= r && !found; dz++) for (int dx = -r; dx <= r && !found; dx++) {
                            if (std::max(std::abs(dx), std::abs(dz)) != r) continue;
                            if (S.desert(sx + dx * 16, sz + dz * 16)) { bx = sx + dx * 16; bz = sz + dz * 16; found = true; }
                        }
                    if (!found) continue;
                    auto a8 = dense_flood_bridged(pred, bx, bz, 8, 16000 / 8, state, stack, true);
                    double v8 = a8.cells * 64.0 * 16 / 1e6;
                    if (v8 > best8) { best8 = v8; best_x = bx; best_z = bz; best_edge = a8.edge; }
                    if (best8 >= 0.6 * e.proxy) break;                  // plausible: this start reached the main desert
                }
                if (best8 < 0) { e.s8 = 0; continue; }
                e.x = best_x; e.z = best_z; e.s8 = best8; e.edge = best_edge;
                int bx = best_x, bz = best_z;
                if (e.s8 >= P.refine_mb) {
                    auto a4 = dense_flood_bridged(pred, bx, bz, 4, 16000 / 4, state, stack, false);
                    e.s4 = a4.cells * 16.0 * 16 / 1e6; e.edge |= a4.edge;
                }
                double v = e.s4 > 0 ? e.s4 : e.s8;
                seed_best = std::max(seed_best, v);
                std::lock_guard<std::mutex> g(mu);
                if (e.s4 >= P.verify_mb)
                    fprintf(stderr, "CANDIDATE step4=%.2fM seed=%lld x=%d z=%d -> verify with exact flood (Y=256)\n", e.s4, (long long)seed, bx * 4 + 2, bz * 4 + 2);
                if (v > best) {
                    best = v;
                    fprintf(stderr, "best %.2fM (step %d) seed=%lld x=%d z=%d proxy=%.1fM\n", v, e.s4 > 0 ? 4 : 8, (long long)seed, bx * 4 + 2, bz * 4 + 2, e.proxy);
                }
            }
            {
                std::lock_guard<std::mutex> g(mu);
                for (auto &e : ev) {
                    if (e.proxy < P.csv_min && e.s8 <= 0 && e.s4 <= 0) continue;
                    const auto &st = sites[e.rank];
                    fprintf(out, "%lld,%.4f,%.4f,%.4f,%.4f,%zu,%d,%d,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%d,%.1f\n", (long long)seed, fy[0], fy[1], maxta, maxtb,
                            e.rank, e.x * 4 + 2, e.z * 4 + 2, st.s, st.t, st.c, st.e, e.proxy, e.s8, e.s4, int(e.edge), e.hot);
                }
            }
            auto t3 = Clock::now();
            t_tab = t_tab + std::chrono::duration<double>(t1 - t0).count();
            t_scan = t_scan + std::chrono::duration<double>(t2 - t1).count();
            t_site = t_site + std::chrono::duration<double>(t3 - t2).count();
            std::lock_guard<std::mutex> g(mu);
            fprintf(sout, "%lld,%.4f,%.4f,%.4f,%.4f,%zu,%zu,%.3f\n", (long long)seed, fy[0], fy[1], maxta, maxtb, cand.size(), sites.size(), seed_best);
            auto now = Clock::now();
            if (std::chrono::duration<double>(now - last).count() > 30) {
                last = now; fflush(out); fflush(sout);
                fprintf(stderr, "%llu seeds, %llu passed, %llu scanned, %llu sites; %.2f passed/s; core-s tables %.0f scan %.0f sites %.0f (proxy %.0f hot %.0f gpu %.0f); hot zones %llu, big %llu\n",
                        (unsigned long long)done.load(), (unsigned long long)passed.load(), (unsigned long long)scanned.load(), (unsigned long long)nsites.load(),
                        passed.load() / std::chrono::duration<double>(now - begin).count(), t_tab.load(), t_scan.load(), t_site.load(), t_proxy.load(), t_hot.load(), t_gpu.load(), (unsigned long long)nhot.load(), (unsigned long long)nbig.load());
            }
        }
    };
    std::thread reader;
#ifdef DRIP_GPU
    if (P.gate) {
        gproxy::start_gate(0, start, count ? count : 100000000000000ULL, float(P.gate_phase), float(P.gate_amp), float(P.gate_pthr),
            [&](uint64_t sd) {
                std::unique_lock<std::mutex> l(qmu);
                qcv.wait(l, [&] { return q.size() < 20000 || interrupted; });   // backpressure on the gate
                q.push_back(sd);
                qcv.notify_all();
            },
            [&] { std::lock_guard<std::mutex> l(qmu); qdone = true; qcv.notify_all(); });
    } else
#endif
    if (P.from_stdin) reader = std::thread([&] {
        char line[256];
        while (!interrupted && fgets(line, sizeof line, stdin)) {
            uint64_t sd = std::strtoull(line, nullptr, 10);
            std::unique_lock<std::mutex> l(qmu);
            qcv.wait(l, [&] { return q.size() < 20000 || interrupted; });   // backpressure on the gate
            q.push_back(sd);
            qcv.notify_all();
        }
        std::lock_guard<std::mutex> l(qmu); qdone = true; qcv.notify_all();
    });
    std::vector<std::thread> pool;
    for (int i = 0; i < threads; i++) pool.emplace_back(worker);
    for (auto &t : pool) t.join();
    if (reader.joinable()) { if (!qdone) { reader.detach(); } else reader.join(); }
    fclose(out); fclose(sout);
    if (zout) fclose(zout);
    double sec = std::chrono::duration<double>(Clock::now() - begin).count();
    FILE *m = fopen((prefix + ".json").c_str(), "w");
    fprintf(m, "{\"start\":%llu,\"done\":%llu,\"passed\":%llu,\"scanned\":%llu,\"amp\":%g,\"sites\":%llu,\"seconds\":%.3f,\"threads\":%d,\"phase\":%g,\"sigma\":%g,"
               "\"wc\":%g,\"we\":%g,\"t\":%g,\"ta\":%g,\"max_sites\":%d,\"nms\":%d,\"cpu_tables\":%.3f,\"cpu_scan\":%.3f,\"cpu_sites\":%.3f}\n",
            (unsigned long long)start, (unsigned long long)done.load(), (unsigned long long)passed.load(), (unsigned long long)scanned.load(), P.amp, (unsigned long long)nsites.load(),
            sec, threads, P.max_phase, P.sigma, P.wc, P.we, P.t, P.ta, P.max_sites, P.nms, t_tab.load(), t_scan.load(), t_site.load());
    fclose(m);
    fprintf(stderr, "done: %llu seeds (%llu passed) in %.1fs\n", (unsigned long long)done.load(), (unsigned long long)passed.load(), sec);
    // Leave immediately: a detached stdin reader or GPU engine thread may still be blocked.
    std::fflush(nullptr);
    std::_Exit(0);
}
