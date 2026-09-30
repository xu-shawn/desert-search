// GPU seed gate for the desert search (standalone tool): prints "seed amp" lines for
// dsearch_desert stdin=1. The kernels live in gate.cuh (also used in-process by dsearch_desert_gpu gate=1).
//
//   tgate START COUNT [phase=0.10] [amp=0.70] [box=1 pthr=0.645 | box=0 margin=0.01] [check=0] [calib=N]
// box=1: anti-aliased pre-gate (4-parity cell mean), 99.5% recall of amp>=0.70 at pthr=0.645
// calib=N: "seed pre full" amplitudes of N phase-passed seeds (pre-gate threshold calibration)
// Seeds are splitmix64(START + k), k in [0, COUNT), matching dsearch_desert.
#include "gate.cuh"
#include <string>
#include <cstring>
using namespace tg;

// ---------------------------------------------------------------- CPU reference (same math as lowfield.hpp)
static float cpu_amp(uint64_t seed, int G, int R, float sig) {
    float amp = 0;
    for (int side = 0; side < 2; side++) {
        PerlinParams p; init_side(seed, side, p);
        std::vector<float> t(size_t(G) * G), h(size_t(G) * G);
        for (int j = 0; j < G; j++) for (int i = 0; i < G; i++) {
            double u = (G == 512 ? i * 0.5 : i) + p.ox, v = (G == 512 ? j * 0.5 : j) + p.oz;
            int ix = int(floor(u)), iz = int(floor(v)); double fx = u - floor(u), fz = v - floor(v);
            double tx = fade(fx), tz = fade(fz), acc = 0;
            for (int c = 0; c < 8; c++) {
                int dx = c & 1, dy = (c >> 1) & 1, dz = c >> 2;
                int hsh = p.perm[(p.perm[(p.perm[(ix + dx) & 255] + p.iy + dy) & 255] + iz + dz) & 255];
                double wc = (dx ? tx : 1 - tx) * (dy ? p.vy : 1 - p.vy) * (dz ? tz : 1 - tz);
                acc += wc * gradd(hsh, fx - dx, p.fy - dy, fz - dz);
            }
            t[size_t(j) * G + i] = float(T_W0 * acc);
        }
        std::vector<float> k(2 * R + 1); float ks = 0;
        for (int d = -R; d <= R; d++) ks += k[d + R] = expf(-0.5f * d * d / (sig * sig));
        for (int j = 0; j < G; j++) for (int i = 0; i < G; i++) { float s = 0; for (int d = -R; d <= R; d++) s += k[d + R] * t[size_t(j) * G + ((i + d) & (G - 1))]; h[size_t(j) * G + i] = s / ks; }
        float m = -9;
        for (int j = 0; j < G; j++) for (int i = 0; i < G; i++) { float s = 0; for (int d = -R; d <= R; d++) s += k[d + R] * h[size_t((j + d) & (G - 1)) * G + i]; m = std::max(m, s / ks); }
        amp += m;
    }
    return amp;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "tgate START COUNT [phase=0.10] [amp=0.70] [box=1 pthr=0.645 | box=0 margin=0.01] [check=0] [calib=N]\n"); return 2; }
    uint64_t start = std::stoull(argv[1]), count = std::stoull(argv[2]);
    GateCfg g; int check = 0, calib = 0;
    for (int i = 3; i < argc; i++) {
        std::string kv = argv[i]; auto e = kv.find('='); std::string k = kv.substr(0, e); double v = std::stod(kv.substr(e + 1));
        if (k == "phase") g.maxph = float(v); else if (k == "amp") g.amp = float(v); else if (k == "margin") g.margin = float(v); else if (k == "check") check = int(v);
        else if (k == "calib") calib = int(v); else if (k == "box") g.box = int(v); else if (k == "sig") g.sig = float(v); else if (k == "pthr") g.pthr = float(v);
        else if (k == "tpb_pre") g.tpb_pre = int(v); else if (k == "tpb_full") g.tpb_full = int(v); else if (k == "perseed") g.per_seed_blocks = v != 0;
    }
    const uint64_t B = 1ull << 26; const unsigned CAP1 = 1u << 22, CAP3 = 1u << 16;
    const float maxph = g.maxph, calib_sig = g.sig; const int calib_box = g.box;
    if (calib) {
        uint64_t *d_s1, *d_s3; float *d_a3; PerlinParams *d_p1; unsigned *d_n;
        CK(cudaMalloc(&d_s1, CAP1 * 8)); CK(cudaMalloc(&d_s3, CAP3 * 8)); CK(cudaMalloc(&d_a3, CAP3 * 4));
        CK(cudaMalloc(&d_p1, size_t(CAP1) * 2 * sizeof(PerlinParams))); CK(cudaMalloc(&d_n, 3 * sizeof(unsigned)));
        set_smem<256, 4, 16, false>(); set_smem<512, 7, 8, false>(); set_smem<256, 4, 8, true>();
    {   // pre-gate and full amplitude of the first `calib` phase-passed seeds: "seed pre full"
        CK(cudaMemset(d_n, 0, 3 * sizeof(unsigned)));
        k_phase<<<unsigned((B + 255) / 256), 256>>>(start, B, maxph, d_s1, d_n, CAP1);
        unsigned n[3]; CK(cudaMemcpy(n, d_n, 3 * sizeof(unsigned), cudaMemcpyDeviceToHost));
        unsigned m = std::min({n[0], unsigned(calib), CAP3});
        {   // atomics collect seeds in arbitrary order: sort for a reproducible sample
            std::vector<uint64_t> all(std::min(n[0], CAP1));
            CK(cudaMemcpy(all.data(), d_s1, all.size() * 8, cudaMemcpyDeviceToHost));
            std::sort(all.begin(), all.end());
            CK(cudaMemcpy(d_s1, all.data(), size_t(m) * 8, cudaMemcpyHostToDevice));
        }
        k_init<<<(2 * m + 255) / 256, 256>>>(d_s1, m, d_p1);
        float *d_a2; uint64_t *d_s4; CK(cudaMalloc(&d_a2, CAP3 * 4)); CK(cudaMalloc(&d_s4, CAP3 * 8));
        if (calib_box) k_amp2<256, 4, 8, true><<<4096, 256, dyn_bytes<256, 4, 8, true>()>>>(d_s1, d_p1, m, calib_sig, -1e9f, d_s4, d_a2, d_n + 1, CAP3);
        else k_amp2<256, 4, 16><<<4096, 256, dyn_bytes<256, 4, 16>()>>>(d_s1, d_p1, m, 1.0f, -1e9f, d_s4, d_a2, d_n + 1, CAP3);
        k_amp2<512, 7, 8><<<4096, 256, dyn_bytes<512, 7, 8>()>>>(d_s1, d_p1, m, 2.0f, -1e9f, d_s3, d_a3, d_n + 2, CAP3);
        std::vector<uint64_t> s2(m), s3(m); std::vector<float> a2(m), a3(m);
        CK(cudaMemcpy(s2.data(), d_s4, m * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(a2.data(), d_a2, m * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s3.data(), d_s3, m * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(a3.data(), d_a3, m * 4, cudaMemcpyDeviceToHost));
        std::vector<std::pair<uint64_t, float>> x(m);
        for (unsigned i = 0; i < m; i++) x[i] = {s2[i], a2[i]};
        std::sort(x.begin(), x.end());
        for (unsigned i = 0; i < m; i++) {
            auto it = std::lower_bound(x.begin(), x.end(), std::make_pair(s3[i], -1e30f));
            printf("%llu %.5f %.5f %.5f\n", (unsigned long long)s3[i], it->second, a3[i], phase_of(s3[i]));
        }
        return 0;
    }
    }
    int checked = 0;
    run_gate(0, start, count, g, [&](uint64_t seed, float amp) {
        printf("%llu %.5f\n", (unsigned long long)seed, amp);
        if (check && checked < 4) { fprintf(stderr, "  check seed %llu: gpu amp %.5f cpu amp %.5f\n", (unsigned long long)seed, amp, cpu_amp(seed, 512, 7, 2.0f)); checked++; }
    }, [] { fflush(stdout); return false; });
    return 0;
}
