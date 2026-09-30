// GPU proxy engine: see gproxy.h.
#include "gproxy.h"
#include "gate.cuh"
#include "../desert_dtable.hpp"
#include <condition_variable>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>
#include <chrono>


namespace gproxy {

__constant__ float c_prob[40][40];
__constant__ float c_cut;

__device__ __forceinline__ float fadef(float t) { return t * t * t * (t * (t * 6.f - 15.f) + 10.f); }
__device__ __forceinline__ float gradf(int h, float x, float y, float z) {
    // Minecraft's 16 gradients without a lookup table (a runtime-indexed local array lands in
    // slow local memory): u/v selection as in Ken Perlin's reference, including the 12..15 duplicates.
    h &= 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}
// lattice coordinate of world quart X for this octave: integer part and fraction, exactly
__device__ __forceinline__ void lat(int X, int k, int side, int o_i, float o_f, int &ii, float &ff) {
    int q; float f;
    if (side == 0) {
        q = X >> k;                                               // floor division by 2^k
        f = float(X - (q << k)) * __int_as_float((127 - k) << 23); // * 2^-k
    } else {
        long long num = (long long)X * 337, den = 331LL << k;
        long long qq = num >= 0 ? num / den : -((-num + den - 1) / den);
        f = float(num - qq * den) / float(den);
        q = int(qq);
    }
    f += o_f;
    if (f >= 1.f) { f -= 1.f; q += 1; }
    ii = q + o_i; ff = f;
}
__device__ float perlin(const Octave &o, const uint8_t *P, int X, int Z) {
    int ix, iz; float fx, fz;
    lat(X, o.k, o.side, o.ox_i, o.ox_f, ix, fx);
    lat(Z, o.k, o.side, o.oz_i, o.oz_f, iz, fz);
    const int iy = o.iy; const float fy = o.fy, ty = o.vy;
    int a0 = P[ix & 255], a1 = P[(ix + 1) & 255];
    int b00 = P[(a0 + iy) & 255], b01 = P[(a0 + iy + 1) & 255], b10 = P[(a1 + iy) & 255], b11 = P[(a1 + iy + 1) & 255];
    float tx = fadef(fx), tz = fadef(fz);
    auto L = [](float t, float a, float b) { return a + t * (b - a); };
    float z0 = L(ty, L(tx, gradf(P[(b00 + iz) & 255], fx, fy, fz), gradf(P[(b10 + iz) & 255], fx - 1, fy, fz)),
                     L(tx, gradf(P[(b01 + iz) & 255], fx, fy - 1, fz), gradf(P[(b11 + iz) & 255], fx - 1, fy - 1, fz)));
    float z1 = L(ty, L(tx, gradf(P[(b00 + iz + 1) & 255], fx, fy, fz - 1), gradf(P[(b10 + iz + 1) & 255], fx - 1, fy, fz - 1)),
                     L(tx, gradf(P[(b01 + iz + 1) & 255], fx, fy - 1, fz - 1), gradf(P[(b11 + iz + 1) & 255], fx - 1, fy - 1, fz - 1)));
    return L(tz, z0, z1);
}

struct LC { int i; float f; };                 // lattice integer part and fraction
struct DZone { int x0, z0, nx, nz; long long off; int seed; long long coff; };   // coff: offset into coords
// Exact lattice coordinate of world quart X for an octave (host, 64-bit arithmetic).
static inline LC lat_host(int X, int k, int side, int o_i, float o_f) {
    long long q; double f;
    if (side == 0) { q = (long long)X >> k; f = double(X - (q << k)) / double(1LL << k); }
    else {
        long long num = (long long)X * 337, den = 331LL << k;
        q = num >= 0 ? num / den : -((-num + den - 1) / den);
        f = double(num - q * den) / double(den);
    }
    float ff = float(f) + o_f;
    if (ff >= 1.f) { ff -= 1.f; q += 1; }
    return {int(q) + o_i, ff};
}

// coords layout per zone: for each of the 14 octaves, nx x-entries then nz z-entries
__device__ float perlin_lc(const Octave &o, const uint8_t *P, LC cx, LC cz) {
    const int ix = cx.i, iz = cz.i, iy = o.iy; const float fx = cx.f, fz = cz.f, fy = o.fy, ty = o.vy;
    int a0 = P[ix & 255], a1 = P[(ix + 1) & 255];
    int b00 = P[(a0 + iy) & 255], b01 = P[(a0 + iy + 1) & 255], b10 = P[(a1 + iy) & 255], b11 = P[(a1 + iy + 1) & 255];
    float tx = fadef(fx), tz = fadef(fz);
    auto L = [](float t, float a, float b) { return a + t * (b - a); };
    float z0 = L(ty, L(tx, gradf(P[(b00 + iz) & 255], fx, fy, fz), gradf(P[(b10 + iz) & 255], fx - 1, fy, fz)),
                     L(tx, gradf(P[(b01 + iz) & 255], fx, fy - 1, fz), gradf(P[(b11 + iz) & 255], fx - 1, fy - 1, fz)));
    float z1 = L(ty, L(tx, gradf(P[(b00 + iz + 1) & 255], fx, fy, fz - 1), gradf(P[(b10 + iz + 1) & 255], fx - 1, fy, fz - 1)),
                     L(tx, gradf(P[(b01 + iz + 1) & 255], fx, fy - 1, fz - 1), gradf(P[(b11 + iz + 1) & 255], fx - 1, fy - 1, fz - 1)));
    return L(tz, z0, z1);
}
// grid (zones, chunks): block (x, y) evaluates cells [y*CHUNK, (y+1)*CHUNK) of zone x, so even a
// single zone spreads over the whole GPU (requests often carry only a few zones)
constexpr int CHUNK = 2048;
__global__ void k_proxy(const SeedNoise *seeds, const DZone *zones, const LC *coords, uint8_t *out) {
    __shared__ Octave O[14];
    const DZone z = zones[blockIdx.x];
    const int stride = z.nx + z.nz, n = z.nx * z.nz, c0 = blockIdx.y * CHUNK, c1 = min(n, c0 + CHUNK);
    if (c0 >= n) return;
    const SeedNoise &sn = seeds[z.seed];
    for (int i = threadIdx.x; i < 14; i += blockDim.x) O[i] = i < 4 ? sn.t[i] : i < 10 ? sn.c[i - 4] : sn.e[i - 10];
    __syncthreads();
    const LC *cd = coords + z.coff;
    for (int c = c0 + threadIdx.x; c < c1; c += blockDim.x) {
        int ci = c % z.nx, cj = c / z.nx;
        float t = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) t += O[i].w * perlin_lc(O[i], O[i].perm, cd[i * stride + ci], cd[i * stride + z.nx + cj]);
        uint8_t pass = 0;
        if (t >= 0.55f) {
            float cv = 0, ev = 0;
            #pragma unroll
            for (int i = 4; i < 10; i++) cv += O[i].w * perlin_lc(O[i], O[i].perm, cd[i * stride + ci], cd[i * stride + z.nx + cj]);
            #pragma unroll
            for (int i = 10; i < 14; i++) ev += O[i].w * perlin_lc(O[i], O[i].perm, cd[i * stride + ci], cd[i * stride + z.nx + cj]);
            int a = min(39, max(0, int(floorf((cv + 0.8f) / 0.05f)))), b = min(39, max(0, int(floorf((ev + 1.0f) / 0.05f))));
            pass = c_prob[a][b] >= c_cut;
        }
        out[z.off + c] = pass;
    }
}


// ---------------------------------------------------------------- period tables + translation scan
// Bit-exact with the CPU (lowfield.hpp tabulate_octave + Blur, dsearch_desert scan): same summation
// order, explicit round-to-nearest multiply/add (no FMA contraction).
constexpr int STN = 512, SCAP_SEEDS = 32, SCAP_CAND = 1 << 16;
__constant__ float c_kT[32], c_kC[32];
__constant__ int c_rT, c_rC, c_nt, c_cell, c_period, c_kmin;
static int16_t *d_ibT, *d_ibC; static int8_t *d_loT, *d_hiT;

__global__ void k_tab(const TabIn *tin, float *tabs) {                 // grid (STN/8, 6, seeds), block STN
    __shared__ uint8_t P[256], col[2][256];
    __shared__ float Tc[2][2][8][16];
    __shared__ int i0[2], k0[2];
    const TabIn &in = tin[blockIdx.z * 6 + blockIdx.y];
    for (int e = threadIdx.x; e < 256; e += blockDim.x) { P[e] = in.perm[e]; col[0][e] = in.col[0][e]; col[1][e] = in.col[1][e]; }
    for (int e = threadIdx.x; e < 512; e += blockDim.x) (&Tc[0][0][0][0])[e] = (&in.T[0][0][0][0])[e];
    if (threadIdx.x < 2) { i0[threadIdx.x] = in.i0[threadIdx.x]; k0[threadIdx.x] = in.k0[threadIdx.x]; }
    __syncthreads();
    float *t = tabs + (size_t(blockIdx.z) * 6 + blockIdx.y) * STN * STN;
    const int i = threadIdx.x, px = i & 1, ix = (i >> 1) + i0[px];
    for (int r = 0; r < 8; r++) {
        const int j = blockIdx.x * 8 + r, pz = j & 1, iz = (j >> 1) + k0[pz];
        float acc = 0;
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            int dx = c & 1, dy = (c >> 1) & 1, dz = c >> 2;
            acc = __fadd_rn(acc, Tc[px][pz][c][P[(col[dy][(ix + dx) & 255] + iz + dz) & 255] & 15]);
        }
        t[size_t(j) * STN + i] = acc;
    }
}
__global__ void k_blurh(const float *a, float *h) {                    // grid (STN, 6, seeds), block STN
    __shared__ float row[STN];
    const size_t base = (size_t(blockIdx.z) * 6 + blockIdx.y) * STN * STN + size_t(blockIdx.x) * STN;
    row[threadIdx.x] = a[base + threadIdx.x];
    __syncthreads();
    const bool T = blockIdx.y < 2; const int r = T ? c_rT : c_rC; const float *k = T ? c_kT : c_kC;
    float s = 0;
    for (int d = 0; d <= 2 * r; d++) s = __fadd_rn(s, __fmul_rn(k[d], row[(threadIdx.x + d - r) & (STN - 1)]));
    h[base + threadIdx.x] = s;
}
__device__ __forceinline__ void atomic_max_f(float *a, float v) {
    if (v >= 0) atomicMax(reinterpret_cast<int *>(a), __float_as_int(v));
    else atomicMin(reinterpret_cast<unsigned *>(a), __float_as_uint(v));
}
__global__ void k_blurv(const float *h, float *a, float *mx) {         // grid (STN, 6, seeds), block STN
    const size_t tb = (size_t(blockIdx.z) * 6 + blockIdx.y) * STN * STN;
    const bool T = blockIdx.y < 2; const int r = T ? c_rT : c_rC; const float *k = T ? c_kT : c_kC;
    const int j = blockIdx.x, i = threadIdx.x;
    float s = 0;
    for (int d = 0; d <= 2 * r; d++) s = __fadd_rn(s, __fmul_rn(k[d], h[tb + size_t((j + d - r) & (STN - 1)) * STN + i]));
    a[tb + size_t(j) * STN + i] = s;
    if (T) {
        float m = s;
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        if ((threadIdx.x & 31) == 0) atomic_max_f(&mx[blockIdx.z * 2 + blockIdx.y], m);
    }
}
__global__ void k_active(const float *tabs, float ta, int *act, unsigned *nact) {   // grid (STN*STN/256, seeds), block 256
    const int cell = blockIdx.x * 256 + threadIdx.x, sd = blockIdx.y;
    if (tabs[size_t(sd) * 6 * STN * STN + cell] >= ta) act[size_t(sd) * STN * STN + atomicAdd(&nact[sd], 1u)] = cell;
}
__global__ void k_scan(const float *tabs, const int *act, const unsigned *nact, const int16_t *ibT, const int16_t *ibC,
                       const int8_t *loT, const int8_t *hiT, ScanParams sp, RawCand *out, unsigned *ncand) {
    // grid (max active cells, seeds), block 64: thread = one z translation of one active side-A T cell
    const int sd = blockIdx.y, nt = c_nt;
    if (blockIdx.x >= nact[sd]) return;
    const int cell = act[size_t(sd) * STN * STN + blockIdx.x], j = cell >> 9, i = cell & (STN - 1);
    const int tz = loT[j] + threadIdx.x;
    if (tz > hiT[j]) return;
    const float *TA = tabs + size_t(sd) * 6 * STN * STN, *TB = TA + STN * STN, *CA = TB + STN * STN, *CB = CA + STN * STN,
                *EA = CB + STN * STN, *EB = EA + STN * STN;
    const float sa = TA[cell];
    const int ca = ((2 * j) & (STN - 1)) * STN + ((2 * i) & (STN - 1));
    const float cA = CA[ca], eA = EA[ca];
    const float *rT = TB + ibT[j * nt + tz] * STN, *rC = CB + ibC[j * nt + tz] * STN, *rE = EB + ibC[j * nt + tz] * STN;
    const float tt = sp.tt, pre = __fadd_rn(sp.tt, -0.3f);
    for (int tx = loT[i]; tx <= hiT[i]; tx++) {
        float tv = __fadd_rn(sa, rT[ibT[i * nt + tx]]);
        if (tv < pre) continue;
        int cb = ibC[i * nt + tx];
        float cv = __fadd_rn(cA, rC[cb]), ev = __fadd_rn(eA, rE[cb]);
        float sc = __fadd_rn(__fadd_rn(tv, __fmul_rn(sp.wc, cv)), __fmul_rn(sp.we, ev));
        if (sc < tt) continue;
        unsigned k = atomicAdd(&ncand[sd], 1u);
        if (k < SCAP_CAND) out[size_t(sd) * SCAP_CAND + k] = {sc, tv, cv, ev, i * c_cell + (tx + c_kmin) * c_period, j * c_cell + (tz + c_kmin) * c_period};
    }
}

struct ScanReq {
    const TabIn *tabs; ScanParams sp; float maxta = 0, maxtb = 0; std::vector<RawCand> *raw; bool ok = true, done = false;
};
static std::mutex smu;
static std::condition_variable scv;
static std::deque<ScanReq *> spending;
static cudaStream_t s_scan;
static bool scan_started = false;
static int scan_device = 0;

static void scan_loop() {
    CK(cudaSetDevice(scan_device));
    const size_t TB = size_t(6) * STN * STN;
    TabIn *d_in; float *d_tab, *d_h, *d_mx; int *d_act; unsigned *d_nact, *d_ncand; RawCand *d_out;
    CK(cudaMalloc(&d_in, SCAP_SEEDS * 6 * sizeof(TabIn))); CK(cudaMalloc(&d_tab, SCAP_SEEDS * TB * 4)); CK(cudaMalloc(&d_h, SCAP_SEEDS * TB * 4));
    CK(cudaMalloc(&d_mx, SCAP_SEEDS * 2 * 4)); CK(cudaMalloc(&d_act, size_t(SCAP_SEEDS) * STN * STN * 4));
    CK(cudaMalloc(&d_nact, SCAP_SEEDS * 4)); CK(cudaMalloc(&d_ncand, SCAP_SEEDS * 4)); CK(cudaMalloc(&d_out, size_t(SCAP_SEEDS) * SCAP_CAND * sizeof(RawCand)));
    TabIn *h_in; float *h_mx; unsigned *h_nact, *h_ncand; RawCand *h_out;
    CK(cudaMallocHost(&h_in, SCAP_SEEDS * 6 * sizeof(TabIn))); CK(cudaMallocHost(&h_mx, SCAP_SEEDS * 2 * 4));
    CK(cudaMallocHost(&h_nact, SCAP_SEEDS * 4)); CK(cudaMallocHost(&h_ncand, SCAP_SEEDS * 4));
    CK(cudaMallocHost(&h_out, size_t(SCAP_SEEDS) * SCAP_CAND * sizeof(RawCand)));
    std::vector<float> ninf(SCAP_SEEDS * 2, -1e30f);
    double seeds_done = 0, kms = 0; auto last = std::chrono::steady_clock::now();
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (;;) {
        std::vector<ScanReq *> batch;
        {
            std::unique_lock<std::mutex> l(smu);
            scv.wait(l, [] { return !spending.empty(); });
            while (!spending.empty() && batch.size() < size_t(SCAP_SEEDS)) { batch.push_back(spending.front()); spending.pop_front(); }
        }
        const unsigned S = unsigned(batch.size());
        const ScanParams sp = batch[0]->sp;                 // all requests share the search parameters
        for (unsigned k = 0; k < S; k++) std::memcpy(h_in + 6 * k, batch[k]->tabs, 6 * sizeof(TabIn));
        std::memcpy(h_mx, ninf.data(), S * 2 * 4);
        CK(cudaEventRecord(e0, s_scan));
        CK(cudaMemcpyAsync(d_in, h_in, S * 6 * sizeof(TabIn), cudaMemcpyHostToDevice, s_scan));
        CK(cudaMemcpyAsync(d_mx, h_mx, S * 2 * 4, cudaMemcpyHostToDevice, s_scan));
        CK(cudaMemsetAsync(d_nact, 0, S * 4, s_scan)); CK(cudaMemsetAsync(d_ncand, 0, S * 4, s_scan));
        k_tab<<<dim3(STN / 8, 6, S), STN, 0, s_scan>>>(d_in, d_tab);
        k_blurh<<<dim3(STN, 6, S), STN, 0, s_scan>>>(d_tab, d_h);
        k_blurv<<<dim3(STN, 6, S), STN, 0, s_scan>>>(d_h, d_tab, d_mx);
        k_active<<<dim3(STN * STN / 256, S), 256, 0, s_scan>>>(d_tab, sp.ta, d_act, d_nact);
        CK(cudaMemcpyAsync(h_nact, d_nact, S * 4, cudaMemcpyDeviceToHost, s_scan)); CK(cudaStreamSynchronize(s_scan));
        unsigned maxa = 0; for (unsigned k = 0; k < S; k++) maxa = std::max(maxa, h_nact[k]);
        if (maxa) k_scan<<<dim3(maxa, S), 64, 0, s_scan>>>(d_tab, d_act, d_nact, d_ibT, d_ibC, d_loT, d_hiT, sp, d_out, d_ncand);
        CK(cudaGetLastError());
        CK(cudaMemcpyAsync(h_ncand, d_ncand, S * 4, cudaMemcpyDeviceToHost, s_scan));
        CK(cudaMemcpyAsync(h_mx, d_mx, S * 2 * 4, cudaMemcpyDeviceToHost, s_scan)); CK(cudaStreamSynchronize(s_scan));
        for (unsigned k = 0; k < S; k++) {
            unsigned n = std::min(h_ncand[k], unsigned(SCAP_CAND));
            if (n) CK(cudaMemcpyAsync(h_out + size_t(k) * SCAP_CAND, d_out + size_t(k) * SCAP_CAND, n * sizeof(RawCand), cudaMemcpyDeviceToHost, s_scan));
        }
        CK(cudaEventRecord(e1, s_scan)); CK(cudaStreamSynchronize(s_scan));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); kms += ms; seeds_done += S;
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last).count() > 30) {
            last = now;
            fprintf(stderr, "gscan: %.0f seeds, %.2f ms/seed wall on stream (batch %u)\n", seeds_done, kms / seeds_done, S);
        }
        std::lock_guard<std::mutex> l(smu);
        for (unsigned k = 0; k < S; k++) {
            auto *r = batch[k];
            r->maxta = h_mx[2 * k]; r->maxtb = h_mx[2 * k + 1];
            r->ok = h_ncand[k] <= unsigned(SCAP_CAND);
            r->raw->assign(h_out + size_t(k) * SCAP_CAND, h_out + size_t(k) * SCAP_CAND + std::min(h_ncand[k], unsigned(SCAP_CAND)));
            r->done = true;
        }
        scv.notify_all();
    }
}

void scan_init(const int16_t *ibT, const int8_t *loT, const int8_t *hiT, const int16_t *ibC, int nt,
               const float *kernT, int rT, const float *kernC, int rC, int cell, int period, int kmin) {
    CK(cudaGetDevice(&scan_device));
    CK(cudaMalloc(&d_ibT, size_t(STN) * nt * 2)); CK(cudaMalloc(&d_ibC, size_t(STN) * nt * 2));
    CK(cudaMalloc(&d_loT, STN)); CK(cudaMalloc(&d_hiT, STN));
    CK(cudaMemcpy(d_ibT, ibT, size_t(STN) * nt * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_ibC, ibC, size_t(STN) * nt * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_loT, loT, STN, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_hiT, hiT, STN, cudaMemcpyHostToDevice));
    CK(cudaMemcpyToSymbol(c_kT, kernT, (2 * rT + 1) * 4)); CK(cudaMemcpyToSymbol(c_kC, kernC, (2 * rC + 1) * 4));
    CK(cudaMemcpyToSymbol(c_rT, &rT, 4)); CK(cudaMemcpyToSymbol(c_rC, &rC, 4)); CK(cudaMemcpyToSymbol(c_nt, &nt, 4));
    CK(cudaMemcpyToSymbol(c_cell, &cell, 4)); CK(cudaMemcpyToSymbol(c_period, &period, 4)); CK(cudaMemcpyToSymbol(c_kmin, &kmin, 4));
    int lo, hi; CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    CK(cudaStreamCreateWithPriority(&s_scan, cudaStreamNonBlocking, hi));
    std::lock_guard<std::mutex> l(smu);
    if (!scan_started) { std::thread(scan_loop).detach(); scan_started = true; }
}

bool scan(const TabIn tabs[6], const ScanParams &sp, float &maxta, float &maxtb, std::vector<RawCand> &raw) {
    ScanReq r; r.tabs = tabs; r.sp = sp; r.raw = &raw;
    std::unique_lock<std::mutex> l(smu);
    spending.push_back(&r);
    scv.notify_all();
    scv.wait(l, [&] { return r.done; });
    maxta = r.maxta; maxtb = r.maxtb;
    return r.ok;
}


// ---------------------------------------------------------------- batching engine
struct Request {
    const SeedNoise *sn; const std::vector<Zone> *zones; std::vector<std::vector<uint8_t>> *out;
    std::vector<LC> coords;            // per zone: 14 octaves x (nx + nz)
    bool done = false;
};
static std::mutex mu;
static std::condition_variable cv;
static std::deque<Request *> pending;
static std::thread gpu_thread, gate_thread;
static bool started = false;
static cudaStream_t s_proxy;           // highest priority, non-blocking: proxy batches overtake the gate

static void gpu_loop(int device) {
    CK(cudaSetDevice(device));
    size_t cap_seeds = 256, cap_zones = 16384, cap_cells = size_t(1) << 28;
    size_t cap_coords = size_t(1) << 25;
    SeedNoise *d_seeds; DZone *d_zones; uint8_t *d_out; LC *d_coords;
    CK(cudaMalloc(&d_seeds, cap_seeds * sizeof(SeedNoise))); CK(cudaMalloc(&d_zones, cap_zones * sizeof(DZone))); CK(cudaMalloc(&d_out, cap_cells));
    CK(cudaMalloc(&d_coords, cap_coords * sizeof(LC)));
    std::vector<SeedNoise> h_seeds; std::vector<DZone> h_zones; std::vector<uint8_t> h_out; std::vector<LC> h_coords;
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    double kms = 0, cells_done = 0, zones_done = 0; auto last = std::chrono::steady_clock::now();
    for (;;) {
        std::vector<Request *> batch;
        {
            std::unique_lock<std::mutex> l(mu);
            cv.wait(l, [] { return !pending.empty(); });
            size_t cells = 0, nz = 0, ncd = 0;
            while (!pending.empty() && batch.size() < cap_seeds) {
                Request *r = pending.front();
                size_t c = 0; for (auto &z : *r->zones) c += size_t(z.nx) * z.nz;
                if (!batch.empty() && (cells + c > cap_cells || nz + r->zones->size() > cap_zones || ncd + r->coords.size() > cap_coords)) break;
                cells += c; nz += r->zones->size(); ncd += r->coords.size(); pending.pop_front(); batch.push_back(r);
            }
        }
        h_seeds.clear(); h_zones.clear(); h_coords.clear();
        long long off = 0;
        for (size_t s = 0; s < batch.size(); s++) {
            h_seeds.push_back(*batch[s]->sn);
            long long coff = (long long)h_coords.size();
            for (auto &z : *batch[s]->zones) {
                h_zones.push_back({z.x0, z.z0, z.nx, z.nz, off, int(s), coff});
                off += (long long)z.nx * z.nz; coff += 14LL * (z.nx + z.nz);
            }
            h_coords.insert(h_coords.end(), batch[s]->coords.begin(), batch[s]->coords.end());
        }
        if (!h_zones.empty()) {
            CK(cudaMemcpyAsync(d_seeds, h_seeds.data(), h_seeds.size() * sizeof(SeedNoise), cudaMemcpyHostToDevice, s_proxy));
            CK(cudaMemcpyAsync(d_zones, h_zones.data(), h_zones.size() * sizeof(DZone), cudaMemcpyHostToDevice, s_proxy));
            CK(cudaMemcpyAsync(d_coords, h_coords.data(), h_coords.size() * sizeof(LC), cudaMemcpyHostToDevice, s_proxy));
            CK(cudaEventRecord(e0, s_proxy));
            long long maxc = 0; for (auto &z : h_zones) maxc = std::max(maxc, (long long)z.nx * z.nz);
            k_proxy<<<dim3(unsigned(h_zones.size()), unsigned((maxc + CHUNK - 1) / CHUNK)), 256, 0, s_proxy>>>(d_seeds, d_zones, d_coords, d_out);
            CK(cudaGetLastError());
            CK(cudaEventRecord(e1, s_proxy)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); kms += ms; cells_done += double(off); zones_done += h_zones.size();
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last).count() > 30) {
                last = now;
                fprintf(stderr, "gproxy: %.0f zones, %.1fM cells, kernel %.1f s -> %.1fM cells/s, %.0f zones/s (batch %zu zones)\n",
                        zones_done, cells_done / 1e6, kms / 1e3, cells_done / kms / 1e3, zones_done / (kms / 1e3), h_zones.size());
            }
            h_out.resize(size_t(off));
            CK(cudaMemcpyAsync(h_out.data(), d_out, size_t(off), cudaMemcpyDeviceToHost, s_proxy));
            CK(cudaStreamSynchronize(s_proxy));
        }
        size_t zi = 0;
        std::lock_guard<std::mutex> l(mu);
        for (auto *r : batch) {
            r->out->resize(r->zones->size());
            for (size_t k = 0; k < r->zones->size(); k++, zi++) {
                const auto &z = h_zones[zi];
                (*r->out)[k].assign(h_out.begin() + z.off, h_out.begin() + z.off + (long long)z.nx * z.nz);
            }
            r->done = true;
        }
        cv.notify_all();
    }
}

void init(int device, float cut) {
    CK(cudaSetDevice(device));
    CK(cudaMemcpyToSymbol(c_prob, standalone::desert_pass_prob, sizeof(float) * 40 * 40));
    CK(cudaMemcpyToSymbol(c_cut, &cut, sizeof(float)));
    std::lock_guard<std::mutex> l(mu);
    if (!started) {
        int lo, hi; CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        CK(cudaStreamCreateWithPriority(&s_proxy, cudaStreamNonBlocking, hi));
        gpu_thread = std::thread(gpu_loop, device); gpu_thread.detach(); started = true;
    }
}

void start_gate(int device, uint64_t start, uint64_t count, float phase, float amp, float pthr,
                std::function<void(uint64_t)> push, std::function<void()> done) {
    gate_thread = std::thread([=] {
        CK(cudaSetDevice(device));
        int lo, hi; CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        cudaStream_t st; CK(cudaStreamCreateWithPriority(&st, cudaStreamNonBlocking, lo));
        tg::GateCfg g; g.maxph = phase; g.amp = amp; g.pthr = pthr; g.per_seed_blocks = true;
        tg::run_gate(st, start, count, g, [&](uint64_t seed, float) { push(seed); }, [] { return false; });
        done();
    });
    gate_thread.detach();
}

void evaluate(const SeedNoise &sn, const std::vector<Zone> &zones, std::vector<std::vector<uint8_t>> &bitmaps) {
    if (zones.empty()) { bitmaps.clear(); return; }
    Request r{&sn, &zones, &bitmaps};
    const Octave *oc[14];
    for (int i = 0; i < 14; i++) oc[i] = i < 4 ? &sn.t[i] : i < 10 ? &sn.c[i - 4] : &sn.e[i - 10];
    for (auto &z : zones)
        for (int i = 0; i < 14; i++) {
            const Octave &o = *oc[i];
            for (int a = 0; a < z.nx; a++) r.coords.push_back(lat_host(z.x0 + 32 * a, o.k, o.side, o.ox_i, o.ox_f));
            for (int b = 0; b < z.nz; b++) r.coords.push_back(lat_host(z.z0 + 32 * b, o.k, o.side, o.oz_i, o.oz_f));
        }
    std::unique_lock<std::mutex> l(mu);
    pending.push_back(&r);
    cv.notify_all();
    cv.wait(l, [&] { return r.done; });
}
}
