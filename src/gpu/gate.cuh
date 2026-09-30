#pragma once
// GPU seed gate (kernels + host loop), shared by the standalone tgate tool and the in-process gate of
// dsearch_desert_gpu: temperature octave-0 phase filter -> anti-aliased 256^2 amplitude pre-gate ->
// full smoothed 512^2 period tables and the amplitude gate.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>

#ifndef CK
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#endif

namespace tg {
// ---------------------------------------------------------------- RNG (bit-identical to standalone.hpp)
__host__ __device__ inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
__host__ __device__ inline uint64_t mix64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
struct Rng {
    uint64_t a, b;
    __host__ __device__ static Rng from_seed(uint64_t seed) {
        seed ^= 0x6a09e667f3bcc909ULL;
        return {mix64(seed), mix64(seed + 0x9e3779b97f4a7c15ULL)};
    }
    __host__ __device__ uint64_t next() {
        uint64_t r = rotl64(a + b, 17) + a;
        b ^= a; a = rotl64(a, 49) ^ b ^ (b << 21); b = rotl64(b, 28);
        return r;
    }
    __host__ __device__ double uniform() { return (next() >> 11) * 0x1p-53; }
    __host__ __device__ uint32_t bounded(uint32_t n) {
        uint64_t r = uint64_t(uint32_t(next())) * n;
        if (uint32_t(r) < n) while (uint32_t(r) < uint32_t(-n) % n) r = uint64_t(uint32_t(next())) * n;
        return uint32_t(r >> 32);
    }
};
// minecraft:temperature fork hash and octave_-10 salt (from standalone.hpp)
constexpr uint64_t T_A = 0x5c7e6b29735f0d7fULL, T_B = 0xf7d86f1bbc734988ULL;
constexpr uint64_t S10_A = 0x36d326eed40efeb2ULL, S10_B = 0x5be9ce18223c636aULL;
// octave-0 weight in final temperature units: factor 1.25 * (32/63) * amplitude 1.5
constexpr double T_W0 = 1.25 * (32.0 / 63.0) * 1.5;

struct PerlinParams { uint8_t perm[256]; double ox, oy, oz, fy, vy; int iy; };

__host__ __device__ inline double fade(double t) { return t * t * t * (t * (t * 6. - 15.) + 10.); }
__host__ __device__ inline double gradd(unsigned h, double x, double y, double z) {
    const double gx[16] = {1,-1,1,-1,1,-1,1,-1,0,0,0,0,1,0,-1,0};
    const double gy[16] = {1,1,-1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1};
    const double gz[16] = {0,0,0,0,1,1,-1,-1,1,1,-1,-1,0,1,0,-1};
    h &= 15; return gx[h] * x + gy[h] * y + gz[h] * z;
}
// Temperature octave-0 Perlin for one side, plus that side's octave-0 y fraction.
__host__ __device__ inline void init_side(uint64_t seed, int side, PerlinParams &p) {
    Rng w = Rng::from_seed(seed);
    uint64_t a = w.next(), b = w.next();
    Rng c{a ^ T_A, b ^ T_B};
    uint64_t fa = 0, fb = 0;
    for (int s = 0; s <= side; s++) { fa = c.next(); fb = c.next(); }
    Rng r{fa ^ S10_A, fb ^ S10_B};
    p.ox = r.uniform() * 256; p.oy = r.uniform() * 256; p.oz = r.uniform() * 256;
    for (int i = 0; i < 256; i++) p.perm[i] = uint8_t(i);
    for (int i = 0; i < 256; i++) { int j = i + int(r.bounded(256 - i)); uint8_t t = p.perm[i]; p.perm[i] = p.perm[j]; p.perm[j] = t; }
    p.iy = int(floor(p.oy)); p.fy = p.oy - floor(p.oy); p.vy = fade(p.fy);
}
__host__ __device__ inline double phase_of(uint64_t seed) {
    Rng w = Rng::from_seed(seed);
    uint64_t a = w.next(), b = w.next();
    Rng c{a ^ T_A, b ^ T_B};
    double s = 0;
    for (int side = 0; side < 2; side++) {
        uint64_t fa = c.next(), fb = c.next();
        Rng r{fa ^ S10_A, fb ^ S10_B};
        r.next();
        double y = r.uniform() * 256.;
        s += fabs((y - floor(y)) - .5);
    }
    return s;
}

// ---------------------------------------------------------------- kernels
__global__ void k_phase(uint64_t start, uint64_t count, float maxph, uint64_t *out, unsigned *nout, unsigned cap) {
    uint64_t k = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) return;
    uint64_t seed = mix64(start + k);
    if (phase_of(seed) <= maxph) { unsigned i = atomicAdd(nout, 1u); if (i < cap) out[i] = seed; }
}
__global__ void k_init(const uint64_t *seeds, unsigned n, PerlinParams *pp) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * n) return;
    init_side(seeds[i >> 1], i & 1, pp[i]);
}

// Tiled "tabulate + separable Gaussian blur + max" over one side's period grid.
//  G = grid size per axis (256: one parity, lattice spacing 1; 512: all parities, spacing 1/2)
//  R = blur radius in grid cells, SIG = sigma in grid cells.
template <int G, int R>
__device__ float tile_max(const PerlinParams &p, float sig, float w, float (*V)[32 + 2 * R], float (*H)[32], float *kern, float *Tc,
                          uint8_t *P, uint8_t (*col)[256], float *red) {
    const int tid = threadIdx.x, nt = blockDim.x;
    for (int i = tid; i < 256; i += nt) P[i] = p.perm[i];
    __syncthreads();
    for (int i = tid; i < 256; i += nt) { col[0][i] = P[(P[i] + p.iy) & 255]; col[1][i] = P[(P[i] + p.iy + 1) & 255]; }
    if (tid <= 2 * R) { float d = float(tid - R); kern[tid] = expf(-0.5f * d * d / (sig * sig)); }
    // corner weight tables per parity: Tc[(px*2+pz)*128 + c*16 + h]
    const int NP = (G == 512) ? 2 : 1;
    for (int e = tid; e < NP * NP * 128; e += nt) {
        int par = e / 128, c = (e / 16) & 7, h = e & 15, px = par / 2, pz = par & 1;
        double bx = (G == 512 ? px * 0.5 : 0.0) + p.ox, bz = (G == 512 ? pz * 0.5 : 0.0) + p.oz;
        double fx = bx - floor(bx), fz = bz - floor(bz), tx = fade(fx), tz = fade(fz), ty = p.vy;
        int dx = c & 1, dy = (c >> 1) & 1, dz = c >> 2;
        double wc = (dx ? tx : 1 - tx) * (dy ? ty : 1 - ty) * (dz ? tz : 1 - tz);
        Tc[e] = float(w * wc * gradd(h, fx - dx, p.fy - dy, fz - dz));
    }
    __syncthreads();
    if (tid == 0) { float s = 0; for (int i = 0; i <= 2 * R; i++) s += kern[i]; red[0] = s; }
    __syncthreads();
    float ksum = red[0];
    float best = -1e9f;
    int ix0[2], iz0[2];
    for (int par = 0; par < 2; par++) {
        double bx = (G == 512 ? par * 0.5 : 0.0) + p.ox, bz = (G == 512 ? par * 0.5 : 0.0) + p.oz;
        ix0[par] = int(floor(bx)); iz0[par] = int(floor(bz));
    }
    const int W = 32 + 2 * R;
    for (int tz0 = 0; tz0 < G; tz0 += 32) for (int tx0 = 0; tx0 < G; tx0 += 32) {
        // samples for the tile plus halo (periodic)
        for (int e = tid; e < W * W; e += nt) {
            int a = e % W, b = e / W;
            int gi = (tx0 + a - R) & (G - 1), gj = (tz0 + b - R) & (G - 1);
            int px = (G == 512) ? (gi & 1) : 0, pz = (G == 512) ? (gj & 1) : 0;
            int ix = ((G == 512) ? (gi >> 1) : gi) + ix0[px], iz = ((G == 512) ? (gj >> 1) : gj) + iz0[pz];
            const float *T = Tc + (px * 2 + pz) * 128;
            float acc = 0;
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                int dx = c & 1, dy = (c >> 1) & 1, dz = c >> 2;
                acc += T[c * 16 + (P[(col[dy][(ix + dx) & 255] + iz + dz) & 255] & 15)];
            }
            V[b][a] = acc;
        }
        __syncthreads();
        for (int e = tid; e < W * 32; e += nt) {
            int a = e % 32, b = e / 32; float s = 0;
            for (int d = 0; d <= 2 * R; d++) s += kern[d] * V[b][a + d];
            H[b][a] = s;
        }
        __syncthreads();
        for (int e = tid; e < 32 * 32; e += nt) {
            int a = e % 32, b = e / 32; float s = 0;
            for (int d = 0; d <= 2 * R; d++) s += kern[d] * H[b + d][a];
            best = fmaxf(best, s / (ksum * ksum));
        }
        __syncthreads();
    }
    // block max
    for (int o = 16; o > 0; o >>= 1) best = fmaxf(best, __shfl_xor_sync(0xffffffff, best, o));
    if ((tid & 31) == 0) red[1 + tid / 32] = best;
    __syncthreads();
    if (tid == 0) { float m = red[1]; for (int i = 1; i < nt / 32; i++) m = fmaxf(m, red[1 + i]); red[0] = m; }
    __syncthreads();
    float m = red[0];
    __syncthreads();
    return m;
}

template <int G, int R>
__global__ void k_amp(const uint64_t *seeds, const PerlinParams *pp, unsigned n, float sig, float thr, uint64_t *out, float *outamp,
                      unsigned *nout, unsigned cap) {
    __shared__ float V[32 + 2 * R][32 + 2 * R];
    __shared__ float H[32 + 2 * R][32];
    __shared__ float kern[2 * R + 1];
    __shared__ float Tc[4 * 128];
    __shared__ uint8_t P[256];
    __shared__ uint8_t col[2][256];
    __shared__ float red[33];
    for (unsigned s = blockIdx.x; s < n; s += gridDim.x) {
        float amp = 0;
        for (int side = 0; side < 2; side++)
            amp += tile_max<G, R>(pp[2 * s + side], sig, float(T_W0), V, H, kern, Tc, P, col, red);
        if (threadIdx.x == 0 && amp >= thr) {
            unsigned i = atomicAdd(nout, 1u);
            if (i < cap) { out[i] = seeds[s]; if (outamp) outamp[i] = amp; }
        }
    }
}


// Streaming version: grid rows are produced in blocks of RB rows. Corner hashes are computed once
// per (row, lattice column); horizontally blurred rows go into a ring of NR = RB + 2R rows; each
// completed output row is blurred vertically and reduced to the maximum. No halo is recomputed.
// Dynamic shared memory layout: Hb[NR][G] floats | Vb[RB][G] floats | hx[RB][4][256] bytes
template <int G, int R, int RB, bool BOX = false>
constexpr size_t dyn_bytes() { return size_t(RB + 2 * R) * G * 4 + size_t(RB) * G * 4 + (BOX ? size_t(RB + 2) * 512 : size_t(RB) * 1024); }

// BOX (G = 256 only): each grid value is the mean of the 4 half-lattice parities of that lattice
// cell, an anti-aliasing prefilter that makes the 256^2 estimate track the 512^2 amplitude.
// The 4 samples share lattice corners, so their weights are merged into one table per corner
// offset (u, v) in {0,1,2}^2 and dy: Tb[((u*3+v)*2+dy)*16 + h], 18 lookups per cell.
template <int G, int R, int RB, bool BOX>
__device__ float stream_max(const PerlinParams &p, float sig, float w, float *dyn, float *kern, float *Tc,
                            uint8_t *P, uint8_t (*col)[256], float *red) {
    constexpr int NR = RB + 2 * R;
    const int tid = threadIdx.x, nt = blockDim.x;
    float *Hb = dyn, *Vb = dyn + NR * G;
    uint8_t *hx = reinterpret_cast<uint8_t *>(Vb + RB * G);
    constexpr bool HALF = (G == 512) || BOX;                 // tables for all 4 parities
    for (int i = tid; i < 256; i += nt) P[i] = p.perm[i];
    __syncthreads();
    for (int i = tid; i < 256; i += nt) { col[0][i] = P[(P[i] + p.iy) & 255]; col[1][i] = P[(P[i] + p.iy + 1) & 255]; }
    if (tid <= 2 * R) { float d = float(tid - R); kern[tid] = expf(-0.5f * d * d / (sig * sig)); }
    const int NP = HALF ? 2 : 1;
    if constexpr (BOX) {
        double fr[2][2]; int sh[2][2];                       // [axis][parity]: fraction, lattice shift vs parity 0
        for (int ax = 0; ax < 2; ax++) {
            double o = ax ? p.oz : p.ox;
            for (int q = 0; q < 2; q++) { double b = o + 0.5 * q; fr[ax][q] = b - floor(b); sh[ax][q] = int(floor(b) - floor(o)); }
        }
        for (int e = tid; e < 288; e += nt) {
            int h = e & 15, dy = (e >> 4) & 1, uv = e >> 5, u = uv / 3, v = uv % 3;
            double acc = 0, ty = p.vy;
            for (int px = 0; px < 2; px++) for (int pz = 0; pz < 2; pz++) {
                int dx = u - sh[0][px], dz = v - sh[1][pz];
                if (dx < 0 || dx > 1 || dz < 0 || dz > 1) continue;
                double fx = fr[0][px], fz = fr[1][pz], tx = fade(fx), tz = fade(fz);
                double wc = (dx ? tx : 1 - tx) * (dy ? ty : 1 - ty) * (dz ? tz : 1 - tz);
                acc += wc * gradd(h, fx - dx, p.fy - dy, fz - dz);
            }
            Tc[e] = float(0.25 * w * acc);
        }
    } else
    for (int e = tid; e < NP * NP * 128; e += nt) {
        int par = e / 128, c = (e / 16) & 7, h = e & 15, px = par / 2, pz = par & 1;
        double bx = (HALF ? px * 0.5 : 0.0) + p.ox, bz = (HALF ? pz * 0.5 : 0.0) + p.oz;
        double fx = bx - floor(bx), fz = bz - floor(bz), tx = fade(fx), tz = fade(fz), ty = p.vy;
        int dx = c & 1, dy = (c >> 1) & 1, dz = c >> 2;
        double wc = (dx ? tx : 1 - tx) * (dy ? ty : 1 - ty) * (dz ? tz : 1 - tz);
        Tc[e] = float((BOX ? 0.25 : 1.0) * w * wc * gradd(h, fx - dx, p.fy - dy, fz - dz));
    }
    __syncthreads();
    if (tid == 0) { float s = 0; for (int i = 0; i <= 2 * R; i++) s += kern[i]; red[0] = s; }
    __syncthreads();
    const float norm = 1.f / (red[0] * red[0]);
    int ix0[2], iz0[2];
    for (int par = 0; par < 2; par++) {
        double bx = (HALF ? par * 0.5 : 0.0) + p.ox, bz = (HALF ? par * 0.5 : 0.0) + p.oz;
        ix0[par] = int(floor(bx)); iz0[par] = int(floor(bz));
    }
    // grid rows r0 .. r0+nr-1 (r0 >= -R, periodic in the grid) -> ring slots (r + R) % NR
    auto rows_VH = [&](int r0, int nr) {
        if constexpr (BOX) {                                         // hashes per (z row, dy, lattice column)
            for (int e = tid; e < (nr + 2) * 512; e += nt) {
                int k = e >> 9, dy = (e >> 8) & 1, c = e & 255, iz = r0 + k + iz0[0];
                hx[e] = P[(col[dy][c] + iz) & 255] & 15;
            }
        } else
        for (int e = tid; e < nr * 256; e += nt) {                  // corner hashes per (row, lattice column)
            int k = e >> 8, c = e & 255, j = (r0 + k) & (G - 1);
            int pz = (G == 512) ? (j & 1) : 0, iz = ((G == 512) ? (j >> 1) : j) + iz0[pz];
            uint8_t *hk = hx + k * 1024;
            #pragma unroll
            for (int q = 0; q < 4; q++) hk[q * 256 + c] = P[(col[q >> 1][c] + iz + (q & 1)) & 255] & 15;
        }
        __syncthreads();
        for (int e = tid; e < nr * G; e += nt) {                    // samples
            int k = e / G, i = e % G, j = (r0 + k) & (G - 1);
            float acc = 0;
            if constexpr (BOX) {
                #pragma unroll
                for (int v = 0; v < 3; v++)
                    #pragma unroll
                    for (int dy = 0; dy < 2; dy++) {
                        const uint8_t *hr = hx + (k + v) * 512 + dy * 256;
                        #pragma unroll
                        for (int u = 0; u < 3; u++) acc += Tc[((u * 3 + v) * 2 + dy) * 16 + hr[(i + u + ix0[0]) & 255]];
                    }
            } else {
                int px = (G == 512) ? (i & 1) : 0, pz = (G == 512) ? (j & 1) : 0;
                int ix = ((G == 512) ? (i >> 1) : i) + ix0[px];
                const float *T = Tc + (px * 2 + pz) * 128;
                const uint8_t *hk = hx + k * 1024;
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    int dx = c & 1, q = ((c >> 1) & 1) * 2 + (c >> 2);  // q = dy*2 + dz
                    acc += T[c * 16 + hk[q * 256 + ((ix + dx) & 255)]];
                }
            }
            Vb[k * G + i] = acc;
        }
        __syncthreads();
        for (int e = tid; e < nr * G; e += nt) {                    // horizontal blur (periodic)
            int k = e / G, i = e % G; float s = 0;
            #pragma unroll
            for (int d = 0; d <= 2 * R; d++) s += kern[d] * Vb[k * G + ((i + d - R) & (G - 1))];
            Hb[((r0 + k + R) % NR) * G + i] = s;
        }
        __syncthreads();
    };
    float best = -1e9f;
    for (int k0 = 0; k0 < 2 * R; k0 += RB) rows_VH(-R + k0, min(RB, 2 * R - k0));   // rows -R .. R-1
    for (int b0 = 0; b0 < G; b0 += RB) {
        rows_VH(b0 + R, RB);                                         // rows b0+R .. b0+R+RB-1
        for (int e = tid; e < RB * G; e += nt) {                    // vertical blur of output rows b0 .. b0+RB-1
            int k = e / G, i = e % G, slot = (b0 + k) % NR;          // slot of row b0+k-R
            float s = 0;
            #pragma unroll
            for (int d = 0; d <= 2 * R; d++) { s += kern[d] * Hb[slot * G + i]; slot = (slot + 1 == NR) ? 0 : slot + 1; }
            best = fmaxf(best, s * norm);
        }
        // the next rows_VH only overwrites slots of rows < b0+RB-R, which this pass has finished with
        // after the barrier at the top of rows_VH
    }
    for (int o = 16; o > 0; o >>= 1) best = fmaxf(best, __shfl_xor_sync(0xffffffff, best, o));
    if ((tid & 31) == 0) red[1 + tid / 32] = best;
    __syncthreads();
    if (tid == 0) { float m = red[1]; for (int i = 1; i < nt / 32; i++) m = fmaxf(m, red[1 + i]); red[0] = m; }
    __syncthreads();
    float m = red[0];
    __syncthreads();
    return m;
}

template <int G, int R, int RB, bool BOX = false>
__global__ void k_amp2(const uint64_t *seeds, const PerlinParams *pp, unsigned n, float sig, float thr, uint64_t *out, float *outamp,
                       unsigned *nout, unsigned cap) {
    extern __shared__ float dyn[];
    __shared__ float kern[2 * R + 1];
    __shared__ float Tc[4 * 128];
    __shared__ uint8_t P[256];
    __shared__ uint8_t col[2][256];
    __shared__ float red[33];
    for (unsigned s = blockIdx.x; s < n; s += gridDim.x) {
        float amp = 0;
        for (int side = 0; side < 2; side++)
            amp += stream_max<G, R, RB, BOX>(pp[2 * s + side], sig, float(T_W0), dyn, kern, Tc, P, col, red);
        if (threadIdx.x == 0 && amp >= thr) {
            unsigned i = atomicAdd(nout, 1u);
            if (i < cap) { out[i] = seeds[s]; if (outamp) outamp[i] = amp; }
        }
    }
}


struct GateCfg {
    float maxph = 0.10f, amp = 0.70f;
    int box = 1; float pthr = 0.645f, sig = 1.0f;   // box=1: anti-aliased pre-gate with absolute threshold pthr
    float margin = 0.01f;                           // box=0: legacy pre-gate at amp - margin
    bool per_seed_blocks = false;                   // one short block per seed, so higher-priority streams interleave
    int tpb_pre = 256, tpb_full = 256;              // threads per block of the two amplitude kernels
};
template <int G, int R, int RB, bool BOX>
static void set_smem() { CK(cudaFuncSetAttribute(k_amp2<G, R, RB, BOX>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(dyn_bytes<G, R, RB, BOX>()))); }

// Seed indices [start, start + count) on stream st; emit(seed, amp) for each seed passing the gate,
// stop() polled between batches. Progress lines go to stderr.
template <class Emit, class Stop>
void run_gate(cudaStream_t st, uint64_t start, uint64_t count, const GateCfg &g, Emit emit, Stop stop) {
    const uint64_t B = 1ull << 26;                         // seed indices per batch
    const unsigned CAP1 = 1u << 22, CAP2 = 1u << 20, CAP3 = 1u << 16;
    uint64_t *d_s1, *d_s2, *d_s3; float *d_a3; PerlinParams *d_p1, *d_p2; unsigned *d_n, *h_n;
    CK(cudaMalloc(&d_s1, CAP1 * 8)); CK(cudaMalloc(&d_s2, CAP2 * 8)); CK(cudaMalloc(&d_s3, CAP3 * 8)); CK(cudaMalloc(&d_a3, CAP3 * 4));
    CK(cudaMalloc(&d_p1, size_t(CAP1) * 2 * sizeof(PerlinParams))); CK(cudaMalloc(&d_p2, size_t(CAP2) * 2 * sizeof(PerlinParams)));
    CK(cudaMalloc(&d_n, 3 * sizeof(unsigned))); CK(cudaMallocHost(&h_n, 3 * sizeof(unsigned)));
    uint64_t *h_s; float *h_a; CK(cudaMallocHost(&h_s, CAP3 * 8)); CK(cudaMallocHost(&h_a, CAP3 * 4));
    // the full-table kernel wants ~68 KB of shared memory per block; Turing and older allow 64 KB (48 KB on
    // some parts), so use half-height row blocks there
    int dev, smem_optin; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    const bool small_smem = size_t(smem_optin) < dyn_bytes<512, 7, 8>() + 4096;
    set_smem<256, 4, 8, true>();
    if (small_smem) set_smem<512, 7, 4, false>(); else set_smem<512, 7, 8, false>();
    if (!g.box) set_smem<256, 4, 16, false>();
    cudaEvent_t ev[4]; for (auto &e : ev) CK(cudaEventCreate(&e));
    double ms_pre = 0, ms_full = 0;
    auto grid = [&](unsigned n) { return g.per_seed_blocks ? n : std::min(n, 4096u); };
    auto t0 = std::chrono::steady_clock::now();
    uint64_t tot1 = 0, tot2 = 0, tot3 = 0;
    for (uint64_t base = 0; base < count && !stop(); base += B) {
        uint64_t nb = std::min(B, count - base);
        CK(cudaMemsetAsync(d_n, 0, 3 * sizeof(unsigned), st));
        k_phase<<<unsigned((nb + 255) / 256), 256, 0, st>>>(start + base, nb, g.maxph, d_s1, d_n, CAP1);
        CK(cudaMemcpyAsync(h_n, d_n, 3 * sizeof(unsigned), cudaMemcpyDeviceToHost, st)); CK(cudaStreamSynchronize(st));
        unsigned n[3] = {std::min(h_n[0], CAP1), 0, 0};
        if (n[0]) {
            k_init<<<(2 * n[0] + 255) / 256, 256, 0, st>>>(d_s1, n[0], d_p1);
            CK(cudaEventRecord(ev[0], st));
            if (g.box) k_amp2<256, 4, 8, true><<<grid(n[0]), g.tpb_pre, dyn_bytes<256, 4, 8, true>(), st>>>(d_s1, d_p1, n[0], g.sig, g.pthr, d_s2, nullptr, d_n + 1, CAP2);
            else k_amp2<256, 4, 16><<<grid(n[0]), 256, dyn_bytes<256, 4, 16>(), st>>>(d_s1, d_p1, n[0], 1.0f, g.amp - g.margin, d_s2, nullptr, d_n + 1, CAP2);
            CK(cudaEventRecord(ev[1], st));
            CK(cudaMemcpyAsync(h_n, d_n, 3 * sizeof(unsigned), cudaMemcpyDeviceToHost, st)); CK(cudaStreamSynchronize(st));
            n[1] = std::min(h_n[1], CAP2);
            float t; CK(cudaEventElapsedTime(&t, ev[0], ev[1])); ms_pre += t;
            if (n[1]) {
                k_init<<<(2 * n[1] + 255) / 256, 256, 0, st>>>(d_s2, n[1], d_p2);
                CK(cudaEventRecord(ev[2], st));
                if (small_smem) k_amp2<512, 7, 4><<<grid(n[1]), g.tpb_full, dyn_bytes<512, 7, 4>(), st>>>(d_s2, d_p2, n[1], 2.0f, g.amp, d_s3, d_a3, d_n + 2, CAP3);
                else k_amp2<512, 7, 8><<<grid(n[1]), g.tpb_full, dyn_bytes<512, 7, 8>(), st>>>(d_s2, d_p2, n[1], 2.0f, g.amp, d_s3, d_a3, d_n + 2, CAP3);
                CK(cudaEventRecord(ev[3], st));
                CK(cudaMemcpyAsync(h_n, d_n, 3 * sizeof(unsigned), cudaMemcpyDeviceToHost, st)); CK(cudaStreamSynchronize(st));
                CK(cudaEventElapsedTime(&t, ev[2], ev[3])); ms_full += t;
                n[2] = std::min(h_n[2], CAP3);
                CK(cudaMemcpyAsync(h_s, d_s3, n[2] * 8, cudaMemcpyDeviceToHost, st)); CK(cudaMemcpyAsync(h_a, d_a3, n[2] * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
            }
        }
        for (unsigned i = 0; i < n[2]; i++) emit(h_s[i], h_a[i]);
        tot1 += n[0]; tot2 += n[1]; tot3 += n[2];
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "gate: %llu seeds, %llu phase-passed, %llu pre-gate, %llu passed; %.0f seeds/s, %.1f passed/s; kernel s pre %.1f full %.1f\n",
                (unsigned long long)(base + nb), (unsigned long long)tot1, (unsigned long long)tot2, (unsigned long long)tot3,
                (base + nb) / sec, tot3 / sec, ms_pre / 1e3, ms_full / 1e3);
    }
    cudaFree(d_s1); cudaFree(d_s2); cudaFree(d_s3); cudaFree(d_a3); cudaFree(d_p1); cudaFree(d_p2); cudaFree(d_n);
    cudaFreeHost(h_n); cudaFreeHost(h_s); cudaFreeHost(h_a);
}
}  // namespace tg
