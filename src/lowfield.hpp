#pragma once
// Octave-0 period tables for the world scan (approximate screening only; float precision).
#include "standalone.hpp"
#include <vector>

namespace standalone {
constexpr int TN = 512;                    // table cells per 256-lattice period (0.5 lattice units)

// Tabulate w * perlin(u, 0, v) for u,v = 0, 0.5, ..., 255.5 lattice units.
// Samples of equal parity share fractional offsets, so each is a sum of 8 precomputed
// corner terms indexed by the corner's gradient hash.
// Per-octave setup shared with the GPU table kernel: hashed column rows, parity lattice origins and
// the 8 corner terms per (x parity, z parity, gradient hash).
inline void tab_setup(const Perlin &p, double w, uint8_t col[2][256], int i0[2], int k0[2], float T[2][2][8][16]) {
    const auto &P = p.permutation;
    for (int x = 0; x < 256; x++)
        for (int dy = 0; dy < 2; dy++) col[dy][x] = P[(P[x] + p.iy + dy) & 255];
    double fx[2], fz[2];
    for (int par = 0; par < 2; par++) {
        double bx = par * 0.5 + p.ox, bz = par * 0.5 + p.oz;
        i0[par] = int(std::floor(bx)); fx[par] = bx - std::floor(bx);
        k0[par] = int(std::floor(bz)); fz[par] = bz - std::floor(bz);
    }
    for (int px = 0; px < 2; px++) for (int pz = 0; pz < 2; pz++) {
        double tx = fade(fx[px]), tz = fade(fz[pz]), ty = p.vy;
        for (int c = 0; c < 8; c++) {
            int dx = c & 1, dy = (c >> 1) & 1, dz = c >> 2;
            double wc = (dx ? tx : 1 - tx) * (dy ? ty : 1 - ty) * (dz ? tz : 1 - tz);
            for (int h = 0; h < 16; h++)
                T[px][pz][c][h] = float(w * wc * gradient(h, fx[px] - dx, p.fy - dy, fz[pz] - dz));
        }
    }
}

inline void tabulate_octave(const Perlin &p, double w, std::vector<float> &t) {
    t.resize(size_t(TN) * TN);
    const auto &P = p.permutation;
    uint8_t col[2][256]; int i0[2], k0[2]; float T[2][2][8][16];
    tab_setup(p, w, col, i0, k0, T);
    for (int j = 0; j < TN; j++) {
        int pz = j & 1, iz = (j >> 1) + k0[pz];
        float *row = &t[size_t(j) * TN];
        for (int i = 0; i < TN; i++) {
            int px = i & 1, ix = (i >> 1) + i0[px];
            const auto &Tc = T[px][pz];
            float acc = 0;
            for (int c = 0; c < 8; c++) {
                int dx = c & 1, dy = (c >> 1) & 1, dz = c >> 2;
                int h = P[(col[dy][(ix + dx) & 255] + iz + dz) & 255] & 15;
                acc += Tc[c][h];
            }
            row[i] = acc;
        }
    }
}

// Cheap amplitude pre-gate: only the even/even parity samples (u,v = 0,1,...,255 lattice units),
// Gaussian-blurred with sigma = 1 lattice unit, maximum returned. Aliasing biases it high by ~0.05
// relative to the full-resolution amplitude (corr 0.92), so use it with a small margin.
inline float amplitude_sub(const Perlin &p, double w, std::vector<float> &t, std::vector<float> &tmp) {
    const auto &P = p.permutation;
    uint8_t col[2][256];
    for (int x = 0; x < 256; x++)
        for (int dy = 0; dy < 2; dy++) col[dy][x] = P[(P[x] + p.iy + dy) & 255];
    int i0 = int(std::floor(p.ox)), k0 = int(std::floor(p.oz));
    double fx = p.ox - i0, fz = p.oz - k0;
    float T[8][16];
    double tx = fade(fx), tz = fade(fz), ty = p.vy;
    for (int c = 0; c < 8; c++) {
        int dx = c & 1, dy = (c >> 1) & 1, dz = c >> 2;
        double wc = (dx ? tx : 1 - tx) * (dy ? ty : 1 - ty) * (dz ? tz : 1 - tz);
        for (int h = 0; h < 16; h++) T[c][h] = float(w * wc * gradient(h, fx - dx, p.fy - dy, fz - dz));
    }
    t.resize(256 * 256); tmp.resize(256 * 256);
    for (int j = 0; j < 256; j++) {
        int iz = j + k0;
        for (int i = 0; i < 256; i++) {
            int ix = i + i0; float acc = 0;
            for (int c = 0; c < 8; c++) {
                int dx = c & 1, dy = (c >> 1) & 1, dz = c >> 2;
                acc += T[c][P[(col[dy][(ix + dx) & 255] + iz + dz) & 255] & 15];
            }
            t[j * 256 + i] = acc;
        }
    }
    constexpr int r = 4;
    float k[2 * r + 1]; double s = 0;
    for (int d = -r; d <= r; d++) s += k[d + r] = float(std::exp(-0.5 * d * d));
    for (auto &v : k) v = float(v / s);
    for (int j = 0; j < 256; j++) for (int i = 0; i < 256; i++) {
        float a = 0;
        for (int d = -r; d <= r; d++) a += k[d + r] * t[j * 256 + ((i + d) & 255)];
        tmp[j * 256 + i] = a;
    }
    float m = -9;
    for (int j = 0; j < 256; j++) for (int i = 0; i < 256; i++) {
        float a = 0;
        for (int d = -r; d <= r; d++) a += k[d + r] * tmp[((j + d) & 255) * 256 + i];
        m = std::max(m, a);
    }
    return m;
}

// Periodic separable Gaussian blur of a TN x TN table, in place.
struct Blur {
    std::vector<float> k; int r;
    explicit Blur(double sigma) {
        r = int(3 * sigma + 1); k.resize(2 * r + 1);
        std::vector<double> kd(2 * r + 1); double s = 0;
        for (int i = -r; i <= r; i++) s += kd[i + r] = std::exp(-0.5 * i * i / (sigma * sigma));
        for (int i = 0; i <= 2 * r; i++) k[i] = float(kd[i] / s);
    }
    void operator()(std::vector<float> &a, std::vector<float> &t) const {
        const int M = TN + 2 * r;
        t.resize(size_t(M) * TN);
        std::vector<float> row(M);
        for (int j = 0; j < TN; j++) {
            for (int i = 0; i < M; i++) row[i] = a[size_t(j) * TN + ((i - r) & (TN - 1))];
            float *dst = &t[size_t(j + r) * TN];
            for (int i = 0; i < TN; i++) dst[i] = 0;
            for (int d = 0; d <= 2 * r; d++) {
                float w = k[d]; const float *src = &row[d];
                for (int i = 0; i < TN; i++) dst[i] += w * src[i];
            }
        }
        for (int j = 0; j < r; j++) {
            std::copy_n(&t[size_t(TN + j) * TN], TN, &t[size_t(j) * TN]);
            std::copy_n(&t[size_t(r + j) * TN], TN, &t[size_t(TN + r + j) * TN]);
        }
        for (int j = 0; j < TN; j++) {
            float *dst = &a[size_t(j) * TN];
            for (int i = 0; i < TN; i++) dst[i] = 0;
            for (int d = 0; d <= 2 * r; d++) {
                const float *src = &t[size_t(j + d) * TN]; float w = k[d];
                for (int i = 0; i < TN; i++) dst[i] += w * src[i];
            }
        }
    }
};
}
