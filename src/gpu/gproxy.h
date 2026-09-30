#pragma once
// GPU evaluation of the low-octave desert proxy on zone grids (step 32 quarts).
// Thread-safe and blocking: worker threads submit the zones of one seed; a GPU thread batches
// submissions from all workers into one kernel launch.
#include <cstdint>
#include <vector>
#include <functional>

namespace gproxy {
// One Perlin octave of one noise side, pre-reduced for exact integer lattice arithmetic.
struct Octave {
    uint8_t perm[256];
    int k;                 // frequency = 2^-k (side A); side B additionally scales by 337/331
    int side;              // 0 = A, 1 = B
    int ox_i, oz_i;        // floor(ox), floor(oz)
    float ox_f, oz_f;      // fractional parts
    int iy; float fy, vy;  // y = 0 slice
    float w;               // factor * weight (final noise units)
};
// T: 2 octaves x 2 sides, C: octaves 0-2 x 2 sides, E: octaves 0-1 x 2 sides (A then B within each)
struct SeedNoise { Octave t[4], c[6], e[4]; };
struct Zone { int x0, z0, nx, nz; };      // grid origin in quarts (multiple of 32), cells of 32 quarts

// ---- period tables + translation scan (dsearch_desert tables/scan stages) on the GPU
// One octave-0 period table input, prepared on the host exactly as lowfield.hpp tabulate_octave does.
struct TabIn { uint8_t perm[256]; uint8_t col[2][256]; float T[2][2][8][16]; int i0[2], k0[2]; };
struct ScanParams { float ta, tt, wc, we; };
struct RawCand { float s, t, c, e; int x, z; };
// Translation maps (ib[i*nt + t], lo[i], hi[i] for i < 512) for T and C/E, blur kernels, world lattice constants.
void scan_init(const int16_t *ibT, const int8_t *loT, const int8_t *hiT, const int16_t *ibC, int nt,
               const float *kernT, int rT, const float *kernC, int rC, int cell, int period, int kmin);
// tabs = {TA, TB, CA, CB, EA, EB}. Blocking; batched across threads. Returns false if the candidate buffer overflowed.
bool scan(const TabIn tabs[6], const ScanParams &sp, float &maxta, float &maxtb, std::vector<RawCand> &raw);

void init(int device, float cut);
// bitmaps[i] gets nx*nz bytes (row-major, z outer): 1 = proxy-passable
void evaluate(const SeedNoise &sn, const std::vector<Zone> &zones, std::vector<std::vector<uint8_t>> &bitmaps);
// In-process GPU seed gate (gate.cuh) on a low-priority stream in a background thread over seed
// indices [start, start+count): push(seed) for every passing seed (may block for backpressure), then done().
void start_gate(int device, uint64_t start, uint64_t count, float phase, float amp, float pthr,
                std::function<void(uint64_t)> push, std::function<void()> done);
}
