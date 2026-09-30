// Standalone desert flood (4-connected, rivers bridged the way MC-Measure does) at Y=256.
//   dflood SEED X Z [STEP=4]
// X, Z are block coordinates; STEP is the flood lattice in quarts (1 = exact, same count as MC-Measure).
#include "dense_flood.hpp"
#include "surface.hpp"
#include <chrono>
#include <cstdio>
#include <string>
using namespace standalone;

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "dflood SEED X Z [STEP=4]\n"); return 2; }
    uint64_t seed = std::stoull(argv[1]);
    int qx = std::stoi(argv[2]) >> 2, qz = std::stoi(argv[3]) >> 2, step = argc > 4 ? std::stoi(argv[4]) : 4;
    static SurfaceGrid grid;
    SurfaceSampler S(seed, grid);
    if (S.desert_state(qx, qz) != 2) { fprintf(stderr, "(%d, %d) is not desert at Y=256\n", qx * 4, qz * 4); return 1; }
    std::vector<uint8_t> state; std::vector<int> stack;
    auto t0 = std::chrono::steady_clock::now();
    auto a = dense_flood_bridged([&](int x, int z) { return S.desert_state(x, z); }, qx, qz, step, 20000 / step, state, stack, false);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double cells = double(a.cells) * step * step;
    printf("Desert at step %d: %.0f cells (1:4) = %.0f blocks^2%s  [%.1f s]\n", step, cells, cells * 16, a.edge ? " (hit flood window edge)" : "", sec);
    return 0;
}
