// Standalone sampler vs Cubiomes (MC-Measure's copy) at Y=256, the height deserts are measured at:
// climate parameters and biome at random points, the surface-grid lookup, and the desert/river-bridge
// predicate used by every flood. Seed 0 samples around the old desert record.
//   validate [SEEDS=1000] [POINTS=100]
#include "surface.hpp"
#include <math.h>
#define lerp cubiomes_lerp
#include "generator.h"
#undef lerp
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    int seeds = argc > 1 ? atoi(argv[1]) : 1000, points = argc > 2 ? atoi(argv[2]) : 100;
    standalone::Random rng(20260930);
    static standalone::SurfaceGrid grid;
    Generator reference; setupGenerator(&reference, MC_NEWEST, 0);
    long climate_errors = 0, biome_errors = 0, grid_errors = 0, grid_fallbacks = 0, pred_errors = 0, deserts = 0, bridged = 0;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < seeds; i++) {
        uint64_t seed = i == 0 ? 1106000410148237774ULL : rng.next();
        applySeed(&reference, DIM_OVERWORLD, seed);
        standalone::Sampler own(seed);
        standalone::SurfaceSampler surf(seed, grid);
        for (int j = 0; j < points; j++) {
            int x = int(rng.bounded(15000000)) - 7500000, z = int(rng.bounded(15000000)) - 7500000;
            if (i == 0) { x = 7341586 / 4 + int(rng.bounded(4096)) - 2048; z = -22019902 / 4 + int(rng.bounded(4096)) - 2048; }
            int64_t a[6], b[6];
            int expected = sampleBiomeNoise(&reference.bn, a, x, 64, z, nullptr, 0);
            own.climate(x, 64, z, b);
            bool same = true;
            for (int k = 0; k < 6; k++) same &= a[k] == b[k];
            if (!same && climate_errors++ < 3) {
                fprintf(stderr, "climate seed=%lld xz=%d,%d:", (long long)seed, x, z);
                for (int k = 0; k < 6; k++) fprintf(stderr, " %lld/%lld", (long long)a[k], (long long)b[k]);
                fputc('\n', stderr);
            }
            int actual = standalone::classify(b);
            biome_errors += actual != expected;
            int g = grid.lookup(b);
            if (g < 0) grid_fallbacks++; else grid_errors += g != expected;
            // flood predicate: 2 = desert, 1 = river that is desert with weirdness pushed to +-0.0501
            int want = 0;
            if (expected == desert) want = 2;
            else if ((expected == river || expected == frozen_river) && a[5] <= 500 && a[5] >= -500) {
                int64_t c[6]; for (int k = 0; k < 6; k++) c[k] = a[k];
                c[5] = a[5] >= 0 ? 501 : -501;
                want = standalone::classify(c) == desert ? 1 : 0;
            }
            pred_errors += surf.desert_state(x, z) != want;
            deserts += want == 2; bridged += want == 1;
        }
    }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printf("{\"seeds\":%d,\"points_per_seed\":%d,\"climate_errors\":%ld,\"biome_errors\":%ld,\"grid_errors\":%ld,\"grid_fallbacks\":%ld,"
           "\"predicate_errors\":%ld,\"desert_points\":%ld,\"bridged_river_points\":%ld,\"seconds\":%.3f}\n",
           seeds, points, climate_errors, biome_errors, grid_errors, grid_fallbacks, pred_errors, deserts, bridged, sec);
    return !!(climate_errors + biome_errors + grid_errors + pred_errors);
}
