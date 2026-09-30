#pragma once
// Low-octave desert proxy: a point is "proxy-passable" when temperature (both of its octaves) is
// >= 0.55 and the measured P(desert or bridgeable river | C octaves 0-2, E octaves 0-1) >= cut.
// No shift noise, no biome tree: ~14 Perlin samples instead of ~52 plus classification.
#include "standalone.hpp"
#include "desert_dtable.hpp"

namespace standalone {
struct DesertProxy {
    const NormalNoise &T, &C, &E;
    float cut;
    DesertProxy(Sampler &s, float cut_ = 0.5f)
        : T(s.get(Temperature)), C(s.get(Continentalness)), E(s.get(Erosion)), cut(cut_) {}
    static double octaves(const NormalNoise &N, double x, double z, int lo, int hi) {
        double s = 0;
        for (int side = 0; side < 2; side++) {
            double sc = side ? 337.0 / 331.0 : 1.0;
            for (int k = lo; k < hi && k < N.n; k++) {
                const auto &p = N.octave[side][k];
                s += p.weight * p.sample(x * sc * p.frequency, 0, z * sc * p.frequency);
            }
        }
        return s * N.factor;
    }
    float prob(int x, int z) const {
        if (T.sample(x, 0, z) < 0.55) return 0;
        double c = octaves(C, x, z, 0, 3), e = octaves(E, x, z, 0, 2);
        int i = std::clamp(int(std::floor((c + 0.8) / 0.05)), 0, 39), j = std::clamp(int(std::floor((e + 1.0) / 0.05)), 0, 39);
        return desert_pass_prob[i][j];
    }
    // 2 = counted, 0 = boundary (the proxy has no separate river state)
    int operator()(int x, int z) const { return prob(x, z) >= cut ? 2 : 0; }
};
}
