#pragma once
// Surface-biome lookup for Y=256 (quart y=64) without a tree search.
// At that height the depth parameter is always < 0 (terrain offset never exceeds ~1.4), so every
// surface entry (depth box exactly 0 or 1) pays the same depth penalty d^2 for its depth-0 copy and
// cave entries pay more. The winner is therefore the depth-0 surface box containing (T,H,C,E,W).
// Those boxes tile parameter space; we index them on the grid formed by all box boundaries.
// A grid cell covered by exactly one box maps to that biome; anything else falls back to classify().
#include "standalone.hpp"
#include <vector>

namespace standalone {
struct SurfaceGrid {
    std::vector<int32_t> cuts[5];        // per axis (T,H,C,E,W): sorted lower bounds of cells
    std::vector<int16_t> cell;           // biome id, or -1 = ambiguous/uncovered
    int dim[5] = {};
    static constexpr int AX[5] = {0, 1, 2, 3, 5};   // parameter index in q[6] (4 = depth)

    SurfaceGrid() {
        struct Box { int32_t lo[5], hi[5]; int biome; };
        std::vector<Box> boxes;
        for (size_t n = 0; n < std::size(tree); n++) {
            uint64_t e = tree[n];
            if ((e >> 56) != 255) continue;
            const auto &d = parameters[(e >> 32) & 255];              // depth is parameter slot 4
            if (!(d[0] == 0 && d[1] == 0)) continue;
            Box b; b.biome = (e >> 48) & 255;
            for (int a = 0; a < 5; a++) { const auto &r = parameters[(e >> (8 * AX[a])) & 255]; b.lo[a] = r[0]; b.hi[a] = r[1]; }
            boxes.push_back(b);
        }
        for (int a = 0; a < 5; a++) {
            auto &c = cuts[a];
            for (auto &b : boxes) { c.push_back(b.lo[a]); c.push_back(b.hi[a] + 1); }
            std::sort(c.begin(), c.end()); c.erase(std::unique(c.begin(), c.end()), c.end());
            dim[a] = int(c.size()) - 1;                               // cells [c[i], c[i+1])
        }
        size_t total = 1;
        for (int a = 0; a < 5; a++) total *= dim[a];
        cell.assign(total, -2);                                       // -2 = uncovered so far
        for (auto &b : boxes) {
            int lo[5], hi[5];
            for (int a = 0; a < 5; a++) {
                lo[a] = int(std::lower_bound(cuts[a].begin(), cuts[a].end(), b.lo[a]) - cuts[a].begin());
                hi[a] = int(std::lower_bound(cuts[a].begin(), cuts[a].end(), b.hi[a] + 1) - cuts[a].begin());
            }
            for (int i0 = lo[0]; i0 < hi[0]; i0++) for (int i1 = lo[1]; i1 < hi[1]; i1++)
            for (int i2 = lo[2]; i2 < hi[2]; i2++) for (int i3 = lo[3]; i3 < hi[3]; i3++)
            for (int i4 = lo[4]; i4 < hi[4]; i4++) {
                auto &v = cell[index(i0, i1, i2, i3, i4)];
                v = (v == -2) ? int16_t(b.biome) : (v == b.biome ? v : int16_t(-1));
            }
        }
        for (auto &v : cell) if (v == -2) v = -1;
    }
    size_t index(int i0, int i1, int i2, int i3, int i4) const {
        return ((((size_t(i0) * dim[1] + i1) * dim[2] + i2) * dim[3] + i3) * dim[4] + i4);
    }
    // Returns the biome, or -1 if the lookup cannot decide (caller must use classify()).
    int lookup(const int64_t *q) const {
        int idx[5];
        for (int a = 0; a < 5; a++) {
            int64_t v = q[AX[a]];
            const auto &c = cuts[a];
            if (v < c.front() || v >= c.back()) return -1;
            idx[a] = int(std::upper_bound(c.begin(), c.end(), int32_t(v)) - c.begin()) - 1;
            // A value on a cell's lower edge touches the neighbouring box's inclusive upper edge
            // (hi+1 == this cut): the tree resolves such ties by order, so defer to it.
            if (v == c[idx[a]]) return -1;
        }
        return cell[index(idx[0], idx[1], idx[2], idx[3], idx[4])];
    }
};

// Surface biome at quart (x, 64, z) with lazily sampled channels.
struct SurfaceSampler {
    Sampler s;
    const SurfaceGrid &g;
    SurfaceSampler(uint64_t seed, const SurfaceGrid &grid) : s(seed), g(grid) {}
    void params(int x, int z, int64_t *q) { s.climate(x, 64, z, q); }
    int biome(int x, int z) {
        int64_t q[6]; params(x, z, q);
        int b = g.lookup(q);
        return b >= 0 ? b : classify(q);
    }
    // Desert test with a temperature early-out (every desert box has T in [0.55,1]).
    bool desert(int x, int z) {
        const auto &sh = s.get(Shift);
        double px = x + 4. * sh.sample(x, 0, z), pz = z + 4. * sh.sample(z, x, 0);
        float t = s.get(Temperature).sample(px, 0, pz);
        if (int64_t(10000.f * t) < 5500) return false;
        int64_t q[6]; s.finish(px, pz, 64, s.get(Continentalness).sample(px, 0, pz), q);
        int b = g.lookup(q);
        return (b >= 0 ? b : classify(q)) == 2;
    }
    // Flood state for desert with MC-Measure bridging, sampling every noise once:
    // 2 = desert, 1 = river whose biome without the river (weirdness +-0.0501) is desert, 0 = other.
    // Depth is left at a fixed negative value: at Y=256 the true depth is always negative, which
    // cannot change the winning surface box. It is computed exactly only for a tree fallback.
    int desert_state(int x, int z) {
        const auto &sh = s.get(Shift);
        double px = x + 4. * sh.sample(x, 0, z), pz = z + 4. * sh.sample(z, x, 0);
        float t = s.get(Temperature).sample(px, 0, pz);
        if (int64_t(10000.f * t) < 5500) return 0;       // no desert box below T=0.55, for rivers too
        float c = s.get(Continentalness).sample(px, 0, pz), e = s.get(Erosion).sample(px, 0, pz);
        float w = s.get(Weirdness).sample(px, 0, pz), h = s.get(Humidity).sample(px, 0, pz);
        int64_t q[6] = {int64_t(10000.f * t), int64_t(10000.f * h), int64_t(10000.f * c), int64_t(10000.f * e), -10000, int64_t(10000.f * w)};
        bool exact_depth = false;
        auto pick = [&]() {
            int b = g.lookup(q);
            if (b >= 0) return b;
            if (!exact_depth) {
                float v[] = {c, e, -(std::fabs(std::fabs(w) - 0.6666667f) - 0.33333334f) * 3.f, w};
                double off = spline(0, v) + 0.015f;
                q[4] = int64_t(10000.f * float(1. - (64 * 4) / 128. - 83. / 160. + off));
                exact_depth = true;
            }
            return classify(q);
        };
        int b = pick();
        if (b == 2) return 2;
        if (b != 7 && b != 11) return 0;
        if (q[5] > 500 || q[5] < -500) return 0;
        q[5] = q[5] >= 0 ? 501 : -501;
        return pick() == 2 ? 1 : 0;
    }
    // MC-Measure bridging: river whose weirdness pushed to +-0.0501 would give `target`.
    bool bridge(int x, int z, int target) {
        int64_t q[6]; params(x, z, q);
        int b = g.lookup(q); if (b < 0) b = classify(q);
        if (b != 7 && b != 11) return false;
        if (q[5] > 500 || q[5] < -500) return false;
        q[5] = q[5] >= 0 ? 501 : -501;
        int u = g.lookup(q); if (u < 0) u = classify(q);
        return u == target;
    }
};
}
