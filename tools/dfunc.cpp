// P(desert-passable | low-octave C, E) at hot points (T >= 0.55), as 2D histograms.
// Output rows: set, ci, ei, total, desert, passable   (set 0: C0,E0 octave-0 only; set 1: C0-2, E0-1)
#include "../src/surface.hpp"
#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
using namespace standalone;

static double oct(const NormalNoise &N, double x, double z, int lo, int hi) {
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
constexpr int NB_C = 40, NB_E = 40;   // bins of 0.05: C in [-0.8,1.2), E in [-1,1)
static int bc(double c) { int i = int(std::floor((c + 0.8) / 0.05)); return std::clamp(i, 0, NB_C - 1); }
static int be(double e) { int i = int(std::floor((e + 1.0) / 0.05)); return std::clamp(i, 0, NB_E - 1); }

int main(int argc, char **argv) {
    int seeds = std::stoi(argv[1]), pts = std::stoi(argv[2]), threads = std::stoi(argv[3]);
    static SurfaceGrid G;
    std::vector<long> H(2 * NB_C * NB_E * 3, 0);
    std::mutex mu; std::atomic<int> next(0);
    auto work = [&]() {
        std::vector<long> h(H.size(), 0);
        while (true) {
            int k = next++;
            if (k >= seeds) break;
            uint64_t seed = mix(uint64_t(k) * 7 + 99991);
            // temperature octave-0 phase filter (as in the searcher)
            {
                Random world(seed); uint64_t a = world.next(), b = world.next();
                Random c(a ^ configs[Temperature].a, b ^ configs[Temperature].b); double s = 0;
                for (int side = 0; side < 2; side++) { uint64_t fa = c.next(), fb = c.next(); Random r(fa ^ octave_salt[0][0], fb ^ octave_salt[0][1]); r.next(); double y = r.uniform() * 256.; s += std::abs(y - std::floor(y) - .5); }
                if (s > 0.36) continue;
            }
            SurfaceSampler S(seed, G);
            const auto &T = S.s.get(Temperature), &C = S.s.get(Continentalness), &E = S.s.get(Erosion), &sh = S.s.get(Shift);
            Random rng(seed ^ 0x1234);
            int got = 0, tries = 0;
            while (got < pts && tries < pts * 200) {
                tries++;
                int x = int(rng.bounded(15000000)) - 7500000, z = int(rng.bounded(15000000)) - 7500000;
                double px = x + 4. * sh.sample(x, 0, z), pz = z + 4. * sh.sample(z, x, 0);
                if (T.sample(px, 0, pz) < 0.55) continue;
                got++;
                int64_t q[6]; S.params(x, z, q);
                int b = G.lookup(q); if (b < 0) b = classify(q);
                bool des = b == 2, pass = des || S.bridge(x, z, 2);
                double c0 = oct(C, px, pz, 0, 1), c02 = c0 + oct(C, px, pz, 1, 3), e0 = oct(E, px, pz, 0, 1), e01 = e0 + oct(E, px, pz, 1, 2);
                for (int set = 0; set < 2; set++) {
                    int i = bc(set ? c02 : c0), j = be(set ? e01 : e0);
                    long *cell = &h[((set * NB_C + i) * NB_E + j) * 3];
                    cell[0]++; cell[1] += des; cell[2] += pass;
                }
            }
        }
        std::lock_guard<std::mutex> g(mu);
        for (size_t i = 0; i < H.size(); i++) H[i] += h[i];
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) pool.emplace_back(work);
    for (auto &t : pool) t.join();
    for (int set = 0; set < 2; set++) for (int i = 0; i < NB_C; i++) for (int j = 0; j < NB_E; j++) {
        long *cell = &H[((set * NB_C + i) * NB_E + j) * 3];
        if (cell[0]) printf("%d %d %d %ld %ld %ld\n", set, i, j, cell[0], cell[1], cell[2]);
    }
}
