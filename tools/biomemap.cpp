// Surface biome map around a block at Y=256, with the desert component containing the centre marked.
//   biomemap SEED X Z [HALF_BLOCKS=16000] [STEP_BLOCKS=16] > map.bin      (render: scripts/plot_map.py)
// Output: int32 n, then n*n bytes row-major (z outer): biome id, or 255 for the centre's desert component
// (4-connected on the map grid, bridged rivers included like MC-Measure).
#include "surface.hpp"
#include <cstdio>
#include <string>
#include <vector>
using namespace standalone;

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "biomemap SEED X Z [HALF_BLOCKS] [STEP_BLOCKS]\n"); return 2; }
    uint64_t seed = std::stoull(argv[1]);
    int X = std::stoi(argv[2]), Z = std::stoi(argv[3]);
    int half = argc > 4 ? std::stoi(argv[4]) : 16000, step = argc > 5 ? std::stoi(argv[5]) : 16;
    int n = 2 * half / step + 1;
    static SurfaceGrid grid;
    SurfaceSampler S(seed, grid);
    std::vector<uint8_t> b(size_t(n) * n), st(size_t(n) * n);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            int qx = (X - half + i * step) >> 2, qz = (Z - half + j * step) >> 2;
            b[size_t(j) * n + i] = uint8_t(S.biome(qx, qz));
            st[size_t(j) * n + i] = uint8_t(S.desert_state(qx, qz));   // 2 desert, 1 bridgeable river
        }
    std::vector<int> stack{(n / 2) * n + n / 2};
    if (st[stack[0]] != 2) { fprintf(stderr, "centre is not desert\n"); return 1; }
    std::vector<uint8_t> seen(size_t(n) * n);
    seen[stack[0]] = 1;
    while (!stack.empty()) {
        int c = stack.back(); stack.pop_back();
        if (st[c] == 2) b[c] = 255;
        int ci = c % n, cj = c / n;
        const int di[4] = {1, -1, 0, 0}, dj[4] = {0, 0, 1, -1};
        for (int k = 0; k < 4; k++) {
            int i = ci + di[k], j = cj + dj[k];
            if (i < 0 || j < 0 || i >= n || j >= n) continue;
            int d = j * n + i;
            if (!seen[d] && st[d]) { seen[d] = 1; stack.push_back(d); }
        }
    }
    fwrite(&n, 4, 1, stdout);
    fwrite(b.data(), 1, b.size(), stdout);
    return 0;
}
