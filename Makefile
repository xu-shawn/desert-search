# -ffp-contract=off is required: the standalone sampler must round exactly like Java/Cubiomes.
CC       ?= gcc
CXX      ?= g++
NVCC     ?= nvcc
OPT      ?= -O3 -march=native -ffp-contract=off
ARCH     ?= -arch=native
THREADS  ?= $(shell nproc)
MCM      := MC-Measure
CFLAGS   := $(OPT) -fwrapv -I$(MCM)/lib
CXXFLAGS := $(OPT) -std=c++20 -pthread -Isrc
LIBOBJ   := $(addprefix build/mcm_,$(addsuffix .o,biomenoise biomes generator layers noise terrainnoise))
HEADERS  := src/standalone.hpp src/worldgen_data.hpp src/surface.hpp src/dense_flood.hpp src/lowfield.hpp \
            src/desert_proxy.hpp src/desert_dtable.hpp
GPUSRC   := src/gpu/gate.cuh src/gpu/gproxy.cu src/gpu/gproxy.h
CUDA     ?= /usr/local/cuda

.PHONY: all gpu check data dtable bundle clean
all: bin/dsearch_desert bin/mcmeasure bin/validate bin/dflood bin/dfunc bin/biomemap
gpu: bin/dsearch_desert_gpu bin/tgate bin/mcmeasure

build bin:
	mkdir -p $@

# MC-Measure (git submodule) = the spreadsheet's measuring tool, and its Cubiomes copy for validation
build/mcm_%.o: $(MCM)/lib/%.c | build
	$(CC) $(CFLAGS) -c $< -o $@
bin/mcmeasure: $(MCM)/measure.c $(LIBOBJ) | bin
	sed 's/^#define THREADS .*/#define THREADS $(THREADS)/' $< > build/measure.c
	$(CC) $(CFLAGS) -I$(MCM) build/measure.c $(LIBOBJ) -lm -pthread -o $@

bin/dsearch_desert: src/dsearch_desert.cpp $(HEADERS) | bin
	$(CXX) $(CXXFLAGS) $< -o $@
bin/validate: src/validate.cpp $(HEADERS) $(LIBOBJ) | bin
	$(CXX) $(CXXFLAGS) -I$(MCM)/lib $< $(LIBOBJ) -lm -o $@
bin/%: tools/%.cpp $(HEADERS) | bin
	$(CXX) $(CXXFLAGS) $< -o $@

# GPU build (CUDA toolkit): gate + proxy + tables/scan on the GPU, one process per GPU.
build/gproxy.o: $(GPUSRC) src/desert_dtable.hpp | build
	$(NVCC) -O3 -std=c++20 $(ARCH) -c src/gpu/gproxy.cu -o $@
bin/dsearch_desert_gpu: src/dsearch_desert.cpp $(HEADERS) build/gproxy.o | bin
	$(CXX) $(CXXFLAGS) -DDRIP_GPU -I$(CUDA)/include $< build/gproxy.o -L$(CUDA)/lib64 -lcudart_static -ldl -lrt -o $@
bin/tgate: src/gpu/tgate.cu src/gpu/gate.cuh | bin
	$(NVCC) -O3 -std=c++20 $(ARCH) $< -o $@

# Portable fat binaries (sm_70..sm_90 + PTX, generic AVX2 host code) for boxes without a toolkit.
bundle:
	$(MAKE) clean && $(MAKE) gpu OPT="-O3 -march=x86-64-v3 -ffp-contract=off" THREADS=32 \
	  ARCH="-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 \
	        -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=[sm_90,compute_90]"
	tar czf dist.tgz bin/dsearch_desert_gpu bin/tgate bin/mcmeasure deploy/gpipe.sh deploy/prun.sh

# Regenerate the biome tree / spline tables from Cubiomes, and the desert-probability table.
data: bin/export_constants
	bin/export_constants > src/worldgen_data.hpp
bin/export_constants: src/export_constants.c $(LIBOBJ) | bin
	$(CC) $(CFLAGS) $< $(LIBOBJ) -lm -o $@
dtable: bin/dfunc                # fresh sample -> build/desert_dtable.hpp (copy over src/ to adopt it)
	bin/dfunc 2000 2000 $(THREADS) > build/dfunc.txt && python3 tools/gen_dtable.py build/dfunc.txt > build/desert_dtable.hpp

# Sampler vs Cubiomes at Y=256 (1M points), then the old record (70,927,696) and the 100M find measured
# by MC-Measure and by the exact standalone flood (identical cell counts).
check: bin/validate bin/mcmeasure bin/dflood
	bin/validate 10000 100
	bin/mcmeasure 1106000410148237774 7341586 -22019902
	bin/dflood 1106000410148237774 7341586 -22019902 1
	bin/mcmeasure -8125596009223775875 -22910974 2295170
	bin/dflood -8125596009223775875 -22910974 2295170 1

clean:
	rm -rf bin build dist.tgz
