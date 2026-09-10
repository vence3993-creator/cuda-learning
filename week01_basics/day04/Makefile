NVCC    ?= nvcc
# -I 告诉编译器去哪找 benchmark.h。.h 文件本身不参与编译。
NVCCFLAGS ?= -O2 -std=c++17 -I../../common
# 按自己的卡改，比如 3090 是 sm_86、4090 是 sm_89、H100 是 sm_90
ARCH ?= -arch=sm_120

TARGETS = vecadd

all: $(TARGETS)

vecadd: vecadd.cu ../../common/benchmark.h
	$(NVCC) $(NVCCFLAGS) $(ARCH) $< -o $@

# ncu 必须在大 N 下跑，否则 DRAM 字节数会因为命中 L2 而偏小
profile: vecadd
	ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,\
dram__bytes.sum.per_second,gpu__time_duration.sum ./vecadd

clean:
	rm -f $(TARGETS)

.PHONY: all profile clean
