# week03/day01/Makefile
NVCC   ?= nvcc
ARCH   ?= sm_120
FLAGS  := -O2 -std=c++17 -arch=$(ARCH) -I../../common
TARGET := reduce_basic

# ncu 要看的指标：分支统一率 + shared 读/写 bank conflict
METRICS := smsp__sass_average_branch_targets_threads_uniform.pct,$\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,$\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum

.PHONY: all run ncu ncu-full clean

all: $(TARGET)

$(TARGET): reduce_basic.cu ../../common/benchmark.h
	$(NVCC) $(FLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

# --once：每个 kernel 只跑一次正确性，ncu 正好抓这 4 次
ncu: $(TARGET)
	ncu -k regex:reduce_v -c 4 --section SourceCounters --metrics $(METRICS) ./$(TARGET) --once

# 全量报告，拿到 Windows 的 Nsight Compute 里看
ncu-full: $(TARGET)
	ncu --set full -k regex:reduce_v -c 4 -f -o reduce_basic_report ./$(TARGET) --once

clean:
	rm -f $(TARGET) *.ncu-rep
