NVCC     ?= nvcc
NCU      ?= ncu
ARCH     ?= sm_120
NVFLAGS  := -O2 -std=c++17 -arch=$(ARCH) -lineinfo -I../../common

TARGET   := transpose_smem

# ---- ncu 指标 ----
# global：sectors / requests，读写分开 -> 算 sectors/request（合并 = 4，完全不合并 = 32）
# shared：bank conflict 与 wavefront（理想情况 wavefronts = requests）
# occupancy：理论值 vs 实际值
METRIC_LIST := \
  gpu__time_duration.sum \
  dram__bytes_op_read.sum \
  dram__bytes_op_write.sum \
  l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
  l1tex__t_requests_pipe_lsu_mem_global_op_st.sum \
  l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum \
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum \
  sm__maximum_warps_per_active_cycle_pct \
  sm__warps_active.avg.pct_of_peak_sustained_active

# make 的反斜杠续行会变成空格，这里把空格换回逗号
empty :=
space := $(empty) $(empty)
comma := ,
METRICS := $(subst $(space),$(comma),$(strip $(METRIC_LIST)))

.PHONY: all run ncu ncu-rep clean

all: $(TARGET)

$(TARGET): transpose_smem.cu ../../common/benchmark.h
	$(NVCC) $(NVFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

# 指标表，直接看数
ncu: $(TARGET)
	$(NCU) --metrics $(METRICS) ./$(TARGET) profile

# 完整报告，拿回 Windows 用 Nsight Compute GUI 打开
ncu-rep: $(TARGET)
	$(NCU) --set full -f -o w2d2 ./$(TARGET) profile

clean:
	rm -f $(TARGET) *.ncu-rep
