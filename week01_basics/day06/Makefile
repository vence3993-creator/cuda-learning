# ============================================================================
# week01_basics/day06/Makefile
# ============================================================================
NVCC    := nvcc
COMMON  := ../../common
# benchmark.h 所在目录
TARGET  := transpose

# -arch=native : CUDA 11.5+ 支持，自动按本机 GPU 生成，不用手填 sm_XX
# -lineinfo    : 必须。没有它 ncu 无法把 Uncoalesced 警告定位到源码行
# -O3          : 打开优化，否则测的是编译器摸鱼后的结果
NVCCFLAGS := -O3 -std=c++17 -arch=native -lineinfo -I$(COMMON)

$(TARGET): $(TARGET).cu $(COMMON)/benchmark.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

# ---------------------------------------------------------------------------
# ncu 采集。用 profile 模式 —— 每个 kernel 只 launch 一次。
# 直接 profile bench 模式会让 ncu replay 几千次，跑到天黑。
# ---------------------------------------------------------------------------
results:
	mkdir -p results

# 全量报告，之后用 ncu-ui 打开看规则和 Source Counters
ncu-full: $(TARGET) results
	ncu --set full -f -o results/day06_full ./$(TARGET) profile

# 今天的主角指标，直接打在终端
ncu-metrics: $(TARGET)
	ncu --metrics \
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,\
dram__bytes_op_read.sum,\
dram__bytes_op_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sectors_op_write.sum \
	./$(TARGET) profile

clean:
	rm -f $(TARGET)
	rm -rf results

.PHONY: run ncu-full ncu-metrics clean results