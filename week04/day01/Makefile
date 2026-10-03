NVCC   := nvcc
# -Xptxas -v：把 -v 传给 ptxas（PTX → SASS 的汇编器，寄存器就是它分配的），
#             编译时打印每个 kernel 的寄存器数、栈帧、spill 字节数
FLAGS  := -O3 -std=c++17 -arch=sm_120 -I../../common -Xptxas -v

# 每行长度：make N=4096 / make ncu N=8192；不写默认 1024
N      ?= 1024
TARGET := layernorm_warp_$(N)

METRICS := gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_bytes.sum,$\
l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,launch__registers_per_thread,$\
sm__warps_active.avg.pct_of_peak_sustained_active,$\
smsp__inst_executed.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum

all: $(TARGET)

# 不同 N 编成不同文件名，改 N 一定会重新编译
$(TARGET): layernorm_warp.cu
	$(NVCC) $(FLAGS) -DNCOLS=$(N) -o $@ $<

# 正确性 + 计时
run: $(TARGET)
	./$(TARGET)

# 每个 kernel 只跑一次，只抓 layernorm 开头的 kernel
ncu: $(TARGET)
	ncu -k regex:layernorm --metrics $(METRICS) ./$(TARGET) ncu

# 按 kernel 统计 SASS 里 local memory 指令（LDL/STL）条数；没有输出 = 0 条
sass: $(TARGET)
	@cuobjdump -sass $(TARGET) | awk '/Function :/{f=$$3} /LDL|STL/{c[f]++} END{for(k in c) print c[k], "条 LDL/STL :", k}' | c++filt

# 一次编译 1024 / 4096 / 8192，只看寄存器和 spill
regs:
	@for n in 1024 4096 8192; do \
	    echo "======== N=$$n ========"; \
	    $(NVCC) $(FLAGS) -DNCOLS=$$n -o layernorm_warp_$$n layernorm_warp.cu 2>&1 \
	        | c++filt | grep -E "Compiling entry|registers|spill"; \
	done

clean:
	rm -f layernorm_warp_*

.PHONY: all run ncu sass regs clean