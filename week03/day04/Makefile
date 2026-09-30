NVCC   := nvcc
FLAGS  := -O3 -std=c++17 -arch=sm_120 -I../../common -Xptxas -v
TARGET := softmax_warp

METRICS := gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_bytes.sum,$\
l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,launch__registers_per_thread,$\
sm__warps_active.avg.pct_of_peak_sustained_active,$\
smsp__inst_executed.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum

all: $(TARGET)

$(TARGET): softmax_warp.cu
	$(NVCC) $(FLAGS) -o $@ $<

# 每个 kernel 只跑一次，只抓 softmax 开头的 kernel
ncu: $(TARGET)
	ncu -k regex:softmax --metrics $(METRICS) ./$(TARGET) ncu

clean:
	rm -f $(TARGET)

.PHONY: all ncu clean
