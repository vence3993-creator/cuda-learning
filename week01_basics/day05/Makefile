NVCC   := nvcc
FLAGS  := -O3 -std=c++17 -arch=sm_120 -I../../common -lineinfo

all: peaks roofline_sweep

%: %.cu
	$(NVCC) $(FLAGS) $< -o $@

sass: roofline_sweep
	@echo "FFMA 指令数：" && cuobjdump -sass roofline_sweep | grep -c FFMA
	@echo "FMUL / FADD（应该很少）：" && cuobjdump -sass roofline_sweep | grep -cE 'FMUL|FADD'

clean:
	rm -f peaks roofline_sweep roofline.csv *.ncu-rep