项	值	什么时候用
Compute Capability	12.0	-arch=sm_120
SM 数量	36	算 grid size、occupancy
CUDA Cores	4608 (128/SM)	算理论算力
Warp size	32	全程
Max threads / block	1024	block 配置上限
Max threads / SM	1536	← 见下方重点
Shared mem / block	48 KB	W2 transpose、W6 sgemm tile
Shared mem / SM	100 KB	算能同时驻留几个 block
Registers / block	65536	W6 调参
L2 Cache	32 MB	访存分析（很大，注意）
显存	8 GB, 128-bit	问题规模上限
理论带宽	448 GB/s	benchmark 分母