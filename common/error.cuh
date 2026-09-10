#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                        \
do {                                                                       \
    const cudaError_t ec = (call);                                         \
    if (ec != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA Error %s:%d\n  %s\n  %s\n",                  \
                __FILE__, __LINE__,                                        \
                cudaGetErrorName(ec), cudaGetErrorString(ec));             \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

#define CHECK_KERNEL()                  \
do {                                    \
    CHECK(cudaGetLastError());          \
    CHECK(cudaDeviceSynchronize());     \
} while (0)
