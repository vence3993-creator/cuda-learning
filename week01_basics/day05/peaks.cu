#include "benchmark.h"
int main() {
    printf("=== 驱动报告（未校准）===\n");
    print_device_info();

    printf("=== 实测校准（2.87 GHz）===\n");
    print_device_info(0, 2.87);
    return 0;
}