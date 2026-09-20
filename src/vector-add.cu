#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t result, const char *operation) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "FAIL %s: %d %s: %s\n", operation, int(result),
                     cudaGetErrorName(result), cudaGetErrorString(result));
        std::exit(1);
    }
    std::printf("OK %s\n", operation);
}
#define CUDA(call) check((call), #call)

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int driver = 0, runtime = 0, count = 0;
    CUDA(cudaRuntimeGetVersion(&runtime));
    std::printf("CUDA runtime version: %d; compiled runtime: %d\n", runtime, CUDART_VERSION);
    CUDA(cudaDriverGetVersion(&driver));
    std::printf("CUDA driver API version: %d\n", driver);
    CUDA(cudaGetDeviceCount(&count));
    if (count < 1) {
        std::fprintf(stderr, "FAIL no CUDA devices\n");
        return 1;
    }
    // An incomplete final block exercises the kernel's bounds check.
    constexpr int n = 1048579;
    constexpr size_t bytes = size_t(n) * sizeof(float);
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = float(i % 1024) * 0.25f;
        b[i] = float(i % 257) * -0.5f;
    }
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp prop{};
        CUDA(cudaGetDeviceProperties(&prop, device));
        std::printf("Device %d: %s; SM %d.%d; total memory %zu bytes\n",
                    device, prop.name, prop.major, prop.minor, prop.totalGlobalMem);
        CUDA(cudaSetDevice(device));
        float *da = nullptr, *db = nullptr, *dc = nullptr;
        CUDA(cudaMalloc(&da, bytes));
        CUDA(cudaMalloc(&db, bytes));
        CUDA(cudaMalloc(&dc, bytes));
        CUDA(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
        CUDA(cudaMemset(dc, 0xff, bytes));
        vector_add<<<(n + 255) / 256, 256>>>(da, db, dc, n);
        CUDA(cudaGetLastError());
        CUDA(cudaDeviceSynchronize());
        CUDA(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            const float expected = a[i] + b[i];
            if (!std::isfinite(c[i]) || c[i] != expected) {
                std::fprintf(stderr, "FAIL device=%d index=%d expected=%g actual=%g\n",
                             device, i, expected, c[i]);
                return 1;
            }
        }
        CUDA(cudaFree(dc));
        CUDA(cudaFree(db));
        CUDA(cudaFree(da));
        CUDA(cudaDeviceReset());
        std::printf("PASS device=%d: verified all %d vector-add results\n", device, n);
    }
    std::puts("PASS CUDA GPU execution (allocation, copies, kernel, synchronization, verification)");
}
