#include <cuda.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

template <typename T> T symbol(void *library, const char *name) {
    dlerror();
    void *p = dlsym(library, name);
    const char *error = dlerror();
    if (error) {
        std::fprintf(stderr, "FAIL dlsym(%s): %s\n", name, error);
        std::exit(1);
    }
    return reinterpret_cast<T>(p);
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("Driver API probe: uid=%u gid=%u, compiled CUDA headers=%d\n",
                getuid(), getgid(), CUDA_VERSION);
    void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        std::fprintf(stderr, "FAIL dlopen(libcuda.so.1): %s\n", dlerror());
        return 1;
    }
    auto init = symbol<decltype(&cuInit)>(lib, "cuInit");
    auto error_name = symbol<decltype(&cuGetErrorName)>(lib, "cuGetErrorName");
    auto error_string = symbol<decltype(&cuGetErrorString)>(lib, "cuGetErrorString");
    auto check = [&](CUresult result, const char *operation) {
        if (result != CUDA_SUCCESS) {
            const char *name = "unknown", *description = "unknown";
            error_name(result, &name);
            error_string(result, &description);
            std::fprintf(stderr, "FAIL %s: %d %s: %s\n", operation,
                         static_cast<int>(result), name, description);
            std::exit(1);
        }
        std::printf("OK %s\n", operation);
    };
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(init), &info))
        std::printf("Loaded cuInit from: %s\n", info.dli_fname);
    auto version = symbol<decltype(&cuDriverGetVersion)>(lib, "cuDriverGetVersion");
    int driver_version = 0;
    check(version(&driver_version), "cuDriverGetVersion");
    std::printf("CUDA driver API version: %d (%d.%d)\n", driver_version,
                driver_version / 1000, (driver_version % 1000) / 10);
    check(init(0), "cuInit");
    auto count_fn = symbol<decltype(&cuDeviceGetCount)>(lib, "cuDeviceGetCount");
    auto device_fn = symbol<decltype(&cuDeviceGet)>(lib, "cuDeviceGet");
    auto name_fn = symbol<decltype(&cuDeviceGetName)>(lib, "cuDeviceGetName");
    auto attr_fn = symbol<decltype(&cuDeviceGetAttribute)>(lib, "cuDeviceGetAttribute");
    auto retain = symbol<decltype(&cuDevicePrimaryCtxRetain)>(lib, "cuDevicePrimaryCtxRetain");
    auto release = symbol<decltype(&cuDevicePrimaryCtxRelease)>(lib, "cuDevicePrimaryCtxRelease_v2");
    int count = 0;
    check(count_fn(&count), "cuDeviceGetCount");
    std::printf("CUDA devices: %d\n", count);
    if (count == 0) {
        std::fprintf(stderr, "FAIL no CUDA devices\n");
        return 1;
    }
    for (int i = 0; i < count; ++i) {
        CUdevice device;
        char name[256]{};
        int major = 0, minor = 0;
        check(device_fn(&device, i), "cuDeviceGet");
        check(name_fn(name, sizeof(name), device), "cuDeviceGetName");
        check(attr_fn(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "compute major");
        check(attr_fn(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "compute minor");
        std::printf("Device %d: %s, compute capability %d.%d\n", i, name, major, minor);
        CUcontext context;
        check(retain(&context, device), "cuDevicePrimaryCtxRetain (context creation)");
        check(release(device), "cuDevicePrimaryCtxRelease");
    }
    dlclose(lib);
    std::puts("PASS driver initialization, enumeration and context creation; run vector-add to prove execution.");
}
