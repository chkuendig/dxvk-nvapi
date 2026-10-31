#include <thread>
#include "nvapi_tests_private.h"
#include "nvapi/default_test_environment.h"
#include "nvapi/extended_test_environment.h"

using namespace trompeloeil;

namespace {
    constexpr auto luid1 = LUID{0x04030211, 0x08070655};
    constexpr auto luid2 = LUID{0x14131211, 0x18171615};
    constexpr auto luid3 = LUID{0x24232221, 0x28272625};

    void ConfigureVkProps(ConfigureProps vkProps, VkDriverId driverId, const LUID& luid) {
        vkProps.props->vendorID = driverId == VK_DRIVER_ID_NVIDIA_PROPRIETARY ? 0x10de : 0x1002;
        vkProps.driverProps->driverID = driverId;
        memcpy(&vkProps.idProps->deviceLUID, &luid, sizeof(luid));
        vkProps.idProps->deviceLUIDValid = VK_TRUE;
        vkProps.idProps->deviceNodeMask = 1;
    }

    // Reports one CUDA device with node mask 1 per given LUID
    std::vector<std::unique_ptr<expectation>> ConfigureCudaDevices(NvCudaMock& cuda, std::vector<LUID> luids) {
        std::vector<std::unique_ptr<expectation>> e;
        e.emplace_back(NAMED_ALLOW_CALL(cuda, IsAvailable()).RETURN(true));
        e.emplace_back(NAMED_ALLOW_CALL(cuda, Init(0U)).RETURN(CUDA_SUCCESS));
        e.emplace_back(NAMED_ALLOW_CALL(cuda, DeviceGetCount(_))
                .SIDE_EFFECT(*_1 = static_cast<int>(luids.size()))
                .RETURN(CUDA_SUCCESS));
        // Use device handles that differ from the ordinals
        e.emplace_back(NAMED_ALLOW_CALL(cuda, DeviceGet(_, _))
                .SIDE_EFFECT(*_1 = _2 + 0x10)
                .RETURN(CUDA_SUCCESS));
        e.emplace_back(NAMED_ALLOW_CALL(cuda, DeviceGetLuid(_, _, _))
                .WITH(_3 >= 0x10 && static_cast<size_t>(_3 - 0x10) < luids.size())
                .SIDE_EFFECT(memcpy(_1, &luids[_3 - 0x10], sizeof(LUID)))
                .SIDE_EFFECT(*_2 = 1)
                .RETURN(CUDA_SUCCESS));
        return e;
    }

    NV_COMPUTE_GPU_TOPOLOGY_V1 EnumComputeCapableGpus() {
        NV_COMPUTE_GPU_TOPOLOGY_V1 topology{};
        topology.version = NV_COMPUTE_GPU_TOPOLOGY_VER1;
        REQUIRE(NvAPI_GPU_CudaEnumComputeCapableGpus(reinterpret_cast<NV_COMPUTE_GPU_TOPOLOGY*>(&topology)) == NVAPI_OK);
        return topology;
    }
}

TEST_CASE("CUDA topology correlates CUDA devices by LUID", "[.sysinfo-cuda]") {
    auto t = std::make_unique<DefaultTestEnvironment>();
    auto e = t->ConfigureExpectations();

    ::SetEnvironmentVariableA("DXVK_NVAPI_ALLOW_OTHER_DRIVERS", "1");

    auto driverId = GENERATE(VK_DRIVER_ID_MESA_RADV, VK_DRIVER_ID_NVIDIA_PROPRIETARY);
    ALLOW_CALL(*t->Vk(), GetDeviceExtensions(_, _))
        .RETURN(std::set<std::string>{VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME, VK_NV_VIEWPORT_ARRAY2_EXTENSION_NAME});
    ALLOW_CALL(*t->Vk(), GetPhysicalDeviceProperties2(_, _, _))
        .SIDE_EFFECT(VkMock::ConfigureGetPhysicalDeviceProperties2(_3,
            [driverId](auto vkProps) { ConfigureVkProps(vkProps, driverId, luid1); }));

    // Upstream behavior without CUDA: only NVIDIA's proprietary driver is reported, based on its architecture
    auto heuristicGpuCount = driverId == VK_DRIVER_ID_NVIDIA_PROPRIETARY ? 1U : 0U;

    SECTION("CUDA device with matching LUID and node mask is reported") {
        auto c = ConfigureCudaDevices(*t->Cuda(), {luid1});

        REQUIRE(NvAPI_Initialize() == NVAPI_OK);
        REQUIRE(t->CudaCreationCount() == 0);

        auto topology = EnumComputeCapableGpus();
        REQUIRE(topology.gpuCount == 1);
        REQUIRE(topology.computeGpus[0].hPhysicalGpu == reinterpret_cast<NvPhysicalGpuHandle>(nvapiAdapterRegistry->GetFirstAdapter()));
        REQUIRE(topology.computeGpus[0].flags == 0x0b);
        REQUIRE(nvapiAdapterRegistry->GetComputeGpuTopology().hasCudaDevices);
        REQUIRE(t->CudaCreationCount() == 1);
    }

    SECTION("CUDA device with different LUID or node mask is not reported") {
        auto mismatch = GENERATE(std::make_pair(luid2, 1U), std::make_pair(luid1, 2U));
        ALLOW_CALL(*t->Cuda(), IsAvailable()).RETURN(true);
        ALLOW_CALL(*t->Cuda(), Init(0U)).RETURN(CUDA_SUCCESS);
        ALLOW_CALL(*t->Cuda(), DeviceGetCount(_)).SIDE_EFFECT(*_1 = 1).RETURN(CUDA_SUCCESS);
        ALLOW_CALL(*t->Cuda(), DeviceGet(_, 0)).SIDE_EFFECT(*_1 = 0).RETURN(CUDA_SUCCESS);
        ALLOW_CALL(*t->Cuda(), DeviceGetLuid(_, _, 0))
            .LR_SIDE_EFFECT(memcpy(_1, &mismatch.first, sizeof(LUID)))
            .LR_SIDE_EFFECT(*_2 = mismatch.second)
            .RETURN(CUDA_SUCCESS);

        REQUIRE(NvAPI_Initialize() == NVAPI_OK);
        REQUIRE(EnumComputeCapableGpus().gpuCount == heuristicGpuCount);
        REQUIRE_FALSE(nvapiAdapterRegistry->GetComputeGpuTopology().hasCudaDevices);
    }

    SECTION("Unavailable nvcuda falls back to upstream behavior") {
        ALLOW_CALL(*t->Cuda(), IsAvailable()).RETURN(false);
        FORBID_CALL(*t->Cuda(), Init(_));

        REQUIRE(NvAPI_Initialize() == NVAPI_OK);
        REQUIRE(EnumComputeCapableGpus().gpuCount == heuristicGpuCount);
        REQUIRE_FALSE(nvapiAdapterRegistry->GetComputeGpuTopology().hasCudaDevices);
        REQUIRE(t->CudaCreationCount() == 1);
    }

    SECTION("Failing cuInit falls back to upstream behavior") {
        ALLOW_CALL(*t->Cuda(), IsAvailable()).RETURN(true);
        REQUIRE_CALL(*t->Cuda(), Init(0U)).RETURN(100); // CUDA_ERROR_NO_DEVICE
        FORBID_CALL(*t->Cuda(), DeviceGetCount(_));

        REQUIRE(NvAPI_Initialize() == NVAPI_OK);
        REQUIRE(EnumComputeCapableGpus().gpuCount == heuristicGpuCount);
        REQUIRE_FALSE(nvapiAdapterRegistry->GetComputeGpuTopology().hasCudaDevices);
    }

    SECTION("Failing device count query falls back to upstream behavior") {
        ALLOW_CALL(*t->Cuda(), IsAvailable()).RETURN(true);
        ALLOW_CALL(*t->Cuda(), Init(0U)).RETURN(CUDA_SUCCESS);
        ALLOW_CALL(*t->Cuda(), DeviceGetCount(_)).RETURN(3); // CUDA_ERROR_NOT_INITIALIZED
        FORBID_CALL(*t->Cuda(), DeviceGet(_, _));

        REQUIRE(NvAPI_Initialize() == NVAPI_OK);
        REQUIRE(EnumComputeCapableGpus().gpuCount == heuristicGpuCount);
    }

    SECTION("Failing device or LUID queries skip that device") {
        auto failingDeviceGet = GENERATE(true, false);
        ALLOW_CALL(*t->Cuda(), IsAvailable()).RETURN(true);
        ALLOW_CALL(*t->Cuda(), Init(0U)).RETURN(CUDA_SUCCESS);
        ALLOW_CALL(*t->Cuda(), DeviceGetCount(_)).SIDE_EFFECT(*_1 = 1).RETURN(CUDA_SUCCESS);
        ALLOW_CALL(*t->Cuda(), DeviceGet(_, 0))
            .SIDE_EFFECT(*_1 = 0)
            .RETURN(failingDeviceGet ? 101 : CUDA_SUCCESS); // CUDA_ERROR_INVALID_DEVICE
        ALLOW_CALL(*t->Cuda(), DeviceGetLuid(_, _, 0))
            .SIDE_EFFECT(memcpy(_1, &luid1, sizeof(luid1)))
            .SIDE_EFFECT(*_2 = 1)
            .RETURN(801); // CUDA_ERROR_NOT_SUPPORTED, outputs must be ignored

        REQUIRE(NvAPI_Initialize() == NVAPI_OK);
        REQUIRE(EnumComputeCapableGpus().gpuCount == heuristicGpuCount);
        REQUIRE_FALSE(nvapiAdapterRegistry->GetComputeGpuTopology().hasCudaDevices);
    }

    SECTION("Discovery result is cached until unload") {
        ALLOW_CALL(*t->Cuda(), IsAvailable()).RETURN(true);
        REQUIRE_CALL(*t->Cuda(), Init(0U)).RETURN(CUDA_SUCCESS);
        REQUIRE_CALL(*t->Cuda(), DeviceGetCount(_)).SIDE_EFFECT(*_1 = 1).RETURN(CUDA_SUCCESS);
        REQUIRE_CALL(*t->Cuda(), DeviceGet(_, 0)).SIDE_EFFECT(*_1 = 0).RETURN(CUDA_SUCCESS);
        REQUIRE_CALL(*t->Cuda(), DeviceGetLuid(_, _, 0))
            .SIDE_EFFECT(memcpy(_1, &luid1, sizeof(luid1)))
            .SIDE_EFFECT(*_2 = 1)
            .RETURN(CUDA_SUCCESS);

        REQUIRE(NvAPI_Initialize() == NVAPI_OK);
        auto& snapshot = nvapiAdapterRegistry->GetComputeGpuTopology();
        for (auto i = 0; i < 3; i++)
            REQUIRE(EnumComputeCapableGpus().gpuCount == 1);

        REQUIRE(&nvapiAdapterRegistry->GetComputeGpuTopology() == &snapshot);
        REQUIRE(t->CudaCreationCount() == 1);
    }

    SECTION("Concurrent first queries discover CUDA devices once") {
        ALLOW_CALL(*t->Cuda(), IsAvailable()).RETURN(true);
        REQUIRE_CALL(*t->Cuda(), Init(0U))
            .SIDE_EFFECT(std::this_thread::sleep_for(std::chrono::milliseconds(50)))
            .RETURN(CUDA_SUCCESS);
        REQUIRE_CALL(*t->Cuda(), DeviceGetCount(_)).SIDE_EFFECT(*_1 = 1).RETURN(CUDA_SUCCESS);
        REQUIRE_CALL(*t->Cuda(), DeviceGet(_, 0)).SIDE_EFFECT(*_1 = 0).RETURN(CUDA_SUCCESS);
        REQUIRE_CALL(*t->Cuda(), DeviceGetLuid(_, _, 0))
            .SIDE_EFFECT(memcpy(_1, &luid1, sizeof(luid1)))
            .SIDE_EFFECT(*_2 = 1)
            .RETURN(CUDA_SUCCESS);

        REQUIRE(NvAPI_Initialize() == NVAPI_OK);

        constexpr auto threadCount = 8U;
        std::array<NvAPI_Status, threadCount> statuses{};
        std::array<NV_COMPUTE_GPU_TOPOLOGY_V1, threadCount> topologies{};
        std::vector<std::thread> threads;
        for (auto i = 0U; i < threadCount; i++)
            threads.emplace_back([i, &statuses, &topologies] {
                topologies[i].version = NV_COMPUTE_GPU_TOPOLOGY_VER1;
                statuses[i] = NvAPI_GPU_CudaEnumComputeCapableGpus(reinterpret_cast<NV_COMPUTE_GPU_TOPOLOGY*>(&topologies[i]));
            });

        for (auto& thread : threads)
            thread.join();

        for (auto i = 0U; i < threadCount; i++) {
            REQUIRE(statuses[i] == NVAPI_OK);
            REQUIRE(topologies[i].gpuCount == 1);
            REQUIRE(topologies[i].computeGpus[0].hPhysicalGpu == reinterpret_cast<NvPhysicalGpuHandle>(nvapiAdapterRegistry->GetFirstAdapter()));
        }

        REQUIRE(t->CudaCreationCount() == 1);
    }

    SECTION("Reentrant calls from the CUDA provider do not deadlock") {
        NvAPI_Status nestedInitialize = NVAPI_ERROR, nestedEnum = NVAPI_ERROR, nestedUnload = NVAPI_ERROR;
        NV_COMPUTE_GPU_TOPOLOGY_V1 nested{};
        nested.version = NV_COMPUTE_GPU_TOPOLOGY_VER1;

        ALLOW_CALL(*t->Cuda(), IsAvailable()).RETURN(true);
        ALLOW_CALL(*t->Cuda(), Init(0U))
            .LR_SIDE_EFFECT(nestedInitialize = NvAPI_Initialize())
            .LR_SIDE_EFFECT(nestedEnum = NvAPI_GPU_CudaEnumComputeCapableGpus(reinterpret_cast<NV_COMPUTE_GPU_TOPOLOGY*>(&nested)))
            .LR_SIDE_EFFECT(nestedUnload = NvAPI_Unload())
            .RETURN(CUDA_SUCCESS);
        ALLOW_CALL(*t->Cuda(), DeviceGetCount(_)).SIDE_EFFECT(*_1 = 1).RETURN(CUDA_SUCCESS);
        ALLOW_CALL(*t->Cuda(), DeviceGet(_, 0)).SIDE_EFFECT(*_1 = 0).RETURN(CUDA_SUCCESS);
        ALLOW_CALL(*t->Cuda(), DeviceGetLuid(_, _, 0))
            .SIDE_EFFECT(memcpy(_1, &luid1, sizeof(luid1)))
            .SIDE_EFFECT(*_2 = 1)
            .RETURN(CUDA_SUCCESS);

        REQUIRE(NvAPI_Initialize() == NVAPI_OK);
        REQUIRE(EnumComputeCapableGpus().gpuCount == 1);

        // The nested query is answered without CUDA, and is not cached
        REQUIRE(nestedInitialize == NVAPI_OK);
        REQUIRE(nestedEnum == NVAPI_OK);
        REQUIRE(nested.gpuCount == heuristicGpuCount);
        REQUIRE(nestedUnload == NVAPI_OK);
        REQUIRE(EnumComputeCapableGpus().gpuCount == 1);
        REQUIRE(t->CudaCreationCount() == 1);
    }
}

TEST_CASE("CUDA topology does not load nvcuda unnecessarily", "[.sysinfo-cuda]") {
    auto t = std::make_unique<DefaultTestEnvironment>();
    auto e = t->ConfigureExpectations();

    ::SetEnvironmentVariableA("DXVK_NVAPI_ALLOW_OTHER_DRIVERS", "1");

    SECTION("Other NVAPI methods do not load nvcuda") {
        ALLOW_CALL(*t->Vk(), GetPhysicalDeviceProperties2(_, _, _))
            .SIDE_EFFECT(VkMock::ConfigureGetPhysicalDeviceProperties2(_3,
                [](auto vkProps) { ConfigureVkProps(vkProps, VK_DRIVER_ID_MESA_RADV, luid1); }));

        REQUIRE(NvAPI_Initialize() == NVAPI_OK);

        NvPhysicalGpuHandle handles[NVAPI_MAX_PHYSICAL_GPUS]{};
        NvU32 count = 0;
        REQUIRE(NvAPI_EnumPhysicalGPUs(handles, &count) == NVAPI_OK);
        NvU32 gpuId = 0;
        REQUIRE(NvAPI_GetGPUIDfromPhysicalGPU(handles[0], &gpuId) == NVAPI_OK);
        NvPhysicalGpuHandle handle{};
        REQUIRE(NvAPI_GetPhysicalGPUFromGPUID(gpuId, &handle) == NVAPI_OK);
        NvU32 version = 0;
        NvAPI_ShortString branch;
        REQUIRE(NvAPI_SYS_GetDriverAndBranchVersion(&version, branch) == NVAPI_OK);
        LUID luid{};
        REQUIRE(NvAPI_GPU_GetAdapterIdFromPhysicalGpu(handle, &luid) == NVAPI_OK);
        REQUIRE(NvAPI_Unload() == NVAPI_OK);

        REQUIRE(t->CudaCreationCount() == 0);
    }

    SECTION("Adapters without LUID do not load nvcuda") {
        REQUIRE(NvAPI_Initialize() == NVAPI_OK);
        EnumComputeCapableGpus();
        REQUIRE(t->CudaCreationCount() == 0);
    }
}

TEST_CASE("CUDA topology reports multiple adapters in CUDA device order", "[.sysinfo-cuda]") {
    auto t = std::make_unique<ExtendedTestEnvironment>();
    auto e = t->ConfigureExpectations();

    ::SetEnvironmentVariableA("DXVK_NVAPI_ALLOW_OTHER_DRIVERS", "1");

    auto adapter1DriverId = GENERATE(VK_DRIVER_ID_MESA_RADV, VK_DRIVER_ID_NVIDIA_PROPRIETARY);
    ALLOW_CALL(*t->Vk(), GetDeviceExtensions(_, _))
        .RETURN(std::set<std::string>{VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME, VK_NV_VIEWPORT_ARRAY2_EXTENSION_NAME});
    ALLOW_CALL(*t->Vk(), GetPhysicalDeviceProperties2(_, reinterpret_cast<VkPhysicalDevice>(0x01), _))
        .SIDE_EFFECT(VkMock::ConfigureGetPhysicalDeviceProperties2(_3,
            [adapter1DriverId](auto vkProps) { ConfigureVkProps(vkProps, adapter1DriverId, luid1); }));
    ALLOW_CALL(*t->Vk(), GetPhysicalDeviceProperties2(_, reinterpret_cast<VkPhysicalDevice>(0x02), _))
        .SIDE_EFFECT(VkMock::ConfigureGetPhysicalDeviceProperties2(_3,
            [](auto vkProps) { ConfigureVkProps(vkProps, VK_DRIVER_ID_MESA_RADV, luid2); }));

    // CUDA device 0 is adapter 2, devices 1 and 2 are adapter 1 (only reported once), device 3 is not visible to NVAPI
    auto cudaSeesAdapter1 = GENERATE(true, false);
    auto c = ConfigureCudaDevices(*t->Cuda(), {luid2, cudaSeesAdapter1 ? luid1 : luid3, cudaSeesAdapter1 ? luid1 : luid3, luid3});

    REQUIRE(NvAPI_Initialize() == NVAPI_OK);

    // Adapters that CUDA does not report keep upstream behavior and follow the CUDA devices
    auto adapter1Reported = cudaSeesAdapter1 || adapter1DriverId == VK_DRIVER_ID_NVIDIA_PROPRIETARY;
    auto topology = EnumComputeCapableGpus();
    REQUIRE(topology.gpuCount == (adapter1Reported ? 2 : 1));
    REQUIRE(topology.computeGpus[0].hPhysicalGpu == reinterpret_cast<NvPhysicalGpuHandle>(nvapiAdapterRegistry->GetAdapter(1)));
    if (adapter1Reported)
        REQUIRE(topology.computeGpus[1].hPhysicalGpu == reinterpret_cast<NvPhysicalGpuHandle>(nvapiAdapterRegistry->GetAdapter(0)));
    REQUIRE(nvapiAdapterRegistry->GetComputeGpuTopology().hasCudaDevices);
}

TEST_CASE("CUDA topology V2 returns GPUs confirmed by CUDA", "[.sysinfo-cuda]") {
    auto t = std::make_unique<DefaultTestEnvironment>();
    auto e = t->ConfigureExpectations();

    ::SetEnvironmentVariableA("DXVK_NVAPI_ALLOW_OTHER_DRIVERS", "1");

    auto driverId = GENERATE(VK_DRIVER_ID_MESA_RADV, VK_DRIVER_ID_NVIDIA_PROPRIETARY);
    ALLOW_CALL(*t->Vk(), GetDeviceExtensions(_, _))
        .RETURN(std::set<std::string>{VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME, VK_NV_VIEWPORT_ARRAY2_EXTENSION_NAME});
    ALLOW_CALL(*t->Vk(), GetPhysicalDeviceProperties2(_, _, _))
        .SIDE_EFFECT(VkMock::ConfigureGetPhysicalDeviceProperties2(_3,
            [driverId](auto vkProps) { ConfigureVkProps(vkProps, driverId, luid1); }));

    // Legacy PhysX initializes only the version
    NV_COMPUTE_GPU_TOPOLOGY_V2 topology;
    topology.version = NV_COMPUTE_GPU_TOPOLOGY_VER;
    topology.gpuCount = 0xdeadbeef;
    topology.computeGpus = reinterpret_cast<NV_COMPUTE_GPU*>(0x1);

    SECTION("V2 returns an NVAPI owned array") {
        auto c = ConfigureCudaDevices(*t->Cuda(), {luid1});

        REQUIRE(NvAPI_Initialize() == NVAPI_OK);

        auto status = NVAPI_ERROR;
        std::thread([&status, &topology] { status = NvAPI_GPU_CudaEnumComputeCapableGpus(&topology); }).join();
        REQUIRE(status == NVAPI_OK);
        REQUIRE(topology.version == NV_COMPUTE_GPU_TOPOLOGY_VER);
        REQUIRE(topology.gpuCount == 1);
        REQUIRE(topology.computeGpus == nvapiAdapterRegistry->GetComputeGpuTopology().gpus.data());
        REQUIRE(topology.computeGpus[0].hPhysicalGpu == reinterpret_cast<NvPhysicalGpuHandle>(nvapiAdapterRegistry->GetFirstAdapter()));
        REQUIRE(topology.computeGpus[0].flags == 0x21001b);

        // The array outlives the producing thread and is not changed by subsequent queries
        NV_COMPUTE_GPU_TOPOLOGY_V2 repeated{};
        repeated.version = NV_COMPUTE_GPU_TOPOLOGY_VER;
        REQUIRE(NvAPI_GPU_CudaEnumComputeCapableGpus(&repeated) == NVAPI_OK);
        REQUIRE(repeated.computeGpus == topology.computeGpus);
        REQUIRE(EnumComputeCapableGpus().computeGpus[0].flags == 0x0b);
        REQUIRE(topology.computeGpus[0].hPhysicalGpu == reinterpret_cast<NvPhysicalGpuHandle>(nvapiAdapterRegistry->GetFirstAdapter()));
        REQUIRE(topology.computeGpus[0].flags == 0x21001b);
    }

    SECTION("V2 returns incompatible-struct-version without GPUs confirmed by CUDA") {
        ALLOW_CALL(*t->Cuda(), IsAvailable()).RETURN(false);

        REQUIRE(NvAPI_Initialize() == NVAPI_OK);
        REQUIRE(NvAPI_GPU_CudaEnumComputeCapableGpus(&topology) == NVAPI_INCOMPATIBLE_STRUCT_VERSION);
        REQUIRE(topology.gpuCount == 0xdeadbeef);
        REQUIRE(topology.computeGpus == reinterpret_cast<NV_COMPUTE_GPU*>(0x1));
    }

    SECTION("Unknown struct version returns incompatible-struct-version") {
        FORBID_CALL(*t->Cuda(), IsAvailable());

        REQUIRE(NvAPI_Initialize() == NVAPI_OK);
        topology.version = NV_COMPUTE_GPU_TOPOLOGY_VER + 1;
        REQUIRE(NvAPI_GPU_CudaEnumComputeCapableGpus(&topology) == NVAPI_INCOMPATIBLE_STRUCT_VERSION);
        REQUIRE(t->CudaCreationCount() == 0);
    }
}
