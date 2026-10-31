#pragma once

#include <condition_variable>
#include <thread>
#include "../nvapi_private.h"
#include "nvapi_resource_factory.h"
#include "nvapi_adapter.h"
#include "nvapi_output.h"
#include "../interfaces/dxvk_interfaces.h"

namespace dxvk {
    struct NvapiComputeGpuTopology {
        // Ordered by CUDA device ordinal, followed by adapters that are only known by heuristic,
        // the vector is returned to callers of topology V2 and must not change after discovery
        std::vector<NV_COMPUTE_GPU> gpus;
        // True when at least one entry was confirmed by the CUDA driver API
        bool hasCudaDevices{};
    };

    class NvapiAdapterRegistry {

      public:
        explicit NvapiAdapterRegistry(NvapiResourceFactory& resourceFactory);
        ~NvapiAdapterRegistry();

        bool Initialize();

        [[nodiscard]] uint32_t GetAdapterCount() const;
        [[nodiscard]] NvapiAdapter* GetAdapter(uint32_t index) const;
        [[nodiscard]] NvapiAdapter* GetFirstAdapter() const;
        [[nodiscard]] NvapiAdapter* FindAdapter(const LUID& luid) const;
        [[nodiscard]] bool IsAdapter(NvapiAdapter* handle) const;
        [[nodiscard]] const NvapiComputeGpuTopology& GetComputeGpuTopology();

        [[nodiscard]] uint32_t GetOutputCount(NvapiAdapter* handle) const;
        [[nodiscard]] NvapiOutput* GetOutput(uint32_t index) const;
        [[nodiscard]] NvapiOutput* GetOutput(NvapiAdapter* handle, uint32_t index) const;
        [[nodiscard]] NvapiOutput* FindOutput(const std::string& displayName) const;
        [[nodiscard]] NvapiOutput* FindOutput(uint32_t id) const;
        [[nodiscard]] NvapiOutput* FindPrimaryOutput() const;
        [[nodiscard]] bool IsOutput(NvapiOutput* handle) const;

        [[nodiscard]] const Com<IDXGIVkInteropFactory1>& GetInteropFactory() const { return m_dxgiVkInterop; }

        // Those flags match NVAPI on Windows for a normal desktop machine
        constexpr static NvU32 ComputeGpuFlags = NV_COMPUTE_GPU_TOPOLOGY_PHYSICS_CAPABLE | NV_COMPUTE_GPU_TOPOLOGY_PHYSICS_ENABLE | NV_COMPUTE_GPU_TOPOLOGY_PHYSICS_RECOMMENDED;
        // GPUs confirmed by CUDA additionally report CUDA and PhysX availability, but only for topology V2
        constexpr static NvU32 CudaComputeGpuFlags = ComputeGpuFlags | NV_COMPUTE_GPU_TOPOLOGY_CUDA_AVAILABLE | NV_COMPUTE_GPU_TOPOLOGY_CUDA_CAPABLE | NV_COMPUTE_GPU_TOPOLOGY_PHYSICS_AVAILABLE;

      private:
        NvapiResourceFactory& m_resourceFactory;
        Com<IDXGIFactory1> m_dxgiFactory;
        Com<IDXGIVkInteropFactory1> m_dxgiVkInterop;
        std::unique_ptr<Vk> m_vk;
        std::unique_ptr<Nvml> m_nvml;
        std::vector<NvapiAdapter*> m_nvapiAdapters;
        std::vector<NvapiOutput*> m_nvapiOutputs;

        std::mutex m_computeGpuTopologyMutex;
        std::condition_variable m_computeGpuTopologyCondition;
        std::optional<std::thread::id> m_computeGpuTopologyThread;
        std::optional<NvapiComputeGpuTopology> m_computeGpuTopology;
        NvapiComputeGpuTopology m_heuristicComputeGpuTopology;
        std::unique_ptr<NvCuda> m_nvcuda;

        [[nodiscard]] NvapiComputeGpuTopology DiscoverComputeGpuTopology(std::unique_ptr<NvCuda>& nvcuda) const;
    };
}
