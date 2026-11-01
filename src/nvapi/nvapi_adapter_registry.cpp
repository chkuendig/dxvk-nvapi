#include "nvapi_adapter_registry.h"
#include "nvapi_d3d11_device.h"
#include "nvapi_d3d12_device.h"
#include "nvapi_d3d12_command_queue.h"
#include "nvapi_d3d12_graphics_command_list.h"
#include "nvapi_d3d_low_latency_device.h"
#include "nvapi_vulkan_low_latency_device_factory.h"
#include "../util/util_string.h"
#include "../util/util_log.h"

namespace dxvk {

    NvapiAdapterRegistry::NvapiAdapterRegistry(NvapiResourceFactory& resourceFactory)
        : m_resourceFactory(resourceFactory) {}

    NvapiAdapterRegistry::~NvapiAdapterRegistry() {
        NvapiD3d11Device::Reset();
        NvapiD3d12Device::Reset();
        NvapiD3d12CommandQueue::Reset();
        NvapiD3d12GraphicsCommandList::Reset();
        NvapiD3dLowLatencyDevice::Reset();
        NvapiVulkanLowLatencyDeviceFactory::Reset();

        for (const auto output : m_nvapiOutputs)
            delete output;

        for (const auto adapter : m_nvapiAdapters)
            delete adapter;

        m_nvapiOutputs.clear();
        m_nvapiAdapters.clear();
    }

    bool NvapiAdapterRegistry::Initialize() {
        m_dxgiFactory = m_resourceFactory.CreateDXGIFactory1();
        if (m_dxgiFactory == nullptr)
            return false;

        m_vk = m_resourceFactory.CreateVulkan(m_dxgiFactory);
        if (!m_vk || !m_vk->IsAvailable())
            return false;

        m_nvml = m_resourceFactory.CreateNvml();
        if (m_nvml->IsAvailable())
            log::info("NVML loaded and initialized successfully");

        Com<IDXGIVkInteropFactory1> dxgiVkInteropFactory;
        if (SUCCEEDED(m_dxgiFactory->QueryInterface(IID_PPV_ARGS(&dxgiVkInteropFactory))))
            m_dxgiVkInterop = dxgiVkInteropFactory;

        // Query all D3D11 adapter from DXVK to honor any DXVK device filtering
        Com<IDXGIAdapter1> dxgiAdapter;
        for (auto i = 0U; m_dxgiFactory->EnumAdapters1(i, &dxgiAdapter) != DXGI_ERROR_NOT_FOUND; i++) {
            Com<IDXGIAdapter3> dxgiAdapter3;
            if (FAILED(dxgiAdapter->QueryInterface(IID_PPV_ARGS(&dxgiAdapter3))))
                continue;

            auto nvapiAdapter = new NvapiAdapter(*m_vk, *m_nvml, dxgiAdapter3);
            if (nvapiAdapter->Initialize(i, m_nvapiOutputs))
                m_nvapiAdapters.push_back(nvapiAdapter);
            else
                delete nvapiAdapter;
        }

        for (const auto adapter : m_nvapiAdapters) {
            if (!adapter->HasNvProprietaryDriver() || adapter->GetArchitectureId() < NV_GPU_ARCHITECTURE_GM000) // Maxwell is the oldest generation we can detect
                continue;

            m_heuristicComputeGpuTopology.gpus.push_back({reinterpret_cast<NvPhysicalGpuHandle>(adapter), ComputeGpuFlags});
        }

        return !m_nvapiAdapters.empty();
    }

    uint32_t NvapiAdapterRegistry::GetAdapterCount() const {
        return m_nvapiAdapters.size();
    }

    NvapiAdapter* NvapiAdapterRegistry::GetAdapter(const uint32_t index) const {
        return index < m_nvapiAdapters.size() ? m_nvapiAdapters[index] : nullptr;
    }

    NvapiAdapter* NvapiAdapterRegistry::GetFirstAdapter() const {
        return m_nvapiAdapters.front();
    }

    NvapiAdapter* NvapiAdapterRegistry::FindAdapter(const LUID& luid) const {
        auto it = std::find_if(m_nvapiAdapters.begin(), m_nvapiAdapters.end(),
            [luid](const auto& adapter) {
                auto adapterLuid = adapter->GetLuid();
                return adapterLuid.has_value()
                    && adapterLuid.value().HighPart == luid.HighPart
                    && adapterLuid.value().LowPart == luid.LowPart;
            });

        return it != m_nvapiAdapters.end() ? *it : nullptr;
    }

    bool NvapiAdapterRegistry::IsAdapter(NvapiAdapter* handle) const {
        return std::find(m_nvapiAdapters.begin(), m_nvapiAdapters.end(), handle) != m_nvapiAdapters.end();
    }

    const NvapiComputeGpuTopology& NvapiAdapterRegistry::GetComputeGpuTopology() {
        std::unique_lock lock(m_computeGpuTopologyMutex);

        // A CUDA provider may call back into NVAPI on the discovering thread,
        // answer such calls without CUDA instead of waiting for ourselves
        if (m_computeGpuTopologyThread == std::this_thread::get_id())
            return m_heuristicComputeGpuTopology;

        m_computeGpuTopologyCondition.wait(lock, [this] { return !m_computeGpuTopologyThread.has_value(); });
        if (m_computeGpuTopology.has_value())
            return m_computeGpuTopology.value();

        // Load and query CUDA outside of the lock, the provider may call back into NVAPI
        m_computeGpuTopologyThread = std::this_thread::get_id();
        lock.unlock();

        std::unique_ptr<NvCuda> nvcuda;
        auto topology = DiscoverComputeGpuTopology(nvcuda);

        lock.lock();
        m_nvcuda = std::move(nvcuda);
        m_computeGpuTopology = std::move(topology);
        m_computeGpuTopologyThread.reset();
        m_computeGpuTopologyCondition.notify_all();

        return m_computeGpuTopology.value();
    }

    NvapiComputeGpuTopology NvapiAdapterRegistry::DiscoverComputeGpuTopology(std::unique_ptr<NvCuda>& nvcuda) const {
        auto topology = NvapiComputeGpuTopology{};

        // Without any PCI location there is nothing to correlate, so do not load CUDA at all
        if (std::none_of(m_nvapiAdapters.begin(), m_nvapiAdapters.end(),
                [](const auto& adapter) { return adapter->HasPciBusInfo(); }))
            return m_heuristicComputeGpuTopology;

        nvcuda = m_resourceFactory.CreateNvCuda();
        if (!nvcuda->IsAvailable()) {
            nvcuda.reset();
            return m_heuristicComputeGpuTopology;
        }

        // Once initialized, keep nvcuda.dll loaded until the final NvAPI_Unload
        if (auto result = nvcuda->Init(0); result != CUDA_SUCCESS) {
            log::info(str::format("CUDA initialization failed with error: ", result));
            return m_heuristicComputeGpuTopology;
        }

        auto deviceCount = 0;
        if (auto result = nvcuda->DeviceGetCount(&deviceCount); result != CUDA_SUCCESS) {
            log::info(str::format("Querying CUDA device count failed with error: ", result));
            return m_heuristicComputeGpuTopology;
        }

        for (auto ordinal = 0; ordinal < deviceCount; ordinal++) {
            CUdevice device{};
            if (auto result = nvcuda->DeviceGet(&device, ordinal); result != CUDA_SUCCESS) {
                log::info(str::format("Querying CUDA device ", ordinal, " failed with error: ", result));
                continue;
            }

            // Same identity as used for NVML, the PCI function is not reported by CUDA
            auto domain = 0, bus = 0, slot = 0;
            if (auto result = nvcuda->DeviceGetAttribute(&domain, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device); result != CUDA_SUCCESS
                || (result = nvcuda->DeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device)) != CUDA_SUCCESS
                || (result = nvcuda->DeviceGetAttribute(&slot, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device)) != CUDA_SUCCESS) {
                log::info(str::format("Querying PCI location of CUDA device ", ordinal, " failed with error: ", result));
                continue;
            }

            std::vector<NvapiAdapter*> matches;
            std::copy_if(m_nvapiAdapters.begin(), m_nvapiAdapters.end(), std::back_inserter(matches),
                [domain, bus, slot](const auto& adapter) {
                    return adapter->HasPciBusInfo()
                        && adapter->GetPciDomainId() == static_cast<uint32_t>(domain)
                        && adapter->GetPciBusId() == static_cast<uint32_t>(bus)
                        && adapter->GetPciDeviceId() == static_cast<uint32_t>(slot);
                });

            if (matches.size() != 1) {
                log::info(str::format("CUDA device ", ordinal, " matches ", matches.size(), " adapters, ignoring"));
                continue;
            }

            auto handle = reinterpret_cast<NvPhysicalGpuHandle>(matches.front());
            if (std::any_of(topology.gpus.begin(), topology.gpus.end(),
                    [handle](const auto& gpu) { return gpu.hPhysicalGpu == handle; }))
                continue; // Report every adapter only once

            log::info(str::format("CUDA device ", ordinal, " corresponds to ", matches.front()->GetDeviceName()));
            topology.gpus.push_back({handle, CudaComputeGpuFlags});
        }

        topology.hasCudaDevices = !topology.gpus.empty();

        for (const auto& gpu : m_heuristicComputeGpuTopology.gpus)
            if (std::none_of(topology.gpus.begin(), topology.gpus.end(),
                    [&gpu](const auto& cudaGpu) { return cudaGpu.hPhysicalGpu == gpu.hPhysicalGpu; }))
                topology.gpus.push_back(gpu);

        return topology;
    }

    uint32_t NvapiAdapterRegistry::GetOutputCount(NvapiAdapter* handle) const {
        return std::count_if(m_nvapiOutputs.begin(), m_nvapiOutputs.end(),
            [&handle](const auto& output) {
                return output->GetParent() == handle;
            });
    }

    NvapiOutput* NvapiAdapterRegistry::GetOutput(const uint32_t index) const {
        return index < m_nvapiOutputs.size() ? m_nvapiOutputs[index] : nullptr;
    }

    NvapiOutput* NvapiAdapterRegistry::GetOutput(NvapiAdapter* handle, const uint32_t index) const {
        std::vector<NvapiOutput*> filtered;
        std::copy_if(m_nvapiOutputs.begin(), m_nvapiOutputs.end(), std::back_inserter(filtered),
            [&handle](const auto& output) {
                return output->GetParent() == handle;
            });

        return index < filtered.size() ? filtered[index] : nullptr;
    }

    NvapiOutput* NvapiAdapterRegistry::FindOutput(const std::string& displayName) const {
        auto it = std::find_if(m_nvapiOutputs.begin(), m_nvapiOutputs.end(),
            [&displayName](const auto& output) {
                return output->GetDeviceName() == displayName;
            });

        return it != m_nvapiOutputs.end() ? *it : nullptr;
    }

    NvapiOutput* NvapiAdapterRegistry::FindOutput(uint32_t id) const {
        auto it = std::find_if(m_nvapiOutputs.begin(), m_nvapiOutputs.end(),
            [id](const auto& output) {
                return output->GetId() == id;
            });

        return it != m_nvapiOutputs.end() ? *it : nullptr;
    }

    NvapiOutput* NvapiAdapterRegistry::FindPrimaryOutput() const {
        auto it = std::find_if(m_nvapiOutputs.begin(), m_nvapiOutputs.end(),
            [](const auto& output) {
                return output->IsPrimary();
            });

        return it != m_nvapiOutputs.end() ? *it : nullptr;
    }

    bool NvapiAdapterRegistry::IsOutput(NvapiOutput* handle) const {
        return std::find(m_nvapiOutputs.begin(), m_nvapiOutputs.end(), handle) != m_nvapiOutputs.end();
    }
}
