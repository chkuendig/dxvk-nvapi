#include "nvcuda.h"
#include "../util/util_string.h"
#include "../util/util_log.h"

namespace dxvk {
    NvCuda::NvCuda() {
        const auto nvcudaModuleName = "nvcuda.dll";
        m_nvcudaModule = ::LoadLibraryA(nvcudaModuleName);
        if (m_nvcudaModule)
            log::info(str::format("Successfully loaded ", nvcudaModuleName));

        if (!m_nvcudaModule) {
            auto lastError = ::GetLastError();
            if (lastError != ERROR_MOD_NOT_FOUND) // Ignore library not found
                log::info(str::format("Loading ", nvcudaModuleName, " failed with error code: ", lastError));

            return;
        }

        m_cuInit = GetProcAddress<PFN_cuInit>("cuInit");
        m_cuDeviceGetCount = GetProcAddress<PFN_cuDeviceGetCount>("cuDeviceGetCount");
        m_cuDeviceGet = GetProcAddress<PFN_cuDeviceGet>("cuDeviceGet");
        m_cuDeviceGetAttribute = GetProcAddress<PFN_cuDeviceGetAttribute>("cuDeviceGetAttribute");

        if (m_cuInit && m_cuDeviceGetCount && m_cuDeviceGet && m_cuDeviceGetAttribute)
            return;

        log::info(str::format(nvcudaModuleName, " loaded but does not provide the required CUDA driver API entry points"));

        ::FreeLibrary(m_nvcudaModule);
        m_nvcudaModule = nullptr;
    }

    NvCuda::~NvCuda() {
        if (!m_nvcudaModule)
            return;

        ::FreeLibrary(m_nvcudaModule);
        m_nvcudaModule = nullptr;
    }

    bool NvCuda::IsAvailable() const {
        return m_nvcudaModule != nullptr;
    }

    CUresult NvCuda::Init(unsigned int flags) const {
        return m_cuInit(flags);
    }

    CUresult NvCuda::DeviceGetCount(int* count) const {
        return m_cuDeviceGetCount(count);
    }

    CUresult NvCuda::DeviceGet(CUdevice* device, int ordinal) const {
        return m_cuDeviceGet(device, ordinal);
    }

    CUresult NvCuda::DeviceGetAttribute(int* value, CUdevice_attribute attribute, CUdevice device) const {
        return m_cuDeviceGetAttribute(value, attribute, device);
    }

    template <typename T>
    T NvCuda::GetProcAddress(const char* name) {
        return reinterpret_cast<T>(reinterpret_cast<void*>(::GetProcAddress(m_nvcudaModule, name)));
    }
}
