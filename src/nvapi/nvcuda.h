#pragma once

#include "../nvapi_private.h"

namespace dxvk {
    // Minimal subset of the public CUDA driver API (cuda.h) that is needed
    // to correlate CUDA devices with DXGI adapters, all available since CUDA 4.0
    using CUresult = int;
    using CUdevice = int;
    using CUdevice_attribute = int;
    constexpr CUresult CUDA_SUCCESS = 0;
    constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33;
    constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34;
    constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50;

    class NvCuda {

      public:
        NvCuda();
        virtual ~NvCuda();

        [[nodiscard]] virtual bool IsAvailable() const;
        [[nodiscard]] virtual CUresult Init(unsigned int flags) const;
        [[nodiscard]] virtual CUresult DeviceGetCount(int* count) const;
        [[nodiscard]] virtual CUresult DeviceGet(CUdevice* device, int ordinal) const;
        [[nodiscard]] virtual CUresult DeviceGetAttribute(int* value, CUdevice_attribute attribute, CUdevice device) const;

      private:
        HMODULE m_nvcudaModule{};

        // CUDAAPI is __stdcall on Windows
        using PFN_cuInit = CUresult(__stdcall*)(unsigned int);
        using PFN_cuDeviceGetCount = CUresult(__stdcall*)(int*);
        using PFN_cuDeviceGet = CUresult(__stdcall*)(CUdevice*, int);
        using PFN_cuDeviceGetAttribute = CUresult(__stdcall*)(int*, CUdevice_attribute, CUdevice);

        PFN_cuInit m_cuInit{};
        PFN_cuDeviceGetCount m_cuDeviceGetCount{};
        PFN_cuDeviceGet m_cuDeviceGet{};
        PFN_cuDeviceGetAttribute m_cuDeviceGetAttribute{};

        template <typename T>
        T GetProcAddress(const char* name);
    };
}
