#pragma once

#include "../nvapi_private.h"

namespace dxvk {
    // Minimal subset of the public CUDA driver API (cuda.h) that is needed
    // to correlate CUDA devices with DXGI adapters.
    using CUresult = int;
    using CUdevice = int;
    constexpr CUresult CUDA_SUCCESS = 0;

    class NvCuda {

      public:
        NvCuda();
        virtual ~NvCuda();

        [[nodiscard]] virtual bool IsAvailable() const;
        [[nodiscard]] virtual CUresult Init(unsigned int flags) const;
        [[nodiscard]] virtual CUresult DeviceGetCount(int* count) const;
        [[nodiscard]] virtual CUresult DeviceGet(CUdevice* device, int ordinal) const;
        [[nodiscard]] virtual CUresult DeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice device) const;

      private:
        HMODULE m_nvcudaModule{};

        // CUDAAPI is __stdcall on Windows
        using PFN_cuInit = CUresult(__stdcall*)(unsigned int);
        using PFN_cuDeviceGetCount = CUresult(__stdcall*)(int*);
        using PFN_cuDeviceGet = CUresult(__stdcall*)(CUdevice*, int);
        using PFN_cuDeviceGetLuid = CUresult(__stdcall*)(char*, unsigned int*, CUdevice);

        PFN_cuInit m_cuInit{};
        PFN_cuDeviceGetCount m_cuDeviceGetCount{};
        PFN_cuDeviceGet m_cuDeviceGet{};
        PFN_cuDeviceGetLuid m_cuDeviceGetLuid{};

        template <typename T>
        T GetProcAddress(const char* name);
    };
}
