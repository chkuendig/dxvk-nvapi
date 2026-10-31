#pragma once

#include "nvapi_private.h"

// Undocumented entry points that are not part of the NVAPI SDK. Their signatures and
// semantics are inferred from callers, the native ones are not known to be identical.

// Queried by legacy PhysX, e.g. PhysXDevice 18.12.7.2 calls it with three additional zero
// arguments. Only the first argument is used, the cdecl variadic signature accepts both shapes.
NVAPI_FUNCTION NvAPI_CUDA_EnumComputeCapableByTopology(NV_COMPUTE_GPU_TOPOLOGY_V2* pComputeTopo, ...);

static constexpr struct {
    const char* func;
    NvU32 id;
} nvapi_undocumented_interface_table[] = {
    { "NvAPI_CUDA_EnumComputeCapableByTopology", 0x36e39e6b },
};
