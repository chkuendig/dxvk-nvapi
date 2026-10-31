#pragma once

#include "../../src/nvapi_private.h"
#include "../../src/nvapi/nvcuda.h"

class NvCudaMock final : public trompeloeil::mock_interface<dxvk::NvCuda> {
    IMPLEMENT_CONST_MOCK0(IsAvailable);
    IMPLEMENT_CONST_MOCK1(Init);
    IMPLEMENT_CONST_MOCK1(DeviceGetCount);
    IMPLEMENT_CONST_MOCK2(DeviceGet);
    IMPLEMENT_CONST_MOCK3(DeviceGetLuid);
};
