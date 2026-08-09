#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "harpq/native/stage_codec.hpp"

namespace harpq::native_runtime {

// Performs a real CUDA runtime probe.
// On failure, reason receives a human-readable explanation when it is non-null.
[[nodiscard]] bool CudaDeviceAvailable(std::string* reason = nullptr) noexcept;

// Correctness-first CUDA implementation of the progressive linear contract.
//
// The selected residual prefix is decoded from packed signed-int2 codes and
// materialized as an FP32 row-major weight matrix on the GPU.  The linear
// operation is then evaluated with cuBLAS SGEMM.  This deliberately simple
// path is an oracle for future fused kernels, not a throughput implementation.
// Prefix zero is valid and represents an all-zero weight matrix.  Bias, when
// supplied, is added exactly once.
[[nodiscard]] std::vector<float> ProgressiveLinearCudaOracle(
    const std::vector<float>& inputs,
    std::size_t batch_size,
    const ProgressiveMatrix& matrix,
    std::size_t prefix,
    const std::vector<float>& bias = {});

}  // namespace harpq::native_runtime
