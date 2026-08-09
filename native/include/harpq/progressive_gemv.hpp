#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>

#include "harpq/native/stage_codec.hpp"

namespace harpq::native_runtime {

// Geometry copied by value into a GEMV launch. Scale storage follows the
// StageCodec layout [tile-grid-row, tile-grid-column, row-in-tile, group].
struct ProgressiveGemvGeometry {
  std::uint64_t original_rows = 0;
  std::uint64_t original_columns = 0;
  std::uint64_t padded_rows = 0;
  std::uint64_t padded_columns = 0;
  std::uint64_t tile_rows = 0;
  std::uint64_t tile_columns = 0;
  std::uint64_t group_size = 0;
};

// Both pointers refer to device-resident storage. scale_fp16_bits contains the
// raw IEEE-754 binary16 bit patterns, not host-decoded float values.
struct ProgressiveGemvStageDeviceView {
  const std::uint8_t* packed_codes = nullptr;
  std::uint64_t packed_code_bytes = 0;
  const std::uint16_t* scale_fp16_bits = nullptr;
  std::uint64_t scale_count = 0;
};

struct ProgressiveGemvDeviceView {
  ProgressiveGemvGeometry geometry{};
  ProgressiveGemvStageDeviceView stages[kMaximumStageCount]{};
  std::size_t stage_count = 0;
};

struct ProgressiveGemvIoDeviceView {
  const float* input = nullptr;
  std::uint64_t input_count = 0;
  const float* bias = nullptr;
  std::uint64_t bias_count = 0;
  float* output = nullptr;
  std::uint64_t output_count = 0;
};

// Enqueues a batch-one progressive GEMV on stream. Input, bias (when non-null),
// output, packed codes, and FP16 scale bits must all be device-resident and
// remain alive until the kernel has completed (normally proven by a recorded
// completion event). No FP32 weight matrix is materialized. Prefix zero is
// valid and writes bias or exact zeros.
//
// This entry point validates scalar geometry and required pointers, but cannot
// validate device allocation lengths or inspect device values. The caller owns
// stream synchronization and asynchronous error handling.
[[nodiscard]] cudaError_t LaunchProgressiveGemv(
    const ProgressiveGemvIoDeviceView& io,
    const ProgressiveGemvDeviceView& matrix,
    std::size_t prefix,
    cudaStream_t stream) noexcept;

// Correctness-first owning wrapper for tests and integration bring-up. It
// validates the complete host representation, converts its exactly representable
// scale values back to raw FP16, owns all temporary device allocations, and
// synchronizes before returning. Unlike the production launch, this wrapper is
// intentionally not a performance measurement path.
[[nodiscard]] std::vector<float> ProgressiveGemvCuda(
    const std::vector<float>& input,
    const ProgressiveMatrix& matrix,
    std::size_t prefix,
    const std::vector<float>& bias = {});

}  // namespace harpq::native_runtime
