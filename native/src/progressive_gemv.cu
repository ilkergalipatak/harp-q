#include "harpq/progressive_gemv.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace harpq::native_runtime {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaximumBlocks = 65535;

static_assert(kThreadsPerBlock % 32 == 0,
              "progressive GEMV reduction requires whole warps");
static_assert(kMaximumStageCount == 4,
              "the fused kernel is specialized for four progressive stages");
static_assert(std::is_trivially_copyable_v<ProgressiveGemvDeviceView>,
              "the device view must be safe to pass as a kernel argument");

[[noreturn]] void ThrowCudaError(cudaError_t status, const char* operation) {
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorName(status) << " ("
          << cudaGetErrorString(status) << ')';
  throw std::runtime_error(message.str());
}

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    ThrowCudaError(status, operation);
  }
}

std::size_t CheckedMultiply(std::size_t left,
                            std::size_t right,
                            const char* label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error(std::string(label) + " size overflows size_t");
  }
  return left * right;
}

std::size_t CeilDiv(std::size_t numerator, std::size_t denominator) {
  return numerator / denominator + (numerator % denominator != 0 ? 1U : 0U);
}

struct ValidatedGeometry {
  std::size_t original_rows;
  std::size_t original_columns;
  std::size_t padded_rows;
  std::size_t padded_columns;
  std::size_t tile_rows;
  std::size_t tile_columns;
  std::size_t group_size;
  std::size_t tile_grid_rows;
  std::size_t tile_grid_columns;
  std::size_t groups_per_tile;
  std::size_t expected_code_bytes;
  std::size_t expected_scale_count;
};

ValidatedGeometry ValidateHostMatrix(const ProgressiveMatrix& matrix,
                                     std::size_t prefix) {
  const MatrixGeometry& geometry = matrix.geometry;
  if (geometry.original_rows == 0 || geometry.original_columns == 0 ||
      geometry.padded_rows == 0 || geometry.padded_columns == 0 ||
      geometry.tile_rows == 0 || geometry.tile_columns == 0 ||
      geometry.group_size == 0) {
    throw std::invalid_argument("matrix geometry dimensions must be positive");
  }
  if (geometry.tile_columns % geometry.group_size != 0) {
    throw std::invalid_argument("tile width must be divisible by group size");
  }

  const std::size_t expected_padded_rows =
      CheckedMultiply(CeilDiv(geometry.original_rows, geometry.tile_rows),
                      geometry.tile_rows, "padded rows");
  const std::size_t expected_padded_columns =
      CheckedMultiply(CeilDiv(geometry.original_columns, geometry.tile_columns),
                      geometry.tile_columns, "padded columns");
  if (geometry.padded_rows != expected_padded_rows ||
      geometry.padded_columns != expected_padded_columns) {
    throw std::invalid_argument(
        "padded matrix shape disagrees with tile geometry");
  }
  if (matrix.stages.empty() || matrix.stages.size() > kMaximumStageCount) {
    throw std::invalid_argument(
        "a progressive matrix must contain between one and four stages");
  }
  if (prefix > matrix.stages.size()) {
    throw std::invalid_argument("prefix exceeds the available stage count");
  }

  ValidatedGeometry result{
      geometry.original_rows,
      geometry.original_columns,
      geometry.padded_rows,
      geometry.padded_columns,
      geometry.tile_rows,
      geometry.tile_columns,
      geometry.group_size,
      geometry.padded_rows / geometry.tile_rows,
      geometry.padded_columns / geometry.tile_columns,
      geometry.tile_columns / geometry.group_size,
      0,
      0,
  };
  const std::size_t padded_elements =
      CheckedMultiply(result.padded_rows, result.padded_columns,
                      "padded matrix");
  result.expected_code_bytes =
      padded_elements / 4 + (padded_elements % 4 != 0 ? 1U : 0U);
  result.expected_scale_count =
      CheckedMultiply(result.tile_grid_rows, result.tile_grid_columns,
                      "scale grid");
  result.expected_scale_count =
      CheckedMultiply(result.expected_scale_count, result.tile_rows,
                      "scale rows");
  result.expected_scale_count =
      CheckedMultiply(result.expected_scale_count, result.groups_per_tile,
                      "scale groups");

  const ScaleShape expected_shape = {
      result.tile_grid_rows,
      result.tile_grid_columns,
      result.tile_rows,
      result.groups_per_tile,
  };
  for (std::size_t stage_index = 0; stage_index < matrix.stages.size();
       ++stage_index) {
    const DecodedStage& stage = matrix.stages[stage_index];
    if (stage.stage_index != stage_index) {
      throw std::invalid_argument(
          "stages must form a zero-based contiguous prefix");
    }
    if (stage.packed_codes.size() != result.expected_code_bytes) {
      throw std::invalid_argument(
          "stage packed-code byte count disagrees with geometry");
    }
    if (stage.scale_shape != expected_shape ||
        stage.scales.size() != result.expected_scale_count) {
      throw std::invalid_argument(
          "stage scale shape disagrees with geometry");
    }
    for (float scale : stage.scales) {
      if (!std::isfinite(scale) || scale < 0.0F) {
        throw std::invalid_argument(
            "stage scales must be finite and non-negative");
      }
      const __half encoded = __float2half_rn(scale);
      const float decoded = __half2float(encoded);
      if (!std::isfinite(decoded)) {
        throw std::invalid_argument(
            "stage scale is outside the finite FP16 range");
      }
      if (decoded != scale) {
        throw std::invalid_argument(
            "stage scale is not exactly representable as FP16");
      }
    }
  }
  return result;
}

void ValidateFinite(const std::vector<float>& values, const char* label) {
  for (float value : values) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(std::string(label) +
                                  " must contain only finite values");
    }
  }
}

template <typename T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ != 0) {
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                           CheckedMultiply(count_, sizeof(T),
                                           "device allocation")),
                "cudaMalloc");
    }
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      static_cast<void>(cudaFree(data_));
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        static_cast<void>(cudaFree(data_));
      }
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] const T* get() const noexcept { return data_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

class CudaStream final {
 public:
  CudaStream() {
    CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags");
  }

  ~CudaStream() {
    if (stream_ != nullptr) {
      // On exceptional exits, finish queued operations before host staging
      // vectors and device buffers are destroyed.
      static_cast<void>(cudaStreamSynchronize(stream_));
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

bool CheckedMultiplyU64(std::uint64_t left,
                        std::uint64_t right,
                        std::uint64_t* result) noexcept {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

bool ValidateDeviceView(const ProgressiveGemvIoDeviceView& io,
                        const ProgressiveGemvDeviceView& matrix,
                        std::size_t prefix) noexcept {
  const ProgressiveGemvGeometry& geometry = matrix.geometry;
  if (geometry.original_rows == 0 || geometry.original_columns == 0 ||
      geometry.padded_rows == 0 || geometry.padded_columns == 0 ||
      geometry.tile_rows == 0 || geometry.tile_columns == 0 ||
      geometry.group_size == 0 ||
      geometry.tile_columns % geometry.group_size != 0 ||
      matrix.stage_count == 0 || matrix.stage_count > kMaximumStageCount ||
      prefix > matrix.stage_count || io.input == nullptr ||
      io.input_count != geometry.original_columns || io.output == nullptr ||
      io.output_count != geometry.original_rows ||
      ((io.bias == nullptr) != (io.bias_count == 0)) ||
      (io.bias != nullptr && io.bias_count != geometry.original_rows)) {
    return false;
  }

  const std::uint64_t grid_rows =
      geometry.original_rows / geometry.tile_rows +
      (geometry.original_rows % geometry.tile_rows != 0 ? 1U : 0U);
  const std::uint64_t grid_columns =
      geometry.original_columns / geometry.tile_columns +
      (geometry.original_columns % geometry.tile_columns != 0 ? 1U : 0U);
  std::uint64_t expected_padded_rows = 0;
  std::uint64_t expected_padded_columns = 0;
  std::uint64_t padded_elements = 0;
  std::uint64_t scale_grid = 0;
  if (!CheckedMultiplyU64(grid_rows, geometry.tile_rows,
                          &expected_padded_rows) ||
      !CheckedMultiplyU64(grid_columns, geometry.tile_columns,
                          &expected_padded_columns) ||
      expected_padded_rows != geometry.padded_rows ||
      expected_padded_columns != geometry.padded_columns ||
      !CheckedMultiplyU64(geometry.padded_rows, geometry.padded_columns,
                          &padded_elements) ||
      !CheckedMultiplyU64(grid_rows, grid_columns, &scale_grid) ||
      !CheckedMultiplyU64(scale_grid, geometry.tile_rows, &scale_grid) ||
      !CheckedMultiplyU64(scale_grid,
                          geometry.tile_columns / geometry.group_size,
                          &scale_grid)) {
    return false;
  }
  const std::uint64_t expected_code_bytes =
      padded_elements / 4 + (padded_elements % 4 != 0 ? 1U : 0U);
  const std::uint64_t expected_scale_count = scale_grid;
  for (std::size_t stage_index = 0; stage_index < prefix; ++stage_index) {
    if (matrix.stages[stage_index].packed_codes == nullptr ||
        matrix.stages[stage_index].packed_code_bytes != expected_code_bytes ||
        matrix.stages[stage_index].scale_fp16_bits == nullptr ||
        matrix.stages[stage_index].scale_count != expected_scale_count) {
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ float WarpSum(float value) {
  constexpr unsigned int kFullWarpMask = 0xffffffffU;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return value;
}

__device__ __forceinline__ float BlockSum(float value) {
  __shared__ float warp_sums[kThreadsPerBlock / 32];
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  value = WarpSum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
    value = lane < (blockDim.x + 31U) / 32U ? warp_sums[lane] : 0.0F;
    value = WarpSum(value);
  }
  return value;
}

__global__ void ProgressiveGemvKernel(const float* input,
                                      ProgressiveGemvDeviceView matrix,
                                      std::uint64_t prefix,
                                      const float* bias,
                                      float* output) {
  const ProgressiveGemvGeometry geometry = matrix.geometry;
  const std::uint64_t tile_grid_columns =
      geometry.padded_columns / geometry.tile_columns;
  const std::uint64_t groups_per_tile =
      geometry.tile_columns / geometry.group_size;

  for (std::uint64_t row = blockIdx.x; row < geometry.original_rows;) {
    float thread_sum = 0.0F;
    for (std::uint64_t column = threadIdx.x;
         column < geometry.original_columns;) {
      const std::uint64_t padded_index = row * geometry.padded_columns + column;
      const std::uint64_t tile_row = row / geometry.tile_rows;
      const std::uint64_t row_in_tile = row % geometry.tile_rows;
      const std::uint64_t tile_column = column / geometry.tile_columns;
      const std::uint64_t group_in_tile =
          (column % geometry.tile_columns) / geometry.group_size;
      const std::uint64_t scale_index =
          (((tile_row * tile_grid_columns + tile_column) * geometry.tile_rows +
            row_in_tile) *
               groups_per_tile +
           group_in_tile);

      float progressive_weight = 0.0F;
#pragma unroll
      for (std::uint64_t stage_index = 0; stage_index < kMaximumStageCount;
           ++stage_index) {
        if (stage_index < prefix) {
          const std::uint8_t encoded = static_cast<std::uint8_t>(
              (matrix.stages[stage_index].packed_codes[padded_index / 4] >>
               (2 * (padded_index % 4))) &
              0x03U);
          const int code = encoded >= 2U ? static_cast<int>(encoded) - 4
                                         : static_cast<int>(encoded);
          const __half scale = __ushort_as_half(
              matrix.stages[stage_index].scale_fp16_bits[scale_index]);
          progressive_weight = fmaf(static_cast<float>(code),
                                    __half2float(scale), progressive_weight);
        }
      }
      thread_sum = fmaf(input[column], progressive_weight, thread_sum);
      if (geometry.original_columns - column <= blockDim.x) {
        break;
      }
      column += blockDim.x;
    }

    const float sum = BlockSum(thread_sum);
    if (threadIdx.x == 0) {
      output[row] = sum + (bias == nullptr ? 0.0F : bias[row]);
    }
    // Blocks may cover more than one row when original_rows exceeds the grid
    // limit. Do not reuse reduction shared memory until every warp is done.
    __syncthreads();
    if (geometry.original_rows - row <= gridDim.x) {
      break;
    }
    row += gridDim.x;
  }
}

std::uint16_t EncodeFp16(float value) {
  const __half_raw encoded = __float2half_rn(value);
  return encoded.x;
}

}  // namespace

cudaError_t LaunchProgressiveGemv(const ProgressiveGemvIoDeviceView& io,
                                  const ProgressiveGemvDeviceView& matrix,
                                  std::size_t prefix,
                                  cudaStream_t stream) noexcept {
  if (!ValidateDeviceView(io, matrix, prefix)) {
    return cudaErrorInvalidValue;
  }
  // A stale thread-local CUDA error must not be reported as this kernel's
  // launch result. Surface it without enqueueing new work, then use the
  // standard clearing launch check below.
  const cudaError_t prior_error = cudaGetLastError();
  if (prior_error != cudaSuccess) {
    return prior_error;
  }
  const unsigned int blocks = static_cast<unsigned int>(
      std::min<std::uint64_t>(matrix.geometry.original_rows, kMaximumBlocks));
  ProgressiveGemvKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      io.input, matrix, static_cast<std::uint64_t>(prefix), io.bias, io.output);
  return cudaGetLastError();
}

std::vector<float> ProgressiveGemvCuda(const std::vector<float>& input,
                                       const ProgressiveMatrix& matrix,
                                       std::size_t prefix,
                                       const std::vector<float>& bias) {
  const ValidatedGeometry geometry = ValidateHostMatrix(matrix, prefix);
  if (input.size() != geometry.original_columns) {
    throw std::invalid_argument(
        "batch-one input element count disagrees with matrix geometry");
  }
  if (!bias.empty() && bias.size() != geometry.original_rows) {
    throw std::invalid_argument(
        "bias element count disagrees with output features");
  }
  ValidateFinite(input, "input");
  ValidateFinite(bias, "bias");

  DeviceBuffer<float> device_input(input.size());
  DeviceBuffer<float> device_output(geometry.original_rows);
  DeviceBuffer<float> device_bias(bias.size());
  std::array<DeviceBuffer<std::uint8_t>, kMaximumStageCount> device_codes;
  std::array<DeviceBuffer<std::uint16_t>, kMaximumStageCount> device_scales;
  std::array<std::vector<std::uint16_t>, kMaximumStageCount> encoded_scales;
  // Declared after every resource used by queued work, so its synchronizing
  // destructor runs first during stack unwinding.
  CudaStream stream;

  CheckCuda(cudaMemcpyAsync(device_input.get(), input.data(),
                            CheckedMultiply(input.size(), sizeof(float),
                                            "input copy"),
                            cudaMemcpyHostToDevice, stream.get()),
            "cudaMemcpyAsync(input H2D)");
  if (!bias.empty()) {
    CheckCuda(cudaMemcpyAsync(device_bias.get(), bias.data(),
                              CheckedMultiply(bias.size(), sizeof(float),
                                              "bias copy"),
                              cudaMemcpyHostToDevice, stream.get()),
              "cudaMemcpyAsync(bias H2D)");
  }

  ProgressiveGemvDeviceView device_view{};
  device_view.geometry = {
      geometry.original_rows,
      geometry.original_columns,
      geometry.padded_rows,
      geometry.padded_columns,
      geometry.tile_rows,
      geometry.tile_columns,
      geometry.group_size,
  };
  device_view.stage_count = matrix.stages.size();

  for (std::size_t stage_index = 0; stage_index < prefix; ++stage_index) {
    const DecodedStage& stage = matrix.stages[stage_index];
    encoded_scales[stage_index].reserve(stage.scales.size());
    for (float scale : stage.scales) {
      encoded_scales[stage_index].push_back(EncodeFp16(scale));
    }
    device_codes[stage_index] =
        DeviceBuffer<std::uint8_t>(stage.packed_codes.size());
    device_scales[stage_index] =
        DeviceBuffer<std::uint16_t>(encoded_scales[stage_index].size());
    CheckCuda(cudaMemcpyAsync(device_codes[stage_index].get(),
                              stage.packed_codes.data(),
                              stage.packed_codes.size(), cudaMemcpyHostToDevice,
                              stream.get()),
              "cudaMemcpyAsync(stage codes H2D)");
    CheckCuda(cudaMemcpyAsync(
                  device_scales[stage_index].get(),
                  encoded_scales[stage_index].data(),
                  CheckedMultiply(encoded_scales[stage_index].size(),
                                  sizeof(std::uint16_t), "stage scale copy"),
                  cudaMemcpyHostToDevice, stream.get()),
              "cudaMemcpyAsync(stage scales H2D)");
    device_view.stages[stage_index] = {
        device_codes[stage_index].get(),
        static_cast<std::uint64_t>(stage.packed_codes.size()),
        device_scales[stage_index].get(),
        static_cast<std::uint64_t>(encoded_scales[stage_index].size())};
  }

  const ProgressiveGemvIoDeviceView io{
      device_input.get(),
      static_cast<std::uint64_t>(input.size()),
      bias.empty() ? nullptr : device_bias.get(),
      static_cast<std::uint64_t>(bias.size()),
      device_output.get(),
      static_cast<std::uint64_t>(geometry.original_rows),
  };
  CheckCuda(LaunchProgressiveGemv(io, device_view, prefix, stream.get()),
            "ProgressiveGemvKernel launch");

  std::vector<float> output(geometry.original_rows);
  CheckCuda(cudaMemcpyAsync(output.data(), device_output.get(),
                            CheckedMultiply(output.size(), sizeof(float),
                                            "output copy"),
                            cudaMemcpyDeviceToHost, stream.get()),
            "cudaMemcpyAsync(output D2H)");
  CheckCuda(cudaStreamSynchronize(stream.get()),
            "ProgressiveGemvKernel execution");
  return output;
}

}  // namespace harpq::native_runtime
