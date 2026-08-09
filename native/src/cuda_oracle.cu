#include "harpq/cuda_oracle.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace harpq::native_runtime {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaxOracleBlocks = 65535;

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

const char* CublasStatusName(cublasStatus_t status) noexcept {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
#if defined(CUBLAS_STATUS_NOT_SUPPORTED)
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if defined(CUBLAS_STATUS_LICENSE_ERROR)
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    default:
      return "CUBLAS_STATUS_UNKNOWN";
  }
}

void CheckCublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream message;
    message << operation << " failed: " << CublasStatusName(status) << " ("
            << static_cast<int>(status) << ')';
    throw std::runtime_error(message.str());
  }
}

std::size_t CheckedMultiply(std::size_t left, std::size_t right, const char* label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error(std::string(label) + " size overflows size_t");
  }
  return left * right;
}

int CheckedCublasDimension(std::size_t value, const char* label) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(std::string(label) + " exceeds the cuBLAS int range");
  }
  return static_cast<int>(value);
}

unsigned int OracleBlockCount(std::size_t element_count) {
  const std::size_t required =
      element_count / static_cast<std::size_t>(kThreadsPerBlock) +
      (element_count % static_cast<std::size_t>(kThreadsPerBlock) != 0 ? 1U : 0U);
  return static_cast<unsigned int>(
      std::min<std::size_t>(required, kMaxOracleBlocks));
}

template <typename T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ != 0) {
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data_), CheckedMultiply(count_, sizeof(T),
                                                                     "device allocation")),
                "cudaMalloc");
    }
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      // Destructors must not throw.  Every operation which matters to the
      // caller is synchronized and checked before this cleanup path.
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
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

class CublasHandle final {
 public:
  CublasHandle() {
    CheckCublas(cublasCreate(&handle_), "cublasCreate");
    cublasStatus_t status = cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH);
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasSetAtomicsMode(handle_, CUBLAS_ATOMICS_NOT_ALLOWED);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
      static_cast<void>(cublasDestroy(handle_));
      handle_ = nullptr;
      CheckCublas(status, "configure deterministic cuBLAS oracle");
    }
  }

  ~CublasHandle() {
    if (handle_ != nullptr) {
      static_cast<void>(cublasDestroy(handle_));
    }
  }

  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;

  [[nodiscard]] cublasHandle_t get() const noexcept { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

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
  std::size_t padded_elements;
  std::size_t original_elements;
  std::size_t expected_code_bytes;
  std::size_t expected_scale_count;
};

ValidatedGeometry ValidateMatrix(const ProgressiveMatrix& matrix, std::size_t prefix) {
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
  const auto ceil_div = [](std::size_t numerator, std::size_t denominator) {
    return numerator / denominator + (numerator % denominator != 0 ? 1U : 0U);
  };
  const std::size_t expected_padded_rows =
      CheckedMultiply(ceil_div(geometry.original_rows, geometry.tile_rows),
                      geometry.tile_rows, "padded rows");
  const std::size_t expected_padded_columns =
      CheckedMultiply(ceil_div(geometry.original_columns, geometry.tile_columns),
                      geometry.tile_columns, "padded columns");
  if (geometry.padded_rows != expected_padded_rows ||
      geometry.padded_columns != expected_padded_columns) {
    throw std::invalid_argument("padded matrix shape disagrees with tile geometry");
  }
  if (matrix.stages.empty() || matrix.stages.size() > kMaximumStageCount) {
    throw std::invalid_argument("a progressive matrix must contain between one and four stages");
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
      0,
      0,
  };
  result.padded_elements =
      CheckedMultiply(result.padded_rows, result.padded_columns, "padded matrix");
  result.original_elements =
      CheckedMultiply(result.original_rows, result.original_columns, "original matrix");
  result.expected_code_bytes =
      result.padded_elements / 4 + (result.padded_elements % 4 != 0 ? 1U : 0U);
  result.expected_scale_count =
      CheckedMultiply(result.tile_grid_rows, result.tile_grid_columns, "scale grid");
  result.expected_scale_count =
      CheckedMultiply(result.expected_scale_count, result.tile_rows, "scale rows");
  result.expected_scale_count =
      CheckedMultiply(result.expected_scale_count, result.groups_per_tile, "scale groups");

  const ScaleShape expected_shape = {
      result.tile_grid_rows,
      result.tile_grid_columns,
      result.tile_rows,
      result.groups_per_tile,
  };
  for (std::size_t stage_index = 0; stage_index < matrix.stages.size(); ++stage_index) {
    const DecodedStage& stage = matrix.stages[stage_index];
    if (stage.stage_index != stage_index) {
      throw std::invalid_argument("stages must form a zero-based contiguous prefix");
    }
    if (stage.packed_codes.size() != result.expected_code_bytes) {
      throw std::invalid_argument("stage packed-code byte count disagrees with geometry");
    }
    if (stage.scale_shape != expected_shape ||
        stage.scales.size() != result.expected_scale_count) {
      throw std::invalid_argument("stage scale shape disagrees with geometry");
    }
    for (float scale : stage.scales) {
      if (!std::isfinite(scale) || scale < 0.0F) {
        throw std::invalid_argument("stage scales must be finite and non-negative");
      }
    }
  }
  return result;
}

void ValidateFinite(const std::vector<float>& values, const char* label) {
  for (float value : values) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(std::string(label) + " must contain only finite values");
    }
  }
}

__global__ void AccumulateDequantizedStageKernel(
    const std::uint8_t* packed_codes,
    const float* scales,
    float* weights,
    std::uint64_t original_rows,
    std::uint64_t original_columns,
    std::uint64_t padded_columns,
    std::uint64_t tile_rows,
    std::uint64_t tile_columns,
    std::uint64_t group_size,
    std::uint64_t tile_grid_columns,
    std::uint64_t groups_per_tile) {
  const std::uint64_t output_elements = original_rows * original_columns;
  const std::uint64_t start =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t output_index = start; output_index < output_elements;
       output_index += stride) {
    const std::uint64_t row = output_index / original_columns;
    const std::uint64_t column = output_index % original_columns;
    const std::uint64_t padded_index = row * padded_columns + column;
    const std::uint8_t encoded = static_cast<std::uint8_t>(
        (packed_codes[padded_index / 4] >> (2 * (padded_index % 4))) & 0x03U);
    const int code = encoded >= 2U ? static_cast<int>(encoded) - 4
                                   : static_cast<int>(encoded);

    const std::uint64_t tile_row = row / tile_rows;
    const std::uint64_t row_in_tile = row % tile_rows;
    const std::uint64_t tile_column = column / tile_columns;
    const std::uint64_t group_in_tile =
        (column % tile_columns) / group_size;
    const std::uint64_t scale_index =
        (((tile_row * tile_grid_columns + tile_column) * tile_rows + row_in_tile) *
             groups_per_tile +
         group_in_tile);
    weights[output_index] += static_cast<float>(code) * scales[scale_index];
  }
}

__global__ void AddBiasKernel(float* output,
                              const float* bias,
                              std::uint64_t batch_size,
                              std::uint64_t output_features) {
  const std::uint64_t element_count = batch_size * output_features;
  const std::uint64_t start =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = start; index < element_count; index += stride) {
    output[index] += bias[index % output_features];
  }
}

}  // namespace

bool CudaDeviceAvailable(std::string* reason) noexcept {
  try {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
      if (reason != nullptr) {
        std::ostringstream message;
        message << cudaGetErrorName(status) << ": " << cudaGetErrorString(status);
        *reason = message.str();
      }
      // Clear CUDA's per-thread last-error state after a diagnostic probe.
      static_cast<void>(cudaGetLastError());
      return false;
    }
    if (device_count <= 0) {
      if (reason != nullptr) {
        *reason = "CUDA runtime reported no devices";
      }
      return false;
    }
    if (reason != nullptr) {
      reason->clear();
    }
    return true;
  } catch (...) {
    if (reason != nullptr) {
      try {
        *reason = "CUDA device probe failed while formatting its diagnostic";
      } catch (...) {
        // Preserve noexcept even if the diagnostic allocation itself fails.
      }
    }
    return false;
  }
}

std::vector<float> ProgressiveLinearCudaOracle(const std::vector<float>& inputs,
                                               std::size_t batch_size,
                                               const ProgressiveMatrix& matrix,
                                               std::size_t prefix,
                                               const std::vector<float>& bias) {
  const ValidatedGeometry geometry = ValidateMatrix(matrix, prefix);
  const std::size_t expected_inputs =
      CheckedMultiply(batch_size, geometry.original_columns, "input");
  if (inputs.size() != expected_inputs) {
    throw std::invalid_argument("input element count disagrees with batch and matrix geometry");
  }
  if (!bias.empty() && bias.size() != geometry.original_rows) {
    throw std::invalid_argument("bias element count disagrees with output features");
  }
  ValidateFinite(inputs, "inputs");
  ValidateFinite(bias, "bias");

  // Match the CPU reference's well-defined empty-batch behavior without
  // issuing a zero-sized cuBLAS operation.
  if (batch_size == 0) {
    return {};
  }

  const std::size_t output_elements =
      CheckedMultiply(batch_size, geometry.original_rows, "output");
  const int output_features =
      CheckedCublasDimension(geometry.original_rows, "output feature count");
  const int input_features =
      CheckedCublasDimension(geometry.original_columns, "input feature count");
  const int batches = CheckedCublasDimension(batch_size, "batch size");

  DeviceBuffer<float> device_weights(geometry.original_elements);
  DeviceBuffer<float> device_inputs(inputs.size());
  DeviceBuffer<float> device_output(output_elements);
  CheckCuda(cudaMemset(device_weights.get(), 0,
                       CheckedMultiply(device_weights.size(), sizeof(float),
                                       "weight memset")),
            "cudaMemset(weights)");
  CheckCuda(cudaMemcpy(device_inputs.get(), inputs.data(),
                       CheckedMultiply(inputs.size(), sizeof(float), "input copy"),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(inputs H2D)");

  for (std::size_t stage_index = 0; stage_index < prefix; ++stage_index) {
    const DecodedStage& stage = matrix.stages[stage_index];
    DeviceBuffer<std::uint8_t> device_codes(stage.packed_codes.size());
    DeviceBuffer<float> device_scales(stage.scales.size());
    CheckCuda(cudaMemcpy(device_codes.get(), stage.packed_codes.data(),
                         stage.packed_codes.size(), cudaMemcpyHostToDevice),
              "cudaMemcpy(stage codes H2D)");
    CheckCuda(cudaMemcpy(device_scales.get(), stage.scales.data(),
                         CheckedMultiply(stage.scales.size(), sizeof(float),
                                         "stage scale copy"),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(stage scales H2D)");

    AccumulateDequantizedStageKernel<<<OracleBlockCount(geometry.original_elements),
                                       kThreadsPerBlock>>>(
        device_codes.get(), device_scales.get(), device_weights.get(),
        geometry.original_rows, geometry.original_columns, geometry.padded_columns,
        geometry.tile_rows, geometry.tile_columns, geometry.group_size,
        geometry.tile_grid_columns, geometry.groups_per_tile);
    CheckCuda(cudaGetLastError(), "AccumulateDequantizedStageKernel launch");
    // The stage buffers are scoped to this iteration.  Synchronization makes
    // their lifetime explicit and turns asynchronous kernel faults into a
    // precise exception at the offending stage.
    CheckCuda(cudaDeviceSynchronize(), "AccumulateDequantizedStageKernel execution");
  }

  CublasHandle handle;
  const float alpha = 1.0F;
  const float beta = 0.0F;
  // Row-major W[O,I] is column-major W_col[I,O], and row-major X[B,I]
  // is column-major X_col[I,B].  C_col[O,B] = W_col^T * X_col has the
  // same storage as the desired row-major Y[B,O].
  CheckCublas(cublasSgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, output_features,
                          batches, input_features, &alpha, device_weights.get(),
                          input_features, device_inputs.get(), input_features, &beta,
                          device_output.get(), output_features),
              "cublasSgemm(progressive linear oracle)");

  DeviceBuffer<float> device_bias;
  if (!bias.empty()) {
    device_bias = DeviceBuffer<float>(bias.size());
    CheckCuda(cudaMemcpy(device_bias.get(), bias.data(),
                         CheckedMultiply(bias.size(), sizeof(float), "bias copy"),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(bias H2D)");
    AddBiasKernel<<<OracleBlockCount(output_elements), kThreadsPerBlock>>>(
        device_output.get(), device_bias.get(), batch_size, geometry.original_rows);
    CheckCuda(cudaGetLastError(), "AddBiasKernel launch");
  }

  std::vector<float> output(output_elements);
  CheckCuda(cudaMemcpy(output.data(), device_output.get(),
                       CheckedMultiply(output.size(), sizeof(float), "output copy"),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy(output D2H)");
  return output;
}

}  // namespace harpq::native_runtime
