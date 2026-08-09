#include "harpq/progressive_gemv.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using harpq::native_runtime::LaunchProgressiveGemv;
using harpq::native_runtime::ProgressiveGemvDeviceView;
using harpq::native_runtime::ProgressiveGemvIoDeviceView;

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaximumBlocks = 65535;
constexpr std::size_t kMaximumStageCount = 4;
constexpr std::size_t kMaximumDimension = 16384;
constexpr std::size_t kMaximumPaddedElements = std::size_t{1} << 26;
constexpr std::size_t kMaximumWarmup = 1000;
constexpr std::size_t kMaximumIterations = 10000;
constexpr std::size_t kMaximumRepeats = 31;
constexpr std::size_t kMaximumWeightedElementLaunches =
    std::size_t{1} << 41;
constexpr std::uint32_t kDataSeed = 0x48515250U;
constexpr float kStaticOracleAbsoluteTolerance = 2.0e-5F;
constexpr float kStaticOracleRelativeTolerance = 2.0e-6F;

static_assert(kThreadsPerBlock % 32 == 0,
              "matched GEMV reductions require whole warps");

struct Configuration {
  std::size_t rows = 4096;
  std::size_t columns = 4096;
  std::size_t tile = 128;
  std::size_t group = 128;
  std::size_t prefix = 2;
  std::size_t warmup = 10;
  std::size_t iterations = 100;
  std::size_t repeats = 7;
  int device = 0;
  bool json = false;
  bool help = false;
};

struct Geometry {
  std::size_t rows;
  std::size_t columns;
  std::size_t padded_rows;
  std::size_t padded_columns;
  std::size_t tile;
  std::size_t group;
  std::size_t tile_grid_rows;
  std::size_t tile_grid_columns;
  std::size_t groups_per_tile;
  std::size_t padded_elements;
  std::size_t int2_code_bytes;
  std::size_t int4_code_bytes;
  std::size_t scale_count;
};

struct TimingSummary {
  std::vector<double> microseconds_per_launch;
  double median_microseconds = 0.0;
  double minimum_microseconds = 0.0;
  double maximum_microseconds = 0.0;
};

struct PayloadSummary {
  std::size_t progressive_code_bytes;
  std::size_t progressive_scale_bytes;
  std::size_t progressive_model_bytes;
  std::size_t static_code_bytes;
  std::size_t static_scale_bytes;
  std::size_t static_model_bytes;
  std::size_t input_bytes;
  std::size_t output_bytes;
  std::size_t total_allocated_device_bytes;
};

struct OracleValidation {
  double maximum_absolute_error;
  double tolerance_at_maximum_error;
  std::size_t maximum_error_index;
};

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
    throw std::overflow_error(std::string(label) + " overflows size_t");
  }
  return left * right;
}

std::size_t CheckedAdd(std::size_t left,
                       std::size_t right,
                       const char* label) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error(std::string(label) + " overflows size_t");
  }
  return left + right;
}

std::size_t CeilDiv(std::size_t numerator, std::size_t denominator) {
  return numerator / denominator + (numerator % denominator != 0 ? 1U : 0U);
}

std::string JsonEscape(std::string_view value) {
  std::ostringstream escaped;
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        escaped << "\\\"";
        break;
      case '\\':
        escaped << "\\\\";
        break;
      case '\b':
        escaped << "\\b";
        break;
      case '\f':
        escaped << "\\f";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        if (character < 0x20U) {
          escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                  << static_cast<unsigned int>(character) << std::dec
                  << std::setfill(' ');
        } else {
          escaped << static_cast<char>(character);
        }
    }
  }
  return escaped.str();
}

std::size_t ParseUnsigned(std::string_view text, const char* option) {
  if (text.empty() || text.front() == '-' || text.front() == '+') {
    throw std::invalid_argument(std::string(option) +
                                " requires an unsigned integer");
  }
  std::size_t value = 0;
  const char* begin = text.data();
  const char* end = text.data() + text.size();
  const auto parsed = std::from_chars(begin, end, value);
  if (parsed.ec != std::errc{} || parsed.ptr != end) {
    throw std::invalid_argument(std::string(option) +
                                " requires an unsigned integer");
  }
  return value;
}

std::pair<std::string_view, std::string_view> SplitOption(
    std::string_view argument) {
  const std::size_t equals = argument.find('=');
  if (equals == std::string_view::npos) {
    return {argument, {}};
  }
  return {argument.substr(0, equals), argument.substr(equals + 1)};
}

Configuration ParseArguments(int argc, char** argv) {
  Configuration config;
  std::array<bool, 9> value_seen{};
  bool json_seen = false;
  bool help_seen = false;

  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    const auto [name, inline_value] = SplitOption(argument);
    if (name == "--json") {
      if (!inline_value.empty() || json_seen) {
        throw std::invalid_argument("--json is duplicated or has a value");
      }
      json_seen = true;
      config.json = true;
      continue;
    }
    if (name == "--help" || name == "-h") {
      if (!inline_value.empty() || help_seen) {
        throw std::invalid_argument("--help is duplicated or has a value");
      }
      help_seen = true;
      config.help = true;
      continue;
    }

    std::size_t option_index = 0;
    std::size_t* destination = nullptr;
    if (name == "--rows") {
      option_index = 0;
      destination = &config.rows;
    } else if (name == "--columns" || name == "--cols") {
      option_index = 1;
      destination = &config.columns;
    } else if (name == "--tile") {
      option_index = 2;
      destination = &config.tile;
    } else if (name == "--group") {
      option_index = 3;
      destination = &config.group;
    } else if (name == "--prefix") {
      option_index = 4;
      destination = &config.prefix;
    } else if (name == "--warmup") {
      option_index = 5;
      destination = &config.warmup;
    } else if (name == "--iterations") {
      option_index = 6;
      destination = &config.iterations;
    } else if (name == "--repeats") {
      option_index = 7;
      destination = &config.repeats;
    } else if (name == "--device") {
      option_index = 8;
    } else {
      throw std::invalid_argument("unknown option: " + std::string(name));
    }

    const std::size_t seen_index = option_index;
    if (value_seen[seen_index]) {
      throw std::invalid_argument("duplicate option: " + std::string(name));
    }
    value_seen[seen_index] = true;

    std::string_view value = inline_value;
    if (value.empty()) {
      if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for " + std::string(name));
      }
      value = argv[++index];
    }
    const std::size_t parsed = ParseUnsigned(value, std::string(name).c_str());
    if (name == "--device") {
      if (parsed > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("--device exceeds the int range");
      }
      config.device = static_cast<int>(parsed);
    } else {
      *destination = parsed;
    }
  }
  return config;
}

Geometry ValidateConfiguration(const Configuration& config) {
  if (config.rows == 0 || config.rows > kMaximumDimension ||
      config.columns == 0 || config.columns > kMaximumDimension) {
    throw std::invalid_argument(
        "--rows and --columns must be in [1, 16384]");
  }
  if (config.tile == 0 || config.tile > 1024) {
    throw std::invalid_argument("--tile must be in [1, 1024]");
  }
  if (config.group == 0 || config.group > config.tile ||
      config.tile % config.group != 0) {
    throw std::invalid_argument(
        "--group must be positive and divide --tile exactly");
  }
  if (config.prefix == 0 || config.prefix > kMaximumStageCount) {
    throw std::invalid_argument("--prefix must be in [1, 4]");
  }
  if (config.warmup > kMaximumWarmup) {
    throw std::invalid_argument("--warmup must be in [0, 1000]");
  }
  if (config.iterations == 0 || config.iterations > kMaximumIterations) {
    throw std::invalid_argument("--iterations must be in [1, 10000]");
  }
  if (config.repeats < 3 || config.repeats > kMaximumRepeats) {
    throw std::invalid_argument("--repeats must be in [3, 31]");
  }

  Geometry geometry{
      config.rows,
      config.columns,
      CheckedMultiply(CeilDiv(config.rows, config.tile), config.tile,
                      "padded rows"),
      CheckedMultiply(CeilDiv(config.columns, config.tile), config.tile,
                      "padded columns"),
      config.tile,
      config.group,
      0,
      0,
      config.tile / config.group,
      0,
      0,
      0,
      0,
  };
  geometry.tile_grid_rows = geometry.padded_rows / geometry.tile;
  geometry.tile_grid_columns = geometry.padded_columns / geometry.tile;
  geometry.padded_elements =
      CheckedMultiply(geometry.padded_rows, geometry.padded_columns,
                      "padded elements");
  if (geometry.padded_elements > kMaximumPaddedElements) {
    throw std::invalid_argument(
        "padded matrix exceeds the 67,108,864-element benchmark bound");
  }
  geometry.int2_code_bytes =
      geometry.padded_elements / 4 +
      (geometry.padded_elements % 4 != 0 ? 1U : 0U);
  geometry.int4_code_bytes =
      geometry.padded_elements / 2 +
      (geometry.padded_elements % 2 != 0 ? 1U : 0U);
  geometry.scale_count =
      CheckedMultiply(geometry.tile_grid_rows, geometry.tile_grid_columns,
                      "scale grid");
  geometry.scale_count =
      CheckedMultiply(geometry.scale_count, geometry.tile, "scale rows");
  geometry.scale_count = CheckedMultiply(
      geometry.scale_count, geometry.groups_per_tile, "scale groups");

  const std::size_t measured_launches =
      CheckedMultiply(config.iterations, config.repeats,
                      "measured launches per path");
  const std::size_t launches_per_path =
      CheckedAdd(config.warmup, measured_launches, "launches per path");
  const std::size_t stage_weight = CheckedAdd(config.prefix, 1, "path weight");
  const std::size_t weighted_launches =
      CheckedMultiply(launches_per_path, stage_weight, "weighted launches");
  const std::size_t weighted_work = CheckedMultiply(
      geometry.padded_elements, weighted_launches, "weighted element launches");
  if (weighted_work > kMaximumWeightedElementLaunches) {
    throw std::invalid_argument(
        "requested shape and launch counts exceed the bounded work budget");
  }
  return geometry;
}

PayloadSummary ComputePayload(const Configuration& config,
                              const Geometry& geometry) {
  PayloadSummary payload{};
  payload.progressive_code_bytes =
      CheckedMultiply(config.prefix, geometry.int2_code_bytes,
                      "progressive code payload");
  payload.progressive_scale_bytes = CheckedMultiply(
      CheckedMultiply(config.prefix, geometry.scale_count,
                      "progressive scale values"),
      sizeof(std::uint16_t), "progressive scale payload");
  payload.progressive_model_bytes =
      CheckedAdd(payload.progressive_code_bytes,
                 payload.progressive_scale_bytes, "progressive model payload");
  payload.static_code_bytes = geometry.int4_code_bytes;
  payload.static_scale_bytes =
      CheckedMultiply(geometry.scale_count, sizeof(std::uint16_t),
                      "static scale payload");
  payload.static_model_bytes =
      CheckedAdd(payload.static_code_bytes, payload.static_scale_bytes,
                 "static model payload");
  payload.input_bytes =
      CheckedMultiply(geometry.columns, sizeof(float), "input payload");
  payload.output_bytes =
      CheckedMultiply(geometry.rows, sizeof(float), "output payload");
  payload.total_allocated_device_bytes = CheckedAdd(
      CheckedAdd(payload.progressive_model_bytes, payload.static_model_bytes,
                 "model allocations"),
      CheckedAdd(payload.input_bytes,
                 CheckedMultiply(payload.output_bytes, 2, "output allocations"),
                 "activation allocations"),
      "total benchmark allocations");
  return payload;
}

std::uint32_t Mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return static_cast<std::uint32_t>(value ^ (value >> 31U));
}

std::uint16_t Fp16Bits(float value) {
  const __half_raw encoded = __float2half_rn(value);
  return encoded.x;
}

std::vector<float> MakeInput(std::size_t columns) {
  std::vector<float> input(columns);
  for (std::size_t index = 0; index < columns; ++index) {
    const int centered = static_cast<int>(Mix(kDataSeed + index) % 257U) - 128;
    input[index] = static_cast<float>(centered) / 128.0F;
  }
  return input;
}

std::vector<std::uint8_t> MakeInt2Codes(std::size_t elements,
                                        std::size_t stage) {
  const std::size_t byte_count =
      elements / 4 + (elements % 4 != 0 ? 1U : 0U);
  std::vector<std::uint8_t> packed(byte_count, 0);
  for (std::size_t index = 0; index < elements; ++index) {
    const std::uint8_t encoded = static_cast<std::uint8_t>(
        Mix(kDataSeed + index + stage * 0x100000001b3ULL) & 0x03U);
    packed[index / 4] |= static_cast<std::uint8_t>(
        encoded << static_cast<unsigned int>(2 * (index % 4)));
  }
  return packed;
}

std::vector<std::uint8_t> MakeInt4Codes(std::size_t elements) {
  const std::size_t byte_count =
      elements / 2 + (elements % 2 != 0 ? 1U : 0U);
  std::vector<std::uint8_t> packed(byte_count, 0);
  for (std::size_t index = 0; index < elements; ++index) {
    const std::uint8_t encoded = static_cast<std::uint8_t>(
        Mix(kDataSeed + index + 0x5744000000000000ULL) & 0x0fU);
    packed[index / 2] |= static_cast<std::uint8_t>(
        encoded << static_cast<unsigned int>(4 * (index % 2)));
  }
  return packed;
}

std::vector<std::uint16_t> MakeScales(std::size_t count,
                                      std::size_t stage,
                                      bool static_w4) {
  std::vector<std::uint16_t> scales(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int variation = static_cast<int>(Mix(kDataSeed + index) % 3U);
    const int exponent = static_w4 ? -7 - variation
                                  : -6 - static_cast<int>(stage) - variation;
    scales[index] = Fp16Bits(std::ldexp(1.0F, exponent));
  }
  return scales;
}

float DecodeFp16Bits(std::uint16_t bits) {
  __half_raw raw{};
  raw.x = bits;
  return __half2float(__half(raw));
}

float HostWarpSum(std::array<float, 32> lanes) {
  // CUDA shuffles read the pre-instruction value from every source lane. Use a
  // snapshot at each step so the CPU oracle reproduces that reduction tree.
  for (std::size_t offset = 16; offset != 0; offset /= 2) {
    const std::array<float, 32> before = lanes;
    for (std::size_t lane = 0; lane + offset < lanes.size(); ++lane) {
      lanes[lane] = before[lane] + before[lane + offset];
    }
  }
  return lanes[0];
}

std::vector<float> StaticW4CpuOracle(
    const std::vector<float>& input,
    const std::vector<std::uint8_t>& packed_codes,
    const std::vector<std::uint16_t>& scale_fp16_bits,
    const Geometry& geometry) {
  std::vector<float> output(geometry.rows);
  for (std::size_t row = 0; row < geometry.rows; ++row) {
    std::array<float, kThreadsPerBlock> lane_sums{};
    for (std::size_t lane = 0; lane < lane_sums.size(); ++lane) {
      float sum = 0.0F;
      for (std::size_t column = lane; column < geometry.columns;
           column += lane_sums.size()) {
        const std::size_t padded_index =
            row * geometry.padded_columns + column;
        const std::uint8_t encoded = static_cast<std::uint8_t>(
            (packed_codes[padded_index / 2] >>
             (4 * (padded_index % 2))) &
            0x0fU);
        const int code = encoded >= 8U ? static_cast<int>(encoded) - 16
                                       : static_cast<int>(encoded);
        const std::size_t tile_row = row / geometry.tile;
        const std::size_t row_in_tile = row % geometry.tile;
        const std::size_t tile_column = column / geometry.tile;
        const std::size_t group_in_tile =
            (column % geometry.tile) / geometry.group;
        const std::size_t scale_index =
            (((tile_row * geometry.tile_grid_columns + tile_column) *
                  geometry.tile +
              row_in_tile) *
                 geometry.groups_per_tile +
             group_in_tile);
        const float scale = DecodeFp16Bits(scale_fp16_bits[scale_index]);
        sum = std::fma(input[column], static_cast<float>(code) * scale, sum);
      }
      lane_sums[lane] = sum;
    }

    std::array<float, 32> block_warp{};
    for (std::size_t warp = 0; warp < lane_sums.size() / 32; ++warp) {
      std::array<float, 32> warp_lanes{};
      std::copy_n(lane_sums.begin() + warp * 32, 32, warp_lanes.begin());
      block_warp[warp] = HostWarpSum(warp_lanes);
    }
    output[row] = HostWarpSum(block_warp);
  }
  return output;
}

OracleValidation ValidateStaticW4AgainstCpuOracle(
    const std::vector<float>& actual,
    const std::vector<float>& input,
    const std::vector<std::uint8_t>& packed_codes,
    const std::vector<std::uint16_t>& scale_fp16_bits,
    const Geometry& geometry) {
  const std::vector<float> expected = StaticW4CpuOracle(
      input, packed_codes, scale_fp16_bits, geometry);
  if (actual.size() != expected.size()) {
    throw std::logic_error("static-W4 CPU oracle output size mismatch");
  }

  OracleValidation validation{0.0, kStaticOracleAbsoluteTolerance, 0};
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(expected[index])) {
      throw std::runtime_error(
          "static-W4 CPU oracle produced a non-finite output");
    }
    const double absolute_error = std::abs(
        static_cast<double>(actual[index]) - static_cast<double>(expected[index]));
    const double tolerance =
        static_cast<double>(kStaticOracleAbsoluteTolerance) +
        static_cast<double>(kStaticOracleRelativeTolerance) *
            std::abs(static_cast<double>(expected[index]));
    if (index == 0 || absolute_error > validation.maximum_absolute_error) {
      validation.maximum_absolute_error = absolute_error;
      validation.tolerance_at_maximum_error = tolerance;
      validation.maximum_error_index = index;
    }
    if (absolute_error > tolerance) {
      std::ostringstream message;
      message << "static-W4 GPU output failed its deterministic CPU oracle at "
              << index << ": GPU=" << actual[index]
              << ", CPU=" << expected[index]
              << ", abs_error=" << absolute_error
              << ", tolerance=" << tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return validation;
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
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class CudaEvent final {
 public:
  CudaEvent() { CheckCuda(cudaEventCreate(&event_), "cudaEventCreate"); }

  ~CudaEvent() {
    if (event_ != nullptr) {
      static_cast<void>(cudaEventDestroy(event_));
    }
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

// Declared after every device buffer so its destructor runs first during
// exception unwinding. This keeps buffers and pageable host staging memory
// alive until all previously enqueued work has stopped using them.
class StreamCompletionGuard final {
 public:
  explicit StreamCompletionGuard(cudaStream_t stream) : stream_(stream) {}

  ~StreamCompletionGuard() {
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamSynchronize(stream_));
    }
  }

  StreamCompletionGuard(const StreamCompletionGuard&) = delete;
  StreamCompletionGuard& operator=(const StreamCompletionGuard&) = delete;

 private:
  cudaStream_t stream_;
};

template <typename T>
void CopyToDevice(DeviceBuffer<T>* destination,
                  const std::vector<T>& source,
                  cudaStream_t stream,
                  const char* operation) {
  if (source.empty()) {
    return;
  }
  CheckCuda(cudaMemcpyAsync(destination->get(), source.data(),
                            CheckedMultiply(source.size(), sizeof(T),
                                            "host-to-device copy"),
                            cudaMemcpyHostToDevice, stream),
            operation);
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

// Benchmark-local naive static-W4 baseline. It intentionally mirrors the
// production progressive kernel's output-row mapping and FP32 reduction, but
// it is not a vendor/library W4 kernel and is not an acceptance baseline.
__global__ void MatchedNaiveStaticW4Kernel(
    const float* input,
    const std::uint8_t* packed_codes,
    const std::uint16_t* scale_fp16_bits,
    std::uint64_t rows,
    std::uint64_t columns,
    std::uint64_t padded_columns,
    std::uint64_t tile,
    std::uint64_t group,
    float* output) {
  const std::uint64_t tile_grid_columns = padded_columns / tile;
  const std::uint64_t groups_per_tile = tile / group;
  for (std::uint64_t row = blockIdx.x; row < rows;) {
    float thread_sum = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns;) {
      const std::uint64_t padded_index = row * padded_columns + column;
      const std::uint8_t encoded = static_cast<std::uint8_t>(
          (packed_codes[padded_index / 2] >>
           (4 * (padded_index % 2))) &
          0x0fU);
      const int code = encoded >= 8U ? static_cast<int>(encoded) - 16
                                     : static_cast<int>(encoded);
      const std::uint64_t tile_row = row / tile;
      const std::uint64_t row_in_tile = row % tile;
      const std::uint64_t tile_column = column / tile;
      const std::uint64_t group_in_tile = (column % tile) / group;
      const std::uint64_t scale_index =
          (((tile_row * tile_grid_columns + tile_column) * tile + row_in_tile) *
               groups_per_tile +
           group_in_tile);
      const float scale =
          __half2float(__ushort_as_half(scale_fp16_bits[scale_index]));
      thread_sum = fmaf(input[column], static_cast<float>(code) * scale,
                        thread_sum);
      if (columns - column <= blockDim.x) {
        break;
      }
      column += blockDim.x;
    }
    const float sum = BlockSum(thread_sum);
    if (threadIdx.x == 0) {
      output[row] = sum;
    }
    __syncthreads();
    if (rows - row <= gridDim.x) {
      break;
    }
    row += gridDim.x;
  }
}

cudaError_t LaunchMatchedNaiveStaticW4(const float* input,
                                       const std::uint8_t* packed_codes,
                                       const std::uint16_t* scales,
                                       const Geometry& geometry,
                                       float* output,
                                       cudaStream_t stream) noexcept {
  if (input == nullptr || packed_codes == nullptr || scales == nullptr ||
      output == nullptr || geometry.rows == 0 || geometry.columns == 0) {
    return cudaErrorInvalidValue;
  }
  // Match the production launch policy: do not attribute an existing
  // thread-local CUDA error to this kernel, and clear/check this launch's own
  // immediate status after enqueueing it.
  const cudaError_t prior_error = cudaGetLastError();
  if (prior_error != cudaSuccess) {
    return prior_error;
  }
  const unsigned int blocks = static_cast<unsigned int>(
      std::min<std::size_t>(geometry.rows, kMaximumBlocks));
  MatchedNaiveStaticW4Kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      input, packed_codes, scales, geometry.rows, geometry.columns,
      geometry.padded_columns, geometry.tile, geometry.group, output);
  return cudaGetLastError();
}

template <typename Launcher>
double MeasureMicroseconds(std::size_t iterations,
                           cudaStream_t stream,
                           cudaEvent_t start,
                           cudaEvent_t stop,
                           Launcher&& launch) {
  CheckCuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    CheckCuda(launch(), "timed GEMV launch");
  }
  CheckCuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
  CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
  float elapsed_milliseconds = 0.0F;
  CheckCuda(cudaEventElapsedTime(&elapsed_milliseconds, start, stop),
            "cudaEventElapsedTime");
  if (!std::isfinite(elapsed_milliseconds) || elapsed_milliseconds <= 0.0F) {
    throw std::runtime_error(
        "CUDA event duration is not positive; increase --iterations");
  }
  return static_cast<double>(elapsed_milliseconds) * 1000.0 /
         static_cast<double>(iterations);
}

TimingSummary Summarize(std::vector<double> samples) {
  if (samples.empty()) {
    throw std::logic_error("cannot summarize an empty timing sample");
  }
  TimingSummary summary;
  summary.microseconds_per_launch = std::move(samples);
  std::vector<double> sorted = summary.microseconds_per_launch;
  std::sort(sorted.begin(), sorted.end());
  summary.minimum_microseconds = sorted.front();
  summary.maximum_microseconds = sorted.back();
  const std::size_t middle = sorted.size() / 2;
  summary.median_microseconds = sorted.size() % 2 == 0
                                   ? (sorted[middle - 1] + sorted[middle]) / 2.0
                                   : sorted[middle];
  return summary;
}

double VerifyFiniteAndChecksum(const std::vector<float>& output,
                               const char* label) {
  double checksum = 0.0;
  for (std::size_t index = 0; index < output.size(); ++index) {
    if (!std::isfinite(output[index])) {
      throw std::runtime_error(std::string(label) +
                               " produced a non-finite output at index " +
                               std::to_string(index));
    }
    checksum += std::abs(static_cast<double>(output[index]));
  }
  if (!std::isfinite(checksum)) {
    throw std::runtime_error(std::string(label) +
                             " checksum is non-finite");
  }
  return checksum;
}

void PrintSamples(std::ostream& output, const std::vector<double>& samples) {
  output << '[';
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << samples[index];
  }
  output << ']';
}

void PrintHelp(std::ostream& output) {
  output
      << "Usage: harpq_gemv_benchmark [options]\n"
      << "  --rows N          output rows, default 4096\n"
      << "  --columns N       input columns, default 4096 (--cols alias)\n"
      << "  --tile N          square tile extent, default 128\n"
      << "  --group N         columns per scale group, default 128\n"
      << "  --prefix N        progressive int2 stage count [1,4], default 2\n"
      << "  --warmup N        untimed launches per path [0,1000], default 10\n"
      << "  --iterations N    launches per timed repeat [1,10000], default 100\n"
      << "  --repeats N       timed repeats [3,31], default 7\n"
      << "  --device N        CUDA device ordinal, default 0\n"
      << "  --json            emit one JSON object\n"
      << "  --help, -h        show this help\n";
}

void PrintJsonError(std::string_view status, std::string_view message) {
  std::cout << "{\"status\":\"" << JsonEscape(status)
            << "\",\"classification\":\"exploratory_matched_naive_baseline\""
            << ",\"message\":\"" << JsonEscape(message) << "\"}\n";
}

}  // namespace

int main(int argc, char** argv) {
  const bool json_requested = [&] {
    for (int index = 1; index < argc; ++index) {
      if (std::string_view(argv[index]) == "--json") {
        return true;
      }
    }
    return false;
  }();

  try {
    const Configuration config = ParseArguments(argc, argv);
    if (config.help) {
      PrintHelp(std::cout);
      return 0;
    }
    const Geometry geometry = ValidateConfiguration(config);

    int device_count = 0;
    const cudaError_t probe_status = cudaGetDeviceCount(&device_count);
    if (probe_status != cudaSuccess || device_count == 0) {
      const std::string reason =
          probe_status == cudaSuccess
              ? "CUDA runtime reported no devices"
              : std::string(cudaGetErrorName(probe_status)) + ": " +
                    cudaGetErrorString(probe_status);
      if (config.json) {
        PrintJsonError("skipped", reason);
      } else {
        std::cout << "SKIP: " << reason << '\n';
      }
      static_cast<void>(cudaGetLastError());
      return 77;
    }
    if (config.device < 0 || config.device >= device_count) {
      throw std::invalid_argument(
          "--device does not name an available CUDA device");
    }
    CheckCuda(cudaSetDevice(config.device), "cudaSetDevice");
    CheckCuda(cudaFree(nullptr), "initialize CUDA context");

    cudaDeviceProp properties{};
    int driver_version = 0;
    int runtime_version = 0;
    CheckCuda(cudaGetDeviceProperties(&properties, config.device),
              "cudaGetDeviceProperties");
    CheckCuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    CheckCuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");

    const PayloadSummary payload = ComputePayload(config, geometry);
    std::size_t free_device_bytes = 0;
    std::size_t total_device_bytes = 0;
    CheckCuda(cudaMemGetInfo(&free_device_bytes, &total_device_bytes),
              "cudaMemGetInfo");
    if (payload.total_allocated_device_bytes > free_device_bytes -
            free_device_bytes / 5) {
      throw std::runtime_error(
          "benchmark allocations exceed 80% of currently free device memory");
    }

    const std::vector<float> host_input = MakeInput(geometry.columns);
    std::array<std::vector<std::uint8_t>, kMaximumStageCount>
        host_progressive_codes;
    std::array<std::vector<std::uint16_t>, kMaximumStageCount>
        host_progressive_scales;
    for (std::size_t stage = 0; stage < config.prefix; ++stage) {
      host_progressive_codes[stage] =
          MakeInt2Codes(geometry.padded_elements, stage);
      host_progressive_scales[stage] =
          MakeScales(geometry.scale_count, stage, false);
    }
    const std::vector<std::uint8_t> host_static_codes =
        MakeInt4Codes(geometry.padded_elements);
    const std::vector<std::uint16_t> host_static_scales =
        MakeScales(geometry.scale_count, 0, true);

    CudaStream stream;
    CudaEvent start;
    CudaEvent stop;
    DeviceBuffer<float> device_input(host_input.size());
    DeviceBuffer<float> device_progressive_output(geometry.rows);
    DeviceBuffer<float> device_static_output(geometry.rows);
    std::array<DeviceBuffer<std::uint8_t>, kMaximumStageCount>
        device_progressive_codes;
    std::array<DeviceBuffer<std::uint16_t>, kMaximumStageCount>
        device_progressive_scales;
    DeviceBuffer<std::uint8_t> device_static_codes(host_static_codes.size());
    DeviceBuffer<std::uint16_t> device_static_scales(host_static_scales.size());
    StreamCompletionGuard completion_guard(stream.get());

    CopyToDevice(&device_input, host_input, stream.get(),
                 "cudaMemcpyAsync(input H2D)");
    for (std::size_t stage = 0; stage < config.prefix; ++stage) {
      device_progressive_codes[stage] =
          DeviceBuffer<std::uint8_t>(host_progressive_codes[stage].size());
      device_progressive_scales[stage] =
          DeviceBuffer<std::uint16_t>(host_progressive_scales[stage].size());
      CopyToDevice(&device_progressive_codes[stage],
                   host_progressive_codes[stage], stream.get(),
                   "cudaMemcpyAsync(progressive codes H2D)");
      CopyToDevice(&device_progressive_scales[stage],
                   host_progressive_scales[stage], stream.get(),
                   "cudaMemcpyAsync(progressive scales H2D)");
    }
    CopyToDevice(&device_static_codes, host_static_codes, stream.get(),
                 "cudaMemcpyAsync(static codes H2D)");
    CopyToDevice(&device_static_scales, host_static_scales, stream.get(),
                 "cudaMemcpyAsync(static scales H2D)");
    CheckCuda(cudaStreamSynchronize(stream.get()),
              "synchronize benchmark setup");

    ProgressiveGemvDeviceView progressive_view{};
    progressive_view.geometry = {
        geometry.rows,
        geometry.columns,
        geometry.padded_rows,
        geometry.padded_columns,
        geometry.tile,
        geometry.tile,
        geometry.group,
    };
    progressive_view.stage_count = config.prefix;
    for (std::size_t stage = 0; stage < config.prefix; ++stage) {
      progressive_view.stages[stage] = {
          device_progressive_codes[stage].get(),
          static_cast<std::uint64_t>(host_progressive_codes[stage].size()),
          device_progressive_scales[stage].get(),
          static_cast<std::uint64_t>(host_progressive_scales[stage].size()),
      };
    }

    const ProgressiveGemvIoDeviceView progressive_io{
        device_input.get(),
        static_cast<std::uint64_t>(host_input.size()),
        nullptr,
        0,
        device_progressive_output.get(),
        static_cast<std::uint64_t>(geometry.rows),
    };

    const auto launch_progressive = [&] {
      return LaunchProgressiveGemv(
          progressive_io, progressive_view, config.prefix, stream.get());
    };
    const auto launch_static = [&] {
      return LaunchMatchedNaiveStaticW4(
          device_input.get(), device_static_codes.get(),
          device_static_scales.get(), geometry, device_static_output.get(),
          stream.get());
    };

    for (std::size_t iteration = 0; iteration < config.warmup; ++iteration) {
      CheckCuda(launch_progressive(), "progressive warm-up launch");
      CheckCuda(launch_static(), "static W4 warm-up launch");
    }
    CheckCuda(cudaStreamSynchronize(stream.get()), "synchronize warm-up");

    std::vector<double> progressive_samples;
    std::vector<double> static_samples;
    progressive_samples.reserve(config.repeats);
    static_samples.reserve(config.repeats);
    for (std::size_t repeat = 0; repeat < config.repeats; ++repeat) {
      if (repeat % 2 == 0) {
        progressive_samples.push_back(MeasureMicroseconds(
            config.iterations, stream.get(), start.get(), stop.get(),
            launch_progressive));
        static_samples.push_back(MeasureMicroseconds(
            config.iterations, stream.get(), start.get(), stop.get(),
            launch_static));
      } else {
        static_samples.push_back(MeasureMicroseconds(
            config.iterations, stream.get(), start.get(), stop.get(),
            launch_static));
        progressive_samples.push_back(MeasureMicroseconds(
            config.iterations, stream.get(), start.get(), stop.get(),
            launch_progressive));
      }
    }
    const TimingSummary progressive = Summarize(std::move(progressive_samples));
    const TimingSummary static_w4 = Summarize(std::move(static_samples));

    std::vector<float> progressive_output(geometry.rows);
    std::vector<float> static_output(geometry.rows);
    CheckCuda(cudaMemcpyAsync(
                  progressive_output.data(), device_progressive_output.get(),
                  CheckedMultiply(progressive_output.size(), sizeof(float),
                                  "progressive output copy"),
                  cudaMemcpyDeviceToHost, stream.get()),
              "cudaMemcpyAsync(progressive output D2H)");
    CheckCuda(cudaMemcpyAsync(
                  static_output.data(), device_static_output.get(),
                  CheckedMultiply(static_output.size(), sizeof(float),
                                  "static output copy"),
                  cudaMemcpyDeviceToHost, stream.get()),
              "cudaMemcpyAsync(static output D2H)");
    CheckCuda(cudaStreamSynchronize(stream.get()),
              "synchronize output verification");
    const double progressive_checksum =
        VerifyFiniteAndChecksum(progressive_output, "progressive GEMV");
    const double static_checksum =
        VerifyFiniteAndChecksum(static_output, "static W4 GEMV");
    const OracleValidation static_oracle =
        ValidateStaticW4AgainstCpuOracle(
            static_output, host_input, host_static_codes,
            host_static_scales, geometry);

    const double progressive_launches_per_second =
        1.0e6 / progressive.median_microseconds;
    const double static_launches_per_second =
        1.0e6 / static_w4.median_microseconds;
    const double throughput_ratio =
        progressive_launches_per_second / static_launches_per_second;

    std::cout << std::setprecision(9);
    if (config.json) {
      std::cout
          << "{\"status\":\"ok\""
          << ",\"classification\":\"exploratory_matched_naive_baseline\""
          << ",\"acceptance_evidence\":false"
          << ",\"baseline_scope\":\"benchmark-local naive static-W4; not a production or library-parity baseline\""
          << ",\"timing_scope\":\"CUDA events around kernel launches only; allocation and transfers excluded\""
          << ",\"device\":{\"ordinal\":" << config.device
          << ",\"name\":\"" << JsonEscape(properties.name) << "\""
          << ",\"compute_capability\":\"" << properties.major << '.'
          << properties.minor << "\""
          << ",\"multiprocessors\":" << properties.multiProcessorCount
          << ",\"vram_bytes\":" << properties.totalGlobalMem
          << ",\"free_vram_before_allocations_bytes\":" << free_device_bytes
          << ",\"driver_api_version\":" << driver_version
          << ",\"runtime_version\":" << runtime_version
          << ",\"compiled_cudart_version\":" << CUDART_VERSION << '}'
          << ",\"configuration\":{\"rows\":" << geometry.rows
          << ",\"columns\":" << geometry.columns
          << ",\"padded_rows\":" << geometry.padded_rows
          << ",\"padded_columns\":" << geometry.padded_columns
          << ",\"tile\":" << geometry.tile
          << ",\"group\":" << geometry.group
          << ",\"prefix\":" << config.prefix
          << ",\"progressive_bits_per_code\":" << config.prefix * 2
          << ",\"static_bits_per_code\":4"
          << ",\"bit_width_matched\":"
          << (config.prefix == 2 ? "true" : "false")
          << ",\"warmup_per_path\":" << config.warmup
          << ",\"iterations_per_repeat\":" << config.iterations
          << ",\"repeats\":" << config.repeats
          << ",\"seed\":" << kDataSeed << '}'
          << ",\"payload\":{\"progressive_code_bytes\":"
          << payload.progressive_code_bytes
          << ",\"progressive_scale_bytes\":"
          << payload.progressive_scale_bytes
          << ",\"progressive_model_bytes\":"
          << payload.progressive_model_bytes
          << ",\"static_code_bytes\":" << payload.static_code_bytes
          << ",\"static_scale_bytes\":" << payload.static_scale_bytes
          << ",\"static_model_bytes\":" << payload.static_model_bytes
          << ",\"input_bytes\":" << payload.input_bytes
          << ",\"output_bytes_per_path\":" << payload.output_bytes
          << ",\"allocated_device_bytes\":"
          << payload.total_allocated_device_bytes << '}'
          << ",\"progressive\":{\"median_us\":"
          << progressive.median_microseconds
          << ",\"min_us\":" << progressive.minimum_microseconds
          << ",\"max_us\":" << progressive.maximum_microseconds
          << ",\"launches_per_second\":"
          << progressive_launches_per_second << ",\"samples_us\":";
      PrintSamples(std::cout, progressive.microseconds_per_launch);
      std::cout << ",\"finite_output\":true,\"output_l1_checksum\":"
                << progressive_checksum << '}'
                << ",\"static_w4\":{\"median_us\":"
                << static_w4.median_microseconds
                << ",\"min_us\":" << static_w4.minimum_microseconds
                << ",\"max_us\":" << static_w4.maximum_microseconds
                << ",\"launches_per_second\":"
                << static_launches_per_second << ",\"samples_us\":";
      PrintSamples(std::cout, static_w4.microseconds_per_launch);
      std::cout << ",\"finite_output\":true,\"output_l1_checksum\":"
                << static_checksum
                << ",\"cpu_oracle_verified\":true"
                << ",\"max_abs_error\":"
                << static_oracle.maximum_absolute_error
                << ",\"tolerance\":"
                << static_oracle.tolerance_at_maximum_error
                << ",\"absolute_tolerance\":"
                << kStaticOracleAbsoluteTolerance
                << ",\"relative_tolerance\":"
                << kStaticOracleRelativeTolerance
                << ",\"max_error_index\":"
                << static_oracle.maximum_error_index << '}'
                << ",\"progressive_throughput_over_static_w4\":"
                << throughput_ratio
                << ",\"caveats\":[\"exploratory single-device microbenchmark\",\"matched naive baseline is not an optimized static-W4 library kernel\",\"GPU clocks and desktop activity are uncontrolled\",\"payload bytes are stored bytes, not measured DRAM traffic\"]}\n";
    } else {
      std::cout
          << "classification: exploratory_matched_naive_baseline\n"
          << "acceptance evidence: no\n"
          << "device: " << properties.name << " (sm_" << properties.major
          << properties.minor << ")\n"
          << "shape: " << geometry.rows << 'x' << geometry.columns
          << ", padded " << geometry.padded_rows << 'x'
          << geometry.padded_columns << ", tile/group " << geometry.tile << '/'
          << geometry.group << ", prefix " << config.prefix << "\n"
          << "progressive median: " << progressive.median_microseconds
          << " us, " << progressive_launches_per_second << " launches/s\n"
          << "matched-naive static-W4 median: "
          << static_w4.median_microseconds << " us, "
          << static_launches_per_second << " launches/s\n"
          << "progressive throughput / static-W4: " << throughput_ratio
          << "\n"
          << "model payload bytes (progressive/static): "
          << payload.progressive_model_bytes << '/'
          << payload.static_model_bytes << "\n"
          << "finite outputs: yes; L1 checksums (progressive/static): "
          << progressive_checksum << '/' << static_checksum << "\n"
          << "static-W4 CPU oracle: verified; max_abs_error "
          << static_oracle.maximum_absolute_error << ", tolerance "
          << static_oracle.tolerance_at_maximum_error << " (abs "
          << kStaticOracleAbsoluteTolerance << " + rel "
          << kStaticOracleRelativeTolerance << ")\n"
          << "caveat: benchmark-local naive baseline; not production/library parity or acceptance evidence\n";
    }
    return 0;
  } catch (const std::invalid_argument& error) {
    if (json_requested) {
      PrintJsonError("invalid_arguments", error.what());
    } else {
      std::cerr << "invalid arguments: " << error.what() << '\n';
      PrintHelp(std::cerr);
    }
    return 2;
  } catch (const std::exception& error) {
    if (json_requested) {
      PrintJsonError("error", error.what());
    } else {
      std::cerr << "benchmark failed: " << error.what() << '\n';
    }
    return 1;
  }
}
