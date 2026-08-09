#include "harpq/progressive_gemv.hpp"

#include "generated/golden_vectors.hpp"
#include "harpq/cuda_oracle.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using harpq::native_runtime::CudaDeviceAvailable;
using harpq::native_runtime::DecodeStagePayload;
using harpq::native_runtime::LaunchProgressiveGemv;
using harpq::native_runtime::ProgressiveGemvCuda;
using harpq::native_runtime::ProgressiveGemvDeviceView;
using harpq::native_runtime::ProgressiveGemvIoDeviceView;
using harpq::native_runtime::ProgressiveLinearCudaOracle;
using harpq::native_runtime::ProgressiveMatrix;
using harpq::native_runtime::ScaleShape;
namespace golden = harpq::native_runtime::golden;

constexpr float kAbsoluteTolerance = 2.0e-4F;
constexpr float kRelativeTolerance = 2.0e-4F;

template <typename T>
class TestDeviceBuffer final {
 public:
  explicit TestDeviceBuffer(std::size_t count) {
    const cudaError_t status =
        cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string("test cudaMalloc failed: ") + cudaGetErrorString(status));
    }
  }

  ~TestDeviceBuffer() {
    if (data_ != nullptr) {
      static_cast<void>(cudaFree(data_));
    }
  }

  TestDeviceBuffer(const TestDeviceBuffer&) = delete;
  TestDeviceBuffer& operator=(const TestDeviceBuffer&) = delete;

  [[nodiscard]] T* get() noexcept { return data_; }

 private:
  T* data_ = nullptr;
};

class TestCudaStream final {
 public:
  TestCudaStream() {
    const cudaError_t status =
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string("test stream creation failed: ") +
          cudaGetErrorString(status));
    }
  }

  ~TestCudaStream() {
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  TestCudaStream(const TestCudaStream&) = delete;
  TestCudaStream& operator=(const TestCudaStream&) = delete;

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class TestStreamCompletionGuard final {
 public:
  explicit TestStreamCompletionGuard(cudaStream_t stream) : stream_(stream) {}

  ~TestStreamCompletionGuard() {
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamSynchronize(stream_));
    }
  }

  TestStreamCompletionGuard(const TestStreamCompletionGuard&) = delete;
  TestStreamCompletionGuard& operator=(const TestStreamCompletionGuard&) =
      delete;

 private:
  cudaStream_t stream_;
};

void RequireCudaSuccess(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(status));
  }
}

void RequireInvalidDeviceView(cudaError_t status, const char* label) {
  if (status != cudaErrorInvalidValue) {
    throw std::runtime_error(std::string("production launch accepted ") + label);
  }
}

[[nodiscard]] ProgressiveMatrix MakeGoldenMatrix() {
  ProgressiveMatrix matrix{golden::kGeometry, {}};
  matrix.stages.reserve(golden::kStageCount);
  for (std::size_t stage_index = 0; stage_index < golden::kStageCount;
       ++stage_index) {
    const auto& bytes = golden::kStagePayloads[stage_index];
    matrix.stages.push_back(DecodeStagePayload(
        std::vector<std::uint8_t>(bytes.begin(), bytes.end()),
        static_cast<std::uint16_t>(stage_index), golden::kScaleShape));
  }
  return matrix;
}

[[nodiscard]] ProgressiveMatrix MakeWideMatrix(std::size_t columns) {
  constexpr std::size_t kRows = 3;
  constexpr std::size_t kTileRows = 4;
  constexpr std::size_t kTileColumns = 128;
  constexpr std::size_t kGroupSize = 128;
  const std::size_t padded_columns =
      ((columns + kTileColumns - 1) / kTileColumns) * kTileColumns;
  const std::size_t padded_elements = kTileRows * padded_columns;
  const std::size_t code_bytes = (padded_elements + 3) / 4;
  const ScaleShape scale_shape{
      1, padded_columns / kTileColumns, kTileRows, 1};
  const std::size_t scale_count =
      scale_shape[0] * scale_shape[1] * scale_shape[2] * scale_shape[3];

  ProgressiveMatrix matrix{
      {kRows, columns, kTileRows, padded_columns, kTileRows, kTileColumns,
       kGroupSize},
      {}};
  matrix.stages.reserve(golden::kStageCount);
  for (std::size_t stage_index = 0; stage_index < golden::kStageCount;
       ++stage_index) {
    std::vector<std::uint8_t> packed_codes(code_bytes, 0);
    for (std::size_t element = 0; element < padded_elements; ++element) {
      const std::uint8_t raw = static_cast<std::uint8_t>(
          (element * 5 + stage_index * 3 + element / 17) & 0x03U);
      packed_codes[element / 4] = static_cast<std::uint8_t>(
          packed_codes[element / 4] | (raw << (2 * (element % 4))));
    }
    std::vector<float> scales(scale_count);
    for (std::size_t index = 0; index < scale_count; ++index) {
      scales[index] = static_cast<float>(1 + ((index + stage_index) % 7)) /
                      64.0F;
    }
    matrix.stages.push_back({
        static_cast<std::uint16_t>(stage_index),
        std::move(packed_codes),
        std::move(scales),
        scale_shape,
    });
  }
  return matrix;
}

template <typename Callable>
void RequireInvalidArgument(Callable&& callable, const std::string& label) {
  try {
    callable();
  } catch (const std::invalid_argument&) {
    return;
  } catch (const std::exception& error) {
    throw std::runtime_error(label + " raised the wrong exception: " +
                             error.what());
  }
  throw std::runtime_error(label + " did not reject invalid input");
}

template <typename Callable>
void RequireOverflowError(Callable&& callable, const std::string& label) {
  try {
    callable();
  } catch (const std::overflow_error&) {
    return;
  } catch (const std::exception& error) {
    throw std::runtime_error(label + " raised the wrong exception: " +
                             error.what());
  }
  throw std::runtime_error(label + " did not reject overflowing input");
}

template <typename Expected>
void RequireNear(const std::vector<float>& actual,
                 const Expected& expected,
                 const std::string& label) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(label + " output size mismatch");
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index])) {
      throw std::runtime_error(label + " produced a non-finite value");
    }
    const float error = std::abs(actual[index] - expected[index]);
    const float tolerance =
        kAbsoluteTolerance + kRelativeTolerance * std::abs(expected[index]);
    if (error > tolerance) {
      std::ostringstream message;
      message << label << " differs at " << index << ": fused="
              << actual[index] << ", expected=" << expected[index]
              << ", error=" << error << ", tolerance=" << tolerance;
      throw std::runtime_error(message.str());
    }
  }
}

std::vector<float> InputRow(std::size_t batch_index) {
  const std::size_t columns = golden::kGeometry.original_columns;
  const auto begin = golden::kInputs.begin() + batch_index * columns;
  return std::vector<float>(begin, begin + columns);
}

std::vector<float> GoldenOutputRow(std::size_t prefix,
                                   std::size_t batch_index) {
  const std::size_t rows = golden::kGeometry.original_rows;
  const auto begin = golden::kPrefixOutputs[prefix].begin() + batch_index * rows;
  return std::vector<float>(begin, begin + rows);
}

void TestAllPrefixesAgainstGoldenAndOracle() {
  static_assert(golden::kStageCount == 4,
                "the canonical fixture must contain four stages");
  const ProgressiveMatrix matrix = MakeGoldenMatrix();
  const std::vector<float> bias(golden::kBias.begin(), golden::kBias.end());
  for (std::size_t prefix = 0; prefix <= golden::kStageCount; ++prefix) {
    for (std::size_t batch_index = 0; batch_index < golden::kBatchSize;
         ++batch_index) {
      const std::vector<float> input = InputRow(batch_index);
      const std::vector<float> fused =
          ProgressiveGemvCuda(input, matrix, prefix, bias);
      const std::vector<float> oracle =
          ProgressiveLinearCudaOracle(input, 1, matrix, prefix, bias);
      RequireNear(fused, GoldenOutputRow(prefix, batch_index),
                  "golden prefix " + std::to_string(prefix));
      RequireNear(fused, oracle,
                  "cuBLAS oracle prefix " + std::to_string(prefix));
    }
  }
}

void TestOneHotInputsObserveEveryWeight() {
  const ProgressiveMatrix matrix = MakeGoldenMatrix();
  const std::size_t columns = golden::kGeometry.original_columns;
  const std::size_t rows = golden::kGeometry.original_rows;
  for (std::size_t prefix = 0; prefix <= golden::kStageCount; ++prefix) {
    for (std::size_t active_column = 0; active_column < columns;
         ++active_column) {
      std::vector<float> input(columns, 0.0F);
      input[active_column] = 1.0F;
      std::vector<float> expected(rows);
      for (std::size_t row = 0; row < rows; ++row) {
        expected[row] = golden::kPrefixWeights[prefix][row * columns +
                                                        active_column];
      }
      const std::vector<float> fused =
          ProgressiveGemvCuda(input, matrix, prefix);
      RequireNear(fused, expected,
                  "one-hot prefix " + std::to_string(prefix) + " column " +
                      std::to_string(active_column));
    }
  }
}

void TestBaseOnlyAndBiasContracts() {
  ProgressiveMatrix matrix = MakeGoldenMatrix();
  matrix.stages.resize(1);
  const std::vector<float> input = InputRow(0);
  const std::vector<float> bias(golden::kBias.begin(), golden::kBias.end());
  RequireNear(ProgressiveGemvCuda(input, matrix, 1, bias),
              GoldenOutputRow(1, 0), "base-only matrix");

  const std::vector<float> zero_prefix =
      ProgressiveGemvCuda(input, matrix, 0);
  if (!std::all_of(zero_prefix.begin(), zero_prefix.end(),
                   [](float value) { return value == 0.0F; })) {
    throw std::runtime_error(
        "prefix zero without bias must produce exact zeros");
  }
  const std::vector<float> bias_only =
      ProgressiveGemvCuda(input, matrix, 0, bias);
  if (bias_only != bias) {
    throw std::runtime_error("prefix zero must add bias exactly once");
  }
}

void TestEdgePaddingIsExercised() {
  if (golden::kGeometry.original_rows >= golden::kGeometry.padded_rows ||
      golden::kGeometry.original_columns >= golden::kGeometry.padded_columns ||
      golden::kScaleShape[0] < 2 || golden::kScaleShape[1] < 2) {
    throw std::runtime_error(
        "canonical CUDA fixture no longer exercises two-dimensional padding");
  }
  const ProgressiveMatrix matrix = MakeGoldenMatrix();
  const std::vector<float> input = InputRow(1);
  RequireNear(ProgressiveGemvCuda(input, matrix, golden::kStageCount),
              ProgressiveLinearCudaOracle(input, 1, matrix,
                                          golden::kStageCount),
              "edge-padded matrix");
}

void TestHostValidation() {
  const ProgressiveMatrix matrix = MakeGoldenMatrix();
  const std::vector<float> input = InputRow(0);
  const std::vector<float> bias(golden::kBias.begin(), golden::kBias.end());

  RequireInvalidArgument(
      [&] { static_cast<void>(ProgressiveGemvCuda({}, matrix, 1, bias)); },
      "input geometry");
  RequireInvalidArgument(
      [&] {
        static_cast<void>(ProgressiveGemvCuda(input, matrix, 1, {0.0F}));
      },
      "bias geometry");
  RequireInvalidArgument(
      [&] {
        static_cast<void>(ProgressiveGemvCuda(
            input, matrix, golden::kStageCount + 1, bias));
      },
      "prefix range");

  std::vector<float> bad_input = input;
  bad_input[0] = std::numeric_limits<float>::infinity();
  RequireInvalidArgument(
      [&] { static_cast<void>(ProgressiveGemvCuda(bad_input, matrix, 1)); },
      "non-finite input");
  std::vector<float> bad_bias = bias;
  bad_bias[0] = std::numeric_limits<float>::quiet_NaN();
  RequireInvalidArgument(
      [&] {
        static_cast<void>(ProgressiveGemvCuda(input, matrix, 1, bad_bias));
      },
      "non-finite bias");

  ProgressiveMatrix non_contiguous = matrix;
  non_contiguous.stages[1].stage_index = 3;
  RequireInvalidArgument(
      [&] {
        static_cast<void>(ProgressiveGemvCuda(input, non_contiguous, 1));
      },
      "non-contiguous stages");

  ProgressiveMatrix short_codes = matrix;
  short_codes.stages[0].packed_codes.pop_back();
  RequireInvalidArgument(
      [&] { static_cast<void>(ProgressiveGemvCuda(input, short_codes, 1)); },
      "packed-code geometry");

  ProgressiveMatrix wrong_scale_shape = matrix;
  ++wrong_scale_shape.stages[0].scale_shape[0];
  RequireInvalidArgument(
      [&] {
        static_cast<void>(ProgressiveGemvCuda(input, wrong_scale_shape, 1));
      },
      "scale geometry");

  ProgressiveMatrix bad_scale = matrix;
  bad_scale.stages[0].scales[0] = -1.0F;
  RequireInvalidArgument(
      [&] { static_cast<void>(ProgressiveGemvCuda(input, bad_scale, 1)); },
      "negative scale");
  bad_scale = matrix;
  bad_scale.stages[0].scales[0] =
      std::numeric_limits<float>::quiet_NaN();
  RequireInvalidArgument(
      [&] { static_cast<void>(ProgressiveGemvCuda(input, bad_scale, 1)); },
      "non-finite scale");
  bad_scale = matrix;
  bad_scale.stages[0].scales[0] =
      std::numeric_limits<float>::max();
  RequireInvalidArgument(
      [&] { static_cast<void>(ProgressiveGemvCuda(input, bad_scale, 1)); },
      "unrepresentable FP16 scale");
  bad_scale = matrix;
  bad_scale.stages[0].scales[0] = 0.1F;
  RequireInvalidArgument(
      [&] { static_cast<void>(ProgressiveGemvCuda(input, bad_scale, 1)); },
      "inexact FP16 scale");

  ProgressiveMatrix empty = matrix;
  empty.stages.clear();
  RequireInvalidArgument(
      [&] { static_cast<void>(ProgressiveGemvCuda(input, empty, 0)); },
      "empty stage set");
  ProgressiveMatrix too_many = matrix;
  too_many.stages.push_back(matrix.stages.back());
  RequireInvalidArgument(
      [&] { static_cast<void>(ProgressiveGemvCuda(input, too_many, 1)); },
      "too many stages");

  ProgressiveMatrix bad_padding = matrix;
  --bad_padding.geometry.padded_columns;
  RequireInvalidArgument(
      [&] { static_cast<void>(ProgressiveGemvCuda(input, bad_padding, 1)); },
      "invalid padding");

  ProgressiveMatrix overflow = matrix;
  overflow.geometry = {
      std::numeric_limits<std::size_t>::max(),
      std::numeric_limits<std::size_t>::max(),
      std::numeric_limits<std::size_t>::max(),
      std::numeric_limits<std::size_t>::max(), 1, 1, 1};
  RequireOverflowError(
      [&] { static_cast<void>(ProgressiveGemvCuda(input, overflow, 1)); },
      "overflowing geometry");
}

void TestWideReductionStrides() {
  for (const std::size_t columns : {257U, 513U, 4096U}) {
    const ProgressiveMatrix matrix = MakeWideMatrix(columns);
    std::vector<float> input(columns);
    for (std::size_t column = 0; column < columns; ++column) {
      input[column] = static_cast<float>(
          static_cast<int>(column % 29) - 14) / 32.0F;
    }
    const std::vector<float> bias{0.125F, -0.25F, 0.375F};
    for (std::size_t prefix = 0; prefix <= golden::kStageCount; ++prefix) {
      const std::vector<float> fused =
          ProgressiveGemvCuda(input, matrix, prefix, bias);
      const std::vector<float> oracle =
          ProgressiveLinearCudaOracle(input, 1, matrix, prefix, bias);
      RequireNear(
          fused, oracle,
          "wide K=" + std::to_string(columns) +
              " prefix " + std::to_string(prefix));
    }
  }
}

void TestRawFp16ScaleBits() {
  TestDeviceBuffer<float> device_input(1);
  TestDeviceBuffer<float> device_output(1);
  TestDeviceBuffer<std::uint8_t> device_code(1);
  TestDeviceBuffer<std::uint16_t> device_scale(1);
  const float input = 1.0F;
  const std::uint8_t code = 0x01U;
  RequireCudaSuccess(
      cudaMemcpy(device_input.get(), &input, sizeof(input), cudaMemcpyHostToDevice),
      "raw-scale input H2D");
  RequireCudaSuccess(
      cudaMemcpy(device_code.get(), &code, sizeof(code), cudaMemcpyHostToDevice),
      "raw-scale code H2D");

  ProgressiveGemvDeviceView matrix{};
  matrix.geometry = {1, 1, 1, 1, 1, 1, 1};
  matrix.stages[0] = {device_code.get(), 1, device_scale.get(), 1};
  matrix.stage_count = 1;
  const ProgressiveGemvIoDeviceView io{
      device_input.get(), 1, nullptr, 0, device_output.get(), 1};

  const std::array<std::pair<std::uint16_t, float>, 2> cases{{
      {0x0001U, std::ldexp(1.0F, -24)},
      {0x7bffU, 65504.0F},
  }};
  for (const auto& [bits, expected] : cases) {
    RequireCudaSuccess(
        cudaMemcpy(
            device_scale.get(), &bits, sizeof(bits), cudaMemcpyHostToDevice),
        "raw-scale bits H2D");
    RequireCudaSuccess(
        LaunchProgressiveGemv(io, matrix, 1, nullptr),
        "raw-scale fused launch");
    RequireCudaSuccess(cudaDeviceSynchronize(), "raw-scale fused execution");
    float output = 0.0F;
    RequireCudaSuccess(
        cudaMemcpy(&output, device_output.get(), sizeof(output),
                   cudaMemcpyDeviceToHost),
        "raw-scale output D2H");
    if (output != expected) {
      throw std::runtime_error("raw FP16 scale bit pattern decoded incorrectly");
    }
  }
}

void TestProductionViewValidation() {
  const std::size_t padded_elements =
      golden::kGeometry.padded_rows * golden::kGeometry.padded_columns;
  const std::size_t code_bytes =
      padded_elements / 4 + (padded_elements % 4 != 0 ? 1U : 0U);
  const std::size_t scale_count = golden::kScaleShape[0] *
                                  golden::kScaleShape[1] *
                                  golden::kScaleShape[2] *
                                  golden::kScaleShape[3];
  TestCudaStream stream;
  TestDeviceBuffer<float> device_input(golden::kGeometry.original_columns);
  TestDeviceBuffer<float> device_output(golden::kGeometry.original_rows);
  TestDeviceBuffer<float> device_bias(golden::kGeometry.original_rows);
  TestDeviceBuffer<std::uint8_t> device_codes(code_bytes);
  TestDeviceBuffer<std::uint16_t> device_scales(scale_count);
  // Constructed after all buffers, hence synchronized before they are freed on
  // every normal or exceptional exit from this validation test.
  TestStreamCompletionGuard completion_guard(stream.get());
  RequireCudaSuccess(
      cudaMemsetAsync(device_input.get(), 0,
                      golden::kGeometry.original_columns * sizeof(float),
                      stream.get()),
      "production-view input memset");
  RequireCudaSuccess(
      cudaMemsetAsync(device_bias.get(), 0,
                      golden::kGeometry.original_rows * sizeof(float),
                      stream.get()),
      "production-view bias memset");
  RequireCudaSuccess(
      cudaMemsetAsync(device_codes.get(), 0, code_bytes, stream.get()),
      "production-view codes memset");
  RequireCudaSuccess(
      cudaMemsetAsync(device_scales.get(), 0,
                      scale_count * sizeof(std::uint16_t), stream.get()),
      "production-view scales memset");

  ProgressiveGemvDeviceView view{};
  view.geometry = {
      golden::kGeometry.original_rows,
      golden::kGeometry.original_columns,
      golden::kGeometry.padded_rows,
      golden::kGeometry.padded_columns,
      golden::kGeometry.tile_rows,
      golden::kGeometry.tile_columns,
      golden::kGeometry.group_size,
  };
  view.stages[0] = {
      device_codes.get(), static_cast<std::uint64_t>(code_bytes),
      device_scales.get(), static_cast<std::uint64_t>(scale_count)};
  view.stage_count = 1;

  const ProgressiveGemvIoDeviceView empty_io{};
  RequireInvalidDeviceView(
      LaunchProgressiveGemv(empty_io, view, 0, stream.get()),
      "null input/output");

  const ProgressiveGemvIoDeviceView io{
      device_input.get(),
      golden::kGeometry.original_columns,
      nullptr,
      0,
      device_output.get(),
      golden::kGeometry.original_rows,
  };
  // Establish that every pointer and count in the base fixture is valid before
  // mutating one field at a time below. Synchronize before any RAII buffer can
  // leave scope so asynchronous kernel accesses cannot outlive allocations.
  RequireCudaSuccess(LaunchProgressiveGemv(io, view, 1, stream.get()),
                     "valid production-view launch");
  RequireCudaSuccess(cudaStreamSynchronize(stream.get()),
                     "valid production-view execution");

  ProgressiveGemvIoDeviceView short_input = io;
  --short_input.input_count;
  RequireInvalidDeviceView(
      LaunchProgressiveGemv(short_input, view, 0, stream.get()),
      "a short input view");

  ProgressiveGemvIoDeviceView short_output = io;
  --short_output.output_count;
  RequireInvalidDeviceView(
      LaunchProgressiveGemv(short_output, view, 0, stream.get()),
      "a short output view");

  ProgressiveGemvIoDeviceView bias_without_count = io;
  bias_without_count.bias = device_bias.get();
  RequireInvalidDeviceView(
      LaunchProgressiveGemv(bias_without_count, view, 0, stream.get()),
      "a bias pointer without a count");

  ProgressiveGemvIoDeviceView count_without_bias = io;
  count_without_bias.bias_count = golden::kGeometry.original_rows;
  RequireInvalidDeviceView(
      LaunchProgressiveGemv(count_without_bias, view, 0, stream.get()),
      "a bias count without a pointer");

  ProgressiveGemvIoDeviceView short_bias = io;
  short_bias.bias = device_bias.get();
  short_bias.bias_count = golden::kGeometry.original_rows - 1;
  RequireInvalidDeviceView(
      LaunchProgressiveGemv(short_bias, view, 0, stream.get()),
      "a short bias view");

  ProgressiveGemvIoDeviceView valid_bias = io;
  valid_bias.bias = device_bias.get();
  valid_bias.bias_count = golden::kGeometry.original_rows;
  RequireCudaSuccess(
      LaunchProgressiveGemv(valid_bias, view, 0, stream.get()),
      "valid bias production-view launch");
  RequireCudaSuccess(cudaStreamSynchronize(stream.get()),
                     "valid bias production-view execution");

  ProgressiveGemvDeviceView short_codes = view;
  --short_codes.stages[0].packed_code_bytes;
  RequireInvalidDeviceView(
      LaunchProgressiveGemv(io, short_codes, 1, stream.get()),
      "an incorrect packed-code byte count");

  ProgressiveGemvDeviceView short_scales = view;
  --short_scales.stages[0].scale_count;
  RequireInvalidDeviceView(
      LaunchProgressiveGemv(io, short_scales, 1, stream.get()),
      "an incorrect scale count");

  ProgressiveGemvDeviceView missing_codes = view;
  missing_codes.stages[0].packed_codes = nullptr;
  RequireInvalidDeviceView(
      LaunchProgressiveGemv(io, missing_codes, 1, stream.get()),
      "missing stage code storage");

  ProgressiveGemvDeviceView missing_scales = view;
  missing_scales.stages[0].scale_fp16_bits = nullptr;
  RequireInvalidDeviceView(
      LaunchProgressiveGemv(io, missing_scales, 1, stream.get()),
      "missing stage scale storage");
}

}  // namespace

int main() {
  std::string unavailable_reason;
  if (!CudaDeviceAvailable(&unavailable_reason)) {
    std::cout << "SKIP: fused progressive GEMV test requires CUDA: "
              << unavailable_reason << '\n';
    return 77;
  }

  try {
    TestAllPrefixesAgainstGoldenAndOracle();
    TestOneHotInputsObserveEveryWeight();
    TestBaseOnlyAndBiasContracts();
    TestEdgePaddingIsExercised();
    TestHostValidation();
    TestWideReductionStrides();
    TestRawFp16ScaleBits();
    TestProductionViewValidation();
    std::cout << "fused progressive GEMV tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fused progressive GEMV test failed: " << error.what() << '\n';
    return 1;
  }
}
