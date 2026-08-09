#include "harpq/cuda_oracle.hpp"

#include "generated/golden_vectors.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using harpq::native_runtime::CudaDeviceAvailable;
using harpq::native_runtime::DecodeStagePayload;
using harpq::native_runtime::ProgressiveLinearCudaOracle;
using harpq::native_runtime::ProgressiveMatrix;
namespace golden = harpq::native_runtime::golden;

constexpr float kAbsoluteTolerance = 1.0e-4F;
constexpr float kRelativeTolerance = 1.0e-4F;

template <typename Callable>
void RequireInvalidArgument(Callable&& callable, const std::string& label) {
  try {
    callable();
  } catch (const std::invalid_argument&) {
    return;
  } catch (const std::exception& error) {
    throw std::runtime_error(label + " raised the wrong exception: " + error.what());
  }
  throw std::runtime_error(label + " did not reject invalid input");
}

void RequireCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(status));
  }
}

void PrintDeviceSummary() {
  int device = 0;
  int driver_version = 0;
  int runtime_version = 0;
  cudaDeviceProp properties{};
  RequireCuda(cudaGetDevice(&device), "cudaGetDevice");
  RequireCuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
  RequireCuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
  RequireCuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  std::cout << "CUDA device=" << properties.name << "; cc=" << properties.major << '.'
            << properties.minor << "; vram_bytes=" << properties.totalGlobalMem
            << "; async_engines=" << properties.asyncEngineCount
            << "; driver_api=" << driver_version
            << "; runtime=" << runtime_version << '\n';
}

[[nodiscard]] ProgressiveMatrix MakeGoldenMatrix() {
  ProgressiveMatrix matrix{golden::kGeometry, {}};
  matrix.stages.reserve(golden::kStageCount);
  for (std::size_t stage_index = 0; stage_index < golden::kStageCount;
       ++stage_index) {
    const auto& bytes = golden::kStagePayloads[stage_index];
    const std::vector<std::uint8_t> payload(bytes.begin(), bytes.end());
    matrix.stages.push_back(DecodeStagePayload(
        payload, static_cast<std::uint16_t>(stage_index), golden::kScaleShape));
  }
  return matrix;
}

template <typename Expected>
void RequireNear(const std::vector<float>& actual,
                 const Expected& expected,
                 std::size_t prefix) {
  if (actual.size() != expected.size()) {
    std::ostringstream message;
    message << "prefix " << prefix << " output size is " << actual.size()
            << "; expected " << expected.size();
    throw std::runtime_error(message.str());
  }

  float maximum_absolute_error = 0.0F;
  std::size_t maximum_error_index = 0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index])) {
      std::ostringstream message;
      message << "prefix " << prefix << " produced a non-finite value at index "
              << index;
      throw std::runtime_error(message.str());
    }
    const float absolute_error = std::abs(actual[index] - expected[index]);
    const float allowed_error =
        kAbsoluteTolerance + kRelativeTolerance * std::abs(expected[index]);
    if (absolute_error > maximum_absolute_error) {
      maximum_absolute_error = absolute_error;
      maximum_error_index = index;
    }
    if (absolute_error > allowed_error) {
      std::ostringstream message;
      message << "prefix " << prefix << " differs at output " << index
              << ": CUDA=" << actual[index] << ", golden=" << expected[index]
              << ", abs_error=" << absolute_error
              << ", tolerance=" << allowed_error;
      throw std::runtime_error(message.str());
    }
  }
  std::cout << "prefix " << prefix << " ok; max_abs_error="
            << maximum_absolute_error << " at output " << maximum_error_index << '\n';
}

void TestGoldenPrefixes() {
  static_assert(golden::kStageCount == 4,
                "the canonical progressive fixture must contain four stages");
  static_assert(golden::kPrefixCount == golden::kStageCount + 1,
                "the canonical fixture must cover prefix zero through four");

  const ProgressiveMatrix matrix = MakeGoldenMatrix();
  const std::vector<float> inputs(golden::kInputs.begin(), golden::kInputs.end());
  const std::vector<float> bias(golden::kBias.begin(), golden::kBias.end());
  for (std::size_t prefix = 0; prefix < golden::kPrefixCount; ++prefix) {
    const std::vector<float> output = ProgressiveLinearCudaOracle(
        inputs, golden::kBatchSize, matrix, prefix, bias);
    RequireNear(output, golden::kPrefixOutputs[prefix], prefix);
  }
}

void TestIdentityInputObservesEveryWeight() {
  const ProgressiveMatrix matrix = MakeGoldenMatrix();
  const std::size_t input_features = golden::kGeometry.original_columns;
  const std::size_t output_features = golden::kGeometry.original_rows;
  std::vector<float> identity(input_features * input_features, 0.0F);
  for (std::size_t index = 0; index < input_features; ++index) {
    identity[index * input_features + index] = 1.0F;
  }

  for (std::size_t prefix = 0; prefix < golden::kPrefixCount; ++prefix) {
    std::vector<float> expected;
    expected.reserve(input_features * output_features);
    for (std::size_t input = 0; input < input_features; ++input) {
      for (std::size_t output = 0; output < output_features; ++output) {
        expected.push_back(
            golden::kPrefixWeights[prefix][output * input_features + input]);
      }
    }
    const std::vector<float> actual = ProgressiveLinearCudaOracle(
        identity, input_features, matrix, prefix);
    RequireNear(actual, expected, prefix);
  }
}

void TestBaseOnlyMatrix() {
  ProgressiveMatrix matrix = MakeGoldenMatrix();
  matrix.stages.resize(1);
  const std::vector<float> inputs(golden::kInputs.begin(), golden::kInputs.end());
  const std::vector<float> bias(golden::kBias.begin(), golden::kBias.end());
  const std::vector<float> output = ProgressiveLinearCudaOracle(
      inputs, golden::kBatchSize, matrix, 1, bias);
  RequireNear(output, golden::kPrefixOutputs[1], 1);
}

void TestCudaValidation() {
  const ProgressiveMatrix matrix = MakeGoldenMatrix();
  const std::vector<float> inputs(golden::kInputs.begin(), golden::kInputs.end());
  const std::vector<float> bias(golden::kBias.begin(), golden::kBias.end());

  RequireInvalidArgument(
      [&] {
        static_cast<void>(ProgressiveLinearCudaOracle(
            inputs, golden::kBatchSize, matrix, golden::kStageCount + 1, bias));
      },
      "prefix range");
  RequireInvalidArgument(
      [&] {
        static_cast<void>(ProgressiveLinearCudaOracle(
            {}, golden::kBatchSize, matrix, 1, bias));
      },
      "input geometry");
  RequireInvalidArgument(
      [&] {
        static_cast<void>(ProgressiveLinearCudaOracle(
            inputs, golden::kBatchSize, matrix, 1, {0.0F}));
      },
      "bias geometry");

  std::vector<float> non_finite_inputs = inputs;
  non_finite_inputs[0] = std::numeric_limits<float>::infinity();
  RequireInvalidArgument(
      [&] {
        static_cast<void>(ProgressiveLinearCudaOracle(
            non_finite_inputs, golden::kBatchSize, matrix, 1, bias));
      },
      "non-finite input");

  ProgressiveMatrix non_contiguous = matrix;
  non_contiguous.stages[1].stage_index = 3;
  RequireInvalidArgument(
      [&] {
        static_cast<void>(ProgressiveLinearCudaOracle(
            inputs, golden::kBatchSize, non_contiguous, 1, bias));
      },
      "non-contiguous stages");

  ProgressiveMatrix short_codes = matrix;
  short_codes.stages[0].packed_codes.pop_back();
  RequireInvalidArgument(
      [&] {
        static_cast<void>(ProgressiveLinearCudaOracle(
            inputs, golden::kBatchSize, short_codes, 1, bias));
      },
      "packed-code geometry");

  ProgressiveMatrix wrong_scale_shape = matrix;
  ++wrong_scale_shape.stages[0].scale_shape[0];
  RequireInvalidArgument(
      [&] {
        static_cast<void>(ProgressiveLinearCudaOracle(
            inputs, golden::kBatchSize, wrong_scale_shape, 1, bias));
      },
      "scale geometry");

  ProgressiveMatrix non_finite_scale = matrix;
  non_finite_scale.stages[0].scales[0] =
      std::numeric_limits<float>::quiet_NaN();
  RequireInvalidArgument(
      [&] {
        static_cast<void>(ProgressiveLinearCudaOracle(
            inputs, golden::kBatchSize, non_finite_scale, 1, bias));
      },
      "non-finite scale");
}

void TestZeroPrefixWithoutBias() {
  const ProgressiveMatrix matrix = MakeGoldenMatrix();
  const std::vector<float> inputs(golden::kInputs.begin(), golden::kInputs.end());
  const std::vector<float> output =
      ProgressiveLinearCudaOracle(inputs, golden::kBatchSize, matrix, 0);
  if (!std::all_of(output.begin(), output.end(),
                   [](float value) { return value == 0.0F; })) {
    throw std::runtime_error("prefix zero without bias must produce exact zeros");
  }
}

void TestEmptyBatchMatchesReferenceContract() {
  const ProgressiveMatrix matrix = MakeGoldenMatrix();
  const std::vector<float> bias(golden::kBias.begin(), golden::kBias.end());
  const std::vector<float> output =
      ProgressiveLinearCudaOracle({}, 0, matrix, golden::kStageCount, bias);
  if (!output.empty()) {
    throw std::runtime_error("an empty batch must produce an empty output");
  }
}

}  // namespace

int main() {
  std::string unavailable_reason;
  if (!CudaDeviceAvailable(&unavailable_reason)) {
    std::cout << "SKIP: CUDA oracle test requires a working CUDA device: "
              << unavailable_reason << '\n';
    return 77;
  }

  try {
    PrintDeviceSummary();
    TestGoldenPrefixes();
    TestIdentityInputObservesEveryWeight();
    TestBaseOnlyMatrix();
    TestZeroPrefixWithoutBias();
    TestEmptyBatchMatchesReferenceContract();
    TestCudaValidation();
    std::cout << "CUDA oracle golden-vector tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CUDA oracle test failed: " << error.what() << '\n';
    return 1;
  }
}
