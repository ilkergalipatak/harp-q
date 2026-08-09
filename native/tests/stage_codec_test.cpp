#include "harpq/native/stage_codec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "generated/golden_vectors.hpp"

namespace {

using harpq::native_runtime::Crc32c;
using harpq::native_runtime::DecodeStagePayload;
using harpq::native_runtime::DequantizePrefix;
using harpq::native_runtime::ProgressiveLinearReference;
using harpq::native_runtime::ProgressiveMatrix;
using harpq::native_runtime::StageCodecError;
namespace golden = harpq::native_runtime::golden;

int failures = 0;

void Check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Callable>
void ExpectStageCodecError(Callable&& callable, const std::string& message) {
  try {
    callable();
    Check(false, message + " (no exception)");
  } catch (const StageCodecError&) {
  } catch (const std::exception& error) {
    Check(false, message + " (wrong exception: " + error.what() + ")");
  }
}

template <typename Expected>
void CheckNear(
    const std::vector<float>& actual,
    const Expected& expected,
    float tolerance,
    const std::string& label) {
  Check(actual.size() == expected.size(), label + " size");
  if (actual.size() != expected.size()) {
    return;
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index]) ||
        std::abs(actual[index] - expected[index]) > tolerance) {
      Check(false, label + " value at index " + std::to_string(index));
      return;
    }
  }
}

ProgressiveMatrix BuildGoldenMatrix() {
  ProgressiveMatrix matrix{golden::kGeometry, {}};
  matrix.stages.reserve(golden::kStageCount);
  for (std::size_t index = 0; index < golden::kStageCount; ++index) {
    const auto& payload = golden::kStagePayloads[index];
    matrix.stages.push_back(DecodeStagePayload(
        std::vector<std::uint8_t>(payload.begin(), payload.end()),
        static_cast<std::uint16_t>(index),
        golden::kScaleShape));
  }
  return matrix;
}

void TestCrc32c() {
  const std::string standard = "123456789";
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(standard.data());
  Check(Crc32c(bytes, standard.size()) == 0xE3069283U, "CRC32C standard vector");
  const std::uint32_t prefix = Crc32c(bytes, 4);
  Check(
      Crc32c(bytes + 4, standard.size() - 4, prefix) == 0xE3069283U,
      "CRC32C incremental update");
  Check(Crc32c(nullptr, 0) == 0U, "CRC32C empty payload");
  for (std::size_t index = 0; index < golden::kStageCount; ++index) {
    const auto& payload = golden::kStagePayloads[index];
    Check(
        Crc32c(payload.data(), payload.size()) == golden::kStageCrc32c[index],
        "CRC32C golden stage " + std::to_string(index));
  }
}

void TestDecodeValidation() {
  const auto& fixture = golden::kStagePayloads[0];
  const std::vector<std::uint8_t> valid(fixture.begin(), fixture.end());
  const auto decoded = DecodeStagePayload(valid, 0, golden::kScaleShape);
  Check(decoded.stage_index == 0, "decoded stage identity");
  Check(decoded.packed_codes.size() == 32, "decoded code byte count");
  Check(decoded.scales.size() == 32, "decoded scale count");

  ExpectStageCodecError(
      [&] { return DecodeStagePayload(std::vector<std::uint8_t>(19), 0, golden::kScaleShape); },
      "reject truncated payload");
  auto invalid_magic = valid;
  invalid_magic[0] = 0;
  ExpectStageCodecError(
      [&] { return DecodeStagePayload(invalid_magic, 0, golden::kScaleShape); },
      "reject invalid magic");
  auto invalid_version = valid;
  invalid_version[8] = 2;
  ExpectStageCodecError(
      [&] { return DecodeStagePayload(invalid_version, 0, golden::kScaleShape); },
      "reject unsupported version");
  ExpectStageCodecError(
      [&] { return DecodeStagePayload(valid, 1, golden::kScaleShape); },
      "reject stage mismatch");
  auto invalid_length = valid;
  invalid_length.pop_back();
  ExpectStageCodecError(
      [&] { return DecodeStagePayload(invalid_length, 0, golden::kScaleShape); },
      "reject payload length mismatch");
  auto trailing_byte = valid;
  trailing_byte.push_back(0);
  ExpectStageCodecError(
      [&] { return DecodeStagePayload(trailing_byte, 0, golden::kScaleShape); },
      "reject trailing payload bytes");
  auto invalid_scale_count = valid;
  invalid_scale_count[16] = 7;
  ExpectStageCodecError(
      [&] { return DecodeStagePayload(invalid_scale_count, 0, golden::kScaleShape); },
      "reject scale count mismatch");
  constexpr std::size_t kFirstScaleOffset =
      20 + (golden::kGeometry.padded_rows * golden::kGeometry.padded_columns + 3) / 4;
  auto negative_scale = valid;
  negative_scale[kFirstScaleOffset + 1] =
      static_cast<std::uint8_t>(negative_scale[kFirstScaleOffset + 1] | 0x80U);
  ExpectStageCodecError(
      [&] { return DecodeStagePayload(negative_scale, 0, golden::kScaleShape); },
      "reject negative scale");
  auto nan_scale = valid;
  nan_scale[kFirstScaleOffset] = 0x00;
  nan_scale[kFirstScaleOffset + 1] = 0x7E;
  ExpectStageCodecError(
      [&] { return DecodeStagePayload(nan_scale, 0, golden::kScaleShape); },
      "reject non-finite scale");
}

void TestKnownSignedInt2Layout() {
  const harpq::native_runtime::MatrixGeometry geometry{
      1, 4, 1, 4, 1, 4, 4};
  harpq::native_runtime::DecodedStage stage{
      0,
      {0x4e},
      {1.0F},
      {1, 1, 1, 1},
  };
  const ProgressiveMatrix matrix{geometry, {stage}};
  const std::array<float, 4> expected{-2.0F, -1.0F, 0.0F, 1.0F};
  CheckNear(
      DequantizePrefix(matrix, 1), expected, 0.0F, "known signed-int2 byte 0x4e");
}

void TestGoldenPrefixes() {
  const ProgressiveMatrix matrix = BuildGoldenMatrix();
  const std::vector<float> inputs(golden::kInputs.begin(), golden::kInputs.end());
  const std::vector<float> bias(golden::kBias.begin(), golden::kBias.end());
  for (std::size_t prefix = 0; prefix < golden::kPrefixCount; ++prefix) {
    CheckNear(
        DequantizePrefix(matrix, prefix),
        golden::kPrefixWeights[prefix],
        0.0F,
        "dequantized prefix " + std::to_string(prefix));
    CheckNear(
        ProgressiveLinearReference(inputs, golden::kBatchSize, matrix, prefix, bias),
        golden::kPrefixOutputs[prefix],
        1.0e-6F,
        "linear prefix " + std::to_string(prefix));
  }
}

void TestRuntimeValidation() {
  const ProgressiveMatrix matrix = BuildGoldenMatrix();
  ExpectStageCodecError(
      [&] { return DequantizePrefix(matrix, golden::kPrefixCount); },
      "reject unavailable prefix");
  ExpectStageCodecError(
      [&] { return ProgressiveLinearReference({}, 1, matrix, 1); },
      "reject input geometry mismatch");
  ExpectStageCodecError(
      [&] {
        return ProgressiveLinearReference(
            std::vector<float>(golden::kGeometry.original_columns, 0.0F),
            1,
            matrix,
            1,
            {0.0F});
      },
      "reject bias geometry mismatch");
  auto invalid_matrix = matrix;
  invalid_matrix.stages[1].stage_index = 3;
  ExpectStageCodecError(
      [&] { return DequantizePrefix(invalid_matrix, 1); },
      "reject non-contiguous stage identity");
  auto non_finite = std::vector<float>(golden::kGeometry.original_columns, 0.0F);
  non_finite[0] = std::numeric_limits<float>::infinity();
  ExpectStageCodecError(
      [&] { return ProgressiveLinearReference(non_finite, 1, matrix, 1); },
      "reject non-finite input");
}

}  // namespace

int main() {
  TestCrc32c();
  TestDecodeValidation();
  TestKnownSignedInt2Layout();
  TestGoldenPrefixes();
  TestRuntimeValidation();
  if (failures != 0) {
    std::cerr << failures << " native test(s) failed\n";
    return 1;
  }
  std::cout << "HARP-Q native CPU tests passed\n";
  return 0;
}
