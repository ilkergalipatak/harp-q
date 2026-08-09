#ifndef HARPQ_NATIVE_STAGE_CODEC_HPP_
#define HARPQ_NATIVE_STAGE_CODEC_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace harpq::native_runtime {

inline constexpr std::uint16_t kStageCodecVersion = 1;
inline constexpr std::size_t kMaximumStageCount = 4;

using ScaleShape = std::array<std::size_t, 4>;

struct MatrixGeometry {
  std::size_t original_rows;
  std::size_t original_columns;
  std::size_t padded_rows;
  std::size_t padded_columns;
  std::size_t tile_rows;
  std::size_t tile_columns;
  std::size_t group_size;
};

struct DecodedStage {
  std::uint16_t stage_index;
  std::vector<std::uint8_t> packed_codes;
  std::vector<float> scales;
  ScaleShape scale_shape;
};

struct ProgressiveMatrix {
  MatrixGeometry geometry;
  std::vector<DecodedStage> stages;
};

class StageCodecError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[nodiscard]] DecodedStage DecodeStagePayload(
    const std::vector<std::uint8_t>& payload,
    std::uint16_t expected_stage,
    const ScaleShape& scale_shape);

[[nodiscard]] std::vector<float> DequantizeStage(
    const ProgressiveMatrix& matrix, std::size_t stage_index);

[[nodiscard]] std::vector<float> DequantizePrefix(
    const ProgressiveMatrix& matrix, std::size_t prefix);

// Inputs and outputs are row-major. The input shape is
// (batch_size, original_columns); output is (batch_size, original_rows).
[[nodiscard]] std::vector<float> ProgressiveLinearReference(
    const std::vector<float>& inputs,
    std::size_t batch_size,
    const ProgressiveMatrix& matrix,
    std::size_t prefix,
    const std::vector<float>& bias = {});

[[nodiscard]] std::uint32_t Crc32c(
    const std::uint8_t* data, std::size_t size, std::uint32_t crc = 0);

[[nodiscard]] inline std::uint32_t Crc32c(
    const std::vector<std::uint8_t>& data, std::uint32_t crc = 0) {
  return Crc32c(data.data(), data.size(), crc);
}

}  // namespace harpq::native_runtime

#endif  // HARPQ_NATIVE_STAGE_CODEC_HPP_
