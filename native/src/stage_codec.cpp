#include "harpq/native/stage_codec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace harpq::native_runtime {
namespace {

constexpr std::array<std::uint8_t, 8> kPayloadMagic{
    'H', 'Q', 'S', 'T', 'A', 'G', 'E', '\0'};
constexpr std::size_t kPayloadHeaderBytes = 20;
constexpr std::uint32_t kCrc32cPolynomial = 0x82F63B78U;

[[nodiscard]] std::size_t CheckedMultiply(
    std::size_t left, std::size_t right, const char* label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw StageCodecError(std::string(label) + " overflows size_t");
  }
  return left * right;
}

[[nodiscard]] std::size_t CheckedAdd(
    std::size_t left, std::size_t right, const char* label) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw StageCodecError(std::string(label) + " overflows size_t");
  }
  return left + right;
}

[[nodiscard]] std::uint16_t ReadLe16(const std::uint8_t* bytes) noexcept {
  return static_cast<std::uint16_t>(bytes[0]) |
         static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[1]) << 8U);
}

[[nodiscard]] std::uint32_t ReadLe32(const std::uint8_t* bytes) noexcept {
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

[[nodiscard]] float HalfToFloat(std::uint16_t half) noexcept {
  const bool negative = (half & 0x8000U) != 0;
  const std::uint16_t exponent = static_cast<std::uint16_t>((half >> 10U) & 0x1FU);
  const std::uint16_t mantissa = static_cast<std::uint16_t>(half & 0x03FFU);

  float value = 0.0F;
  if (exponent == 0) {
    value = std::ldexp(static_cast<float>(mantissa), -24);
  } else if (exponent == 0x1FU) {
    value = mantissa == 0 ? std::numeric_limits<float>::infinity()
                          : std::numeric_limits<float>::quiet_NaN();
  } else {
    value = std::ldexp(
        1.0F + static_cast<float>(mantissa) / 1024.0F,
        static_cast<int>(exponent) - 15);
  }
  return negative ? -value : value;
}

[[nodiscard]] std::size_t ScaleCount(const ScaleShape& shape) {
  std::size_t count = 1;
  for (const std::size_t dimension : shape) {
    if (dimension == 0) {
      throw StageCodecError("scale_shape dimensions must be positive");
    }
    count = CheckedMultiply(count, dimension, "scale count");
  }
  return count;
}

[[nodiscard]] std::size_t CeilDiv(std::size_t numerator, std::size_t denominator) {
  return numerator / denominator + (numerator % denominator != 0 ? 1U : 0U);
}

void ValidateMatrix(const ProgressiveMatrix& matrix) {
  const MatrixGeometry& geometry = matrix.geometry;
  if (geometry.original_rows == 0 || geometry.original_columns == 0) {
    throw StageCodecError("original matrix dimensions must be positive");
  }
  if (geometry.tile_rows == 0 || geometry.tile_columns == 0 ||
      geometry.group_size == 0) {
    throw StageCodecError("tile dimensions and group_size must be positive");
  }
  if (geometry.tile_columns % geometry.group_size != 0) {
    throw StageCodecError("tile width must be divisible by group_size");
  }

  const std::size_t expected_padded_rows = CheckedMultiply(
      CeilDiv(geometry.original_rows, geometry.tile_rows),
      geometry.tile_rows,
      "padded rows");
  const std::size_t expected_padded_columns = CheckedMultiply(
      CeilDiv(geometry.original_columns, geometry.tile_columns),
      geometry.tile_columns,
      "padded columns");
  if (geometry.padded_rows != expected_padded_rows ||
      geometry.padded_columns != expected_padded_columns) {
    throw StageCodecError("padded matrix dimensions do not match tile geometry");
  }
  if (matrix.stages.empty() || matrix.stages.size() > kMaximumStageCount) {
    throw StageCodecError("a progressive matrix must contain between one and four stages");
  }

  const ScaleShape expected_scale_shape{
      geometry.padded_rows / geometry.tile_rows,
      geometry.padded_columns / geometry.tile_columns,
      geometry.tile_rows,
      geometry.tile_columns / geometry.group_size};
  const std::size_t padded_elements = CheckedMultiply(
      geometry.padded_rows, geometry.padded_columns, "padded element count");
  const std::size_t expected_code_bytes = CeilDiv(padded_elements, 4);
  const std::size_t expected_scale_count = ScaleCount(expected_scale_shape);

  for (std::size_t index = 0; index < matrix.stages.size(); ++index) {
    const DecodedStage& stage = matrix.stages[index];
    if (stage.stage_index != index) {
      throw StageCodecError("stages must form a zero-based contiguous prefix");
    }
    if (stage.scale_shape != expected_scale_shape ||
        stage.scales.size() != expected_scale_count) {
      throw StageCodecError("stage scale geometry does not match the matrix");
    }
    if (stage.packed_codes.size() != expected_code_bytes) {
      throw StageCodecError("stage code payload length does not match the matrix");
    }
    if (!std::all_of(stage.scales.begin(), stage.scales.end(), [](float scale) {
          return std::isfinite(scale) && scale >= 0.0F;
        })) {
      throw StageCodecError("stage scales must be finite and non-negative");
    }
  }
}

[[nodiscard]] std::int8_t DecodeCode(
    const std::vector<std::uint8_t>& packed_codes, std::size_t element_index) {
  const std::uint8_t packed = packed_codes[element_index / 4];
  const auto shift = static_cast<unsigned>((element_index % 4) * 2);
  const std::uint8_t raw = static_cast<std::uint8_t>((packed >> shift) & 0x03U);
  return static_cast<std::int8_t>(raw >= 2 ? static_cast<int>(raw) - 4 : raw);
}

[[nodiscard]] std::array<std::uint32_t, 256> MakeCrc32cTable() noexcept {
  std::array<std::uint32_t, 256> table{};
  for (std::uint32_t index = 0; index < table.size(); ++index) {
    std::uint32_t value = index;
    for (int bit = 0; bit < 8; ++bit) {
      value = (value >> 1U) ^ ((value & 1U) != 0 ? kCrc32cPolynomial : 0U);
    }
    table[index] = value;
  }
  return table;
}

}  // namespace

DecodedStage DecodeStagePayload(
    const std::vector<std::uint8_t>& payload,
    std::uint16_t expected_stage,
    const ScaleShape& scale_shape) {
  if (payload.size() < kPayloadHeaderBytes) {
    throw StageCodecError("stage payload is truncated");
  }
  if (!std::equal(kPayloadMagic.begin(), kPayloadMagic.end(), payload.begin())) {
    throw StageCodecError("stage payload magic is invalid");
  }

  const std::uint16_t version = ReadLe16(payload.data() + 8);
  const std::uint16_t stage_index = ReadLe16(payload.data() + 10);
  const std::uint32_t code_bytes_u32 = ReadLe32(payload.data() + 12);
  const std::uint32_t scale_count_u32 = ReadLe32(payload.data() + 16);
  if (version != kStageCodecVersion) {
    throw StageCodecError("unsupported stage codec version " + std::to_string(version));
  }
  if (stage_index != expected_stage) {
    throw StageCodecError("payload stage does not match expected stage");
  }

  const std::size_t expected_scale_count = ScaleCount(scale_shape);
  if (static_cast<std::size_t>(scale_count_u32) != expected_scale_count) {
    throw StageCodecError("scale count does not match metadata");
  }
  const std::size_t code_bytes = code_bytes_u32;
  const std::size_t scale_bytes = CheckedMultiply(expected_scale_count, 2, "scale bytes");
  const std::size_t expected_length = CheckedAdd(
      CheckedAdd(kPayloadHeaderBytes, code_bytes, "payload length"),
      scale_bytes,
      "payload length");
  if (payload.size() != expected_length) {
    throw StageCodecError("stage payload length does not match its header");
  }

  DecodedStage decoded;
  decoded.stage_index = stage_index;
  decoded.scale_shape = scale_shape;
  decoded.packed_codes.assign(
      payload.begin() + static_cast<std::ptrdiff_t>(kPayloadHeaderBytes),
      payload.begin() + static_cast<std::ptrdiff_t>(kPayloadHeaderBytes + code_bytes));
  decoded.scales.reserve(expected_scale_count);
  const std::size_t scales_offset = kPayloadHeaderBytes + code_bytes;
  for (std::size_t index = 0; index < expected_scale_count; ++index) {
    const float scale = HalfToFloat(ReadLe16(payload.data() + scales_offset + index * 2));
    if (!std::isfinite(scale)) {
      throw StageCodecError("stage scales must be finite");
    }
    if (scale < 0.0F) {
      throw StageCodecError("stage scales cannot be negative");
    }
    decoded.scales.push_back(scale);
  }
  return decoded;
}

std::vector<float> DequantizeStage(
    const ProgressiveMatrix& matrix, std::size_t stage_index) {
  ValidateMatrix(matrix);
  if (stage_index >= matrix.stages.size()) {
    throw StageCodecError("stage_index is outside the available prefix");
  }

  const MatrixGeometry& geometry = matrix.geometry;
  const DecodedStage& stage = matrix.stages[stage_index];
  std::vector<float> output(
      CheckedMultiply(
          geometry.original_rows, geometry.original_columns, "output element count"),
      0.0F);
  const std::size_t tile_grid_columns = geometry.padded_columns / geometry.tile_columns;
  const std::size_t groups_per_tile = geometry.tile_columns / geometry.group_size;

  for (std::size_t row = 0; row < geometry.original_rows; ++row) {
    const std::size_t tile_row = row / geometry.tile_rows;
    const std::size_t row_in_tile = row % geometry.tile_rows;
    for (std::size_t column = 0; column < geometry.original_columns; ++column) {
      const std::size_t tile_column = column / geometry.tile_columns;
      const std::size_t group = (column % geometry.tile_columns) / geometry.group_size;
      const std::size_t scale_index =
          (((tile_row * tile_grid_columns + tile_column) * geometry.tile_rows +
            row_in_tile) *
               groups_per_tile +
           group);
      const std::size_t code_index = row * geometry.padded_columns + column;
      output[row * geometry.original_columns + column] =
          static_cast<float>(DecodeCode(stage.packed_codes, code_index)) *
          stage.scales[scale_index];
    }
  }
  return output;
}

std::vector<float> DequantizePrefix(
    const ProgressiveMatrix& matrix, std::size_t prefix) {
  ValidateMatrix(matrix);
  if (prefix > matrix.stages.size()) {
    throw StageCodecError("prefix exceeds the available stage count");
  }
  const std::size_t output_size = CheckedMultiply(
      matrix.geometry.original_rows,
      matrix.geometry.original_columns,
      "output element count");
  std::vector<float> output(output_size, 0.0F);
  for (std::size_t stage_index = 0; stage_index < prefix; ++stage_index) {
    const std::vector<float> stage = DequantizeStage(matrix, stage_index);
    for (std::size_t index = 0; index < output.size(); ++index) {
      output[index] += stage[index];
    }
  }
  return output;
}

std::vector<float> ProgressiveLinearReference(
    const std::vector<float>& inputs,
    std::size_t batch_size,
    const ProgressiveMatrix& matrix,
    std::size_t prefix,
    const std::vector<float>& bias) {
  ValidateMatrix(matrix);
  if (prefix > matrix.stages.size()) {
    throw StageCodecError("prefix exceeds the available stage count");
  }
  const MatrixGeometry& geometry = matrix.geometry;
  const std::size_t expected_inputs = CheckedMultiply(
      batch_size, geometry.original_columns, "input element count");
  if (inputs.size() != expected_inputs) {
    throw StageCodecError("input element count does not match batch geometry");
  }
  if (!bias.empty() && bias.size() != geometry.original_rows) {
    throw StageCodecError("bias length must match output features");
  }
  if (!std::all_of(inputs.begin(), inputs.end(), [](float value) {
        return std::isfinite(value);
      }) ||
      !std::all_of(bias.begin(), bias.end(), [](float value) {
        return std::isfinite(value);
      })) {
    throw StageCodecError("inputs and bias must be finite");
  }

  std::vector<float> output(
      CheckedMultiply(batch_size, geometry.original_rows, "linear output element count"),
      0.0F);
  for (std::size_t stage_index = 0; stage_index < prefix; ++stage_index) {
    const std::vector<float> weights = DequantizeStage(matrix, stage_index);
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
      for (std::size_t row = 0; row < geometry.original_rows; ++row) {
        float dot = 0.0F;
        for (std::size_t column = 0; column < geometry.original_columns; ++column) {
          dot += inputs[batch * geometry.original_columns + column] *
                 weights[row * geometry.original_columns + column];
        }
        output[batch * geometry.original_rows + row] += dot;
      }
    }
  }
  if (!bias.empty()) {
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
      for (std::size_t row = 0; row < geometry.original_rows; ++row) {
        output[batch * geometry.original_rows + row] += bias[row];
      }
    }
  }
  return output;
}

std::uint32_t Crc32c(
    const std::uint8_t* data, std::size_t size, std::uint32_t crc) {
  if (data == nullptr && size != 0) {
    throw std::invalid_argument("CRC32C data cannot be null when size is non-zero");
  }
  static const std::array<std::uint32_t, 256> table = MakeCrc32cTable();
  std::uint32_t value = crc ^ 0xFFFFFFFFU;
  for (std::size_t index = 0; index < size; ++index) {
    value = table[(value ^ data[index]) & 0xFFU] ^ (value >> 8U);
  }
  return value ^ 0xFFFFFFFFU;
}

}  // namespace harpq::native_runtime
