#ifndef HARPQ_NATIVE_PACKET_IO_HPP_
#define HARPQ_NATIVE_PACKET_IO_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "harpq/native/packet_ring.hpp"

namespace harpq::native_runtime {

enum class PacketRequirement : std::uint8_t {
  kRequired = 0,
  kOptional = 1,
};

struct PacketReadRequest {
  std::string packet_id;
  std::uint64_t file_offset = 0;
  std::size_t logical_bytes = 0;
  std::size_t physical_bytes = 0;
  std::uint32_t expected_crc32c = 0;
  PacketRequirement requirement = PacketRequirement::kRequired;
};

struct PacketReadStats {
  std::size_t logical_bytes = 0;
  std::size_t physical_bytes_requested = 0;
  std::size_t file_bytes_returned = 0;
  std::size_t pread_calls = 0;
  bool legacy_unpadded_tail = false;
};

enum class PacketIoErrorCode : std::uint8_t {
  kNone = 0,
  kOpenFailed = 1,
  kStatFailed = 2,
  kPackSizeMismatch = 3,
  kReadFailed = 4,
  kUnexpectedEof = 5,
  kChecksumMismatch = 6,
};

struct PacketIoFailure {
  PacketIoErrorCode code = PacketIoErrorCode::kNone;
  int system_error = 0;
  std::size_t bytes_completed = 0;
  std::uint32_t expected_crc32c = 0;
  std::uint32_t actual_crc32c = 0;
  std::string message;
};

class PacketIoError : public std::runtime_error {
 public:
  explicit PacketIoError(PacketIoFailure failure);
  [[nodiscard]] const PacketIoFailure& failure() const noexcept;

 private:
  PacketIoFailure failure_;
};

class RequiredPacketLoadError final : public PacketIoError {
 public:
  explicit RequiredPacketLoadError(PacketIoFailure failure);
};

enum class PacketLoadStatus : std::uint8_t {
  kReady = 0,
  kWouldBlock = 1,
  kCancelled = 2,
  kOptionalUnavailable = 3,
};

struct PacketLoadResult {
  PacketLoadStatus status = PacketLoadStatus::kCancelled;
  std::optional<ReadySlotLease> ready;
  PacketReadStats stats{};
  PacketIoFailure failure{};
};

namespace detail {
struct CancellationState;
}  // namespace detail

class CancellationToken final {
 public:
  CancellationToken() noexcept = default;
  [[nodiscard]] bool IsCancelled() const noexcept;

 private:
  friend class CancellationSource;
  explicit CancellationToken(
      std::shared_ptr<const detail::CancellationState> state) noexcept;
  std::shared_ptr<const detail::CancellationState> state_;
};

class CancellationSource final {
 public:
  CancellationSource();
  [[nodiscard]] CancellationToken token() const noexcept;
  void Cancel() noexcept;

  CancellationSource(const CancellationSource&) = delete;
  CancellationSource& operator=(const CancellationSource&) = delete;
  CancellationSource(CancellationSource&&) noexcept = default;
  CancellationSource& operator=(CancellationSource&&) noexcept = default;

 private:
  std::shared_ptr<detail::CancellationState> state_;
};

// Owns one immutable pack descriptor and performs correctness-first buffered
// positional reads. This class deliberately does not claim O_DIRECT semantics
// or storage throughput evidence.
class PreadPackFile final {
 public:
  PreadPackFile(std::string path, std::uint64_t declared_pack_size);
  ~PreadPackFile();

  PreadPackFile(const PreadPackFile&) = delete;
  PreadPackFile& operator=(const PreadPackFile&) = delete;
  PreadPackFile(PreadPackFile&& other) noexcept;
  PreadPackFile& operator=(PreadPackFile&& other) noexcept;

  [[nodiscard]] const std::string& source_name() const noexcept;
  [[nodiscard]] std::uint64_t declared_pack_size() const noexcept;

  [[nodiscard]] PacketLoadResult Load(
      PacketRing& ring,
      const PacketReadRequest& request,
      const CancellationToken& cancellation = CancellationToken{});

 private:
  // Kept private so deterministic EINTR/partial-read injection is available to
  // the native unit test without becoming a production backend API.
  using ReadAtFunction = std::function<std::ptrdiff_t(
      std::uint8_t*, std::size_t, std::uint64_t, int&)>;
  friend class PacketIoTestAccess;
  PreadPackFile(std::string source_name,
                std::uint64_t declared_pack_size,
                ReadAtFunction read_at);

  [[nodiscard]] std::ptrdiff_t ReadAt(
      std::uint8_t* destination,
      std::size_t count,
      std::uint64_t offset,
      int& error_number);
  void Close() noexcept;

  std::string source_name_;
  std::uint64_t declared_pack_size_ = 0;
  int file_descriptor_ = -1;
  ReadAtFunction injected_read_at_;
};

}  // namespace harpq::native_runtime

#endif  // HARPQ_NATIVE_PACKET_IO_HPP_
