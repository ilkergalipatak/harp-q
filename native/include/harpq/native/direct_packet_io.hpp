#ifndef HARPQ_NATIVE_DIRECT_PACKET_IO_HPP_
#define HARPQ_NATIVE_DIRECT_PACKET_IO_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "harpq/native/packet_io.hpp"
#include "harpq/native/packet_ring.hpp"

namespace harpq::native_runtime {

struct DirectPacketIoConfig {
  // This is a logical bound on concurrently outstanding packet reads. The
  // kernel ring may round its entry count up, but this backend never admits
  // more than queue_depth packet operations.
  std::uint32_t queue_depth = 8;
};

enum class DirectIoAlignmentSource : std::uint8_t {
  kStatxDioalign = 0,
  kFilesystemBlockSizeFallback = 1,
};

struct DirectIoAlignment {
  std::size_t memory_alignment = 0;
  std::size_t offset_alignment = 0;
  DirectIoAlignmentSource source =
      DirectIoAlignmentSource::kFilesystemBlockSizeFallback;
};

enum class DirectIoUnsupportedReason : std::uint8_t {
  kNone = 0,
  kStatxQueryFailed = 1,
  kStatxDioalignNotReported = 2,
  kStatxDioalignZero = 3,
  kIoUringProbeFailed = 4,
  kIoUringReadUnsupported = 5,
};

struct DirectIoCapability {
  bool supported = false;
  DirectIoAlignment alignment{};
  DirectIoUnsupportedReason unsupported_reason =
      DirectIoUnsupportedReason::kNone;
  int system_error = 0;
};

// kBufferedOnlyLegacyTail is deliberately distinct from an optional-packet
// failure. It means the v1 packet is valid, but its legacy unpadded terminal
// extent cannot be truthfully submitted through O_DIRECT. The caller may
// explicitly route that request to PreadPackFile; this backend never performs
// a silent fallback.
enum class DirectPacketLoadStatus : std::uint8_t {
  kSubmitted = 0,
  kReady = 1,
  kWouldBlock = 2,
  kCancelled = 3,
  kOptionalUnavailable = 4,
  kBufferedOnlyLegacyTail = 5,
  // The filesystem did not return an authoritative STATX_DIOALIGN contract.
  // Fallback geometry is diagnostic only and no O_DIRECT SQE was submitted.
  kDirectUnsupported = 6,
};

struct DirectPacketReadStats {
  std::size_t logical_bytes = 0;
  std::size_t physical_bytes_requested = 0;
  std::size_t file_bytes_returned = 0;
  // Includes retry SQEs after an aligned short completion. bytes_submitted can
  // therefore exceed physical_bytes_requested in an injected/recovery path.
  std::size_t bytes_submitted = 0;
  std::size_t read_submissions = 0;
  std::size_t read_completions = 0;
  // Cooperative cancellation is checked before submission and after the
  // original read CQE. This slice intentionally issues no ASYNC_CANCEL SQEs;
  // these reserved counters therefore remain zero.
  std::size_t cancel_submissions = 0;
  std::size_t cancel_completions = 0;
  bool legacy_unpadded_tail = false;
  // The ordinary IORING_OP_READ destination was the acquired aligned ring
  // slot. This does not claim that io_uring fixed-buffer registration was used.
  bool used_ring_slot_buffer = false;
};

struct DirectPacketLoadResult {
  DirectPacketLoadStatus status = DirectPacketLoadStatus::kCancelled;
  // Zero for an immediate result (cancelled, backpressure, legacy buffered-only,
  // or unsupported direct I/O).
  // Non-zero and stable from kSubmitted through the terminal completion. IDs
  // are monotonic only within one DirectPackFile instance; they are not global.
  std::uint64_t operation_id = 0;
  std::optional<ReadySlotLease> ready;
  DirectPacketReadStats stats{};
  PacketIoFailure failure{};
};

namespace detail {
class DirectPackFileImpl;
}  // namespace detail

class DirectPacketIoTestAccess;

// Linux-only asynchronous immutable-pack reader. It opens the pack with
// O_DIRECT, owns one liburing instance, and submits ordinary IORING_OP_READ
// operations directly into acquired PacketRing slots. Fixed-buffer registration
// is intentionally not implied. PacketRing must outlive this object; declare the
// ring before DirectPackFile so normal reverse destruction drains I/O before the
// ring allocation is released. Do not move that PacketRing facade while this
// backend is alive.
//
// The API is scheduler-thread confined. Submit/Poll/Wait must not be called
// concurrently. A submitted operation owns its FillingSlotLease until it is
// cancelled, fails, or publishes a CRC-verified ReadySlotLease.
// Forking while this object is alive is forbidden, even with no operation in
// flight: io_uring_ring_dontfork protects the kernel-ring mappings, but cannot
// make an inherited ordinary-read destination or allocator state safe. The PID
// guard is defense in depth for an accidental child call, not fork support.
//
// This first slice is a CPU PacketRing direct-I/O correctness adapter, not a
// CUDA ownership bridge. It must not be pointed through CudaPacketRing private
// storage or used to infer that CUDA shared-control lifetime extends through an
// outstanding read CQE; that composition requires a separate owning adapter.
class DirectPackFile final {
 public:
  DirectPackFile(std::string path,
                 std::uint64_t declared_pack_size,
                 PacketRing& ring,
                 DirectPacketIoConfig config = {});
  ~DirectPackFile();

  DirectPackFile(const DirectPackFile&) = delete;
  DirectPackFile& operator=(const DirectPackFile&) = delete;
  DirectPackFile(DirectPackFile&&) = delete;
  DirectPackFile& operator=(DirectPackFile&&) = delete;

  [[nodiscard]] const std::string& source_name() const noexcept;
  [[nodiscard]] std::uint64_t declared_pack_size() const noexcept;
  [[nodiscard]] std::uint32_t queue_depth() const noexcept;
  [[nodiscard]] std::size_t in_flight() const noexcept;
  [[nodiscard]] DirectIoAlignment direct_alignment() const noexcept;
  [[nodiscard]] DirectIoCapability direct_capability() const noexcept;
  [[nodiscard]] bool direct_supported() const noexcept;
  [[nodiscard]] std::size_t ring_slot_count() const noexcept;

  [[nodiscard]] DirectPacketLoadResult Submit(
      const PacketReadRequest& request,
      const CancellationToken& cancellation = CancellationToken{});

  // Poll returns nullopt when no terminal packet completion is currently
  // available. Cancellation is cooperative: a cancelled token becomes visible
  // only after the original read CQE proves the slot is no longer kernel-owned.
  [[nodiscard]] std::optional<DirectPacketLoadResult> Poll();

  // Wait blocks for one terminal packet completion. Calling it with no packet
  // operation in flight is a logic error.
  [[nodiscard]] DirectPacketLoadResult Wait();

 private:
  friend class DirectPacketIoTestAccess;
  // Replaces successive read CQE results after the real kernel completion.
  // It is private so deterministic short/error injection cannot accidentally
  // become a production runtime policy.
  static void InjectCompletionResultsForTest(std::vector<int> results);
  static void ClearCompletionResultsForTest() noexcept;
  static void SuppressStatxDioalignForTestOnce() noexcept;

  std::unique_ptr<detail::DirectPackFileImpl> impl_;
};

}  // namespace harpq::native_runtime

#endif  // HARPQ_NATIVE_DIRECT_PACKET_IO_HPP_
