#ifndef HARPQ_CUDA_DIRECT_PACKET_IO_HPP_
#define HARPQ_CUDA_DIRECT_PACKET_IO_HPP_

#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include "harpq/cuda_packet_ring.hpp"
#include "harpq/native/direct_packet_io.hpp"

namespace harpq::native_runtime {

// Host-observed intervals include polling, CRC, and dispatch delays. They are
// conservative service costs, not CUDA-event bandwidth or overlap measurements.
// The caller assigns globally ordered sequence_id values when exporting these
// fields to Python ServiceObservation; source-scoped operation IDs are not used.
struct PacketServiceObservation {
  static constexpr const char* service_class = "direct-h2d/host-observed-v1";
  std::string packet_id;
  std::size_t logical_bytes = 0;
  std::size_t physical_io_bytes = 0;
  double critical_path_us = 0;
  double io_service_us = 0;
  double h2d_service_us = 0;
};

namespace detail {
class CudaPacketTransferState;
}

class CudaDirectReadyPacket;

// Owns the destination allocation and H2D completion. Destruction waits for H2D
// before freeing it. Consumers must retain this object through their own GPU
// compute completion; MakeStreamWait orders compute but does not own its lifetime.
class CudaPacketTransfer final {
 public:
  CudaPacketTransfer() noexcept;
  ~CudaPacketTransfer();
  CudaPacketTransfer(const CudaPacketTransfer&) = delete;
  CudaPacketTransfer& operator=(const CudaPacketTransfer&) = delete;
  CudaPacketTransfer(CudaPacketTransfer&&) noexcept;
  CudaPacketTransfer& operator=(CudaPacketTransfer&&) noexcept;

  [[nodiscard]] CompletionState Poll() noexcept;
  [[nodiscard]] CompletionState Wait() noexcept;
  // Call before Wait/Poll recycles the completion event.
  [[nodiscard]] cudaError_t MakeStreamWait(cudaStream_t stream) noexcept;
  // Pointer is usable only after successful Wait/Poll or on an ordered stream.
  [[nodiscard]] const void* device_data() const noexcept;
  [[nodiscard]] std::size_t size_bytes() const noexcept;
  // Null until successful completion; failed or abandoned transfers emit none.
  [[nodiscard]] const PacketServiceObservation* observation() const noexcept;

 private:
  friend class CudaDirectReadyPacket;
  explicit CudaPacketTransfer(std::unique_ptr<detail::CudaPacketTransferState>) noexcept;
  std::unique_ptr<detail::CudaPacketTransferState> state_;
};

class CudaDirectReadyPacket final {
 public:
  CudaDirectReadyPacket(const CudaDirectReadyPacket&) = delete;
  CudaDirectReadyPacket& operator=(const CudaDirectReadyPacket&) = delete;
  CudaDirectReadyPacket(CudaDirectReadyPacket&&) noexcept = default;
  CudaDirectReadyPacket& operator=(CudaDirectReadyPacket&&) noexcept = default;

  [[nodiscard]] bool valid() const noexcept { return ready_.valid(); }
  [[nodiscard]] ReadyPacketView view() const { return ready_.view(); }
  // Allocation/enqueue failure preserves the ready lease for retry or release.
  [[nodiscard]] CudaPacketTransfer EnqueueH2d() &&;

 private:
  friend class CudaDirectPackFile;
  CudaDirectReadyPacket(CudaReadySlotLease ready, DirectPacketReadStats stats,
                       std::chrono::steady_clock::time_point submitted_at) noexcept;
  CudaReadySlotLease ready_;
  DirectPacketReadStats stats_;
  std::chrono::steady_clock::time_point submitted_at_;
  std::chrono::steady_clock::time_point ready_at_;
};

struct CudaDirectPacketLoadResult {
  DirectPacketLoadStatus status = DirectPacketLoadStatus::kCancelled;
  std::uint64_t operation_id = 0;
  std::optional<CudaDirectReadyPacket> ready;
  DirectPacketReadStats stats{};
  PacketIoFailure failure{};
};

// Linux + CUDA, owner-thread confined, no fork while alive. Shares the ring's
// registration owner, so the original ring facade may be moved or destroyed.
// Destruction drains original read CQEs before releasing that owner. Ready/H2D
// leases retain it independently. Read outcomes match DirectPackFile exactly.
class CudaDirectPackFile final {
 public:
  CudaDirectPackFile(CudaPacketRing& ring, std::string path,
                     std::uint64_t declared_pack_size, DirectPacketIoConfig config = {});
  ~CudaDirectPackFile() = default;
  CudaDirectPackFile(const CudaDirectPackFile&) = delete;
  CudaDirectPackFile& operator=(const CudaDirectPackFile&) = delete;
  CudaDirectPackFile(CudaDirectPackFile&&) = delete;
  CudaDirectPackFile& operator=(CudaDirectPackFile&&) = delete;

  [[nodiscard]] DirectIoCapability direct_capability() const noexcept;
  [[nodiscard]] std::size_t in_flight() const noexcept;
  [[nodiscard]] CudaDirectPacketLoadResult Submit(
      const PacketReadRequest&, const CancellationToken& = CancellationToken{});
  [[nodiscard]] std::optional<CudaDirectPacketLoadResult> Poll();
  [[nodiscard]] CudaDirectPacketLoadResult Wait();

 private:
  [[nodiscard]] CudaDirectPacketLoadResult Wrap(DirectPacketLoadResult);
  // Reverse destruction drains source_ before unregistering ring_.
  CudaPacketRing ring_;
  DirectPackFile source_;
};

}  // namespace harpq::native_runtime

#endif  // HARPQ_CUDA_DIRECT_PACKET_IO_HPP_
