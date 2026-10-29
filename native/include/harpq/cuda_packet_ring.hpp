#ifndef HARPQ_CUDA_PACKET_RING_HPP_
#define HARPQ_CUDA_PACKET_RING_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <cuda_runtime_api.h>

#include "harpq/native/packet_io.hpp"
#include "harpq/native/packet_ring.hpp"

namespace harpq::native_runtime {

namespace detail {
class CudaPacketRingControl;
}  // namespace detail

class CudaReadySlotLease;
class CudaInFlightSlotLease;
class CudaPacketRingTestAccess;
class CudaDirectPackFile;

// CUDA-owning counterpart of FillingSlotLease. The additional shared control
// keeps the whole host registration alive even when the CudaPacketRing facade
// is destroyed before this lease.
class CudaFillingSlotLease final {
 public:
  CudaFillingSlotLease() noexcept = default;
  ~CudaFillingSlotLease() = default;

  CudaFillingSlotLease(const CudaFillingSlotLease&) = delete;
  CudaFillingSlotLease& operator=(const CudaFillingSlotLease&) = delete;
  CudaFillingSlotLease(CudaFillingSlotLease&& other) noexcept;
  CudaFillingSlotLease& operator=(CudaFillingSlotLease&& other) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t slot_index() const;
  [[nodiscard]] std::uint64_t generation() const;
  [[nodiscard]] MutablePacketBytes bytes() const;

  [[nodiscard]] CudaReadySlotLease Publish(PacketSlotMetadata metadata) &&;
  void Cancel() noexcept;

 private:
  friend class CudaPacketRing;
  CudaFillingSlotLease(
      std::shared_ptr<detail::CudaPacketRingControl> control,
      FillingSlotLease lease) noexcept;

  // Keep control before the core lease: reverse destruction releases the slot
  // before the last control reference can unregister the backing allocation.
  std::shared_ptr<detail::CudaPacketRingControl> control_;
  FillingSlotLease lease_;
};

class CudaReadySlotLease final {
 public:
  CudaReadySlotLease() noexcept = default;
  ~CudaReadySlotLease() = default;

  CudaReadySlotLease(const CudaReadySlotLease&) = delete;
  CudaReadySlotLease& operator=(const CudaReadySlotLease&) = delete;
  CudaReadySlotLease(CudaReadySlotLease&& other) noexcept;
  CudaReadySlotLease& operator=(CudaReadySlotLease&& other) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] ReadyPacketView view() const;
  [[nodiscard]] int device_ordinal() const;
  // Retains the registration independently of the ring facade. The caller
  // owns device_destination until completion and any dependent compute ends.
  [[nodiscard]] CudaInFlightSlotLease EnqueueH2d(
      void* device_destination, std::size_t destination_capacity_bytes) &&;
  void Release() noexcept;

 private:
  friend class CudaFillingSlotLease;
  friend class CudaPacketRing;
  friend class CudaDirectPackFile;
  CudaReadySlotLease(std::shared_ptr<detail::CudaPacketRingControl> control,
                     ReadySlotLease lease) noexcept;

  std::shared_ptr<detail::CudaPacketRingControl> control_;
  ReadySlotLease lease_;
};

class CudaInFlightSlotLease final {
 public:
  CudaInFlightSlotLease() noexcept = default;
  ~CudaInFlightSlotLease() = default;

  CudaInFlightSlotLease(const CudaInFlightSlotLease&) = delete;
  CudaInFlightSlotLease& operator=(const CudaInFlightSlotLease&) = delete;
  CudaInFlightSlotLease(CudaInFlightSlotLease&& other) noexcept;
  CudaInFlightSlotLease& operator=(CudaInFlightSlotLease&& other) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t slot_index() const;
  [[nodiscard]] std::uint64_t generation() const;
  [[nodiscard]] CompletionState PollAndRecycle() noexcept;
  [[nodiscard]] CompletionState WaitAndRecycle() noexcept;

  // Orders subsequent work on compute_stream after this slot's recorded H2D
  // completion. The wait must be enqueued before this lease is recycled and
  // its per-slot event becomes eligible for reuse.
  [[nodiscard]] cudaError_t MakeStreamWait(cudaStream_t compute_stream) noexcept;

 private:
  friend class CudaPacketRing;
  friend class CudaReadySlotLease;
  CudaInFlightSlotLease(
      std::shared_ptr<detail::CudaPacketRingControl> control,
      InFlightSlotLease lease,
      cudaEvent_t completion_event) noexcept;

  std::shared_ptr<detail::CudaPacketRingControl> control_;
  InFlightSlotLease lease_;
  cudaEvent_t completion_event_ = nullptr;
};

struct CudaPacketLoadResult {
  PacketLoadStatus status = PacketLoadStatus::kCancelled;
  std::optional<CudaReadySlotLease> ready;
  PacketReadStats stats{};
  PacketIoFailure failure{};
};

// Owns one page-aligned PacketRing allocation, registers that complete backing
// allocation exactly once with cudaHostRegister, and owns a nonblocking H2D
// stream plus one completion event per slot. Registration failure is fatal;
// there is deliberately no pageable-memory fallback.
class CudaPacketRing final {
 public:
  explicit CudaPacketRing(PacketRingConfig config);
  ~CudaPacketRing() = default;

  CudaPacketRing(const CudaPacketRing&) = delete;
  CudaPacketRing& operator=(const CudaPacketRing&) = delete;
  CudaPacketRing(CudaPacketRing&&) noexcept = default;
  CudaPacketRing& operator=(CudaPacketRing&&) noexcept = default;

  [[nodiscard]] std::optional<CudaFillingSlotLease> TryAcquireFilling();

  // Enqueues exactly ReadyPacketView::logical_bytes from the registered slot.
  // Validation and enqueue failures leave ready valid. On every error after an
  // asynchronous operation may have been accepted, the owned stream is safely
  // synchronized before this function throws.
  [[nodiscard]] CudaInFlightSlotLease EnqueueH2d(
      CudaReadySlotLease&& ready,
      void* device_destination,
      std::size_t destination_capacity_bytes);

  // Loads through the CUDA ring's registered core allocation and wraps the
  // resulting core ready lease before it can escape the facade.
  [[nodiscard]] CudaPacketLoadResult Load(
      PreadPackFile& source,
      const PacketReadRequest& request,
      const CancellationToken& cancellation = CancellationToken{});

  [[nodiscard]] std::size_t slot_count() const noexcept;
  [[nodiscard]] std::size_t slot_capacity() const noexcept;
  [[nodiscard]] std::size_t io_alignment() const noexcept;
  [[nodiscard]] std::size_t allocation_alignment() const noexcept;
  [[nodiscard]] std::size_t slot_stride() const noexcept;
  [[nodiscard]] std::vector<PacketSlotSnapshot> Snapshot() const;

 private:
  friend class CudaPacketRingTestAccess;
  friend class CudaDirectPackFile;
  // Only the owning direct-I/O adapter may bind a reader to this retained core.
  explicit CudaPacketRing(std::shared_ptr<detail::CudaPacketRingControl> control) noexcept;
  [[nodiscard]] PacketRing& DirectIoCore();
  enum class TestFault : std::uint32_t {
    kHostRegister = 1U << 0U,
    kEventRecord = 1U << 1U,
    kEventQuery = 1U << 2U,
    kEventSynchronize = 1U << 3U,
    kStreamSynchronize = 1U << 4U,
    kDeviceSynchronize = 1U << 5U,
    kHostUnregister = 1U << 6U,
  };

  static void InjectFaultsForTest(std::uint32_t fault_mask) noexcept;
  [[nodiscard]] cudaStream_t CopyStreamForTest() const noexcept;

  std::shared_ptr<detail::CudaPacketRingControl> control_;
};

}  // namespace harpq::native_runtime

#endif  // HARPQ_CUDA_PACKET_RING_HPP_
