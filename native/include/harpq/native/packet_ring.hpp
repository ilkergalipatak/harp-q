#ifndef HARPQ_NATIVE_PACKET_RING_HPP_
#define HARPQ_NATIVE_PACKET_RING_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace harpq::native_runtime {

enum class PacketSlotState : std::uint8_t {
  kFree = 0,
  kFilling = 1,
  kReady = 2,
  kH2dInFlight = 3,
};

struct PacketRingConfig {
  std::size_t slot_count = 0;
  std::size_t slot_capacity = 0;
  std::size_t io_alignment = 4096;
};

struct MutablePacketBytes {
  std::uint8_t* data = nullptr;
  std::size_t capacity = 0;
};

struct PacketSlotMetadata {
  std::string packet_id;
  std::size_t logical_bytes = 0;
  std::size_t physical_bytes = 0;
  std::uint32_t crc32c = 0;
};

struct ReadyPacketView {
  const std::uint8_t* data = nullptr;
  std::string packet_id;
  std::size_t logical_bytes = 0;
  std::size_t physical_bytes = 0;
  std::uint32_t crc32c = 0;
  std::size_t slot_index = 0;
  std::uint64_t generation = 0;
};

struct PacketSlotSnapshot {
  PacketSlotState state = PacketSlotState::kFree;
  std::uint64_t generation = 0;
};

struct RegistrationStorageView {
  std::uint8_t* data = nullptr;
  std::size_t bytes = 0;
  std::size_t alignment = 0;
};

enum class CompletionState : std::uint8_t {
  kPending = 0,
  kComplete = 1,
  kFailed = 2,
};

// CUDA-free completion boundary. A future CUDA implementation records an event
// after enqueueing H2D, reports whether that event has completed, and makes
// WaitUntilSafe block until the registered host slot is no longer referenced.
class TransferCompletion {
 public:
  virtual ~TransferCompletion() = default;
  [[nodiscard]] virtual CompletionState Poll() noexcept = 0;
  // On return the producer no longer references the slot, even when the
  // returned state is Failed. Implementations must use a safe synchronization
  // fallback rather than hiding an event-query error, and must never return
  // Pending from this blocking operation.
  [[nodiscard]] virtual CompletionState WaitUntilSafe() noexcept = 0;
};

namespace detail {
class PacketRingStorage;
}  // namespace detail

class ReadySlotLease;
class InFlightSlotLease;

class FillingSlotLease final {
 public:
  FillingSlotLease() noexcept = default;
  ~FillingSlotLease();

  FillingSlotLease(const FillingSlotLease&) = delete;
  FillingSlotLease& operator=(const FillingSlotLease&) = delete;
  FillingSlotLease(FillingSlotLease&& other) noexcept;
  FillingSlotLease& operator=(FillingSlotLease&& other) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t slot_index() const;
  [[nodiscard]] std::uint64_t generation() const;
  [[nodiscard]] MutablePacketBytes bytes() const;

  // Publish is the producer's integrity linearization point. The caller must
  // have fully initialized physical_bytes and verified crc32c over exactly
  // logical_bytes before invoking it; consumers cannot observe Filling bytes.
  [[nodiscard]] ReadySlotLease Publish(PacketSlotMetadata metadata) &&;
  void Cancel() noexcept;

 private:
  friend class PacketRing;
  FillingSlotLease(std::shared_ptr<detail::PacketRingStorage> storage,
                   std::size_t slot_index,
                   std::uint64_t generation) noexcept;

  std::shared_ptr<detail::PacketRingStorage> storage_;
  std::size_t slot_index_ = 0;
  std::uint64_t generation_ = 0;
};

class ReadySlotLease final {
 public:
  ReadySlotLease() noexcept = default;
  ~ReadySlotLease();

  ReadySlotLease(const ReadySlotLease&) = delete;
  ReadySlotLease& operator=(const ReadySlotLease&) = delete;
  ReadySlotLease(ReadySlotLease&& other) noexcept;
  ReadySlotLease& operator=(ReadySlotLease&& other) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] ReadyPacketView view() const;
  void Release() noexcept;

  [[nodiscard]] InFlightSlotLease BeginH2d(
      std::unique_ptr<TransferCompletion> completion) &&;

 private:
  friend class FillingSlotLease;
  ReadySlotLease(std::shared_ptr<detail::PacketRingStorage> storage,
                 std::size_t slot_index,
                 std::uint64_t generation) noexcept;

  std::shared_ptr<detail::PacketRingStorage> storage_;
  std::size_t slot_index_ = 0;
  std::uint64_t generation_ = 0;
};

class InFlightSlotLease final {
 public:
  InFlightSlotLease() noexcept = default;
  ~InFlightSlotLease();

  InFlightSlotLease(const InFlightSlotLease&) = delete;
  InFlightSlotLease& operator=(const InFlightSlotLease&) = delete;
  InFlightSlotLease(InFlightSlotLease&& other) noexcept;
  InFlightSlotLease& operator=(InFlightSlotLease&& other) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t slot_index() const;
  [[nodiscard]] std::uint64_t generation() const;

  // Pending retains the in-flight lease. Complete recycles the slot. Failed
  // first executes the completion's safe-wait fallback, then recycles while
  // preserving the failure result for the caller.
  [[nodiscard]] CompletionState PollAndRecycle() noexcept;
  [[nodiscard]] CompletionState WaitAndRecycle() noexcept;

 private:
  friend class ReadySlotLease;
  InFlightSlotLease(std::shared_ptr<detail::PacketRingStorage> storage,
                    std::size_t slot_index,
                    std::uint64_t generation,
                    std::unique_ptr<TransferCompletion> completion) noexcept;

  std::shared_ptr<detail::PacketRingStorage> storage_;
  std::unique_ptr<TransferCompletion> completion_;
  std::size_t slot_index_ = 0;
  std::uint64_t generation_ = 0;
};

class PacketRing final {
 public:
  explicit PacketRing(PacketRingConfig config);
  ~PacketRing() = default;

  PacketRing(const PacketRing&) = delete;
  PacketRing& operator=(const PacketRing&) = delete;
  PacketRing(PacketRing&&) noexcept = default;
  PacketRing& operator=(PacketRing&&) noexcept = default;

  [[nodiscard]] std::optional<FillingSlotLease> TryAcquireFilling();

  [[nodiscard]] std::size_t slot_count() const noexcept;
  [[nodiscard]] std::size_t slot_capacity() const noexcept;
  [[nodiscard]] std::size_t io_alignment() const noexcept;
  [[nodiscard]] std::size_t allocation_alignment() const noexcept;
  [[nodiscard]] std::size_t slot_stride() const noexcept;
  // This stable view exists only so an owning CUDA adapter can register and
  // unregister the complete allocation. It must never be used to bypass slot
  // lease state or to read/write packet contents directly.
  [[nodiscard]] RegistrationStorageView registration_storage() const noexcept;
  [[nodiscard]] std::vector<PacketSlotSnapshot> Snapshot() const;

 private:
  std::shared_ptr<detail::PacketRingStorage> storage_;
};

}  // namespace harpq::native_runtime

#endif  // HARPQ_NATIVE_PACKET_RING_HPP_
