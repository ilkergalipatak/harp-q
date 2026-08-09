#include "harpq/native/packet_ring.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

#include <unistd.h>

namespace harpq::native_runtime {
namespace {

[[nodiscard]] bool IsPowerOfTwo(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1U)) == 0;
}

[[nodiscard]] std::size_t CheckedAlignUp(
    std::size_t value, std::size_t alignment, const char* label) {
  const std::size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const std::size_t increment = alignment - remainder;
  if (value > std::numeric_limits<std::size_t>::max() - increment) {
    throw std::invalid_argument(std::string(label) + " overflows size_t");
  }
  return value + increment;
}

[[nodiscard]] std::size_t CheckedMultiply(
    std::size_t left, std::size_t right, const char* label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::invalid_argument(std::string(label) + " overflows size_t");
  }
  return left * right;
}

[[nodiscard]] std::size_t HostPageSize() {
  errno = 0;
  const long result = sysconf(_SC_PAGESIZE);
  if (result <= 0) {
    throw std::runtime_error(
        std::string("sysconf(_SC_PAGESIZE) failed: ") + std::strerror(errno));
  }
  const auto page_size = static_cast<std::size_t>(result);
  if (!IsPowerOfTwo(page_size)) {
    throw std::runtime_error("host page size is not a power of two");
  }
  return page_size;
}

[[noreturn]] void ThrowInvalidLease(const char* operation) {
  throw std::logic_error(std::string(operation) + " called on an empty packet-slot lease");
}

}  // namespace

namespace detail {

class PacketRingStorage final {
 public:
  explicit PacketRingStorage(PacketRingConfig config) : config_(config) {
    if (config_.slot_count == 0) {
      throw std::invalid_argument("packet ring slot_count must be positive");
    }
    if (config_.slot_capacity == 0) {
      throw std::invalid_argument("packet ring slot_capacity must be positive");
    }
    if (!IsPowerOfTwo(config_.io_alignment) ||
        config_.io_alignment < sizeof(void*)) {
      throw std::invalid_argument(
          "packet ring io_alignment must be a power of two at least sizeof(void*)");
    }
    if (config_.slot_capacity % config_.io_alignment != 0) {
      throw std::invalid_argument(
          "packet ring slot_capacity must be a multiple of io_alignment");
    }

    allocation_alignment_ = std::max(config_.io_alignment, HostPageSize());
    slot_stride_ = CheckedAlignUp(
        config_.slot_capacity, allocation_alignment_, "packet ring slot stride");
    allocation_bytes_ = CheckedMultiply(
        config_.slot_count, slot_stride_, "packet ring allocation");

    void* allocation = nullptr;
    const int status =
        posix_memalign(&allocation, allocation_alignment_, allocation_bytes_);
    if (status != 0) {
      throw std::runtime_error(
          std::string("posix_memalign for packet ring failed: ") +
          std::strerror(status));
    }
    allocation_ = static_cast<std::uint8_t*>(allocation);
    std::memset(allocation_, 0, allocation_bytes_);
    slots_.resize(config_.slot_count);
  }

  ~PacketRingStorage() { std::free(allocation_); }

  PacketRingStorage(const PacketRingStorage&) = delete;
  PacketRingStorage& operator=(const PacketRingStorage&) = delete;

  struct Acquisition {
    std::size_t slot_index;
    std::uint64_t generation;
  };

  [[nodiscard]] std::optional<Acquisition> TryAcquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::size_t attempt = 0; attempt < slots_.size(); ++attempt) {
      const std::size_t index = (next_slot_ + attempt) % slots_.size();
      Slot& slot = slots_[index];
      if (slot.state != PacketSlotState::kFree) {
        continue;
      }
      if (slot.generation == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("packet slot generation counter exhausted");
      }
      ++slot.generation;
      slot.state = PacketSlotState::kFilling;
      slot.metadata = {};
      next_slot_ = (index + 1U) % slots_.size();
      return Acquisition{index, slot.generation};
    }
    return std::nullopt;
  }

  [[nodiscard]] MutablePacketBytes MutableBytes(
      std::size_t slot_index, std::uint64_t generation) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      static_cast<void>(RequireState(
          slot_index, generation, PacketSlotState::kFilling, "bytes"));
    }
    return MutablePacketBytes{
        allocation_ + slot_index * slot_stride_, config_.slot_capacity};
  }

  void Publish(std::size_t slot_index,
               std::uint64_t generation,
               PacketSlotMetadata metadata) {
    if (metadata.packet_id.empty() ||
        metadata.packet_id.find('\0') != std::string::npos) {
      throw std::invalid_argument("ready packet ID must be non-empty and contain no NUL");
    }
    if (metadata.logical_bytes == 0) {
      throw std::invalid_argument("ready packet logical_bytes must be positive");
    }
    if (metadata.physical_bytes < metadata.logical_bytes ||
        metadata.physical_bytes > config_.slot_capacity) {
      throw std::invalid_argument("ready packet physical extent is outside the slot");
    }
    if (metadata.physical_bytes % config_.io_alignment != 0) {
      throw std::invalid_argument(
          "ready packet physical_bytes must be a multiple of io_alignment");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    Slot& slot = RequireState(
        slot_index, generation, PacketSlotState::kFilling, "publish");
    slot.metadata = std::move(metadata);
    slot.state = PacketSlotState::kReady;
  }

  [[nodiscard]] ReadyPacketView ReadyView(
      std::size_t slot_index, std::uint64_t generation) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const Slot& slot = RequireState(
        slot_index, generation, PacketSlotState::kReady, "ready view");
    return ReadyPacketView{
        allocation_ + slot_index * slot_stride_,
        slot.metadata.packet_id,
        slot.metadata.logical_bytes,
        slot.metadata.physical_bytes,
        slot.metadata.crc32c,
        slot_index,
        generation,
    };
  }

  void BeginH2d(std::size_t slot_index, std::uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot& slot = RequireState(
        slot_index, generation, PacketSlotState::kReady, "begin H2D");
    slot.state = PacketSlotState::kH2dInFlight;
  }

  void ReleaseNoThrow(std::size_t slot_index,
                      std::uint64_t generation,
                      PacketSlotState expected) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot_index >= slots_.size()) {
      return;
    }
    Slot& slot = slots_[slot_index];
    if (slot.generation != generation || slot.state != expected) {
      return;
    }
    slot.metadata = {};
    slot.state = PacketSlotState::kFree;
  }

  [[nodiscard]] std::vector<PacketSlotSnapshot> Snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<PacketSlotSnapshot> result;
    result.reserve(slots_.size());
    for (const Slot& slot : slots_) {
      result.push_back(PacketSlotSnapshot{slot.state, slot.generation});
    }
    return result;
  }

  [[nodiscard]] std::size_t slot_count() const noexcept {
    return config_.slot_count;
  }
  [[nodiscard]] std::size_t slot_capacity() const noexcept {
    return config_.slot_capacity;
  }
  [[nodiscard]] std::size_t io_alignment() const noexcept {
    return config_.io_alignment;
  }
  [[nodiscard]] std::size_t allocation_alignment() const noexcept {
    return allocation_alignment_;
  }
  [[nodiscard]] std::size_t slot_stride() const noexcept { return slot_stride_; }
  [[nodiscard]] RegistrationStorageView registration_storage() const noexcept {
    return RegistrationStorageView{
        allocation_, allocation_bytes_, allocation_alignment_};
  }

 private:
  struct Slot {
    PacketSlotState state = PacketSlotState::kFree;
    std::uint64_t generation = 0;
    PacketSlotMetadata metadata{};
  };

  [[nodiscard]] Slot& RequireState(std::size_t slot_index,
                                   std::uint64_t generation,
                                   PacketSlotState expected,
                                   const char* operation) {
    if (slot_index >= slots_.size()) {
      throw std::logic_error(std::string(operation) + " has an invalid slot index");
    }
    Slot& slot = slots_[slot_index];
    if (slot.generation != generation || slot.state != expected) {
      throw std::logic_error(std::string(operation) +
                             " used a stale lease or invalid slot state");
    }
    return slot;
  }

  [[nodiscard]] const Slot& RequireState(std::size_t slot_index,
                                         std::uint64_t generation,
                                         PacketSlotState expected,
                                         const char* operation) const {
    if (slot_index >= slots_.size()) {
      throw std::logic_error(std::string(operation) + " has an invalid slot index");
    }
    const Slot& slot = slots_[slot_index];
    if (slot.generation != generation || slot.state != expected) {
      throw std::logic_error(std::string(operation) +
                             " used a stale lease or invalid slot state");
    }
    return slot;
  }

  PacketRingConfig config_;
  std::size_t allocation_alignment_ = 0;
  std::size_t slot_stride_ = 0;
  std::size_t allocation_bytes_ = 0;
  std::uint8_t* allocation_ = nullptr;
  mutable std::mutex mutex_;
  std::vector<Slot> slots_;
  std::size_t next_slot_ = 0;
};

}  // namespace detail

FillingSlotLease::FillingSlotLease(
    std::shared_ptr<detail::PacketRingStorage> storage,
    std::size_t slot_index,
    std::uint64_t generation) noexcept
    : storage_(std::move(storage)),
      slot_index_(slot_index),
      generation_(generation) {}

FillingSlotLease::~FillingSlotLease() { Cancel(); }

FillingSlotLease::FillingSlotLease(FillingSlotLease&& other) noexcept
    : storage_(std::move(other.storage_)),
      slot_index_(other.slot_index_),
      generation_(other.generation_) {}

FillingSlotLease& FillingSlotLease::operator=(FillingSlotLease&& other) noexcept {
  if (this != &other) {
    Cancel();
    storage_ = std::move(other.storage_);
    slot_index_ = other.slot_index_;
    generation_ = other.generation_;
  }
  return *this;
}

bool FillingSlotLease::valid() const noexcept { return storage_ != nullptr; }

std::size_t FillingSlotLease::slot_index() const {
  if (!valid()) {
    ThrowInvalidLease("slot_index");
  }
  return slot_index_;
}

std::uint64_t FillingSlotLease::generation() const {
  if (!valid()) {
    ThrowInvalidLease("generation");
  }
  return generation_;
}

MutablePacketBytes FillingSlotLease::bytes() const {
  if (!valid()) {
    ThrowInvalidLease("bytes");
  }
  return storage_->MutableBytes(slot_index_, generation_);
}

ReadySlotLease FillingSlotLease::Publish(PacketSlotMetadata metadata) && {
  if (!valid()) {
    ThrowInvalidLease("publish");
  }
  storage_->Publish(slot_index_, generation_, std::move(metadata));
  ReadySlotLease ready(storage_, slot_index_, generation_);
  storage_.reset();
  return ready;
}

void FillingSlotLease::Cancel() noexcept {
  if (storage_ != nullptr) {
    storage_->ReleaseNoThrow(
        slot_index_, generation_, PacketSlotState::kFilling);
    storage_.reset();
  }
}

ReadySlotLease::ReadySlotLease(
    std::shared_ptr<detail::PacketRingStorage> storage,
    std::size_t slot_index,
    std::uint64_t generation) noexcept
    : storage_(std::move(storage)),
      slot_index_(slot_index),
      generation_(generation) {}

ReadySlotLease::~ReadySlotLease() { Release(); }

ReadySlotLease::ReadySlotLease(ReadySlotLease&& other) noexcept
    : storage_(std::move(other.storage_)),
      slot_index_(other.slot_index_),
      generation_(other.generation_) {}

ReadySlotLease& ReadySlotLease::operator=(ReadySlotLease&& other) noexcept {
  if (this != &other) {
    Release();
    storage_ = std::move(other.storage_);
    slot_index_ = other.slot_index_;
    generation_ = other.generation_;
  }
  return *this;
}

bool ReadySlotLease::valid() const noexcept { return storage_ != nullptr; }

ReadyPacketView ReadySlotLease::view() const {
  if (!valid()) {
    ThrowInvalidLease("ready view");
  }
  return storage_->ReadyView(slot_index_, generation_);
}

void ReadySlotLease::Release() noexcept {
  if (storage_ != nullptr) {
    storage_->ReleaseNoThrow(slot_index_, generation_, PacketSlotState::kReady);
    storage_.reset();
  }
}

InFlightSlotLease ReadySlotLease::BeginH2d(
    std::unique_ptr<TransferCompletion> completion) && {
  if (!valid()) {
    ThrowInvalidLease("begin H2D");
  }
  if (completion == nullptr) {
    throw std::invalid_argument("H2D completion cannot be null");
  }
  storage_->BeginH2d(slot_index_, generation_);
  InFlightSlotLease in_flight(
      storage_, slot_index_, generation_, std::move(completion));
  storage_.reset();
  return in_flight;
}

InFlightSlotLease::InFlightSlotLease(
    std::shared_ptr<detail::PacketRingStorage> storage,
    std::size_t slot_index,
    std::uint64_t generation,
    std::unique_ptr<TransferCompletion> completion) noexcept
    : storage_(std::move(storage)),
      completion_(std::move(completion)),
      slot_index_(slot_index),
      generation_(generation) {}

InFlightSlotLease::~InFlightSlotLease() {
  static_cast<void>(WaitAndRecycle());
}

InFlightSlotLease::InFlightSlotLease(InFlightSlotLease&& other) noexcept
    : storage_(std::move(other.storage_)),
      completion_(std::move(other.completion_)),
      slot_index_(other.slot_index_),
      generation_(other.generation_) {}

InFlightSlotLease& InFlightSlotLease::operator=(InFlightSlotLease&& other) noexcept {
  if (this != &other) {
    static_cast<void>(WaitAndRecycle());
    storage_ = std::move(other.storage_);
    completion_ = std::move(other.completion_);
    slot_index_ = other.slot_index_;
    generation_ = other.generation_;
  }
  return *this;
}

bool InFlightSlotLease::valid() const noexcept {
  return storage_ != nullptr && completion_ != nullptr;
}

std::size_t InFlightSlotLease::slot_index() const {
  if (!valid()) {
    ThrowInvalidLease("slot_index");
  }
  return slot_index_;
}

std::uint64_t InFlightSlotLease::generation() const {
  if (!valid()) {
    ThrowInvalidLease("generation");
  }
  return generation_;
}

CompletionState InFlightSlotLease::PollAndRecycle() noexcept {
  if (!valid()) {
    return CompletionState::kComplete;
  }
  CompletionState result = completion_->Poll();
  if (result == CompletionState::kPending) {
    return result;
  }
  if (result != CompletionState::kComplete &&
      result != CompletionState::kFailed) {
    result = CompletionState::kFailed;
  }
  if (result == CompletionState::kFailed) {
    static_cast<void>(completion_->WaitUntilSafe());
  }
  storage_->ReleaseNoThrow(
      slot_index_, generation_, PacketSlotState::kH2dInFlight);
  completion_.reset();
  storage_.reset();
  return result;
}

CompletionState InFlightSlotLease::WaitAndRecycle() noexcept {
  if (!valid()) {
    return CompletionState::kComplete;
  }
  CompletionState result = completion_->WaitUntilSafe();
  if (result == CompletionState::kPending) {
    // The completion violated its blocking contract. The slot is safe by the
    // WaitUntilSafe contract, but surface the protocol failure to the caller.
    result = CompletionState::kFailed;
  }
  storage_->ReleaseNoThrow(
      slot_index_, generation_, PacketSlotState::kH2dInFlight);
  completion_.reset();
  storage_.reset();
  return result;
}

PacketRing::PacketRing(PacketRingConfig config)
    : storage_(std::make_shared<detail::PacketRingStorage>(config)) {}

std::optional<FillingSlotLease> PacketRing::TryAcquireFilling() {
  if (storage_ == nullptr) {
    throw std::logic_error("cannot acquire from a moved-from packet ring");
  }
  const auto acquisition = storage_->TryAcquire();
  if (!acquisition.has_value()) {
    return std::nullopt;
  }
  return FillingSlotLease(
      storage_, acquisition->slot_index, acquisition->generation);
}

std::size_t PacketRing::slot_count() const noexcept {
  return storage_ == nullptr ? 0 : storage_->slot_count();
}

std::size_t PacketRing::slot_capacity() const noexcept {
  return storage_ == nullptr ? 0 : storage_->slot_capacity();
}

std::size_t PacketRing::io_alignment() const noexcept {
  return storage_ == nullptr ? 0 : storage_->io_alignment();
}

std::size_t PacketRing::allocation_alignment() const noexcept {
  return storage_ == nullptr ? 0 : storage_->allocation_alignment();
}

std::size_t PacketRing::slot_stride() const noexcept {
  return storage_ == nullptr ? 0 : storage_->slot_stride();
}

RegistrationStorageView PacketRing::registration_storage() const noexcept {
  return storage_ == nullptr ? RegistrationStorageView{}
                             : storage_->registration_storage();
}

std::vector<PacketSlotSnapshot> PacketRing::Snapshot() const {
  if (storage_ == nullptr) {
    return {};
  }
  return storage_->Snapshot();
}

}  // namespace harpq::native_runtime
