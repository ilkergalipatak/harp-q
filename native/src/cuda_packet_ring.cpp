#include "harpq/cuda_packet_ring.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace harpq::native_runtime {
namespace {

constexpr std::uint32_t kFaultHostRegister = 1U << 0U;
constexpr std::uint32_t kFaultEventRecord = 1U << 1U;
constexpr std::uint32_t kFaultEventQuery = 1U << 2U;
constexpr std::uint32_t kFaultEventSynchronize = 1U << 3U;
constexpr std::uint32_t kFaultStreamSynchronize = 1U << 4U;
constexpr std::uint32_t kFaultDeviceSynchronize = 1U << 5U;
constexpr std::uint32_t kFaultHostUnregister = 1U << 6U;

std::atomic<std::uint32_t> g_test_fault_mask{0};

[[nodiscard]] bool ConsumeTestFault(std::uint32_t fault) noexcept {
  std::uint32_t observed = g_test_fault_mask.load(std::memory_order_relaxed);
  while ((observed & fault) != 0U) {
    if (g_test_fault_mask.compare_exchange_weak(
            observed,
            observed & ~fault,
            std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] cudaError_t PacketCudaHostRegister(
    void* pointer, std::size_t bytes, unsigned int flags) noexcept {
  if (ConsumeTestFault(kFaultHostRegister)) {
    return cudaErrorMemoryAllocation;
  }
  return cudaHostRegister(pointer, bytes, flags);
}

[[nodiscard]] cudaError_t PacketCudaHostUnregister(void* pointer) noexcept {
  if (ConsumeTestFault(kFaultHostUnregister)) {
    return cudaErrorUnknown;
  }
  return cudaHostUnregister(pointer);
}

[[nodiscard]] cudaError_t PacketCudaEventRecord(
    cudaEvent_t event, cudaStream_t stream) noexcept {
  if (ConsumeTestFault(kFaultEventRecord)) {
    return cudaErrorUnknown;
  }
  return cudaEventRecord(event, stream);
}

[[nodiscard]] cudaError_t PacketCudaEventQuery(cudaEvent_t event) noexcept {
  if (ConsumeTestFault(kFaultEventQuery)) {
    return cudaErrorUnknown;
  }
  return cudaEventQuery(event);
}

[[nodiscard]] cudaError_t PacketCudaEventSynchronize(
    cudaEvent_t event) noexcept {
  if (ConsumeTestFault(kFaultEventSynchronize)) {
    return cudaErrorUnknown;
  }
  return cudaEventSynchronize(event);
}

[[nodiscard]] cudaError_t PacketCudaStreamSynchronize(
    cudaStream_t stream) noexcept {
  if (ConsumeTestFault(kFaultStreamSynchronize)) {
    return cudaErrorUnknown;
  }
  return cudaStreamSynchronize(stream);
}

[[nodiscard]] cudaError_t PacketCudaDeviceSynchronize() noexcept {
  if (ConsumeTestFault(kFaultDeviceSynchronize)) {
    return cudaErrorUnknown;
  }
  return cudaDeviceSynchronize();
}

[[noreturn]] void TerminateForUnprovenCudaSafety() noexcept {
  // Continuing could recycle or free registered storage still referenced by
  // DMA. A process-fatal policy is intentionally safer than a use-after-free.
  std::abort();
}

[[noreturn]] void ThrowCudaError(cudaError_t status, const char* operation) {
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorName(status) << " ("
          << cudaGetErrorString(status) << ')';
  throw std::runtime_error(message.str());
}

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    ThrowCudaError(status, operation);
  }
}

class ScopedCudaDevice final {
 public:
  explicit ScopedCudaDevice(int target_device) noexcept {
    status_ = cudaGetDevice(&previous_device_);
    if (status_ != cudaSuccess) {
      return;
    }
    if (previous_device_ == target_device) {
      return;
    }
    status_ = cudaSetDevice(target_device);
    restore_ = status_ == cudaSuccess;
  }

  ~ScopedCudaDevice() {
    if (restore_) {
      static_cast<void>(cudaSetDevice(previous_device_));
    }
  }

  ScopedCudaDevice(const ScopedCudaDevice&) = delete;
  ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

  [[nodiscard]] cudaError_t status() const noexcept { return status_; }

 private:
  int previous_device_ = 0;
  cudaError_t status_ = cudaSuccess;
  bool restore_ = false;
};

}  // namespace

namespace detail {

class CudaPacketRingControl final {
 public:
  explicit CudaPacketRingControl(PacketRingConfig config)
      : ring(std::move(config)) {}

  ~CudaPacketRingControl() noexcept {
    ScopedCudaDevice device_scope(device_ordinal);
    const bool owns_cuda_resources = registered || copy_stream != nullptr ||
                                     !completion_events.empty();
    if (owns_cuda_resources && device_scope.status() != cudaSuccess) {
      TerminateForUnprovenCudaSafety();
    }
    if (copy_stream != nullptr && !SynchronizeForSafety().proven_safe) {
      TerminateForUnprovenCudaSafety();
    }
    for (cudaEvent_t event : completion_events) {
      if (event != nullptr && cudaEventDestroy(event) != cudaSuccess) {
        TerminateForUnprovenCudaSafety();
      }
    }
    if (copy_stream != nullptr &&
        cudaStreamDestroy(copy_stream) != cudaSuccess) {
      TerminateForUnprovenCudaSafety();
    }
    if (registered &&
        PacketCudaHostUnregister(registration.data) != cudaSuccess) {
      TerminateForUnprovenCudaSafety();
    }
    // ring is destroyed after this destructor body, so its backing allocation
    // remains alive through cudaHostUnregister even during partial setup.
  }

  CudaPacketRingControl(const CudaPacketRingControl&) = delete;
  CudaPacketRingControl& operator=(const CudaPacketRingControl&) = delete;

  struct SafetySynchronization {
    bool proven_safe = false;
    cudaError_t stream_status = cudaSuccess;
    cudaError_t device_status = cudaSuccess;
  };

  [[nodiscard]] SafetySynchronization SynchronizeForSafety() noexcept {
    if (copy_stream == nullptr) {
      return SafetySynchronization{true, cudaSuccess, cudaSuccess};
    }
    ScopedCudaDevice device_scope(device_ordinal);
    if (device_scope.status() != cudaSuccess) {
      return SafetySynchronization{
          false, device_scope.status(), device_scope.status()};
    }
    const cudaError_t stream_status =
        PacketCudaStreamSynchronize(copy_stream);
    if (stream_status == cudaSuccess) {
      return SafetySynchronization{true, cudaSuccess, cudaSuccess};
    }
    const cudaError_t device_status = PacketCudaDeviceSynchronize();
    return SafetySynchronization{
        device_status == cudaSuccess, stream_status, device_status};
  }

  PacketRing ring;
  RegistrationStorageView registration{};
  int device_ordinal = 0;
  cudaStream_t copy_stream = nullptr;
  std::vector<cudaEvent_t> completion_events;
  bool registered = false;
};

}  // namespace detail

namespace {

class CudaEventCompletion final : public TransferCompletion {
 public:
  CudaEventCompletion(
      std::shared_ptr<detail::CudaPacketRingControl> control,
      cudaEvent_t event) noexcept
      : control_(std::move(control)), event_(event) {}

  ~CudaEventCompletion() override {
    if (armed_ && !safe_) {
      static_cast<void>(WaitUntilSafe());
    }
  }

  CudaEventCompletion(const CudaEventCompletion&) = delete;
  CudaEventCompletion& operator=(const CudaEventCompletion&) = delete;

  void Arm() noexcept { armed_ = true; }

  [[nodiscard]] CompletionState Poll() noexcept override {
    if (!armed_) {
      failed_ = true;
      safe_ = true;
      return CompletionState::kFailed;
    }
    if (safe_) {
      return failed_ ? CompletionState::kFailed : CompletionState::kComplete;
    }

    ScopedCudaDevice device_scope(control_->device_ordinal);
    if (device_scope.status() != cudaSuccess) {
      failed_ = true;
      static_cast<void>(WaitUntilSafe());
      return CompletionState::kFailed;
    }
    const cudaError_t status = PacketCudaEventQuery(event_);
    if (status == cudaSuccess) {
      safe_ = true;
      return failed_ ? CompletionState::kFailed : CompletionState::kComplete;
    }
    if (status == cudaErrorNotReady) {
      return CompletionState::kPending;
    }

    failed_ = true;
    static_cast<void>(WaitUntilSafe());
    return CompletionState::kFailed;
  }

  [[nodiscard]] CompletionState WaitUntilSafe() noexcept override {
    if (!armed_) {
      failed_ = true;
      safe_ = true;
      return CompletionState::kFailed;
    }
    if (safe_) {
      return failed_ ? CompletionState::kFailed : CompletionState::kComplete;
    }

    ScopedCudaDevice device_scope(control_->device_ordinal);
    if (device_scope.status() != cudaSuccess) {
      failed_ = true;
      if (!control_->SynchronizeForSafety().proven_safe) {
        TerminateForUnprovenCudaSafety();
      }
      safe_ = true;
      return CompletionState::kFailed;
    }
    const cudaError_t status = PacketCudaEventSynchronize(event_);
    if (status != cudaSuccess) {
      failed_ = true;
      if (!control_->SynchronizeForSafety().proven_safe) {
        TerminateForUnprovenCudaSafety();
      }
    }
    safe_ = true;
    return failed_ ? CompletionState::kFailed : CompletionState::kComplete;
  }

 private:
  std::shared_ptr<detail::CudaPacketRingControl> control_;
  cudaEvent_t event_ = nullptr;
  bool armed_ = false;
  bool safe_ = false;
  bool failed_ = false;
};

[[nodiscard]] std::shared_ptr<detail::CudaPacketRingControl> CreateControl(
    PacketRingConfig config) {
  auto control =
      std::make_shared<detail::CudaPacketRingControl>(std::move(config));

  int device_count = 0;
  CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count <= 0) {
    throw std::runtime_error("CUDA packet ring requires a visible CUDA device");
  }
  CheckCuda(cudaGetDevice(&control->device_ordinal), "cudaGetDevice");

  control->registration = control->ring.registration_storage();
  if (control->registration.data == nullptr ||
      control->registration.bytes == 0 ||
      control->registration.alignment == 0) {
    throw std::logic_error("packet ring returned an invalid registration view");
  }
  if (reinterpret_cast<std::uintptr_t>(control->registration.data) %
          control->registration.alignment !=
      0) {
    throw std::logic_error("packet ring registration view is not aligned");
  }

  const cudaError_t registration_status = PacketCudaHostRegister(
      control->registration.data,
      control->registration.bytes,
      cudaHostRegisterDefault);
  if (registration_status != cudaSuccess) {
    std::ostringstream message;
    message << "cudaHostRegister for the complete packet-ring backing failed; "
               "pageable fallback is intentionally disabled: "
            << cudaGetErrorName(registration_status) << " ("
            << cudaGetErrorString(registration_status) << "), bytes="
            << control->registration.bytes;
    throw std::runtime_error(message.str());
  }
  control->registered = true;

  CheckCuda(cudaStreamCreateWithFlags(
                &control->copy_stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags(packet-ring copy stream)");

  control->completion_events.assign(control->ring.slot_count(), nullptr);
  for (std::size_t index = 0; index < control->completion_events.size(); ++index) {
    CheckCuda(cudaEventCreateWithFlags(
                  &control->completion_events[index], cudaEventDisableTiming),
              "cudaEventCreateWithFlags(packet-ring completion)");
  }
  return control;
}

[[noreturn]] void ThrowInvalidCudaLease(const char* operation) {
  throw std::logic_error(
      std::string(operation) + " called on an empty CUDA packet-slot lease");
}

}  // namespace

CudaFillingSlotLease::CudaFillingSlotLease(
    std::shared_ptr<detail::CudaPacketRingControl> control,
    FillingSlotLease lease) noexcept
    : control_(std::move(control)), lease_(std::move(lease)) {}

CudaFillingSlotLease::CudaFillingSlotLease(
    CudaFillingSlotLease&& other) noexcept
    : control_(std::move(other.control_)), lease_(std::move(other.lease_)) {}

CudaFillingSlotLease& CudaFillingSlotLease::operator=(
    CudaFillingSlotLease&& other) noexcept {
  if (this != &other) {
    lease_.Cancel();
    control_.reset();
    control_ = std::move(other.control_);
    lease_ = std::move(other.lease_);
  }
  return *this;
}

bool CudaFillingSlotLease::valid() const noexcept {
  return control_ != nullptr && lease_.valid();
}

std::size_t CudaFillingSlotLease::slot_index() const {
  if (!valid()) {
    ThrowInvalidCudaLease("slot_index");
  }
  return lease_.slot_index();
}

std::uint64_t CudaFillingSlotLease::generation() const {
  if (!valid()) {
    ThrowInvalidCudaLease("generation");
  }
  return lease_.generation();
}

MutablePacketBytes CudaFillingSlotLease::bytes() const {
  if (!valid()) {
    ThrowInvalidCudaLease("bytes");
  }
  return lease_.bytes();
}

CudaReadySlotLease CudaFillingSlotLease::Publish(
    PacketSlotMetadata metadata) && {
  if (!valid()) {
    ThrowInvalidCudaLease("publish");
  }
  auto control = control_;
  ReadySlotLease ready = std::move(lease_).Publish(std::move(metadata));
  control_.reset();
  return CudaReadySlotLease(std::move(control), std::move(ready));
}

void CudaFillingSlotLease::Cancel() noexcept {
  lease_.Cancel();
  control_.reset();
}

CudaReadySlotLease::CudaReadySlotLease(
    std::shared_ptr<detail::CudaPacketRingControl> control,
    ReadySlotLease lease) noexcept
    : control_(std::move(control)), lease_(std::move(lease)) {}

CudaReadySlotLease::CudaReadySlotLease(CudaReadySlotLease&& other) noexcept
    : control_(std::move(other.control_)), lease_(std::move(other.lease_)) {}

CudaReadySlotLease& CudaReadySlotLease::operator=(
    CudaReadySlotLease&& other) noexcept {
  if (this != &other) {
    lease_.Release();
    control_.reset();
    control_ = std::move(other.control_);
    lease_ = std::move(other.lease_);
  }
  return *this;
}

bool CudaReadySlotLease::valid() const noexcept {
  return control_ != nullptr && lease_.valid();
}

ReadyPacketView CudaReadySlotLease::view() const {
  if (!valid()) {
    ThrowInvalidCudaLease("ready view");
  }
  return lease_.view();
}

int CudaReadySlotLease::device_ordinal() const {
  if (!valid()) {
    ThrowInvalidCudaLease("device_ordinal");
  }
  return control_->device_ordinal;
}

void CudaReadySlotLease::Release() noexcept {
  lease_.Release();
  control_.reset();
}

CudaInFlightSlotLease::CudaInFlightSlotLease(
    std::shared_ptr<detail::CudaPacketRingControl> control,
    InFlightSlotLease lease,
    cudaEvent_t completion_event) noexcept
    : control_(std::move(control)),
      lease_(std::move(lease)),
      completion_event_(completion_event) {}

CudaInFlightSlotLease::CudaInFlightSlotLease(
    CudaInFlightSlotLease&& other) noexcept
    : control_(std::move(other.control_)),
      lease_(std::move(other.lease_)),
      completion_event_(other.completion_event_) {
  other.completion_event_ = nullptr;
}

CudaInFlightSlotLease& CudaInFlightSlotLease::operator=(
    CudaInFlightSlotLease&& other) noexcept {
  if (this != &other) {
    static_cast<void>(lease_.WaitAndRecycle());
    control_.reset();
    control_ = std::move(other.control_);
    lease_ = std::move(other.lease_);
    completion_event_ = other.completion_event_;
    other.completion_event_ = nullptr;
  }
  return *this;
}

bool CudaInFlightSlotLease::valid() const noexcept {
  return control_ != nullptr && lease_.valid();
}

std::size_t CudaInFlightSlotLease::slot_index() const {
  if (!valid()) {
    ThrowInvalidCudaLease("slot_index");
  }
  return lease_.slot_index();
}

std::uint64_t CudaInFlightSlotLease::generation() const {
  if (!valid()) {
    ThrowInvalidCudaLease("generation");
  }
  return lease_.generation();
}

CompletionState CudaInFlightSlotLease::PollAndRecycle() noexcept {
  const CompletionState result = lease_.PollAndRecycle();
  if (!lease_.valid()) {
    control_.reset();
    completion_event_ = nullptr;
  }
  return result;
}

CompletionState CudaInFlightSlotLease::WaitAndRecycle() noexcept {
  const CompletionState result = lease_.WaitAndRecycle();
  if (!lease_.valid()) {
    control_.reset();
    completion_event_ = nullptr;
  }
  return result;
}

cudaError_t CudaInFlightSlotLease::MakeStreamWait(
    cudaStream_t compute_stream) noexcept {
  if (!valid() || completion_event_ == nullptr) {
    return cudaErrorInvalidResourceHandle;
  }
  ScopedCudaDevice device_scope(control_->device_ordinal);
  if (device_scope.status() != cudaSuccess) {
    return device_scope.status();
  }
  return cudaStreamWaitEvent(compute_stream, completion_event_, 0);
}

CudaPacketRing::CudaPacketRing(PacketRingConfig config)
    : control_(CreateControl(std::move(config))) {}

CudaPacketRing::CudaPacketRing(
    std::shared_ptr<detail::CudaPacketRingControl> control) noexcept
    : control_(std::move(control)) {}

PacketRing& CudaPacketRing::DirectIoCore() {
  if (control_ == nullptr) {
    throw std::logic_error("cannot bind a moved-from CUDA packet ring");
  }
  return control_->ring;
}

std::optional<CudaFillingSlotLease> CudaPacketRing::TryAcquireFilling() {
  if (control_ == nullptr) {
    throw std::logic_error("cannot acquire from a moved-from CUDA packet ring");
  }
  auto lease = control_->ring.TryAcquireFilling();
  if (!lease.has_value()) {
    return std::nullopt;
  }
  return CudaFillingSlotLease(control_, std::move(*lease));
}

CudaInFlightSlotLease CudaPacketRing::EnqueueH2d(
    CudaReadySlotLease&& ready,
    void* device_destination,
    std::size_t destination_capacity_bytes) {
  if (control_ == nullptr) {
    throw std::logic_error("cannot enqueue on a moved-from CUDA packet ring");
  }
  if (!ready.valid()) {
    ThrowInvalidCudaLease("enqueue H2D");
  }
  if (ready.control_.get() != control_.get()) {
    throw std::invalid_argument(
        "CUDA ready lease belongs to a different packet ring");
  }
  return std::move(ready).EnqueueH2d(device_destination, destination_capacity_bytes);
}

CudaInFlightSlotLease CudaReadySlotLease::EnqueueH2d(
    void* device_destination, std::size_t destination_capacity_bytes) && {
  if (!valid()) {
    ThrowInvalidCudaLease("enqueue H2D");
  }
  const ReadyPacketView view = lease_.view();
  if (device_destination == nullptr) {
    throw std::invalid_argument("H2D device destination cannot be null");
  }
  if (destination_capacity_bytes < view.logical_bytes) {
    throw std::invalid_argument(
        "H2D destination capacity is smaller than packet logical bytes");
  }
  if (view.slot_index >= control_->completion_events.size()) {
    throw std::logic_error("ready packet has an invalid CUDA event slot");
  }

  auto completion = std::make_unique<CudaEventCompletion>(
      control_, control_->completion_events[view.slot_index]);

  ScopedCudaDevice device_scope(control_->device_ordinal);
  CheckCuda(device_scope.status(), "select packet-ring CUDA device");

  const cudaError_t copy_status = cudaMemcpyAsync(
      device_destination,
      view.data,
      view.logical_bytes,
      cudaMemcpyHostToDevice,
      control_->copy_stream);
  if (copy_status != cudaSuccess) {
    if (!control_->SynchronizeForSafety().proven_safe) {
      TerminateForUnprovenCudaSafety();
    }
    ThrowCudaError(copy_status, "cudaMemcpyAsync(packet-ring H2D)");
  }

  const cudaError_t event_status = PacketCudaEventRecord(
      control_->completion_events[view.slot_index], control_->copy_stream);
  if (event_status != cudaSuccess) {
    if (!control_->SynchronizeForSafety().proven_safe) {
      TerminateForUnprovenCudaSafety();
    }
    ThrowCudaError(event_status, "cudaEventRecord(packet-ring H2D completion)");
  }
  completion->Arm();

  InFlightSlotLease in_flight =
      std::move(lease_).BeginH2d(std::move(completion));
  const cudaEvent_t event = control_->completion_events[view.slot_index];
  auto retained_control = std::move(control_);
  return CudaInFlightSlotLease(
      std::move(retained_control),
      std::move(in_flight),
      event);
}

CudaPacketLoadResult CudaPacketRing::Load(
    PreadPackFile& source,
    const PacketReadRequest& request,
    const CancellationToken& cancellation) {
  if (control_ == nullptr) {
    throw std::logic_error("cannot load into a moved-from CUDA packet ring");
  }
  PacketLoadResult core_result =
      source.Load(control_->ring, request, cancellation);

  CudaPacketLoadResult result;
  result.status = core_result.status;
  result.stats = core_result.stats;
  result.failure = std::move(core_result.failure);
  if (core_result.ready.has_value()) {
    CudaReadySlotLease wrapped(control_, std::move(*core_result.ready));
    result.ready.emplace(std::move(wrapped));
  }
  return result;
}

std::size_t CudaPacketRing::slot_count() const noexcept {
  return control_ == nullptr ? 0 : control_->ring.slot_count();
}

std::size_t CudaPacketRing::slot_capacity() const noexcept {
  return control_ == nullptr ? 0 : control_->ring.slot_capacity();
}

std::size_t CudaPacketRing::io_alignment() const noexcept {
  return control_ == nullptr ? 0 : control_->ring.io_alignment();
}

std::size_t CudaPacketRing::allocation_alignment() const noexcept {
  return control_ == nullptr ? 0 : control_->ring.allocation_alignment();
}

std::size_t CudaPacketRing::slot_stride() const noexcept {
  return control_ == nullptr ? 0 : control_->ring.slot_stride();
}

std::vector<PacketSlotSnapshot> CudaPacketRing::Snapshot() const {
  return control_ == nullptr ? std::vector<PacketSlotSnapshot>{}
                             : control_->ring.Snapshot();
}

cudaStream_t CudaPacketRing::CopyStreamForTest() const noexcept {
  return control_ == nullptr ? nullptr : control_->copy_stream;
}

void CudaPacketRing::InjectFaultsForTest(
    std::uint32_t fault_mask) noexcept {
  g_test_fault_mask.store(fault_mask, std::memory_order_relaxed);
}

}  // namespace harpq::native_runtime
