#include "harpq/cuda_direct_packet_io.hpp"

#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace harpq::native_runtime {
namespace {
using Clock = std::chrono::steady_clock;

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

class DeviceScope final {
 public:
  explicit DeviceScope(int device) {
    CheckCuda(cudaGetDevice(&previous_), "get CUDA device");
    if (previous_ != device) {
      CheckCuda(cudaSetDevice(device), "select packet CUDA device");
      changed_ = true;
    }
  }
  ~DeviceScope() {
    if (changed_ && cudaSetDevice(previous_) != cudaSuccess) {
      std::abort();
    }
  }
 private:
  int previous_ = 0;
  bool changed_ = false;
};

class DeviceAllocation final {
 public:
  DeviceAllocation(int device, std::size_t bytes) : device_(device) {
    DeviceScope scope(device_);
    // shortcut: per-packet cudaMalloc may serialize, replace when adding the GPU cache.
    CheckCuda(cudaMalloc(&data_, bytes), "allocate packet destination");
  }
  ~DeviceAllocation() noexcept {
    try {
      DeviceScope scope(device_);
      if (cudaFree(data_) != cudaSuccess) {
        std::abort();
      }
    } catch (...) {
      std::abort();
    }
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  void* data() const noexcept { return data_; }
 private:
  int device_;
  void* data_ = nullptr;
};

double Microseconds(Clock::time_point begin, Clock::time_point end) noexcept {
  return std::chrono::duration<double, std::micro>(end - begin).count();
}
}  // namespace

namespace detail {
class CudaPacketTransferState final {
 public:
  CudaPacketTransferState(CudaReadySlotLease& ready, const DirectPacketReadStats& stats,
                          Clock::time_point submitted_at, Clock::time_point ready_at)
      : observation_{ready.view().packet_id, stats.logical_bytes, stats.file_bytes_returned,
                     0, Microseconds(submitted_at, ready_at), 0},
        submitted_at_(submitted_at),
        destination_(ready.device_ordinal(), stats.logical_bytes) {
    h2d_started_at_ = Clock::now();
    in_flight_ = std::move(ready).EnqueueH2d(destination_.data(), stats.logical_bytes);
  }

  CompletionState Complete(bool wait) noexcept {
    if (status_ != CompletionState::kPending) {
      return status_;
    }
    status_ = wait ? in_flight_.WaitAndRecycle() : in_flight_.PollAndRecycle();
    if (status_ == CompletionState::kComplete) {
      const auto completed_at = Clock::now();
      observation_.critical_path_us = Microseconds(submitted_at_, completed_at);
      observation_.h2d_service_us = Microseconds(h2d_started_at_, completed_at);
    }
    return status_;
  }

  PacketServiceObservation observation_;
  Clock::time_point submitted_at_;
  Clock::time_point h2d_started_at_{};
  // in_flight_ is destroyed first and proves DMA completion before cudaFree.
  DeviceAllocation destination_;
  CudaInFlightSlotLease in_flight_;
  CompletionState status_ = CompletionState::kPending;
};
}  // namespace detail

CudaPacketTransfer::CudaPacketTransfer() noexcept = default;
CudaPacketTransfer::~CudaPacketTransfer() = default;
CudaPacketTransfer::CudaPacketTransfer(CudaPacketTransfer&&) noexcept = default;
CudaPacketTransfer& CudaPacketTransfer::operator=(CudaPacketTransfer&&) noexcept = default;
CudaPacketTransfer::CudaPacketTransfer(
    std::unique_ptr<detail::CudaPacketTransferState> state) noexcept : state_(std::move(state)) {}

CompletionState CudaPacketTransfer::Poll() noexcept {
  return state_ ? state_->Complete(false) : CompletionState::kFailed;
}
CompletionState CudaPacketTransfer::Wait() noexcept {
  return state_ ? state_->Complete(true) : CompletionState::kFailed;
}
cudaError_t CudaPacketTransfer::MakeStreamWait(cudaStream_t stream) noexcept {
  return state_ ? state_->in_flight_.MakeStreamWait(stream) : cudaErrorInvalidResourceHandle;
}
const void* CudaPacketTransfer::device_data() const noexcept {
  return state_ ? state_->destination_.data() : nullptr;
}
std::size_t CudaPacketTransfer::size_bytes() const noexcept {
  return state_ ? state_->observation_.logical_bytes : 0;
}
const PacketServiceObservation* CudaPacketTransfer::observation() const noexcept {
  return state_ && state_->status_ == CompletionState::kComplete ? &state_->observation_ : nullptr;
}

CudaDirectReadyPacket::CudaDirectReadyPacket(
    CudaReadySlotLease ready, DirectPacketReadStats stats, Clock::time_point submitted_at) noexcept
    : ready_(std::move(ready)), stats_(stats), submitted_at_(submitted_at), ready_at_(Clock::now()) {}

CudaPacketTransfer CudaDirectReadyPacket::EnqueueH2d() && {
  return CudaPacketTransfer(std::make_unique<detail::CudaPacketTransferState>(
      ready_, stats_, submitted_at_, ready_at_));
}

CudaDirectPackFile::CudaDirectPackFile(CudaPacketRing& ring, std::string path,
                                       std::uint64_t declared_pack_size, DirectPacketIoConfig config)
    : ring_(ring.control_),
      source_(std::move(path), declared_pack_size, ring_.DirectIoCore(), config) {}

DirectIoCapability CudaDirectPackFile::direct_capability() const noexcept {
  return source_.direct_capability();
}
std::size_t CudaDirectPackFile::in_flight() const noexcept { return source_.in_flight(); }

CudaDirectPacketLoadResult CudaDirectPackFile::Wrap(DirectPacketLoadResult core) {
  CudaDirectPacketLoadResult result;
  result.status = core.status;
  result.operation_id = core.operation_id;
  result.stats = core.stats;
  result.failure = std::move(core.failure);
  if (core.ready) {
    if (!core.submitted_at) {
      throw std::logic_error("direct ready packet has no submission timestamp");
    }
    CudaDirectReadyPacket ready(
        CudaReadySlotLease(ring_.control_, std::move(*core.ready)), core.stats, *core.submitted_at);
    result.ready.emplace(std::move(ready));
  }
  return result;
}
CudaDirectPacketLoadResult CudaDirectPackFile::Submit(
    const PacketReadRequest& request, const CancellationToken& cancellation) {
  return Wrap(source_.Submit(request, cancellation));
}
std::optional<CudaDirectPacketLoadResult> CudaDirectPackFile::Poll() {
  auto core = source_.Poll();
  if (!core) {
    return std::nullopt;
  }
  return Wrap(std::move(*core));
}
CudaDirectPacketLoadResult CudaDirectPackFile::Wait() { return Wrap(source_.Wait()); }

}  // namespace harpq::native_runtime
