#include "harpq/cuda_packet_ring.hpp"

#include "harpq/native/stage_codec.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace harpq::native_runtime {

class CudaPacketRingTestAccess final {
 public:
  [[nodiscard]] static cudaStream_t CopyStream(
      const CudaPacketRing& ring) noexcept {
    return ring.CopyStreamForTest();
  }

  static void FailHostRegisterOnce() noexcept {
    Inject(CudaPacketRing::TestFault::kHostRegister);
  }
  static void FailEventRecordOnce() noexcept {
    Inject(CudaPacketRing::TestFault::kEventRecord);
  }
  static void FailEventQueryOnce() noexcept {
    Inject(CudaPacketRing::TestFault::kEventQuery);
  }
  static void FailEventSynchronizeOnce() noexcept {
    Inject(CudaPacketRing::TestFault::kEventSynchronize);
  }
  static void FailEventAndStreamSynchronizeOnce() noexcept {
    CudaPacketRing::InjectFaultsForTest(
        Mask(CudaPacketRing::TestFault::kEventSynchronize) |
        Mask(CudaPacketRing::TestFault::kStreamSynchronize));
  }
  static void FailAllSafetySynchronizationsOnce() noexcept {
    CudaPacketRing::InjectFaultsForTest(
        Mask(CudaPacketRing::TestFault::kEventSynchronize) |
        Mask(CudaPacketRing::TestFault::kStreamSynchronize) |
        Mask(CudaPacketRing::TestFault::kDeviceSynchronize));
  }
  static void FailStreamSynchronizeOnce() noexcept {
    Inject(CudaPacketRing::TestFault::kStreamSynchronize);
  }
  static void FailHostUnregisterOnce() noexcept {
    Inject(CudaPacketRing::TestFault::kHostUnregister);
  }

 private:
  [[nodiscard]] static constexpr std::uint32_t Mask(
      CudaPacketRing::TestFault fault) noexcept {
    return static_cast<std::uint32_t>(fault);
  }

  static void Inject(CudaPacketRing::TestFault fault) noexcept {
    CudaPacketRing::InjectFaultsForTest(Mask(fault));
  }
};

}  // namespace harpq::native_runtime

namespace {

using harpq::native_runtime::CompletionState;
using harpq::native_runtime::CudaFillingSlotLease;
using harpq::native_runtime::CudaInFlightSlotLease;
using harpq::native_runtime::CudaPacketRing;
using harpq::native_runtime::CudaPacketRingTestAccess;
using harpq::native_runtime::CudaReadySlotLease;
using harpq::native_runtime::Crc32c;
using harpq::native_runtime::PacketLoadStatus;
using harpq::native_runtime::PacketReadRequest;
using harpq::native_runtime::PacketRequirement;
using harpq::native_runtime::PacketRingConfig;
using harpq::native_runtime::PacketSlotMetadata;
using harpq::native_runtime::PacketSlotState;
using harpq::native_runtime::PreadPackFile;

constexpr std::size_t kAlignment = 4096;
constexpr std::size_t kSlotCapacity = 2 * kAlignment;
constexpr std::size_t kLogicalBytes = kAlignment;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void RequireCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorName(status) +
        " (" + cudaGetErrorString(status) + ')');
  }
}

template <typename Callable>
void RequireInvalidArgument(Callable&& callable, const std::string& label) {
  try {
    callable();
  } catch (const std::invalid_argument&) {
    return;
  } catch (const std::exception& error) {
    throw std::runtime_error(
        label + " raised the wrong exception: " + error.what());
  }
  throw std::runtime_error(label + " did not reject invalid input");
}

class DeviceBytes final {
 public:
  explicit DeviceBytes(std::size_t bytes) : bytes_(bytes) {
    RequireCuda(cudaMalloc(&data_, bytes_), "cudaMalloc(test destination)");
  }

  ~DeviceBytes() {
    if (data_ != nullptr) {
      static_cast<void>(cudaFree(data_));
    }
  }

  DeviceBytes(const DeviceBytes&) = delete;
  DeviceBytes& operator=(const DeviceBytes&) = delete;

  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0;
};

std::uint8_t PatternByte(std::size_t index, std::uint8_t salt = 0) noexcept {
  return static_cast<std::uint8_t>(
      (static_cast<std::uint64_t>(index) * 131ULL +
       (static_cast<std::uint64_t>(index) >> 5U) * 17ULL + 0x5AULL + salt) &
      0xFFU);
}

void FillPattern(CudaFillingSlotLease& filling, std::uint8_t salt = 0) {
  const auto bytes = filling.bytes();
  Require(bytes.data != nullptr, "filling lease returned a null byte view");
  Require(bytes.capacity == kSlotCapacity,
          "filling lease returned the wrong slot capacity");
  for (std::size_t index = 0; index < kLogicalBytes; ++index) {
    bytes.data[index] = PatternByte(index, salt);
  }
  std::fill(bytes.data + kLogicalBytes, bytes.data + bytes.capacity, 0);
}

CudaReadySlotLease PublishPattern(CudaPacketRing& ring,
                                  const std::string& packet_id,
                                  std::uint8_t salt = 0) {
  auto acquired = ring.TryAcquireFilling();
  Require(acquired.has_value(), "expected a free CUDA packet-ring slot");
  FillPattern(*acquired, salt);
  return std::move(*acquired).Publish(PacketSlotMetadata{
      packet_id, kLogicalBytes, kLogicalBytes, 0x12345678U});
}

void RequireRegisteredHostPointer(const void* pointer) {
  cudaPointerAttributes attributes{};
  RequireCuda(cudaPointerGetAttributes(&attributes, pointer),
              "cudaPointerGetAttributes(registered host slot)");
#if CUDART_VERSION >= 10000
  Require(attributes.type == cudaMemoryTypeHost,
          "packet-ring slot is not reported as registered host memory");
#else
  Require(attributes.memoryType == cudaMemoryTypeHost,
          "packet-ring slot is not reported as registered host memory");
#endif
}

std::vector<std::uint8_t> CopyBack(const DeviceBytes& device,
                                   std::size_t bytes) {
  std::vector<std::uint8_t> result(bytes);
  RequireCuda(cudaMemcpy(result.data(), device.data(), bytes,
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(test D2H)");
  return result;
}

void RequirePattern(const std::vector<std::uint8_t>& bytes,
                    std::uint8_t salt = 0) {
  Require(bytes.size() == kLogicalBytes, "D2H result has the wrong size");
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    if (bytes[index] != PatternByte(index, salt)) {
      throw std::runtime_error(
          "D2H result differs from the registered host pattern at byte " +
          std::to_string(index));
    }
  }
}

void TestAlignmentRegistrationCopyAndGeneration() {
  CudaPacketRing ring(PacketRingConfig{2, kSlotCapacity, kAlignment});
  Require(ring.slot_count() == 2, "CUDA ring reported the wrong slot count");
  Require(ring.slot_capacity() == kSlotCapacity,
          "CUDA ring reported the wrong capacity");
  Require(ring.io_alignment() == kAlignment,
          "CUDA ring reported the wrong I/O alignment");
  Require(ring.allocation_alignment() >= kAlignment,
          "CUDA ring allocation alignment is too small");
  Require(ring.slot_stride() >= kSlotCapacity,
          "CUDA ring slot stride is too small");

  auto first = ring.TryAcquireFilling();
  auto second = ring.TryAcquireFilling();
  Require(first.has_value() && second.has_value(),
          "failed to acquire both CUDA ring slots");
  const auto first_bytes = first->bytes();
  const auto second_bytes = second->bytes();
  const auto first_address =
      reinterpret_cast<std::uintptr_t>(first_bytes.data);
  const auto second_address =
      reinterpret_cast<std::uintptr_t>(second_bytes.data);
  Require(first_address % ring.allocation_alignment() == 0,
          "first CUDA ring slot is not allocation-aligned");
  Require(second_address % ring.allocation_alignment() == 0,
          "second CUDA ring slot is not allocation-aligned");
  Require(second_address - first_address == ring.slot_stride(),
          "CUDA ring slots do not follow the advertised stride");
  RequireRegisteredHostPointer(first_bytes.data);
  RequireRegisteredHostPointer(second_bytes.data);

  const std::size_t original_slot = first->slot_index();
  const std::uint64_t original_generation = first->generation();
  FillPattern(*first);
  second->Cancel();
  CudaReadySlotLease ready = std::move(*first).Publish(PacketSlotMetadata{
      "alignment-copy", kLogicalBytes, kLogicalBytes, 0xCAFEBABEU});

  DeviceBytes device(kLogicalBytes);
  CudaInFlightSlotLease in_flight =
      ring.EnqueueH2d(std::move(ready), device.data(), device.size());
  Require(in_flight.valid(), "H2D enqueue did not return an active lease");
  const CompletionState completion = in_flight.WaitAndRecycle();
  Require(completion == CompletionState::kComplete,
          "registered H2D completion failed");
  Require(!in_flight.valid(), "completed H2D lease did not become empty");
  RequirePattern(CopyBack(device, kLogicalBytes));

  const auto snapshots = ring.Snapshot();
  Require(snapshots.size() == 2, "CUDA ring snapshot has the wrong size");
  Require(snapshots[original_slot].state == PacketSlotState::kFree,
          "completed H2D slot was not recycled");

  auto reused = ring.TryAcquireFilling();
  Require(reused.has_value(), "completed CUDA ring slot was not reusable");
  Require(reused->slot_index() == original_slot,
          "deterministic ring did not reuse the expected slot");
  Require(reused->generation() == original_generation + 1U,
          "reused CUDA ring slot did not advance its generation");
  reused->Cancel();
}

struct StreamGate {
  std::atomic<bool> entered{false};
  std::atomic<bool> release{false};
};

void CUDART_CB BlockCopyStream(void* opaque) {
  auto* gate = static_cast<StreamGate*>(opaque);
  gate->entered.store(true, std::memory_order_release);
  while (!gate->release.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

class GateRelease final {
 public:
  explicit GateRelease(StreamGate* gate) noexcept : gate_(gate) {}
  ~GateRelease() { gate_->release.store(true, std::memory_order_release); }

 private:
  StreamGate* gate_;
};

void TestPendingTransferCannotRecycle() {
  CudaPacketRing ring(PacketRingConfig{1, kSlotCapacity, kAlignment});
  CudaReadySlotLease ready = PublishPattern(ring, "pending-gate", 7);
  const std::uint64_t generation = ready.view().generation;
  DeviceBytes device(kLogicalBytes);

  StreamGate gate;
  GateRelease release_on_error(&gate);
  RequireCuda(cudaLaunchHostFunc(
                  CudaPacketRingTestAccess::CopyStream(ring),
                  BlockCopyStream,
                  &gate),
              "cudaLaunchHostFunc(test gate)");
  CudaInFlightSlotLease in_flight =
      ring.EnqueueH2d(std::move(ready), device.data(), device.size());

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!gate.entered.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  Require(gate.entered.load(std::memory_order_acquire),
          "CUDA copy-stream gate did not start");
  Require(in_flight.PollAndRecycle() == CompletionState::kPending,
          "blocked H2D event did not report Pending");
  Require(in_flight.valid(), "pending H2D lease was cleared prematurely");
  Require(!ring.TryAcquireFilling().has_value(),
          "pending H2D slot was recycled before event completion");
  const auto snapshot = ring.Snapshot();
  Require(snapshot.size() == 1 &&
              snapshot[0].state == PacketSlotState::kH2dInFlight,
          "pending CUDA slot does not remain H2D-in-flight");

  gate.release.store(true, std::memory_order_release);
  Require(in_flight.WaitAndRecycle() == CompletionState::kComplete,
          "unblocked H2D event did not complete");
  RequirePattern(CopyBack(device, kLogicalBytes), 7);
  auto reused = ring.TryAcquireFilling();
  Require(reused.has_value(), "unblocked H2D slot was not recycled");
  Require(reused->generation() == generation + 1U,
          "recycled pending slot did not advance its generation");
  reused->Cancel();
}

class TempPackFile final {
 public:
  explicit TempPackFile(const std::vector<std::uint8_t>& bytes) {
    char pattern[] = "/tmp/harpq-cuda-packet-ring.XXXXXX";
    const int descriptor = mkstemp(pattern);
    if (descriptor < 0) {
      throw std::runtime_error(
          std::string("mkstemp failed: ") + std::strerror(errno));
    }
    path_ = pattern;

    std::size_t completed = 0;
    while (completed < bytes.size()) {
      const ssize_t result = write(
          descriptor, bytes.data() + completed, bytes.size() - completed);
      if (result < 0 && errno == EINTR) {
        continue;
      }
      if (result <= 0) {
        const int saved_errno = errno;
        static_cast<void>(close(descriptor));
        static_cast<void>(unlink(path_.c_str()));
        throw std::runtime_error(
            std::string("write temp pack failed: ") +
            std::strerror(saved_errno));
      }
      completed += static_cast<std::size_t>(result);
    }
    if (close(descriptor) != 0) {
      const int saved_errno = errno;
      static_cast<void>(unlink(path_.c_str()));
      throw std::runtime_error(
          std::string("close temp pack failed: ") +
          std::strerror(saved_errno));
    }
  }

  ~TempPackFile() {
    if (!path_.empty()) {
      static_cast<void>(unlink(path_.c_str()));
    }
  }

  TempPackFile(const TempPackFile&) = delete;
  TempPackFile& operator=(const TempPackFile&) = delete;

  [[nodiscard]] const std::string& path() const noexcept { return path_; }

 private:
  std::string path_;
};

class Stream final {
 public:
  Stream() {
    RequireCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags(test compute stream)");
  }

  ~Stream() {
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamSynchronize(stream_));
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

void TestPreadToRegisteredH2dAndStreamWait() {
  constexpr std::size_t logical_bytes = 3000;
  std::vector<std::uint8_t> logical(logical_bytes);
  for (std::size_t index = 0; index < logical.size(); ++index) {
    logical[index] = PatternByte(index, 53);
  }
  std::vector<std::uint8_t> pack_bytes(kAlignment, 0);
  std::copy(logical.begin(), logical.end(), pack_bytes.begin());
  TempPackFile pack(pack_bytes);

  PreadPackFile source(pack.path(), pack_bytes.size());
  CudaPacketRing ring(PacketRingConfig{1, kSlotCapacity, kAlignment});
  const PacketReadRequest request{
      "pread-to-h2d",
      0,
      logical.size(),
      kAlignment,
      Crc32c(logical),
      PacketRequirement::kRequired,
  };
  auto loaded = ring.Load(source, request);
  Require(loaded.status == PacketLoadStatus::kReady,
          "CUDA pread facade did not return Ready");
  Require(loaded.ready.has_value(),
          "CUDA pread facade did not retain its ready lease");
  Require(loaded.stats.logical_bytes == logical.size() &&
              loaded.stats.physical_bytes_requested == kAlignment &&
              loaded.stats.file_bytes_returned == kAlignment,
          "CUDA pread facade did not preserve read statistics");

  DeviceBytes destination(logical.size());
  DeviceBytes ordered_copy(logical.size());
  CudaInFlightSlotLease in_flight = ring.EnqueueH2d(
      std::move(*loaded.ready), destination.data(), destination.size());
  Stream compute_stream;
  RequireCuda(in_flight.MakeStreamWait(compute_stream.get()),
              "CudaInFlightSlotLease::MakeStreamWait");
  RequireCuda(cudaMemcpyAsync(
                  ordered_copy.data(),
                  destination.data(),
                  logical.size(),
                  cudaMemcpyDeviceToDevice,
                  compute_stream.get()),
              "cudaMemcpyAsync(ordered D2D)");
  RequireCuda(cudaStreamSynchronize(compute_stream.get()),
              "cudaStreamSynchronize(ordered D2D)");
  Require(in_flight.WaitAndRecycle() == CompletionState::kComplete,
          "pread-backed H2D completion failed");

  const std::vector<std::uint8_t> returned =
      CopyBack(ordered_copy, logical.size());
  Require(returned == logical,
          "pread -> registered ring -> H2D -> stream wait changed payload bytes");
  Require(in_flight.MakeStreamWait(compute_stream.get()) ==
              cudaErrorInvalidResourceHandle,
          "recycled H2D lease accepted a stream wait");
}

void TestInvalidDestinationPreservesReadyLease() {
  CudaPacketRing ring(PacketRingConfig{1, kSlotCapacity, kAlignment});
  CudaPacketRing other(PacketRingConfig{1, kSlotCapacity, kAlignment});
  CudaReadySlotLease ready = PublishPattern(ring, "invalid-destination", 13);

  RequireInvalidArgument(
      [&] {
        static_cast<void>(
            ring.EnqueueH2d(std::move(ready), nullptr, kLogicalBytes));
      },
      "null H2D destination");
  Require(ready.valid(), "null destination consumed the ready lease");

  DeviceBytes device(kLogicalBytes);
  RequireInvalidArgument(
      [&] {
        static_cast<void>(ring.EnqueueH2d(
            std::move(ready), device.data(), kLogicalBytes - 1U));
      },
      "short H2D destination");
  Require(ready.valid(), "short destination consumed the ready lease");

  RequireInvalidArgument(
      [&] {
        static_cast<void>(other.EnqueueH2d(
            std::move(ready), device.data(), device.size()));
      },
      "cross-ring CUDA ready lease");
  Require(ready.valid(), "cross-ring enqueue consumed the ready lease");

  CudaInFlightSlotLease in_flight =
      ring.EnqueueH2d(std::move(ready), device.data(), device.size());
  Require(in_flight.WaitAndRecycle() == CompletionState::kComplete,
          "valid H2D after validation failures did not complete");
  RequirePattern(CopyBack(device, kLogicalBytes), 13);
}

void TestInjectedRecoverableCudaFailures() {
  CudaPacketRingTestAccess::FailHostRegisterOnce();
  bool registration_failed = false;
  try {
    CudaPacketRing ring(PacketRingConfig{1, kSlotCapacity, kAlignment});
    static_cast<void>(ring);
  } catch (const std::runtime_error& error) {
    registration_failed =
        std::string(error.what()).find("pageable fallback") != std::string::npos;
  }
  Require(registration_failed,
          "injected host-registration failure was not exact-or-fail");

  {
    CudaPacketRing ring(PacketRingConfig{1, kSlotCapacity, kAlignment});
    CudaReadySlotLease ready = PublishPattern(ring, "event-record-failure", 61);
    DeviceBytes device(kLogicalBytes);
    CudaPacketRingTestAccess::FailEventRecordOnce();
    bool record_failed = false;
    try {
      static_cast<void>(
          ring.EnqueueH2d(std::move(ready), device.data(), device.size()));
    } catch (const std::runtime_error& error) {
      record_failed =
          std::string(error.what()).find("cudaEventRecord") != std::string::npos;
    }
    Require(record_failed, "injected event-record failure did not surface");
    Require(ready.valid(),
            "event-record failure consumed its ready lease after safe sync");
    const auto snapshot = ring.Snapshot();
    Require(snapshot.size() == 1 &&
                snapshot[0].state == PacketSlotState::kReady,
            "event-record failure changed the core slot state");
    auto retry = ring.EnqueueH2d(
        std::move(ready), device.data(), device.size());
    Require(retry.WaitAndRecycle() == CompletionState::kComplete,
            "event-record retry did not complete");
    RequirePattern(CopyBack(device, kLogicalBytes), 61);
  }

  {
    CudaPacketRing ring(PacketRingConfig{1, kSlotCapacity, kAlignment});
    CudaReadySlotLease ready = PublishPattern(ring, "event-query-failure", 67);
    DeviceBytes device(kLogicalBytes);
    auto in_flight =
        ring.EnqueueH2d(std::move(ready), device.data(), device.size());
    CudaPacketRingTestAccess::FailEventQueryOnce();
    Require(in_flight.PollAndRecycle() == CompletionState::kFailed,
            "event-query failure did not remain externally visible");
    Require(!in_flight.valid(),
            "event-query fallback did not recycle after proven event sync");
    RequirePattern(CopyBack(device, kLogicalBytes), 67);
  }

  {
    CudaPacketRing ring(PacketRingConfig{1, kSlotCapacity, kAlignment});
    CudaReadySlotLease ready =
        PublishPattern(ring, "event-sync-stream-fallback", 71);
    DeviceBytes device(kLogicalBytes);
    auto in_flight =
        ring.EnqueueH2d(std::move(ready), device.data(), device.size());
    CudaPacketRingTestAccess::FailEventSynchronizeOnce();
    Require(in_flight.WaitAndRecycle() == CompletionState::kFailed,
            "event-sync failure did not use the safe stream fallback");
    RequirePattern(CopyBack(device, kLogicalBytes), 71);
  }

  {
    CudaPacketRing ring(PacketRingConfig{1, kSlotCapacity, kAlignment});
    CudaReadySlotLease ready =
        PublishPattern(ring, "stream-sync-device-fallback", 73);
    DeviceBytes device(kLogicalBytes);
    auto in_flight =
        ring.EnqueueH2d(std::move(ready), device.data(), device.size());
    CudaPacketRingTestAccess::FailEventAndStreamSynchronizeOnce();
    Require(in_flight.WaitAndRecycle() == CompletionState::kFailed,
            "stream-sync failure did not use the captured-device fallback");
    RequirePattern(CopyBack(device, kLogicalBytes), 73);
  }

  {
    CudaPacketRing ring(PacketRingConfig{1, kSlotCapacity, kAlignment});
    CudaPacketRingTestAccess::FailStreamSynchronizeOnce();
    // Destruction must prove safety through captured-device synchronization
    // before unregistering the backing allocation.
  }
}

[[noreturn]] void ExecFatalChild(const char* executable, const char* mode) {
  execl(executable, executable, mode, static_cast<char*>(nullptr));
  _exit(127);
}

void RequireSubprocessAbort(const char* executable,
                            const char* mode,
                            const std::string& label) {
  const pid_t child = fork();
  if (child < 0) {
    throw std::runtime_error(
        std::string("fork failed: ") + std::strerror(errno));
  }
  if (child == 0) {
    ExecFatalChild(executable, mode);
  }

  int status = 0;
  pid_t waited = -1;
  do {
    waited = waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  if (waited != child) {
    throw std::runtime_error(
        std::string("waitpid failed: ") + std::strerror(errno));
  }
  Require(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
          label + " did not fail closed with SIGABRT");
}

int RunFatalChildMode(const std::string& mode) {
  if (mode == "--fatal-unsafe-sync") {
    CudaPacketRing ring(PacketRingConfig{1, kSlotCapacity, kAlignment});
    CudaReadySlotLease ready = PublishPattern(ring, "fatal-sync", 79);
    DeviceBytes device(kLogicalBytes);
    auto in_flight =
        ring.EnqueueH2d(std::move(ready), device.data(), device.size());
    CudaPacketRingTestAccess::FailAllSafetySynchronizationsOnce();
    static_cast<void>(in_flight.WaitAndRecycle());
    return 90;
  }
  if (mode == "--fatal-unregister") {
    {
      CudaPacketRing ring(PacketRingConfig{1, kSlotCapacity, kAlignment});
      CudaPacketRingTestAccess::FailHostUnregisterOnce();
    }
    return 91;
  }
  return 92;
}

void TestUnprovenSafetyFailsClosed(const char* executable) {
  RequireSubprocessAbort(
      executable,
      "--fatal-unsafe-sync",
      "unproven event/stream/device synchronization");
  RequireSubprocessAbort(
      executable,
      "--fatal-unregister",
      "failed registered-host teardown");
}

void TestFacadeAndLeaseDestructionOrder() {
  CudaFillingSlotLease filling_after_facade;
  const std::uint8_t* registered_pointer = nullptr;
  {
    auto ring = std::make_unique<CudaPacketRing>(
        PacketRingConfig{1, kSlotCapacity, kAlignment});
    auto acquired = ring->TryAcquireFilling();
    Require(acquired.has_value(), "lifetime test could not acquire filling slot");
    filling_after_facade = std::move(*acquired);
    registered_pointer = filling_after_facade.bytes().data;
  }
  Require(filling_after_facade.valid(),
          "destroying facade invalidated an outstanding filling lease");
  RequireRegisteredHostPointer(registered_pointer);
  filling_after_facade.Cancel();

  DeviceBytes device(kLogicalBytes);
  CudaInFlightSlotLease in_flight_after_facade;
  {
    auto ring = std::make_unique<CudaPacketRing>(
        PacketRingConfig{1, kSlotCapacity, kAlignment});
    CudaReadySlotLease ready = PublishPattern(*ring, "facade-destruction", 29);
    in_flight_after_facade =
        ring->EnqueueH2d(std::move(ready), device.data(), device.size());
  }
  Require(in_flight_after_facade.valid(),
          "destroying facade invalidated an outstanding H2D lease");
  Require(in_flight_after_facade.WaitAndRecycle() == CompletionState::kComplete,
          "H2D lease could not complete after facade destruction");
  RequirePattern(CopyBack(device, kLogicalBytes), 29);

  {
    auto ring = std::make_unique<CudaPacketRing>(
        PacketRingConfig{1, kSlotCapacity, kAlignment});
    CudaReadySlotLease ready = PublishPattern(*ring, "destructor-wait", 41);
    CudaInFlightSlotLease destructor_wait =
        ring->EnqueueH2d(std::move(ready), device.data(), device.size());
    ring.reset();
    // destructor_wait leaves scope without an explicit wait. Its core lease
    // must wait before releasing the last registration-control reference.
  }
  RequirePattern(CopyBack(device, kLogicalBytes), 41);
}

bool CudaDeviceAvailable(std::string* reason) noexcept {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    if (reason != nullptr) {
      *reason = std::string(cudaGetErrorName(status)) + ": " +
                cudaGetErrorString(status);
    }
    static_cast<void>(cudaGetLastError());
    return false;
  }
  if (count <= 0) {
    if (reason != nullptr) {
      *reason = "CUDA runtime reported no devices";
    }
    return false;
  }
  if (reason != nullptr) {
    reason->clear();
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string unavailable_reason;
  if (!CudaDeviceAvailable(&unavailable_reason)) {
    std::cout << "SKIP: CUDA packet-ring test requires a working CUDA device: "
              << unavailable_reason << '\n';
    return 77;
  }

  if (argc == 2 && std::string(argv[1]).find("--fatal-") == 0) {
    return RunFatalChildMode(argv[1]);
  }

  try {
    TestAlignmentRegistrationCopyAndGeneration();
    TestPendingTransferCannotRecycle();
    TestInvalidDestinationPreservesReadyLease();
    TestPreadToRegisteredH2dAndStreamWait();
    TestFacadeAndLeaseDestructionOrder();
    TestInjectedRecoverableCudaFailures();
    TestUnprovenSafetyFailsClosed(argv[0]);
    std::cout << "CUDA packet-ring tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CUDA packet-ring test failed: " << error.what() << '\n';
    return 1;
  }
}
