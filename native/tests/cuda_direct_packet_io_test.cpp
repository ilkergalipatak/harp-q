#include "harpq/cuda_direct_packet_io.hpp"
#include "harpq/native/stage_codec.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace harpq::native_runtime {
class CudaPacketRingTestAccess final {
 public:
  static void EventRecordFailure() { CudaPacketRing::InjectFaultsForTest(1U << 1U); }
  static void EventQueryFailure() { CudaPacketRing::InjectFaultsForTest(1U << 2U); }
};
}  // namespace harpq::native_runtime

namespace {
using namespace harpq::native_runtime;
constexpr std::size_t kAlignment = 4096;

void Require(bool condition, const char* message) {
  if (!condition) { throw std::runtime_error(message); }
}
void Check(cudaError_t error) {
  if (error != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(error)); }
}

class Fixture final {
 public:
  Fixture(std::size_t physical = kAlignment, std::size_t logical = 3000)
      : bytes(physical), logical_bytes(logical) {
    for (std::size_t i = 0; i < logical; ++i) { bytes[i] = (i * 37 + 11) % 251; }
    const char* parent = std::getenv("HARPQ_DIRECT_IO_TEST_DIR");
    std::string pattern = std::string(parent && *parent ? parent : "/tmp") + "/harpq-bridge.XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const int fd = mkstemp(name.data());
    if (fd < 0) { throw std::runtime_error("cannot create bridge test pack"); }
    path = name.data();
    try {
      std::size_t written = 0;
      while (written < bytes.size()) {
        const ssize_t count = write(fd, bytes.data() + written, bytes.size() - written);
        if (count < 0 && errno == EINTR) { continue; }
        Require(count > 0, "write bridge fixture failed");
        written += static_cast<std::size_t>(count);
      }
      Require(fsync(fd) == 0, "fsync fixture failed");
    } catch (...) {
      close(fd);
      unlink(path.c_str());
      throw;
    }
    close(fd);
  }
  ~Fixture() { unlink(path.c_str()); }
  PacketReadRequest Request(PacketRequirement requirement = PacketRequirement::kRequired) const {
    return {"packet", 0, logical_bytes, ((logical_bytes + 4095) / 4096) * 4096,
            Crc32c(bytes.data(), logical_bytes), requirement};
  }
  std::string path;
  std::vector<std::uint8_t> bytes;
  std::size_t logical_bytes;
};

void Registered(const void* pointer) {
  cudaPointerAttributes attr{};
  Check(cudaPointerGetAttributes(&attr, pointer));
  Require(attr.type == cudaMemoryTypeHost, "direct destination lost CUDA registration");
}

CudaDirectPacketLoadResult Load(CudaDirectPackFile& source, const Fixture& fixture) {
  auto submitted = source.Submit(fixture.Request());
  Require(submitted.status == DirectPacketLoadStatus::kSubmitted, "read was not submitted");
  auto ready = source.Wait();
  Require(ready.status == DirectPacketLoadStatus::kReady && ready.ready.has_value(), "read not ready");
  Require(ready.operation_id == submitted.operation_id, "operation ID changed");
  Require(ready.stats.logical_bytes == fixture.logical_bytes, "logical bytes changed");
  Require(ready.stats.file_bytes_returned == fixture.bytes.size(), "physical read bytes changed");
  Registered(ready.ready->view().data);
  return ready;
}

PacketServiceObservation Verify(CudaPacketTransfer& transfer, const Fixture& fixture) {
  Require(transfer.Wait() == CompletionState::kComplete, "H2D failed");
  Require(transfer.Wait() == CompletionState::kComplete, "terminal wait changed state");
  std::vector<std::uint8_t> copied(fixture.logical_bytes);
  Check(cudaMemcpy(copied.data(), transfer.device_data(), copied.size(), cudaMemcpyDeviceToHost));
  Require(std::equal(copied.begin(), copied.end(), fixture.bytes.begin()), "SSD to GPU bytes differ");
  const auto* observation = transfer.observation();
  Require(observation != nullptr, "successful transfer has no observation");
  Require(observation->logical_bytes == fixture.logical_bytes, "observation logical bytes differ");
  Require(observation->physical_io_bytes == fixture.bytes.size(), "observation physical bytes differ");
  Require(std::isfinite(observation->critical_path_us) && observation->critical_path_us > 0,
          "invalid critical-path duration");
  Require(observation->io_service_us >= 0 && observation->h2d_service_us >= 0 &&
              observation->io_service_us + observation->h2d_service_us <= observation->critical_path_us + 1e-6,
          "component intervals exceed critical path");
  return *observation;
}

void TestFacadeLifetimes(const Fixture& fixture) {
  auto ring = std::make_unique<CudaPacketRing>(PacketRingConfig{1, 4096, 4096});
  auto source = std::make_unique<CudaDirectPackFile>(*ring, fixture.path, fixture.bytes.size());
  Require(source->Submit(fixture.Request()).status == DirectPacketLoadStatus::kSubmitted, "submit");
  ring.reset();
  auto ready = source->Wait();
  source.reset();
  Registered(ready.ready->view().data);
  auto transfer = std::move(*ready.ready).EnqueueH2d();
  Require(!ready.ready->valid(), "upload did not consume ready lease");
  ready.ready.reset();
  static_cast<void>(Verify(transfer, fixture));

  CudaPacketRing surviving_ring(PacketRingConfig{1, 4096, 4096});
  {
    CudaDirectPackFile abandoned(surviving_ring, fixture.path, fixture.bytes.size());
    Require(abandoned.Submit(fixture.Request()).status == DirectPacketLoadStatus::kSubmitted, "submit");
  }
  auto filling = surviving_ring.TryAcquireFilling();
  Require(filling.has_value(), "reader destructor failed to drain CQE and release slot");
  Registered(filling->bytes().data);
}

void TestPolicies(const Fixture& fixture) {
  CudaPacketRing ring(PacketRingConfig{1, 4096, 4096});
  CudaDirectPackFile source(ring, fixture.path, fixture.bytes.size(), DirectPacketIoConfig{1});
  CancellationSource cancellation;
  cancellation.Cancel();
  auto cancelled = source.Submit(fixture.Request(), cancellation.token());
  Require(cancelled.status == DirectPacketLoadStatus::kCancelled && source.in_flight() == 0,
          "pre-cancel touched I/O");

  CancellationSource in_flight_cancel;
  Require(source.Submit(fixture.Request(), in_flight_cancel.token()).status ==
              DirectPacketLoadStatus::kSubmitted, "submit cancelled read");
  auto blocked = source.Submit(fixture.Request(PacketRequirement::kOptional));
  Require(blocked.status == DirectPacketLoadStatus::kWouldBlock, "backpressure became fallback");
  in_flight_cancel.Cancel();
  Require(!ring.TryAcquireFilling(), "cancel recycled slot before CQE");
  Require(source.Wait().status == DirectPacketLoadStatus::kCancelled, "cancel completion lost");

  auto bad = fixture.Request(PacketRequirement::kOptional);
  bad.expected_crc32c ^= 1;
  static_cast<void>(source.Submit(bad));
  auto failure = source.Wait();
  Require(failure.status == DirectPacketLoadStatus::kOptionalUnavailable && !failure.ready,
          "optional CRC policy changed");
  bad.requirement = PacketRequirement::kRequired;
  static_cast<void>(source.Submit(bad));
  bool threw = false;
  try { static_cast<void>(source.Wait()); }
  catch (const RequiredPacketLoadError&) { threw = true; }
  Require(threw && source.in_flight() == 0, "required error lost or CQE still tracked");

  auto ready = Load(source, fixture);
  CudaPacketRingTestAccess::EventRecordFailure();
  threw = false;
  try { static_cast<void>(std::move(*ready.ready).EnqueueH2d()); }
  catch (const std::runtime_error&) { threw = true; }
  Require(threw && ready.ready->valid(), "enqueue error consumed ready packet");
  auto retry = std::move(*ready.ready).EnqueueH2d();
  CudaPacketRingTestAccess::EventQueryFailure();
  Require(retry.Poll() == CompletionState::kFailed && retry.observation() == nullptr,
          "failed transfer emitted a successful sample");
  Require(ring.TryAcquireFilling().has_value(), "failed transfer did not safely recycle");

  Fixture legacy(37, 37);
  CudaDirectPackFile old_source(ring, legacy.path, legacy.bytes.size());
  auto old = old_source.Submit(legacy.Request());
  Require(old.status == DirectPacketLoadStatus::kBufferedOnlyLegacyTail && !old.ready &&
              old.stats.read_submissions == 0, "legacy tail silently submitted");
}

void TestOrderedConsumer(const Fixture& fixture) {
  CudaPacketRing ring(PacketRingConfig{1, 4096, 4096});
  CudaDirectPackFile source(ring, fixture.path, fixture.bytes.size());
  auto ready = Load(source, fixture);
  auto transfer = std::move(*ready.ready).EnqueueH2d();
  cudaStream_t compute = nullptr;
  Check(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
  try {
    Check(transfer.MakeStreamWait(compute));
    std::vector<std::uint8_t> copy(fixture.logical_bytes);
    Check(cudaMemcpyAsync(copy.data(), transfer.device_data(), copy.size(), cudaMemcpyDeviceToHost, compute));
    Check(cudaStreamSynchronize(compute));
    Require(std::equal(copy.begin(), copy.end(), fixture.bytes.begin()), "event-ordered consumer mismatch");
  } catch (...) {
    cudaStreamSynchronize(compute);
    cudaStreamDestroy(compute);
    throw;
  }
  Check(cudaStreamDestroy(compute));
  static_cast<void>(Verify(transfer, fixture));
}

void TestConcurrentPollAndAbandonment(const Fixture& fixture) {
  CudaPacketRing ring(PacketRingConfig{2, 4096, 4096});
  CudaDirectPackFile source(ring, fixture.path, fixture.bytes.size(), DirectPacketIoConfig{2});
  auto first = fixture.Request();
  auto second = first;
  second.packet_id = "second";
  const auto a = source.Submit(first);
  const auto b = source.Submit(second);
  Require(a.operation_id != b.operation_id, "concurrent IDs not unique");
  std::vector<std::uint64_t> completed;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (completed.size() < 2) {
    Require(std::chrono::steady_clock::now() < deadline, "poll completion timeout");
    auto polled = source.Poll();
    if (!polled) { continue; }
    Require(polled->ready.has_value(), "poll result lost ready lease");
    const bool is_first = polled->operation_id == a.operation_id;
    Require(is_first || polled->operation_id == b.operation_id, "poll returned unknown ID");
    Require(polled->ready->view().packet_id == (is_first ? "packet" : "second"), "poll metadata mismatch");
    completed.push_back(polled->operation_id);
    {
      auto abandoned = std::move(*polled->ready).EnqueueH2d();
      // Destructor must finish DMA before releasing its destination and slot.
    }
  }
  Require(completed[0] != completed[1] && !source.Poll(), "poll duplicated completion");
  auto fill_a = ring.TryAcquireFilling();
  auto fill_b = ring.TryAcquireFilling();
  Require(fill_a.has_value() && fill_b.has_value(), "abandoned transfer retained slots");
  CudaPacketTransfer empty;
  Require(empty.Wait() == CompletionState::kFailed && empty.observation() == nullptr &&
              empty.device_data() == nullptr && empty.size_bytes() == 0, "empty transfer contract");
}

std::vector<PacketServiceObservation> TestMeasuredSizes() {
  std::vector<PacketServiceObservation> observations;
  for (std::size_t size : {65536U, 262144U, 1048576U}) {
    Fixture fixture(size, size - 29);
    PacketRing oracle_ring(PacketRingConfig{1, size, 4096});
    PreadPackFile oracle(fixture.path, fixture.bytes.size());
    auto buffered = oracle.Load(oracle_ring, fixture.Request());
    Require(buffered.ready.has_value(), "buffered oracle failed");
    CudaPacketRing ring(PacketRingConfig{2, size, 4096});
    CudaDirectPackFile source(ring, fixture.path, fixture.bytes.size());
    for (int sample = 0; sample < 3; ++sample) {
      auto ready = Load(source, fixture);
      const auto expected = buffered.ready->view();
      Require(std::equal(expected.data, expected.data + expected.logical_bytes, ready.ready->view().data),
              "direct read differs from pread oracle");
      auto transfer = std::move(*ready.ready).EnqueueH2d();
      Require(transfer.observation() == nullptr, "sample published before event completion");
      observations.push_back(Verify(transfer, fixture));
    }
  }
  return observations;
}

void PrintObservations(const std::vector<PacketServiceObservation>& observations) {
  std::cout << std::setprecision(17) << "{\"observations\":[";
  for (std::size_t index = 0; index < observations.size(); ++index) {
    if (index) { std::cout << ','; }
    const auto& item = observations[index];
    std::cout << "{\"sequence_id\":" << index << ",\"packet_id\":\"packet\",\"service_class\":\""
              << PacketServiceObservation::service_class << "\",\"logical_bytes\":" << item.logical_bytes
              << ",\"physical_io_bytes\":" << item.physical_io_bytes
              << ",\"critical_path_us\":" << item.critical_path_us
              << ",\"io_service_us\":" << item.io_service_us
              << ",\"h2d_service_us\":" << item.h2d_service_us << '}';
  }
  std::cout << "]}\n";
}
}  // namespace

int main(int argc, char** argv) {
  const bool json = argc == 2 && std::strcmp(argv[1], "--json") == 0;
  if (argc != 1 && !json) { std::cerr << "usage: cuda_direct_packet_io_test [--json]\n"; return 2; }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cerr << "CUDA direct-I/O test requires a working GPU\n";
    return HARPQ_TEST_REQUIRE_CUDA ? 1 : 77;
  }
  try {
    Fixture fixture;
    try {
      CudaPacketRing probe_ring(PacketRingConfig{1, 4096, 4096});
      CudaDirectPackFile probe(probe_ring, fixture.path, fixture.bytes.size());
      if (!probe.direct_capability().supported) {
        std::cerr << "direct-I/O capability unavailable; reason="
                  << static_cast<int>(probe.direct_capability().unsupported_reason) << '\n';
        return HARPQ_TEST_REQUIRE_IO ? 1 : 77;
      }
    } catch (const PacketIoError& error) {
      const auto& failure = error.failure();
      if (failure.code == PacketIoErrorCode::kOpenFailed &&
          (failure.system_error == EPERM || failure.system_error == ENOSYS || failure.system_error == EOPNOTSUPP)) {
        std::cerr << "direct-I/O setup unavailable: " << error.what() << '\n';
        return HARPQ_TEST_REQUIRE_IO ? 1 : 77;
      }
      throw;
    }
    TestFacadeLifetimes(fixture);
    TestPolicies(fixture);
    TestOrderedConsumer(fixture);
    TestConcurrentPollAndAbandonment(fixture);
    const auto observations = TestMeasuredSizes();
    if (json) { PrintObservations(observations); }
    else { std::cout << "CUDA direct-I/O bridge tests passed (9 service observations)\n"; }
  } catch (const std::exception& error) {
    std::cerr << "CUDA direct-I/O bridge test failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
