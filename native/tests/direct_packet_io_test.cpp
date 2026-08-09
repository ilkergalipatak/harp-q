#include "harpq/native/direct_packet_io.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "harpq/native/stage_codec.hpp"

namespace harpq::native_runtime {

class DirectPacketIoTestAccess final {
 public:
  static void InjectCompletionResults(std::vector<int> results) {
    DirectPackFile::InjectCompletionResultsForTest(std::move(results));
  }

  static void ClearCompletionResults() noexcept {
    DirectPackFile::ClearCompletionResultsForTest();
  }

  static void SuppressStatxDioalignOnce() noexcept {
    DirectPackFile::SuppressStatxDioalignForTestOnce();
  }
};

}  // namespace harpq::native_runtime

namespace {

using harpq::native_runtime::CancellationSource;
using harpq::native_runtime::Crc32c;
using harpq::native_runtime::DirectIoAlignmentSource;
using harpq::native_runtime::DirectPackFile;
using harpq::native_runtime::DirectPacketIoConfig;
using harpq::native_runtime::DirectPacketIoTestAccess;
using harpq::native_runtime::DirectPacketLoadResult;
using harpq::native_runtime::DirectPacketLoadStatus;
using harpq::native_runtime::PacketIoError;
using harpq::native_runtime::PacketIoErrorCode;
using harpq::native_runtime::PacketReadRequest;
using harpq::native_runtime::PacketRequirement;
using harpq::native_runtime::PacketRing;
using harpq::native_runtime::PacketRingConfig;
using harpq::native_runtime::PreadPackFile;
using harpq::native_runtime::RequiredPacketLoadError;

constexpr std::size_t kAlignment = 4096;
constexpr int kSkipExitCode = 77;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Exception, typename Callable>
void RequireThrows(Callable&& callable, const std::string& label) {
  try {
    callable();
  } catch (const Exception&) {
    return;
  } catch (const std::exception& error) {
    throw std::runtime_error(
        label + " raised the wrong exception: " + error.what());
  }
  throw std::runtime_error(label + " did not throw");
}

class TempDirectory final {
 public:
  TempDirectory() {
    const char* configured = std::getenv("HARPQ_DIRECT_IO_TEST_DIR");
    const std::string parent =
        configured != nullptr && configured[0] != '\0' ? configured : "/tmp";
    std::string pattern = parent + "/harpq-direct-io.XXXXXX";
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error(
          std::string("mkdtemp failed: ") + std::strerror(errno));
    }
    path_ = created;
  }

  ~TempDirectory() {
    for (const std::string& file : files_) {
      static_cast<void>(unlink(file.c_str()));
    }
    static_cast<void>(rmdir(path_.c_str()));
  }

  TempDirectory(const TempDirectory&) = delete;
  TempDirectory& operator=(const TempDirectory&) = delete;

  [[nodiscard]] std::string File(const std::string& name) {
    const std::string path = path_ + "/" + name;
    files_.push_back(path);
    return path;
  }

 private:
  std::string path_;
  std::vector<std::string> files_;
};

void WriteFile(const std::string& path, const std::vector<std::uint8_t>& bytes) {
  const int descriptor =
      open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (descriptor < 0) {
    throw std::runtime_error(
        std::string("open test file failed: ") + std::strerror(errno));
  }
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
      throw std::runtime_error(
          std::string("write test file failed: ") +
          std::strerror(saved_errno));
    }
    completed += static_cast<std::size_t>(result);
  }
  if (fsync(descriptor) != 0) {
    const int saved_errno = errno;
    static_cast<void>(close(descriptor));
    throw std::runtime_error(
        std::string("fsync test file failed: ") +
        std::strerror(saved_errno));
  }
  if (close(descriptor) != 0) {
    throw std::runtime_error(
        std::string("close test file failed: ") + std::strerror(errno));
  }
}

std::vector<std::uint8_t> Pattern(
    std::size_t size, std::uint8_t seed = 17) {
  std::vector<std::uint8_t> result(size);
  for (std::size_t index = 0; index < size; ++index) {
    result[index] = static_cast<std::uint8_t>(
        seed + (static_cast<std::uint64_t>(index) * 29ULL) % 251ULL);
  }
  return result;
}

[[nodiscard]] std::size_t AlignUp(std::size_t value) {
  return ((value + kAlignment - 1U) / kAlignment) * kAlignment;
}

PacketReadRequest Request(
    std::string packet_id,
    std::uint64_t offset,
    const std::vector<std::uint8_t>& logical,
    PacketRequirement requirement = PacketRequirement::kRequired) {
  return PacketReadRequest{
      std::move(packet_id),
      offset,
      logical.size(),
      AlignUp(logical.size()),
      Crc32c(logical),
      requirement,
  };
}

std::vector<std::uint8_t> ReadyBytes(
    const DirectPacketLoadResult& result) {
  Require(result.ready.has_value(), "ready result does not own a lease");
  const auto view = result.ready->view();
  return std::vector<std::uint8_t>(
      view.data, view.data + view.logical_bytes);
}

bool IsSkippableEnvironmentFailure(const PacketIoError& error) {
  const int code = error.failure().system_error;
  return code == EPERM || code == ENOSYS || code == EOPNOTSUPP;
}

void RequireDirectSupport(
    const DirectPackFile& source, PacketRing& packet_ring) {
  Require(source.direct_supported(),
          "filesystem did not expose authoritative STATX_DIOALIGN");
  const auto alignment = source.direct_alignment();
  Require(
      alignment.source == DirectIoAlignmentSource::kStatxDioalign,
      "direct backend enabled without STATX_DIOALIGN provenance");
  Require(alignment.memory_alignment > 0 && alignment.offset_alignment > 0,
          "direct alignment contains zero");
  Require(source.ring_slot_count() == packet_ring.slot_count(),
          "direct source captured the wrong ring slot count");
  auto filling = packet_ring.TryAcquireFilling();
  Require(filling.has_value(), "cannot inspect a direct ring slot");
  Require(
      reinterpret_cast<std::uintptr_t>(filling->bytes().data) %
              alignment.memory_alignment ==
          0,
      "ring slot does not satisfy authoritative memory alignment");
}

void TestDifferentialCanonicalRead(const std::string& path) {
  const std::vector<std::uint8_t> logical = Pattern(3000, 31);
  std::vector<std::uint8_t> file_bytes(2 * kAlignment, 0);
  std::copy(logical.begin(), logical.end(), file_bytes.begin() + kAlignment);
  WriteFile(path, file_bytes);

  PacketRing ring(PacketRingConfig{2, 2 * kAlignment, kAlignment});
  DirectPackFile direct(path, file_bytes.size(), ring, DirectPacketIoConfig{2});
  RequireDirectSupport(direct, ring);
  PreadPackFile buffered(path, file_bytes.size());
  const PacketReadRequest request = Request("base", kAlignment, logical);

  auto buffered_result = buffered.Load(ring, request);
  Require(buffered_result.ready.has_value(), "buffered oracle did not become ready");
  const auto buffered_view = buffered_result.ready->view();
  const std::vector<std::uint8_t> buffered_bytes(
      buffered_view.data, buffered_view.data + buffered_view.logical_bytes);
  buffered_result.ready.reset();

  const auto submitted = direct.Submit(request);
  Require(submitted.status == DirectPacketLoadStatus::kSubmitted,
          "canonical direct read was not submitted");
  Require(submitted.operation_id != 0, "submitted operation ID is zero");
  Require(submitted.stats.read_submissions == 1,
          "initial direct read submission was not counted");
  Require(submitted.stats.bytes_submitted == kAlignment,
          "initial direct submitted bytes are wrong");
  auto completed = direct.Wait();
  Require(completed.status == DirectPacketLoadStatus::kReady,
          "canonical direct read did not become ready");
  Require(completed.operation_id == submitted.operation_id,
          "completion operation ID changed");
  Require(ReadyBytes(completed) == logical,
          "canonical direct payload differs from source");
  Require(ReadyBytes(completed) == buffered_bytes,
          "direct payload differs from buffered oracle");
  Require(completed.stats.logical_bytes == buffered_result.stats.logical_bytes,
          "direct/buffered logical byte stats differ");
  Require(
      completed.stats.physical_bytes_requested ==
          buffered_result.stats.physical_bytes_requested,
      "direct/buffered physical byte stats differ");
  Require(
      completed.stats.file_bytes_returned ==
          buffered_result.stats.file_bytes_returned,
      "direct/buffered returned byte stats differ");
  Require(completed.stats.read_submissions == buffered_result.stats.pread_calls,
          "direct/buffered read-operation stats differ");
  Require(completed.stats.used_ring_slot_buffer,
          "direct completion did not report ring-slot destination");
  Require(!direct.Poll().has_value(), "poll produced a phantom completion");
}

void TestQueueDepthBackpressureAndMonotonicIds(const std::string& path) {
  const std::vector<std::uint8_t> logical = Pattern(kAlignment, 43);
  WriteFile(path, logical);
  PacketRing ring(PacketRingConfig{2, kAlignment, kAlignment});
  DirectPackFile source(path, logical.size(), ring, DirectPacketIoConfig{1});
  RequireDirectSupport(source, ring);
  const PacketReadRequest first_request = Request("base-0", 0, logical);
  PacketReadRequest second_request = first_request;
  second_request.packet_id = "base-1";

  const auto first = source.Submit(first_request);
  Require(first.status == DirectPacketLoadStatus::kSubmitted,
          "first bounded-queue read was not submitted");
  const auto blocked = source.Submit(second_request);
  Require(blocked.status == DirectPacketLoadStatus::kWouldBlock,
          "full logical queue did not report backpressure");
  Require(blocked.status != DirectPacketLoadStatus::kOptionalUnavailable,
          "backpressure became optional fallback");
  auto first_ready = source.Wait();
  Require(first_ready.status == DirectPacketLoadStatus::kReady,
          "first bounded-queue read did not complete");
  first_ready.ready.reset();

  const auto second = source.Submit(second_request);
  Require(second.status == DirectPacketLoadStatus::kSubmitted,
          "second bounded-queue read was not submitted after release");
  Require(second.operation_id > first.operation_id,
          "direct operation IDs are not monotonic");
  auto second_ready = source.Wait();
  Require(second_ready.status == DirectPacketLoadStatus::kReady,
          "second bounded-queue read did not complete");
}

void TestRingSaturationAndConcurrentOperationMapping(const std::string& path) {
  const std::vector<std::uint8_t> first_logical = Pattern(kAlignment, 47);
  const std::vector<std::uint8_t> second_logical = Pattern(kAlignment, 53);
  std::vector<std::uint8_t> file_bytes(2 * kAlignment, 0);
  std::copy(first_logical.begin(), first_logical.end(), file_bytes.begin());
  std::copy(
      second_logical.begin(),
      second_logical.end(),
      file_bytes.begin() + kAlignment);
  WriteFile(path, file_bytes);

  PacketRing ring(PacketRingConfig{2, kAlignment, kAlignment});
  DirectPackFile source(path, file_bytes.size(), ring, DirectPacketIoConfig{3});
  RequireDirectSupport(source, ring);
  const auto first = source.Submit(Request("packet-0", 0, first_logical));
  const auto second = source.Submit(
      Request("packet-1", kAlignment, second_logical));
  Require(first.status == DirectPacketLoadStatus::kSubmitted &&
              second.status == DirectPacketLoadStatus::kSubmitted,
          "concurrent direct requests were not submitted");
  const auto saturated = source.Submit(Request("packet-2", 0, first_logical));
  Require(saturated.status == DirectPacketLoadStatus::kWouldBlock,
          "ring saturation was not reported as backpressure");

  auto completion_a = source.Wait();
  auto completion_b = source.Wait();
  Require(completion_a.status == DirectPacketLoadStatus::kReady &&
              completion_b.status == DirectPacketLoadStatus::kReady,
          "concurrent direct requests did not both become ready");
  const auto verify = [&](const DirectPacketLoadResult& completion) {
    if (completion.operation_id == first.operation_id) {
      Require(ReadyBytes(completion) == first_logical,
              "first operation ID mapped to wrong slot bytes");
      Require(completion.ready->view().packet_id == "packet-0",
              "first operation ID mapped to wrong metadata");
      return;
    }
    if (completion.operation_id == second.operation_id) {
      Require(ReadyBytes(completion) == second_logical,
              "second operation ID mapped to wrong slot bytes");
      Require(completion.ready->view().packet_id == "packet-1",
              "second operation ID mapped to wrong metadata");
      return;
    }
    throw std::runtime_error("unknown concurrent operation ID completed");
  };
  verify(completion_a);
  verify(completion_b);
  Require(completion_a.operation_id != completion_b.operation_id,
          "concurrent completion IDs were duplicated");
}

void TestCancellationWaitsForOriginalCqe(const std::string& path) {
  const std::vector<std::uint8_t> logical = Pattern(kAlignment, 59);
  WriteFile(path, logical);
  PacketRing ring(PacketRingConfig{1, kAlignment, kAlignment});
  DirectPackFile source(path, logical.size(), ring, DirectPacketIoConfig{1});
  RequireDirectSupport(source, ring);
  const PacketReadRequest request = Request("base", 0, logical);

  CancellationSource pre_cancelled;
  pre_cancelled.Cancel();
  const auto immediate = source.Submit(request, pre_cancelled.token());
  Require(immediate.status == DirectPacketLoadStatus::kCancelled,
          "pre-submit cancellation is not distinct");
  Require(immediate.operation_id == 0 && immediate.stats.read_submissions == 0,
          "pre-submit cancellation touched io_uring");

  CancellationSource after_submit;
  const auto submitted = source.Submit(request, after_submit.token());
  Require(submitted.status == DirectPacketLoadStatus::kSubmitted,
          "cancellation test read was not submitted");
  after_submit.Cancel();
  Require(!ring.TryAcquireFilling().has_value(),
          "cancelled token recycled the slot before original CQE consumption");
  const auto cancelled = source.Wait();
  Require(cancelled.status == DirectPacketLoadStatus::kCancelled,
          "post-submit cancellation is not terminally distinct");
  Require(cancelled.operation_id == submitted.operation_id,
          "cancelled completion lost its operation ID");
  Require(cancelled.stats.read_completions == 1,
          "cancelled slot was released before the original CQE");
  Require(source.in_flight() == 0, "cancelled operation remains in flight");
  Require(ring.TryAcquireFilling().has_value(),
          "cancelled direct operation leaked its slot");
}

void TestChecksumFailurePolicies(const std::string& path) {
  const std::vector<std::uint8_t> logical = Pattern(kAlignment, 71);
  WriteFile(path, logical);
  PacketRing ring(PacketRingConfig{1, kAlignment, kAlignment});
  DirectPackFile source(path, logical.size(), ring, DirectPacketIoConfig{1});
  RequireDirectSupport(source, ring);

  PacketReadRequest optional =
      Request("optional-r1", 0, logical, PacketRequirement::kOptional);
  optional.expected_crc32c ^= 1U;
  Require(source.Submit(optional).status == DirectPacketLoadStatus::kSubmitted,
          "optional corrupt packet was not submitted");
  const auto optional_result = source.Wait();
  Require(
      optional_result.status == DirectPacketLoadStatus::kOptionalUnavailable,
      "optional CRC failure did not remain structured");
  Require(optional_result.failure.code == PacketIoErrorCode::kChecksumMismatch,
          "optional CRC failure code is wrong");
  Require(optional_result.failure.actual_crc32c == Crc32c(logical),
          "optional CRC failure did not report actual checksum");

  PacketReadRequest required = optional;
  required.packet_id = "required-base";
  required.requirement = PacketRequirement::kRequired;
  Require(source.Submit(required).status == DirectPacketLoadStatus::kSubmitted,
          "required corrupt packet was not submitted");
  try {
    static_cast<void>(source.Wait());
    throw std::runtime_error("required CRC failure did not throw");
  } catch (const RequiredPacketLoadError& error) {
    Require(error.failure().code == PacketIoErrorCode::kChecksumMismatch,
            "required CRC failure code is wrong");
  }
  Require(ring.TryAcquireFilling().has_value(),
          "CRC failure policy leaked the ring slot");
}

void TestAlignedAndMisalignedShortCompletions(const std::string& path) {
  const std::vector<std::uint8_t> logical = Pattern(6000, 83);
  std::vector<std::uint8_t> file_bytes(2 * kAlignment, 0);
  std::copy(logical.begin(), logical.end(), file_bytes.begin());
  WriteFile(path, file_bytes);
  PacketRing ring(PacketRingConfig{1, 2 * kAlignment, kAlignment});
  DirectPackFile source(path, file_bytes.size(), ring, DirectPacketIoConfig{1});
  RequireDirectSupport(source, ring);

  DirectPacketIoTestAccess::InjectCompletionResults(
      {static_cast<int>(kAlignment)});
  const auto submitted = source.Submit(Request("aligned-short", 0, logical));
  Require(submitted.status == DirectPacketLoadStatus::kSubmitted,
          "aligned-short read was not submitted");
  auto ready = source.Wait();
  DirectPacketIoTestAccess::ClearCompletionResults();
  Require(ready.status == DirectPacketLoadStatus::kReady,
          "aligned short completion was not safely resumed");
  Require(ReadyBytes(ready) == logical,
          "aligned short-completion retry changed payload bytes");
  Require(ready.stats.read_submissions == 2 &&
              ready.stats.read_completions == 2,
          "aligned short completion did not record both operations");
  Require(ready.stats.file_bytes_returned == 2 * kAlignment,
          "aligned short completion returned-byte accounting is wrong");
  Require(ready.stats.bytes_submitted == 3 * kAlignment,
          "aligned short completion submitted-byte accounting is wrong");
  ready.ready.reset();

  DirectPacketIoTestAccess::InjectCompletionResults({1});
  const auto optional_request = Request(
      "misaligned-short", 0, logical, PacketRequirement::kOptional);
  Require(source.Submit(optional_request).status ==
              DirectPacketLoadStatus::kSubmitted,
          "misaligned-short read was not submitted");
  const auto rejected = source.Wait();
  DirectPacketIoTestAccess::ClearCompletionResults();
  Require(rejected.status == DirectPacketLoadStatus::kOptionalUnavailable,
          "misaligned short completion was silently resumed");
  Require(rejected.failure.code == PacketIoErrorCode::kReadFailed,
          "misaligned short completion failure code is wrong");
  Require(rejected.stats.file_bytes_returned == 1,
          "misaligned short completion byte count is wrong");
}

void TestInjectedReadError(const std::string& path) {
  const std::vector<std::uint8_t> logical = Pattern(kAlignment, 97);
  WriteFile(path, logical);
  PacketRing ring(PacketRingConfig{1, kAlignment, kAlignment});
  DirectPackFile source(path, logical.size(), ring, DirectPacketIoConfig{1});
  RequireDirectSupport(source, ring);

  DirectPacketIoTestAccess::InjectCompletionResults({-EIO});
  const auto request =
      Request("optional-eio", 0, logical, PacketRequirement::kOptional);
  Require(source.Submit(request).status == DirectPacketLoadStatus::kSubmitted,
          "injected-error read was not submitted");
  const auto failed = source.Wait();
  DirectPacketIoTestAccess::ClearCompletionResults();
  Require(failed.status == DirectPacketLoadStatus::kOptionalUnavailable,
          "injected direct error did not use optional policy");
  Require(failed.failure.code == PacketIoErrorCode::kReadFailed &&
              failed.failure.system_error == EIO,
          "injected direct error did not preserve EIO");
  Require(failed.stats.read_completions == 1,
          "injected error released before original CQE");
}

void TestInjectedEintrCompletionResubmits(const std::string& path) {
  const std::vector<std::uint8_t> logical = Pattern(kAlignment, 103);
  WriteFile(path, logical);
  PacketRing ring(PacketRingConfig{1, kAlignment, kAlignment});
  DirectPackFile source(path, logical.size(), ring, DirectPacketIoConfig{1});
  RequireDirectSupport(source, ring);

  DirectPacketIoTestAccess::InjectCompletionResults({-EINTR});
  const auto submitted = source.Submit(Request("eintr", 0, logical));
  Require(submitted.status == DirectPacketLoadStatus::kSubmitted,
          "EINTR completion test read was not submitted");
  auto completed = source.Wait();
  DirectPacketIoTestAccess::ClearCompletionResults();
  Require(completed.status == DirectPacketLoadStatus::kReady,
          "EINTR CQE was not resubmitted");
  Require(ReadyBytes(completed) == logical,
          "EINTR CQE resubmission changed payload bytes");
  Require(completed.stats.read_submissions == 2 &&
              completed.stats.read_completions == 2,
          "EINTR CQE resubmission counters are wrong");
  Require(completed.stats.file_bytes_returned == kAlignment,
          "EINTR CQE counted bytes that were not reported complete");
  Require(completed.stats.bytes_submitted == 2 * kAlignment,
          "EINTR CQE resubmission byte accounting is wrong");
}

void TestLegacyTailIsExplicitBufferedOnly(const std::string& path) {
  const std::vector<std::uint8_t> logical = Pattern(37, 109);
  std::vector<std::uint8_t> legacy(kAlignment + logical.size(), 0);
  std::copy(logical.begin(), logical.end(), legacy.begin() + kAlignment);
  WriteFile(path, legacy);
  PacketRing ring(PacketRingConfig{1, kAlignment, kAlignment});
  DirectPackFile direct(path, legacy.size(), ring, DirectPacketIoConfig{1});
  RequireDirectSupport(direct, ring);
  const PacketReadRequest request = Request("legacy-tail", kAlignment, logical);

  const auto unsupported = direct.Submit(request);
  Require(
      unsupported.status ==
          DirectPacketLoadStatus::kBufferedOnlyLegacyTail,
          "legacy unpadded tail did not request explicit buffered routing");
  Require(unsupported.stats.legacy_unpadded_tail,
          "legacy direct result lacks tail telemetry");
  Require(unsupported.stats.read_submissions == 0 &&
              unsupported.stats.file_bytes_returned == 0,
          "legacy tail was silently submitted as direct I/O");

  PreadPackFile buffered(path, legacy.size());
  const auto loaded = buffered.Load(ring, request);
  Require(loaded.ready.has_value(), "buffered legacy oracle did not load");
  Require(loaded.stats.legacy_unpadded_tail,
          "buffered legacy oracle lacks tail telemetry");
  Require(loaded.stats.file_bytes_returned == logical.size(),
          "buffered legacy returned-byte accounting differs");
  const auto view = loaded.ready->view();
  Require(std::equal(logical.begin(), logical.end(), view.data),
          "buffered legacy bytes differ");
}

void TestMissingStatxIsUnsupportedNotHeuristic(const std::string& path) {
  const std::vector<std::uint8_t> logical = Pattern(kAlignment, 127);
  WriteFile(path, logical);
  PacketRing ring(PacketRingConfig{1, kAlignment, kAlignment});
  DirectPacketIoTestAccess::SuppressStatxDioalignOnce();
  DirectPackFile source(path, logical.size(), ring, DirectPacketIoConfig{1});
  Require(!source.direct_supported(),
          "suppressed STATX_DIOALIGN still enabled direct I/O");
  const auto capability = source.direct_capability();
  Require(!capability.supported,
          "unsupported direct capability reports supported");
  Require(
      capability.unsupported_reason ==
          harpq::native_runtime::DirectIoUnsupportedReason::
              kStatxDioalignNotReported,
      "unsupported direct capability lost its reason");
  Require(capability.system_error == 0,
          "suppressed statx mask unexpectedly reported errno");
  Require(
      source.direct_alignment().source ==
          DirectIoAlignmentSource::kFilesystemBlockSizeFallback,
      "missing STATX_DIOALIGN did not preserve diagnostic provenance");
  const auto result = source.Submit(Request("base", 0, logical));
  Require(result.status == DirectPacketLoadStatus::kDirectUnsupported,
          "missing STATX_DIOALIGN was not explicitly unsupported");
  Require(result.stats.read_submissions == 0,
          "heuristic fallback submitted an O_DIRECT request");
  Require(ring.TryAcquireFilling().has_value(),
          "unsupported direct path consumed a ring slot");
}

void TestDestructorDrainsOriginalCqe(const std::string& path) {
  const std::vector<std::uint8_t> logical = Pattern(kAlignment, 151);
  WriteFile(path, logical);
  PacketRing ring(PacketRingConfig{1, kAlignment, kAlignment});
  {
    DirectPackFile source(path, logical.size(), ring, DirectPacketIoConfig{1});
    RequireDirectSupport(source, ring);
    const auto submitted = source.Submit(Request("base", 0, logical));
    Require(submitted.status == DirectPacketLoadStatus::kSubmitted,
            "teardown-drain read was not submitted");
  }
  Require(ring.TryAcquireFilling().has_value(),
          "backend destructor released/freed before draining original CQE");
}

void TestRequestValidation(const std::string& path) {
  const std::vector<std::uint8_t> logical = Pattern(kAlignment, 163);
  WriteFile(path, logical);
  PacketRing ring(PacketRingConfig{1, kAlignment, kAlignment});
  DirectPackFile source(path, logical.size(), ring, DirectPacketIoConfig{1});
  RequireDirectSupport(source, ring);

  PacketReadRequest bad_offset = Request("bad-offset", 0, logical);
  bad_offset.file_offset = 1;
  RequireThrows<std::invalid_argument>(
      [&] { static_cast<void>(source.Submit(bad_offset)); },
      "misaligned direct offset");

  PacketReadRequest bad_physical = Request("bad-physical", 0, logical);
  bad_physical.physical_bytes = 2 * kAlignment;
  RequireThrows<std::invalid_argument>(
      [&] { static_cast<void>(source.Submit(bad_physical)); },
      "non-exact physical length");

  PacketReadRequest nul_id = Request("valid", 0, logical);
  nul_id.packet_id = std::string("bad\0id", 6);
  RequireThrows<std::invalid_argument>(
      [&] { static_cast<void>(source.Submit(nul_id)); },
      "NUL packet ID");

  RequireThrows<std::logic_error>(
      [&] { static_cast<void>(source.Wait()); },
      "wait without in-flight operation");
}

}  // namespace

int main() {
  try {
    TempDirectory directory;
    try {
      TestDifferentialCanonicalRead(directory.File("differential.pack"));
    } catch (const PacketIoError& error) {
      if (IsSkippableEnvironmentFailure(error)) {
        std::cout << "SKIP: io_uring/O_DIRECT unavailable on test filesystem: "
                  << error.what() << '\n';
        return kSkipExitCode;
      }
      throw;
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      if (message.find("authoritative STATX_DIOALIGN") != std::string::npos) {
        std::cout << "SKIP: " << message << '\n';
        return kSkipExitCode;
      }
      throw;
    }

    TestQueueDepthBackpressureAndMonotonicIds(
        directory.File("queue.pack"));
    TestRingSaturationAndConcurrentOperationMapping(
        directory.File("concurrent.pack"));
    TestCancellationWaitsForOriginalCqe(directory.File("cancel.pack"));
    TestChecksumFailurePolicies(directory.File("checksum.pack"));
    TestAlignedAndMisalignedShortCompletions(directory.File("short.pack"));
    TestInjectedReadError(directory.File("error.pack"));
    TestInjectedEintrCompletionResubmits(directory.File("eintr.pack"));
    TestLegacyTailIsExplicitBufferedOnly(directory.File("legacy.pack"));
    TestMissingStatxIsUnsupportedNotHeuristic(
        directory.File("no-statx.pack"));
    TestDestructorDrainsOriginalCqe(directory.File("drain.pack"));
    TestRequestValidation(directory.File("validation.pack"));
  } catch (const std::exception& error) {
    DirectPacketIoTestAccess::ClearCompletionResults();
    std::cerr << "direct packet I/O test failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "direct packet I/O tests passed\n";
  return 0;
}
