#include "harpq/native/packet_io.hpp"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "harpq/native/stage_codec.hpp"

namespace harpq::native_runtime {

class PacketIoTestAccess final {
 public:
  using Callback = std::function<std::ptrdiff_t(
      std::uint8_t*, std::size_t, std::uint64_t, int&)>;

  static PreadPackFile MakeFake(
      std::uint64_t declared_pack_size, Callback callback) {
    return PreadPackFile(
        "injected-pread-source", declared_pack_size, std::move(callback));
  }
};

}  // namespace harpq::native_runtime

namespace {

using harpq::native_runtime::CancellationSource;
using harpq::native_runtime::Crc32c;
using harpq::native_runtime::PacketIoError;
using harpq::native_runtime::PacketIoErrorCode;
using harpq::native_runtime::PacketIoTestAccess;
using harpq::native_runtime::PacketLoadStatus;
using harpq::native_runtime::PacketReadRequest;
using harpq::native_runtime::PacketRequirement;
using harpq::native_runtime::PacketRing;
using harpq::native_runtime::PacketRingConfig;
using harpq::native_runtime::PreadPackFile;
using harpq::native_runtime::RequiredPacketLoadError;

constexpr std::size_t kAlignment = 4096;
int failures = 0;

void Check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Exception, typename Callable>
void ExpectThrows(Callable&& callable, const std::string& message) {
  try {
    callable();
    Check(false, message + " (no exception)");
  } catch (const Exception&) {
  } catch (const std::exception& error) {
    Check(false, message + " (wrong exception: " + error.what() + ")");
  }
}

class TempDirectory final {
 public:
  TempDirectory() {
    char pattern[] = "/tmp/harpq-packet-io.XXXXXX";
    char* created = mkdtemp(pattern);
    if (created == nullptr) {
      throw std::runtime_error(
          std::string("mkdtemp failed: ") + std::strerror(errno));
    }
    path_ = created;
  }

  ~TempDirectory() {
    for (const std::string& path : files_) {
      static_cast<void>(unlink(path.c_str()));
    }
    static_cast<void>(rmdir(path_.c_str()));
  }

  TempDirectory(const TempDirectory&) = delete;
  TempDirectory& operator=(const TempDirectory&) = delete;

  [[nodiscard]] std::string File(const std::string& name) {
    const std::string result = path_ + "/" + name;
    files_.push_back(result);
    return result;
  }

 private:
  std::string path_;
  std::vector<std::string> files_;
};

void WriteFile(const std::string& path, const std::vector<std::uint8_t>& bytes) {
  const int descriptor = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
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
          std::string("write test file failed: ") + std::strerror(saved_errno));
    }
    completed += static_cast<std::size_t>(result);
  }
  if (close(descriptor) != 0) {
    throw std::runtime_error(
        std::string("close test file failed: ") + std::strerror(errno));
  }
}

std::vector<std::uint8_t> Pattern(std::size_t size, std::uint8_t seed = 17) {
  std::vector<std::uint8_t> result(size);
  for (std::size_t index = 0; index < size; ++index) {
    result[index] = static_cast<std::uint8_t>(seed + (index * 29U) % 251U);
  }
  return result;
}

PacketReadRequest Request(
    std::string packet_id,
    std::uint64_t offset,
    const std::vector<std::uint8_t>& logical,
    PacketRequirement requirement = PacketRequirement::kRequired,
    std::size_t physical_bytes = kAlignment) {
  return PacketReadRequest{
      std::move(packet_id),
      offset,
      logical.size(),
      physical_bytes,
      Crc32c(logical),
      requirement,
  };
}

void TestRealPaddedRead() {
  TempDirectory directory;
  const std::string path = directory.File("padded.pack");
  const std::vector<std::uint8_t> payload = Pattern(3000);
  std::vector<std::uint8_t> file_bytes(kAlignment * 2, 0);
  std::copy(payload.begin(), payload.end(), file_bytes.begin() + kAlignment);
  WriteFile(path, file_bytes);

  PreadPackFile source(path, file_bytes.size());
  PacketRing ring(PacketRingConfig{1, kAlignment, kAlignment});
  auto result = source.Load(ring, Request("base", kAlignment, payload));
  Check(result.status == PacketLoadStatus::kReady, "padded read becomes ready");
  Check(result.ready.has_value(), "padded read owns ready lease");
  Check(!result.stats.legacy_unpadded_tail, "padded read is not legacy");
  Check(result.stats.logical_bytes == payload.size(), "padded logical byte count");
  Check(
      result.stats.physical_bytes_requested == kAlignment &&
          result.stats.file_bytes_returned == kAlignment,
      "padded physical and returned byte counts");
  if (result.ready.has_value()) {
    const auto view = result.ready->view();
    Check(
        std::equal(payload.begin(), payload.end(), view.data),
        "padded payload bytes");
    Check(view.logical_bytes == payload.size(), "ready logical extent");
  }
}

void TestLegacyUnpaddedTailIsZeroFilled() {
  TempDirectory directory;
  const std::string path = directory.File("legacy.pack");
  const std::vector<std::uint8_t> payload = Pattern(37, 41);
  std::vector<std::uint8_t> file_bytes(kAlignment + payload.size(), 0);
  std::copy(payload.begin(), payload.end(), file_bytes.begin() + kAlignment);
  WriteFile(path, file_bytes);

  PreadPackFile source(path, file_bytes.size());
  PacketRing ring(PacketRingConfig{1, kAlignment, kAlignment});
  auto dirty = ring.TryAcquireFilling();
  Check(dirty.has_value(), "acquire slot to dirty legacy suffix");
  if (dirty.has_value()) {
    const auto bytes = dirty->bytes();
    std::memset(bytes.data, 0xA5, bytes.capacity);
    dirty->Cancel();
  }

  auto result = source.Load(
      ring, Request("legacy-base", kAlignment, payload));
  Check(result.status == PacketLoadStatus::kReady, "legacy tail remains readable");
  Check(result.stats.legacy_unpadded_tail, "legacy tail is reported");
  Check(
      result.stats.physical_bytes_requested == kAlignment &&
          result.stats.file_bytes_returned == payload.size(),
      "legacy logical file bytes remain separate from physical request");
  if (result.ready.has_value()) {
    const auto view = result.ready->view();
    Check(
        std::equal(payload.begin(), payload.end(), view.data),
        "legacy payload bytes");
    const bool suffix_zero = std::all_of(
        view.data + payload.size(),
        view.data + kAlignment,
        [](std::uint8_t value) { return value == 0; });
    Check(suffix_zero, "legacy unread suffix is deterministically zero-filled");
  }
}

void TestInjectedEintrAndPartialReads() {
  const auto bytes = std::make_shared<std::vector<std::uint8_t>>(Pattern(kAlignment, 73));
  const auto calls = std::make_shared<std::size_t>(0);
  auto source = PacketIoTestAccess::MakeFake(
      bytes->size(),
      [bytes, calls](std::uint8_t* destination,
                     std::size_t count,
                     std::uint64_t offset,
                     int& error_number) -> std::ptrdiff_t {
        const std::size_t call = (*calls)++;
        if (call == 0) {
          error_number = EINTR;
          return -1;
        }
        const std::size_t available = bytes->size() - static_cast<std::size_t>(offset);
        const std::size_t limit = call == 1 ? 97 : count;
        const std::size_t copied = std::min({count, available, limit});
        std::memcpy(destination, bytes->data() + offset, copied);
        error_number = 0;
        return static_cast<std::ptrdiff_t>(copied);
      });
  PacketRing ring(PacketRingConfig{1, kAlignment, kAlignment});
  auto result = source.Load(ring, Request("partial", 0, *bytes));
  Check(result.status == PacketLoadStatus::kReady, "EINTR and partial read recover");
  Check(result.stats.pread_calls == 3, "EINTR and partial calls are counted");
  Check(result.stats.file_bytes_returned == kAlignment, "partial reads complete extent");
}

void TestUnexpectedEofAndReadFailurePolicies() {
  const auto bytes = std::make_shared<std::vector<std::uint8_t>>(Pattern(1000, 91));
  const auto call_index = std::make_shared<std::size_t>(0);
  auto eof_source = PacketIoTestAccess::MakeFake(
      kAlignment,
      [bytes, call_index](std::uint8_t* destination,
                          std::size_t count,
                          std::uint64_t offset,
                          int& error_number) -> std::ptrdiff_t {
        if ((*call_index)++ != 0) {
          error_number = 0;
          return 0;
        }
        const std::size_t copied = std::min<std::size_t>(100, count);
        std::memcpy(destination, bytes->data() + offset, copied);
        error_number = 0;
        return static_cast<std::ptrdiff_t>(copied);
      });
  PacketRing ring(PacketRingConfig{1, kAlignment, kAlignment});
  try {
    static_cast<void>(eof_source.Load(ring, Request("required-base", 0, *bytes)));
    Check(false, "required premature EOF throws");
  } catch (const RequiredPacketLoadError& error) {
    Check(
        error.failure().code == PacketIoErrorCode::kUnexpectedEof,
        "required EOF error code");
    Check(error.failure().bytes_completed == 100, "required EOF completed bytes");
  }
  Check(ring.TryAcquireFilling().has_value(), "required EOF releases filling slot");

  auto io_source = PacketIoTestAccess::MakeFake(
      kAlignment,
      [](std::uint8_t*, std::size_t, std::uint64_t, int& error_number) {
        error_number = EIO;
        return static_cast<std::ptrdiff_t>(-1);
      });
  auto optional_result = io_source.Load(
      ring,
      Request("optional-r1", 0, *bytes, PacketRequirement::kOptional));
  Check(
      optional_result.status == PacketLoadStatus::kOptionalUnavailable,
      "optional read failure is structured fallback");
  Check(
      optional_result.failure.code == PacketIoErrorCode::kReadFailed &&
          optional_result.failure.system_error == EIO,
      "optional read failure preserves errno");
  Check(ring.TryAcquireFilling().has_value(), "optional read error releases slot");
}

void TestChecksumPolicies() {
  const auto bytes = std::make_shared<std::vector<std::uint8_t>>(Pattern(kAlignment, 111));
  const auto make_source = [bytes] {
    return PacketIoTestAccess::MakeFake(
        kAlignment,
        [bytes](std::uint8_t* destination,
                std::size_t count,
                std::uint64_t offset,
                int& error_number) {
          std::memcpy(destination, bytes->data() + offset, count);
          error_number = 0;
          return static_cast<std::ptrdiff_t>(count);
        });
  };
  PacketRing ring(PacketRingConfig{1, kAlignment, kAlignment});
  PacketReadRequest optional = Request(
      "optional-r1", 0, *bytes, PacketRequirement::kOptional);
  optional.expected_crc32c ^= 1U;
  auto optional_source = make_source();
  auto result = optional_source.Load(ring, optional);
  Check(
      result.status == PacketLoadStatus::kOptionalUnavailable,
      "optional CRC corruption falls back");
  Check(
      result.failure.code == PacketIoErrorCode::kChecksumMismatch &&
          result.failure.actual_crc32c == Crc32c(*bytes),
      "optional CRC failure reports both checksums");

  PacketReadRequest required = optional;
  required.packet_id = "required-base";
  required.requirement = PacketRequirement::kRequired;
  auto required_source = make_source();
  try {
    static_cast<void>(required_source.Load(ring, required));
    Check(false, "required CRC corruption throws");
  } catch (const RequiredPacketLoadError& error) {
    Check(
        error.failure().code == PacketIoErrorCode::kChecksumMismatch,
        "required CRC error code");
  }
  Check(ring.TryAcquireFilling().has_value(), "CRC errors do not leak ring slots");
}

void TestCancellationAndBackpressure() {
  const auto bytes = std::make_shared<std::vector<std::uint8_t>>(Pattern(kAlignment, 137));
  const auto calls = std::make_shared<std::size_t>(0);
  auto source = PacketIoTestAccess::MakeFake(
      kAlignment,
      [bytes, calls](std::uint8_t* destination,
                     std::size_t count,
                     std::uint64_t offset,
                     int& error_number) {
        ++*calls;
        std::memcpy(destination, bytes->data() + offset, count);
        error_number = 0;
        return static_cast<std::ptrdiff_t>(count);
      });
  PacketRing ring(PacketRingConfig{1, kAlignment, kAlignment});

  CancellationSource pre_cancel;
  pre_cancel.Cancel();
  auto cancelled = source.Load(
      ring, Request("base", 0, *bytes), pre_cancel.token());
  Check(cancelled.status == PacketLoadStatus::kCancelled, "pre-read cancellation");
  Check(*calls == 0, "pre-read cancellation avoids syscall");

  CancellationSource during_read;
  auto cancelling_source = PacketIoTestAccess::MakeFake(
      kAlignment,
      [bytes, &during_read](std::uint8_t* destination,
                            std::size_t count,
                            std::uint64_t offset,
                            int& error_number) {
        std::memcpy(destination, bytes->data() + offset, count);
        during_read.Cancel();
        error_number = 0;
        return static_cast<std::ptrdiff_t>(count);
      });
  auto cancelled_after_read = cancelling_source.Load(
      ring, Request("base", 0, *bytes), during_read.token());
  Check(
      cancelled_after_read.status == PacketLoadStatus::kCancelled,
      "cancellation before publish releases filling slot");

  auto held = ring.TryAcquireFilling();
  Check(held.has_value(), "hold only ring slot");
  const std::size_t calls_before_backpressure = *calls;
  auto blocked = source.Load(
      ring,
      Request("optional-r1", 0, *bytes, PacketRequirement::kOptional));
  Check(blocked.status == PacketLoadStatus::kWouldBlock, "ring full is backpressure");
  Check(
      blocked.status != PacketLoadStatus::kOptionalUnavailable,
      "backpressure is not optional fallback");
  Check(*calls == calls_before_backpressure, "backpressure avoids pread");
}

void TestRequestAndPackValidation() {
  const auto bytes = std::make_shared<std::vector<std::uint8_t>>(Pattern(kAlignment, 163));
  const auto callback = [bytes](std::uint8_t* destination,
                                std::size_t count,
                                std::uint64_t offset,
                                int& error_number) {
    std::memcpy(destination, bytes->data() + offset, count);
    error_number = 0;
    return static_cast<std::ptrdiff_t>(count);
  };
  PacketRing ring(PacketRingConfig{1, kAlignment, kAlignment});
  auto source = PacketIoTestAccess::MakeFake(kAlignment, callback);

  PacketReadRequest bad_physical = Request("bad", 0, *bytes);
  bad_physical.physical_bytes = 8192;
  ExpectThrows<std::invalid_argument>(
      [&] { static_cast<void>(source.Load(ring, bad_physical)); },
      "reject non-exact physical length");

  PacketReadRequest align_overflow{
      "overflow-align",
      0,
      std::numeric_limits<std::size_t>::max(),
      0,
      0,
      PacketRequirement::kRequired,
  };
  ExpectThrows<std::invalid_argument>(
      [&] { static_cast<void>(source.Load(ring, align_overflow)); },
      "reject physical align-up overflow");

  PacketReadRequest nul_id = Request("valid", 0, *bytes);
  nul_id.packet_id = std::string("bad\0id", 6);
  ExpectThrows<std::invalid_argument>(
      [&] { static_cast<void>(source.Load(ring, nul_id)); },
      "reject packet ID containing NUL");

  std::vector<std::uint8_t> small = Pattern(1000);
  PacketReadRequest misaligned = Request("bad", 1, small);
  ExpectThrows<std::invalid_argument>(
      [&] { static_cast<void>(source.Load(ring, misaligned)); },
      "reject misaligned file offset");

  auto partial_tail_source = PacketIoTestAccess::MakeFake(1001, callback);
  ExpectThrows<std::invalid_argument>(
      [&] {
        static_cast<void>(partial_tail_source.Load(
            ring, Request("bad-tail", 0, small)));
      },
      "reject partially padded tail");

  const auto off_max = static_cast<std::uint64_t>(
      std::numeric_limits<off_t>::max());
  const std::uint64_t aligned_near_end = off_max & ~std::uint64_t{kAlignment - 1};
  auto huge_source = PacketIoTestAccess::MakeFake(off_max, callback);
  ExpectThrows<std::invalid_argument>(
      [&] {
        static_cast<void>(huge_source.Load(
            ring, Request("overflow", aligned_near_end, small)));
      },
      "reject extent beyond off_t");

  auto over_return_source = PacketIoTestAccess::MakeFake(
      kAlignment,
      [](std::uint8_t*, std::size_t count, std::uint64_t, int& error_number) {
        error_number = 0;
        return static_cast<std::ptrdiff_t>(count + 1);
      });
  auto over_return = over_return_source.Load(
      ring,
      Request("optional-over-return", 0, *bytes, PacketRequirement::kOptional));
  Check(
      over_return.status == PacketLoadStatus::kOptionalUnavailable &&
          over_return.failure.code == PacketIoErrorCode::kReadFailed,
      "reject injected pread over-return without leaking slot");

  TempDirectory directory;
  const std::string path = directory.File("size.pack");
  WriteFile(path, small);
  try {
    PreadPackFile wrong_size(path, small.size() + 1);
    Check(false, "pack size mismatch throws");
  } catch (const PacketIoError& error) {
    Check(
        error.failure().code == PacketIoErrorCode::kPackSizeMismatch,
        "pack size mismatch code");
  }
}

}  // namespace

int main() {
  TestRealPaddedRead();
  TestLegacyUnpaddedTailIsZeroFilled();
  TestInjectedEintrAndPartialReads();
  TestUnexpectedEofAndReadFailurePolicies();
  TestChecksumPolicies();
  TestCancellationAndBackpressure();
  TestRequestAndPackValidation();
  if (failures != 0) {
    std::cerr << failures << " packet I/O test(s) failed\n";
    return 1;
  }
  std::cout << "packet I/O tests passed\n";
  return 0;
}
