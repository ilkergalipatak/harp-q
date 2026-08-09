#include "harpq/native/packet_io.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "harpq/native/stage_codec.hpp"

namespace harpq::native_runtime {
namespace {

[[nodiscard]] std::size_t CheckedAlignUp(
    std::size_t value, std::size_t alignment) {
  const std::size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const std::size_t increment = alignment - remainder;
  if (value > std::numeric_limits<std::size_t>::max() - increment) {
    throw std::invalid_argument("packet physical length overflows size_t");
  }
  return value + increment;
}

[[nodiscard]] std::uint64_t CheckedAdd(
    std::uint64_t left, std::size_t right, const char* label) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::invalid_argument(std::string(label) + " overflows uint64_t");
  }
  return left + static_cast<std::uint64_t>(right);
}

void ValidateRequirement(PacketRequirement requirement) {
  if (requirement != PacketRequirement::kRequired &&
      requirement != PacketRequirement::kOptional) {
    throw std::invalid_argument("packet requirement is invalid");
  }
}

[[nodiscard]] PacketIoFailure MakeFailure(
    PacketIoErrorCode code,
    const PacketReadRequest& request,
    const PacketReadStats& stats,
    std::string detail,
    int system_error = 0,
    std::uint32_t actual_crc32c = 0) {
  std::ostringstream message;
  message << "packet " << request.packet_id << " from pread backend failed: " << detail;
  if (system_error != 0) {
    message << ": " << std::strerror(system_error);
  }
  return PacketIoFailure{
      code,
      system_error,
      stats.file_bytes_returned,
      request.expected_crc32c,
      actual_crc32c,
      message.str(),
  };
}

[[nodiscard]] PacketLoadResult ResolveFailure(
    const PacketReadRequest& request,
    const PacketReadStats& stats,
    PacketIoFailure failure) {
  if (request.requirement == PacketRequirement::kRequired) {
    throw RequiredPacketLoadError(std::move(failure));
  }
  return PacketLoadResult{
      PacketLoadStatus::kOptionalUnavailable,
      std::nullopt,
      stats,
      std::move(failure),
  };
}

}  // namespace

namespace detail {

struct CancellationState {
  std::atomic<bool> cancelled{false};
};

}  // namespace detail

PacketIoError::PacketIoError(PacketIoFailure failure)
    : std::runtime_error(failure.message), failure_(std::move(failure)) {}

const PacketIoFailure& PacketIoError::failure() const noexcept { return failure_; }

RequiredPacketLoadError::RequiredPacketLoadError(PacketIoFailure failure)
    : PacketIoError(std::move(failure)) {}

CancellationToken::CancellationToken(
    std::shared_ptr<const detail::CancellationState> state) noexcept
    : state_(std::move(state)) {}

bool CancellationToken::IsCancelled() const noexcept {
  return state_ != nullptr && state_->cancelled.load(std::memory_order_acquire);
}

CancellationSource::CancellationSource()
    : state_(std::make_shared<detail::CancellationState>()) {}

CancellationToken CancellationSource::token() const noexcept {
  return CancellationToken(state_);
}

void CancellationSource::Cancel() noexcept {
  if (state_ != nullptr) {
    state_->cancelled.store(true, std::memory_order_release);
  }
}

PreadPackFile::PreadPackFile(
    std::string path, std::uint64_t declared_pack_size)
    : source_name_(std::move(path)), declared_pack_size_(declared_pack_size) {
  if (source_name_.empty() || source_name_.find('\0') != std::string::npos) {
    throw std::invalid_argument("pack path must be non-empty and contain no NUL");
  }
  if (declared_pack_size_ >
      static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
    throw std::invalid_argument("declared pack size does not fit off_t");
  }

  const int descriptor = open(source_name_.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    const int saved_errno = errno;
    PacketIoFailure failure;
    failure.code = PacketIoErrorCode::kOpenFailed;
    failure.system_error = saved_errno;
    failure.message = "cannot open pack " + source_name_ + ": " +
                      std::strerror(saved_errno);
    throw PacketIoError(std::move(failure));
  }

  struct stat status {};
  if (fstat(descriptor, &status) != 0) {
    const int saved_errno = errno;
    static_cast<void>(close(descriptor));
    PacketIoFailure failure;
    failure.code = PacketIoErrorCode::kStatFailed;
    failure.system_error = saved_errno;
    failure.message = "cannot stat pack " + source_name_ + ": " +
                      std::strerror(saved_errno);
    throw PacketIoError(std::move(failure));
  }
  if (status.st_size < 0 ||
      static_cast<std::uint64_t>(status.st_size) != declared_pack_size_) {
    static_cast<void>(close(descriptor));
    PacketIoFailure failure;
    failure.code = PacketIoErrorCode::kPackSizeMismatch;
    std::ostringstream message;
    message << "pack size mismatch for " << source_name_ << ": expected "
            << declared_pack_size_ << ", got " << status.st_size;
    failure.message = message.str();
    throw PacketIoError(std::move(failure));
  }
  file_descriptor_ = descriptor;
}

PreadPackFile::PreadPackFile(
    std::string source_name,
    std::uint64_t declared_pack_size,
    ReadAtFunction read_at)
    : source_name_(std::move(source_name)),
      declared_pack_size_(declared_pack_size),
      injected_read_at_(std::move(read_at)) {
  if (source_name_.empty()) {
    throw std::invalid_argument("injected pack source name cannot be empty");
  }
  if (!injected_read_at_) {
    throw std::invalid_argument("injected pread function cannot be empty");
  }
  if (declared_pack_size_ >
      static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
    throw std::invalid_argument("declared pack size does not fit off_t");
  }
}

PreadPackFile::~PreadPackFile() { Close(); }

PreadPackFile::PreadPackFile(PreadPackFile&& other) noexcept
    : source_name_(std::move(other.source_name_)),
      declared_pack_size_(other.declared_pack_size_),
      file_descriptor_(other.file_descriptor_),
      injected_read_at_(std::move(other.injected_read_at_)) {
  other.declared_pack_size_ = 0;
  other.file_descriptor_ = -1;
}

PreadPackFile& PreadPackFile::operator=(PreadPackFile&& other) noexcept {
  if (this != &other) {
    Close();
    source_name_ = std::move(other.source_name_);
    declared_pack_size_ = other.declared_pack_size_;
    file_descriptor_ = other.file_descriptor_;
    injected_read_at_ = std::move(other.injected_read_at_);
    other.declared_pack_size_ = 0;
    other.file_descriptor_ = -1;
  }
  return *this;
}

const std::string& PreadPackFile::source_name() const noexcept {
  return source_name_;
}

std::uint64_t PreadPackFile::declared_pack_size() const noexcept {
  return declared_pack_size_;
}

std::ptrdiff_t PreadPackFile::ReadAt(
    std::uint8_t* destination,
    std::size_t count,
    std::uint64_t offset,
    int& error_number) {
  if (injected_read_at_) {
    return injected_read_at_(destination, count, offset, error_number);
  }
  if (file_descriptor_ < 0) {
    error_number = EBADF;
    return -1;
  }
  errno = 0;
  const ssize_t result = pread(
      file_descriptor_, destination, count, static_cast<off_t>(offset));
  error_number = result < 0 ? errno : 0;
  return static_cast<std::ptrdiff_t>(result);
}

void PreadPackFile::Close() noexcept {
  if (file_descriptor_ >= 0) {
    static_cast<void>(close(file_descriptor_));
    file_descriptor_ = -1;
  }
}

PacketLoadResult PreadPackFile::Load(
    PacketRing& ring,
    const PacketReadRequest& request,
    const CancellationToken& cancellation) {
  if (request.packet_id.empty() ||
      request.packet_id.find('\0') != std::string::npos) {
    throw std::invalid_argument("packet ID must be non-empty and contain no NUL");
  }
  ValidateRequirement(request.requirement);
  if (ring.io_alignment() == 0 || ring.slot_capacity() == 0) {
    throw std::invalid_argument("packet ring is moved-from or invalid");
  }
  if (request.logical_bytes == 0) {
    throw std::invalid_argument("packet logical_bytes must be positive");
  }
  if (request.file_offset % ring.io_alignment() != 0) {
    throw std::invalid_argument("packet file offset is not I/O aligned");
  }
  const std::size_t expected_physical =
      CheckedAlignUp(request.logical_bytes, ring.io_alignment());
  if (request.physical_bytes != expected_physical) {
    throw std::invalid_argument(
        "packet physical_bytes must exactly equal aligned logical_bytes");
  }
  if (request.physical_bytes > ring.slot_capacity()) {
    throw std::invalid_argument("packet physical extent exceeds ring slot capacity");
  }

  const std::uint64_t logical_end =
      CheckedAdd(request.file_offset, request.logical_bytes, "packet logical extent");
  const std::uint64_t physical_end =
      CheckedAdd(request.file_offset, request.physical_bytes, "packet physical extent");
  if (physical_end >
      static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
    throw std::invalid_argument("packet physical extent does not fit off_t");
  }
  if (logical_end > declared_pack_size_) {
    throw std::invalid_argument("packet logical extent exceeds declared pack size");
  }

  bool legacy_unpadded_tail = false;
  std::size_t expected_file_bytes = request.physical_bytes;
  if (physical_end > declared_pack_size_) {
    if (logical_end != declared_pack_size_) {
      throw std::invalid_argument(
          "packet physical extent crosses a partially padded or malformed pack tail");
    }
    legacy_unpadded_tail = true;
    expected_file_bytes = request.logical_bytes;
  }

  PacketReadStats stats{
      request.logical_bytes,
      request.physical_bytes,
      0,
      0,
      legacy_unpadded_tail,
  };
  if (cancellation.IsCancelled()) {
    return PacketLoadResult{
        PacketLoadStatus::kCancelled, std::nullopt, stats, {}};
  }

  std::optional<FillingSlotLease> filling = ring.TryAcquireFilling();
  if (!filling.has_value()) {
    if (cancellation.IsCancelled()) {
      return PacketLoadResult{
          PacketLoadStatus::kCancelled, std::nullopt, stats, {}};
    }
    return PacketLoadResult{
        PacketLoadStatus::kWouldBlock, std::nullopt, stats, {}};
  }
  if (cancellation.IsCancelled()) {
    return PacketLoadResult{
        PacketLoadStatus::kCancelled, std::nullopt, stats, {}};
  }

  const MutablePacketBytes destination = filling->bytes();
  std::size_t completed = 0;
  while (completed < expected_file_bytes) {
    const std::size_t request_remaining = request.physical_bytes - completed;
    const std::size_t syscall_bytes = std::min(
        request_remaining,
        static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
    int read_errno = 0;
    const std::ptrdiff_t read_result = ReadAt(
        destination.data + completed,
        syscall_bytes,
        request.file_offset + static_cast<std::uint64_t>(completed),
        read_errno);
    ++stats.pread_calls;
    if (read_result < 0) {
      if (read_errno == EINTR) {
        if (cancellation.IsCancelled()) {
          return PacketLoadResult{
              PacketLoadStatus::kCancelled, std::nullopt, stats, {}};
        }
        continue;
      }
      return ResolveFailure(
          request,
          stats,
          MakeFailure(
              PacketIoErrorCode::kReadFailed,
              request,
              stats,
              "pread syscall",
              read_errno));
    }
    if (read_result == 0) {
      return ResolveFailure(
          request,
          stats,
          MakeFailure(
              PacketIoErrorCode::kUnexpectedEof,
              request,
              stats,
              "unexpected EOF"));
    }
    const auto read_bytes = static_cast<std::size_t>(read_result);
    if (read_bytes > syscall_bytes || read_bytes > expected_file_bytes - completed) {
      return ResolveFailure(
          request,
          stats,
          MakeFailure(
              PacketIoErrorCode::kReadFailed,
              request,
              stats,
              "pread returned more bytes than the declared immutable extent"));
    }
    completed += read_bytes;
    stats.file_bytes_returned = completed;
    if (cancellation.IsCancelled()) {
      return PacketLoadResult{
          PacketLoadStatus::kCancelled, std::nullopt, stats, {}};
    }
  }

  if (legacy_unpadded_tail && request.physical_bytes > request.logical_bytes) {
    std::memset(
        destination.data + request.logical_bytes,
        0,
        request.physical_bytes - request.logical_bytes);
  }
  if (cancellation.IsCancelled()) {
    return PacketLoadResult{
        PacketLoadStatus::kCancelled, std::nullopt, stats, {}};
  }

  const std::uint32_t actual_crc32c = Crc32c(destination.data, request.logical_bytes);
  if (actual_crc32c != request.expected_crc32c) {
    return ResolveFailure(
        request,
        stats,
        MakeFailure(
            PacketIoErrorCode::kChecksumMismatch,
            request,
            stats,
            "CRC32C mismatch",
            0,
            actual_crc32c));
  }
  // This is the cancellation/publish linearization point. Once Publish begins,
  // a fully verified Ready packet is observable and a later cancellation must
  // be handled by releasing that ready lease rather than relabeling the read.
  if (cancellation.IsCancelled()) {
    return PacketLoadResult{
        PacketLoadStatus::kCancelled, std::nullopt, stats, {}};
  }

  ReadySlotLease ready = std::move(*filling).Publish(PacketSlotMetadata{
      request.packet_id,
      request.logical_bytes,
      request.physical_bytes,
      actual_crc32c,
  });
  return PacketLoadResult{
      PacketLoadStatus::kReady,
      std::optional<ReadySlotLease>(std::move(ready)),
      stats,
      {},
  };
}

}  // namespace harpq::native_runtime
