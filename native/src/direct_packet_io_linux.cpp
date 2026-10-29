#include "harpq/native/direct_packet_io.hpp"

#if !defined(__linux__)
#error "direct_packet_io_linux.cpp is Linux-only"
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <fcntl.h>
#include <liburing.h>
#include <linux/stat.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "harpq/native/stage_codec.hpp"

namespace harpq::native_runtime {
namespace {

std::mutex g_completion_fault_mutex;
std::deque<int> g_completion_faults;
std::atomic<bool> g_suppress_statx_dioalign_once{false};

[[nodiscard]] int InjectedCompletionResult(int actual_result) {
  std::lock_guard<std::mutex> lock(g_completion_fault_mutex);
  if (g_completion_faults.empty()) {
    return actual_result;
  }
  const int injected = g_completion_faults.front();
  g_completion_faults.pop_front();
  return injected;
}

[[nodiscard]] bool IsValidRequirement(PacketRequirement requirement) noexcept {
  return requirement == PacketRequirement::kRequired ||
         requirement == PacketRequirement::kOptional;
}

[[nodiscard]] std::size_t CheckedAlignUp(
    std::size_t value, std::size_t alignment) {
  if (alignment == 0) {
    throw std::invalid_argument("direct I/O alignment must be positive");
  }
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

[[nodiscard]] std::size_t CheckedLeastCommonMultiple(
    std::size_t left, std::size_t right) {
  if (left == 0 || right == 0) {
    throw std::invalid_argument("fallback direct-I/O alignment is zero");
  }
  const std::size_t divisor = std::gcd(left, right);
  const std::size_t reduced = left / divisor;
  if (reduced > std::numeric_limits<std::size_t>::max() / right) {
    throw std::invalid_argument("fallback direct-I/O alignment overflows size_t");
  }
  return reduced * right;
}

[[nodiscard]] PacketIoFailure MakeFailure(
    PacketIoErrorCode code,
    const PacketReadRequest& request,
    const DirectPacketReadStats& stats,
    std::string detail,
    int system_error = 0,
    std::uint32_t actual_crc32c = 0) {
  std::ostringstream message;
  message << "packet " << request.packet_id
          << " from io_uring O_DIRECT backend failed: " << detail;
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

[[noreturn]] void ThrowSourceError(
    PacketIoErrorCode code,
    const std::string& message,
    int system_error = 0) {
  PacketIoFailure failure;
  failure.code = code;
  failure.system_error = system_error;
  failure.message = message;
  throw PacketIoError(std::move(failure));
}

[[noreturn]] void AbortForUnprovenDirectIoSafety() noexcept {
  // A packet slot must never be released while the kernel could still write
  // into it. If the CQ ring cannot prove completion during teardown, fail
  // closed rather than risk recycling or freeing the backing allocation.
  std::abort();
}

}  // namespace

namespace detail {

class DirectPackFileImpl final {
 public:
  DirectPackFileImpl(std::string path,
                     std::uint64_t declared_pack_size,
                     PacketRing& packet_ring,
                     DirectPacketIoConfig config)
      : source_name_(std::move(path)),
        declared_pack_size_(declared_pack_size),
        packet_ring_(packet_ring),
        config_(config),
        owner_pid_(getpid()) {
    try {
      Initialize();
    } catch (...) {
      CleanupWithoutPending();
      throw;
    }
  }

  ~DirectPackFileImpl() noexcept {
    if (getpid() != owner_pid_) {
      // io_uring_ring_dontfork deliberately removed the ring mappings in the
      // child. DirectPackFile releases this impl instead of invoking this
      // destructor there, so reaching this branch is a protocol violation.
      AbortForUnprovenDirectIoSafety();
    }
    DrainForTeardown();
    CleanupWithoutPending();
  }

  DirectPackFileImpl(const DirectPackFileImpl&) = delete;
  DirectPackFileImpl& operator=(const DirectPackFileImpl&) = delete;

  [[nodiscard]] bool IsOwnerProcess() const noexcept {
    return getpid() == owner_pid_;
  }

  void RequireOwnerProcess() const {
    if (!IsOwnerProcess()) {
      throw std::logic_error(
          "DirectPackFile cannot be used after fork; construct a new backend "
          "after exec in the child process");
    }
  }

  [[nodiscard]] const std::string& source_name() const noexcept {
    return source_name_;
  }

  [[nodiscard]] std::uint64_t declared_pack_size() const noexcept {
    return declared_pack_size_;
  }

  [[nodiscard]] std::uint32_t queue_depth() const noexcept {
    return config_.queue_depth;
  }

  [[nodiscard]] std::size_t in_flight() const noexcept {
    return pending_.size();
  }

  [[nodiscard]] DirectIoAlignment direct_alignment() const noexcept {
    return alignment_;
  }

  [[nodiscard]] DirectIoCapability direct_capability() const noexcept {
    return capability_;
  }

  [[nodiscard]] bool direct_supported() const noexcept {
    return direct_supported_;
  }

  [[nodiscard]] std::size_t ring_slot_count() const noexcept {
    return ring_slot_count_;
  }

  [[nodiscard]] DirectPacketLoadResult Submit(
      const PacketReadRequest& request,
      const CancellationToken& cancellation) {
    RequireOwnerProcess();
    ValidateRequestShape(request);

    const std::uint64_t logical_end = CheckedAdd(
        request.file_offset, request.logical_bytes, "packet logical extent");
    const std::uint64_t physical_end = CheckedAdd(
        request.file_offset, request.physical_bytes, "packet physical extent");
    if (physical_end >
        static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
      throw std::invalid_argument("packet physical extent does not fit off_t");
    }
    if (logical_end > declared_pack_size_) {
      throw std::invalid_argument(
          "packet logical extent exceeds declared pack size");
    }

    DirectPacketReadStats stats;
    stats.logical_bytes = request.logical_bytes;
    stats.physical_bytes_requested = request.physical_bytes;

    if (physical_end > declared_pack_size_) {
      if (logical_end != declared_pack_size_) {
        throw std::invalid_argument(
            "packet physical extent crosses a partially padded or malformed "
            "pack tail");
      }
      stats.legacy_unpadded_tail = true;
      if (cancellation.IsCancelled()) {
        return ImmediateResult(
            DirectPacketLoadStatus::kCancelled, std::move(stats));
      }
      return ImmediateResult(
          DirectPacketLoadStatus::kBufferedOnlyLegacyTail, std::move(stats));
    }

    if (cancellation.IsCancelled()) {
      return ImmediateResult(
          DirectPacketLoadStatus::kCancelled, std::move(stats));
    }
    if (!direct_supported_) {
      DirectPacketLoadResult unsupported = ImmediateResult(
          DirectPacketLoadStatus::kDirectUnsupported, std::move(stats));
      unsupported.failure.code = PacketIoErrorCode::kNone;
      unsupported.failure.system_error = capability_.system_error;
      unsupported.failure.message = DirectUnsupportedMessage();
      return unsupported;
    }
    ValidateDirectGeometry(request);
    if (pending_.size() >= config_.queue_depth) {
      return ImmediateResult(
          DirectPacketLoadStatus::kWouldBlock, std::move(stats));
    }

    std::optional<FillingSlotLease> filling =
        packet_ring_.TryAcquireFilling();
    if (!filling.has_value()) {
      if (cancellation.IsCancelled()) {
        return ImmediateResult(
            DirectPacketLoadStatus::kCancelled, std::move(stats));
      }
      return ImmediateResult(
          DirectPacketLoadStatus::kWouldBlock, std::move(stats));
    }
    if (cancellation.IsCancelled()) {
      return ImmediateResult(
          DirectPacketLoadStatus::kCancelled, std::move(stats));
    }

    const MutablePacketBytes destination = filling->bytes();
    ValidateDestination(destination, request.physical_bytes);
    const std::uint64_t operation_id = NextOperationId();

    PendingRead pending{
        operation_id,
        request,
        cancellation,
        std::move(*filling),
        std::move(stats),
        0,
    };
    pending.stats.used_ring_slot_buffer = true;
    const auto insertion = pending_.emplace(operation_id, std::move(pending));
    if (!insertion.second) {
      AbortForUnprovenDirectIoSafety();
    }
    PendingRead& tracked = insertion.first->second;
    tracked.submitted_at = std::chrono::steady_clock::now();
    if (!SubmitRead(tracked)) {
      DirectPacketReadStats blocked_stats = tracked.stats;
      pending_.erase(insertion.first);
      return ImmediateResult(
          DirectPacketLoadStatus::kWouldBlock, std::move(blocked_stats));
    }

    DirectPacketLoadResult submitted;
    submitted.status = DirectPacketLoadStatus::kSubmitted;
    submitted.operation_id = operation_id;
    submitted.stats = tracked.stats;
    submitted.submitted_at = tracked.submitted_at;
    return submitted;
  }

  [[nodiscard]] std::optional<DirectPacketLoadResult> Poll() {
    RequireOwnerProcess();
    if (!direct_supported_ || pending_.empty()) {
      return std::nullopt;
    }
    for (;;) {
      io_uring_cqe* cqe = nullptr;
      const int status = io_uring_peek_cqe(&uring_, &cqe);
      if (status == -EAGAIN) {
        return std::nullopt;
      }
      if (status == -EINTR) {
        continue;
      }
      if (status < 0 || cqe == nullptr) {
        ThrowSourceError(
            PacketIoErrorCode::kReadFailed,
            "io_uring_peek_cqe failed for " + source_name_ + ": " +
                std::strerror(status < 0 ? -status : EIO),
            status < 0 ? -status : EIO);
      }
      const std::uint64_t operation_id = io_uring_cqe_get_data64(cqe);
      const int completion_result = InjectedCompletionResult(cqe->res);
      const unsigned int completion_flags = cqe->flags;
      io_uring_cqe_seen(&uring_, cqe);
      std::optional<DirectPacketLoadResult> terminal =
          HandleReadCompletion(
              operation_id, completion_result, completion_flags, false);
      if (terminal.has_value()) {
        return terminal;
      }
    }
  }

  [[nodiscard]] DirectPacketLoadResult Wait() {
    RequireOwnerProcess();
    if (pending_.empty()) {
      throw std::logic_error(
          "DirectPackFile::Wait called with no packet operation in flight");
    }
    for (;;) {
      io_uring_cqe* cqe = nullptr;
      int status = 0;
      do {
        status = io_uring_wait_cqe(&uring_, &cqe);
      } while (status == -EINTR);
      if (status < 0 || cqe == nullptr) {
        ThrowSourceError(
            PacketIoErrorCode::kReadFailed,
            "io_uring_wait_cqe failed for " + source_name_ + ": " +
                std::strerror(status < 0 ? -status : EIO),
            status < 0 ? -status : EIO);
      }
      const std::uint64_t operation_id = io_uring_cqe_get_data64(cqe);
      const int completion_result = InjectedCompletionResult(cqe->res);
      const unsigned int completion_flags = cqe->flags;
      io_uring_cqe_seen(&uring_, cqe);
      std::optional<DirectPacketLoadResult> terminal =
          HandleReadCompletion(
              operation_id, completion_result, completion_flags, false);
      if (terminal.has_value()) {
        return std::move(*terminal);
      }
    }
  }

 private:
  struct PendingRead {
    std::uint64_t operation_id;
    PacketReadRequest request;
    CancellationToken cancellation;
    FillingSlotLease filling;
    DirectPacketReadStats stats;
    std::size_t completed_bytes;
    std::chrono::steady_clock::time_point submitted_at{};
  };

  static_assert(
      std::is_nothrow_move_constructible<PendingRead>::value,
      "terminal CQE handling requires allocation-free PendingRead transfer");

  void Initialize() {
    if (source_name_.empty() || source_name_.find('\0') != std::string::npos) {
      throw std::invalid_argument(
          "pack path must be non-empty and contain no NUL");
    }
    if (declared_pack_size_ >
        static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
      throw std::invalid_argument("declared pack size does not fit off_t");
    }
    if (config_.queue_depth == 0) {
      throw std::invalid_argument("direct I/O queue_depth must be positive");
    }
    if (packet_ring_.slot_count() == 0 ||
        packet_ring_.slot_capacity() == 0 ||
        packet_ring_.io_alignment() == 0) {
      throw std::invalid_argument("packet ring is moved-from or invalid");
    }
    ring_slot_count_ = packet_ring_.slot_count();

    const int descriptor = open(source_name_.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
      const int saved_errno = errno;
      ThrowSourceError(
          PacketIoErrorCode::kOpenFailed,
          "cannot open pack " + source_name_ + ": " +
              std::strerror(saved_errno),
          saved_errno);
    }
    file_descriptor_ = descriptor;

    struct stat status {};
    if (fstat(file_descriptor_, &status) != 0) {
      const int saved_errno = errno;
      ThrowSourceError(
          PacketIoErrorCode::kStatFailed,
          "cannot stat pack " + source_name_ + ": " +
              std::strerror(saved_errno),
          saved_errno);
    }
    if (status.st_size < 0 ||
        static_cast<std::uint64_t>(status.st_size) != declared_pack_size_) {
      std::ostringstream message;
      message << "pack size mismatch for " << source_name_ << ": expected "
              << declared_pack_size_ << ", got " << status.st_size;
      ThrowSourceError(PacketIoErrorCode::kPackSizeMismatch, message.str());
    }
    if (!S_ISREG(status.st_mode)) {
      ThrowSourceError(
          PacketIoErrorCode::kStatFailed,
          "direct pack source is not a regular file: " + source_name_);
    }

    const std::size_t filesystem_block_size =
        status.st_blksize > 0
            ? static_cast<std::size_t>(status.st_blksize)
            : packet_ring_.io_alignment();
    const std::size_t fallback_alignment = CheckedLeastCommonMultiple(
        packet_ring_.io_alignment(), filesystem_block_size);
    alignment_ = DirectIoAlignment{
        fallback_alignment,
        fallback_alignment,
        DirectIoAlignmentSource::kFilesystemBlockSizeFallback,
    };
    capability_ = DirectIoCapability{
        false,
        alignment_,
        DirectIoUnsupportedReason::kStatxDioalignNotReported,
        0,
    };

    bool authoritative_dioalign = false;
#if defined(STATX_DIOALIGN)
    struct statx direct_status {};
    const bool suppress =
        g_suppress_statx_dioalign_once.exchange(false, std::memory_order_relaxed);
    if (!suppress) {
      errno = 0;
      const int statx_status = statx(file_descriptor_,
                                     "",
                                     AT_EMPTY_PATH | AT_STATX_DONT_SYNC,
                                     STATX_DIOALIGN,
                                     &direct_status);
      if (statx_status != 0) {
        capability_.unsupported_reason =
            DirectIoUnsupportedReason::kStatxQueryFailed;
        capability_.system_error = errno;
      } else if ((direct_status.stx_mask & STATX_DIOALIGN) !=
                 STATX_DIOALIGN) {
        capability_.unsupported_reason =
            DirectIoUnsupportedReason::kStatxDioalignNotReported;
      } else if (direct_status.stx_dio_mem_align == 0 ||
                 direct_status.stx_dio_offset_align == 0) {
        capability_.unsupported_reason =
            DirectIoUnsupportedReason::kStatxDioalignZero;
      } else {
        alignment_ = DirectIoAlignment{
            static_cast<std::size_t>(direct_status.stx_dio_mem_align),
            static_cast<std::size_t>(direct_status.stx_dio_offset_align),
            DirectIoAlignmentSource::kStatxDioalign,
        };
        capability_.alignment = alignment_;
        capability_.unsupported_reason = DirectIoUnsupportedReason::kNone;
        authoritative_dioalign = true;
      }
    }
#else
    static_cast<void>(
        g_suppress_statx_dioalign_once.exchange(false, std::memory_order_relaxed));
#endif

    if (!authoritative_dioalign) {
      // Filesystem block size remains useful diagnostics, but it is not an
      // authoritative direct-I/O contract. Do not submit O_DIRECT heuristically.
      static_cast<void>(close(file_descriptor_));
      file_descriptor_ = -1;
      direct_supported_ = false;
      return;
    }

    ValidateRingAlignment();
    const int current_flags = fcntl(file_descriptor_, F_GETFL);
    if (current_flags < 0 ||
        fcntl(file_descriptor_, F_SETFL, current_flags | O_DIRECT) != 0) {
      const int saved_errno = errno;
      ThrowSourceError(
          PacketIoErrorCode::kOpenFailed,
          "cannot enable O_DIRECT for pack " + source_name_ + ": " +
              std::strerror(saved_errno),
          saved_errno);
    }

    const int setup_status = io_uring_queue_init(
        config_.queue_depth, &uring_, 0);
    if (setup_status < 0) {
      ThrowSourceError(
          PacketIoErrorCode::kOpenFailed,
          "io_uring_queue_init failed for " + source_name_ + ": " +
              std::strerror(-setup_status),
          -setup_status);
    }
    queue_initialized_ = true;
    const int dontfork_status = io_uring_ring_dontfork(&uring_);
    if (dontfork_status < 0) {
      ThrowSourceError(
          PacketIoErrorCode::kOpenFailed,
          "io_uring_ring_dontfork failed for " + source_name_ + ": " +
              std::strerror(-dontfork_status),
          -dontfork_status);
    }
    errno = 0;
    io_uring_probe* probe = io_uring_get_probe_ring(&uring_);
    if (probe == nullptr) {
      capability_.unsupported_reason =
          DirectIoUnsupportedReason::kIoUringProbeFailed;
      capability_.system_error = errno != 0 ? errno : EIO;
      io_uring_queue_exit(&uring_);
      queue_initialized_ = false;
      static_cast<void>(close(file_descriptor_));
      file_descriptor_ = -1;
      return;
    }
    const bool read_supported =
        io_uring_opcode_supported(probe, IORING_OP_READ) != 0;
    io_uring_free_probe(probe);
    if (!read_supported) {
      capability_.unsupported_reason =
          DirectIoUnsupportedReason::kIoUringReadUnsupported;
      capability_.system_error = EOPNOTSUPP;
      io_uring_queue_exit(&uring_);
      queue_initialized_ = false;
      static_cast<void>(close(file_descriptor_));
      file_descriptor_ = -1;
      return;
    }
    direct_supported_ = true;
    capability_.supported = true;
    capability_.alignment = alignment_;
    capability_.unsupported_reason = DirectIoUnsupportedReason::kNone;
    capability_.system_error = 0;
  }

  void ValidateRingAlignment() const {
    if (alignment_.memory_alignment == 0 ||
        alignment_.offset_alignment == 0 ||
        alignment_.source != DirectIoAlignmentSource::kStatxDioalign) {
      throw std::logic_error(
          "direct I/O initialization lacks authoritative STATX_DIOALIGN");
    }
    if (packet_ring_.allocation_alignment() == 0 ||
        packet_ring_.slot_stride() == 0) {
      throw std::invalid_argument(
          "packet ring returned invalid allocation geometry");
    }
    if (packet_ring_.allocation_alignment() % alignment_.memory_alignment != 0 ||
        packet_ring_.slot_stride() % alignment_.memory_alignment != 0) {
      throw std::invalid_argument(
          "packet ring allocation geometry does not guarantee STATX_DIOALIGN "
          "memory alignment for every slot");
    }
  }

  void ValidateRequestShape(const PacketReadRequest& request) const {
    if (request.packet_id.empty() ||
        request.packet_id.find('\0') != std::string::npos) {
      throw std::invalid_argument(
          "packet ID must be non-empty and contain no NUL");
    }
    if (!IsValidRequirement(request.requirement)) {
      throw std::invalid_argument("packet requirement is invalid");
    }
    if (request.logical_bytes == 0) {
      throw std::invalid_argument("packet logical_bytes must be positive");
    }
    if (request.file_offset % packet_ring_.io_alignment() != 0) {
      throw std::invalid_argument("packet file offset is not I/O aligned");
    }
    const std::size_t expected_physical = CheckedAlignUp(
        request.logical_bytes, packet_ring_.io_alignment());
    if (request.physical_bytes != expected_physical) {
      throw std::invalid_argument(
          "packet physical_bytes must exactly equal aligned logical_bytes");
    }
    if (request.physical_bytes > packet_ring_.slot_capacity()) {
      throw std::invalid_argument(
          "packet physical extent exceeds ring slot capacity");
    }
    if (request.physical_bytes >
        static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
      throw std::invalid_argument(
          "packet physical extent exceeds io_uring unsigned length field");
    }
  }

  void ValidateDirectGeometry(const PacketReadRequest& request) const {
    if (request.file_offset % alignment_.offset_alignment != 0) {
      throw std::invalid_argument(
          "packet file offset does not satisfy STATX_DIOALIGN");
    }
    if (request.physical_bytes % alignment_.offset_alignment != 0) {
      throw std::invalid_argument(
          "packet physical length does not satisfy STATX_DIOALIGN");
    }
  }

  void ValidateDestination(
      const MutablePacketBytes& destination,
      std::size_t physical_bytes) const {
    if (destination.data == nullptr || physical_bytes > destination.capacity) {
      throw std::logic_error(
          "packet ring returned an invalid filling destination");
    }
    if (reinterpret_cast<std::uintptr_t>(destination.data) %
            alignment_.memory_alignment !=
        0) {
      throw std::invalid_argument(
          "packet ring destination does not satisfy STATX_DIOALIGN memory "
          "alignment");
    }
  }

  [[nodiscard]] std::uint64_t NextOperationId() {
    if (next_operation_id_ == 0 ||
        next_operation_id_ == std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error("direct I/O operation ID counter exhausted");
    }
    return next_operation_id_++;
  }

  [[nodiscard]] bool SubmitRead(PendingRead& pending) {
    const MutablePacketBytes destination = pending.filling.bytes();
    const std::size_t remaining =
        pending.request.physical_bytes - pending.completed_bytes;
    if (remaining == 0 ||
        remaining >
            static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
      AbortForUnprovenDirectIoSafety();
    }

    io_uring_sqe* sqe = io_uring_get_sqe(&uring_);
    if (sqe == nullptr) {
      return false;
    }
    io_uring_prep_read(
        sqe,
        file_descriptor_,
        destination.data + pending.completed_bytes,
        static_cast<unsigned int>(remaining),
        static_cast<off_t>(pending.request.file_offset +
                           pending.completed_bytes));
    io_uring_sqe_set_data64(sqe, pending.operation_id);

    const int submit_status = io_uring_submit(&uring_);
    if (submit_status != 1) {
      // Once an SQE has been prepared, a non-exact submit result is treated as
      // acceptance-ambiguous. The tracked Filling lease deliberately remains
      // owned, and the process fails closed instead of recycling its buffer.
      AbortForUnprovenDirectIoSafety();
    }
    ++pending.stats.read_submissions;
    if (remaining >
        std::numeric_limits<std::size_t>::max() -
            pending.stats.bytes_submitted) {
      AbortForUnprovenDirectIoSafety();
    }
    pending.stats.bytes_submitted += remaining;
    return true;
  }

  [[nodiscard]] bool RemainingGeometryIsDirectSafe(
      const PendingRead& pending) const {
    const MutablePacketBytes destination = pending.filling.bytes();
    const std::size_t remaining =
        pending.request.physical_bytes - pending.completed_bytes;
    const std::uint64_t offset =
        pending.request.file_offset + pending.completed_bytes;
    return remaining != 0 &&
           remaining <= static_cast<std::size_t>(
                            std::numeric_limits<unsigned int>::max()) &&
           reinterpret_cast<std::uintptr_t>(
               destination.data + pending.completed_bytes) %
                   alignment_.memory_alignment ==
               0 &&
           offset % alignment_.offset_alignment == 0 &&
           remaining % alignment_.offset_alignment == 0;
  }

  [[nodiscard]] std::string DirectUnsupportedMessage() const {
    const char* reason = "unknown";
    switch (capability_.unsupported_reason) {
      case DirectIoUnsupportedReason::kNone:
        reason = "none";
        break;
      case DirectIoUnsupportedReason::kStatxQueryFailed:
        reason = "statx query failed";
        break;
      case DirectIoUnsupportedReason::kStatxDioalignNotReported:
        reason = "STATX_DIOALIGN was not reported";
        break;
      case DirectIoUnsupportedReason::kStatxDioalignZero:
        reason = "STATX_DIOALIGN returned a zero alignment";
        break;
      case DirectIoUnsupportedReason::kIoUringProbeFailed:
        reason = "io_uring opcode probe failed";
        break;
      case DirectIoUnsupportedReason::kIoUringReadUnsupported:
        reason = "IORING_OP_READ is unsupported";
        break;
    }
    std::string message =
        "direct I/O is unsupported for " + source_name_ + ": " + reason;
    if (capability_.system_error != 0) {
      message += ": ";
      message += std::strerror(capability_.system_error);
    }
    return message;
  }

  [[nodiscard]] std::optional<DirectPacketLoadResult> HandleReadCompletion(
      std::uint64_t operation_id,
      int completion_result,
      unsigned int completion_flags,
      bool teardown) {
    if ((completion_flags & IORING_CQE_F_MORE) != 0U) {
      // Ordinary IORING_OP_READ is single-shot. Releasing on F_MORE could let
      // a later CQE race a recycled slot, so reject this impossible protocol
      // state with the same fail-closed policy used for ambiguous submission.
      AbortForUnprovenDirectIoSafety();
    }
    auto found = pending_.find(operation_id);
    if (found == pending_.end()) {
      AbortForUnprovenDirectIoSafety();
    }
    PendingRead& pending = found->second;
    ++pending.stats.read_completions;

    if (teardown) {
      pending_.erase(found);
      return std::nullopt;
    }

    if (completion_result > 0) {
      const auto completed_now = static_cast<std::size_t>(completion_result);
      const std::size_t remaining =
          pending.request.physical_bytes - pending.completed_bytes;
      if (completed_now > remaining) {
        return FinishFailure(
            found,
            PacketIoErrorCode::kReadFailed,
            "completion returned more bytes than the immutable extent");
      }
      pending.completed_bytes += completed_now;
      pending.stats.file_bytes_returned += completed_now;
    }

    if (pending.cancellation.IsCancelled()) {
      DirectPacketReadStats stats = pending.stats;
      pending_.erase(found);
      DirectPacketLoadResult result;
      result.status = DirectPacketLoadStatus::kCancelled;
      result.operation_id = operation_id;
      result.stats = std::move(stats);
      return result;
    }

    if (completion_result == -EINTR) {
      if (!RemainingGeometryIsDirectSafe(pending)) {
        return FinishFailure(
            found,
            PacketIoErrorCode::kReadFailed,
            "EINTR completion leaves non-aligned direct-I/O geometry");
      }
      if (!SubmitRead(pending)) {
        return FinishFailure(
            found,
            PacketIoErrorCode::kReadFailed,
            "EINTR completion retry has no submission queue entry",
            EAGAIN);
      }
      return std::nullopt;
    }
    if (completion_result < 0) {
      const int system_error = -completion_result;
      return FinishFailure(
          found,
          PacketIoErrorCode::kReadFailed,
          "read completion",
          system_error);
    }
    if (completion_result == 0) {
      return FinishFailure(
          found,
          PacketIoErrorCode::kUnexpectedEof,
          "unexpected EOF completion");
    }

    if (pending.completed_bytes < pending.request.physical_bytes) {
      if (!RemainingGeometryIsDirectSafe(pending)) {
        return FinishFailure(
            found,
            PacketIoErrorCode::kReadFailed,
            "short completion leaves a non-aligned direct-I/O remainder");
      }
      if (!SubmitRead(pending)) {
        return FinishFailure(
            found,
            PacketIoErrorCode::kReadFailed,
            "aligned short-completion retry has no submission queue entry",
            EAGAIN);
      }
      return std::nullopt;
    }

    const MutablePacketBytes destination = pending.filling.bytes();
    const std::uint32_t actual_crc32c =
        Crc32c(destination.data, pending.request.logical_bytes);
    if (actual_crc32c != pending.request.expected_crc32c) {
      return FinishFailure(
          found,
          PacketIoErrorCode::kChecksumMismatch,
          "CRC32C mismatch",
          0,
          actual_crc32c);
    }
    if (pending.cancellation.IsCancelled()) {
      DirectPacketReadStats stats = pending.stats;
      pending_.erase(found);
      DirectPacketLoadResult result;
      result.status = DirectPacketLoadStatus::kCancelled;
      result.operation_id = operation_id;
      result.stats = std::move(stats);
      return result;
    }

    PendingRead completed = std::move(pending);
    pending_.erase(found);
    ReadySlotLease ready = std::move(completed.filling).Publish(
        PacketSlotMetadata{
            completed.request.packet_id,
            completed.request.logical_bytes,
            completed.request.physical_bytes,
            actual_crc32c,
        });
    DirectPacketLoadResult result;
    result.status = DirectPacketLoadStatus::kReady;
    result.operation_id = operation_id;
    result.ready.emplace(std::move(ready));
    result.stats = std::move(completed.stats);
    result.submitted_at = completed.submitted_at;
    return result;
  }

  using PendingIterator = std::map<std::uint64_t, PendingRead>::iterator;

  [[nodiscard]] std::optional<DirectPacketLoadResult> FinishFailure(
      PendingIterator found,
      PacketIoErrorCode code,
      const char* detail,
      int system_error = 0,
      std::uint32_t actual_crc32c = 0) {
    PendingRead completed = std::move(found->second);
    pending_.erase(found);
    PacketIoFailure failure = MakeFailure(
        code,
        completed.request,
        completed.stats,
        detail,
        system_error,
        actual_crc32c);
    if (completed.request.requirement == PacketRequirement::kRequired) {
      throw RequiredPacketLoadError(std::move(failure));
    }
    DirectPacketLoadResult result;
    result.status = DirectPacketLoadStatus::kOptionalUnavailable;
    result.operation_id = completed.operation_id;
    result.stats = std::move(completed.stats);
    result.failure = std::move(failure);
    return result;
  }

  [[nodiscard]] static DirectPacketLoadResult ImmediateResult(
      DirectPacketLoadStatus status,
      DirectPacketReadStats stats) {
    DirectPacketLoadResult result;
    result.status = status;
    result.stats = std::move(stats);
    return result;
  }

  void DrainForTeardown() noexcept {
    if (!queue_initialized_) {
      if (!pending_.empty()) {
        AbortForUnprovenDirectIoSafety();
      }
      return;
    }
    while (!pending_.empty()) {
      io_uring_cqe* cqe = nullptr;
      int status = 0;
      do {
        status = io_uring_wait_cqe(&uring_, &cqe);
      } while (status == -EINTR);
      if (status < 0 || cqe == nullptr) {
        AbortForUnprovenDirectIoSafety();
      }
      const std::uint64_t operation_id = io_uring_cqe_get_data64(cqe);
      const int completion_result = cqe->res;
      const unsigned int completion_flags = cqe->flags;
      io_uring_cqe_seen(&uring_, cqe);
      try {
        static_cast<void>(HandleReadCompletion(
            operation_id, completion_result, completion_flags, true));
      } catch (...) {
        AbortForUnprovenDirectIoSafety();
      }
    }
  }

  void CleanupWithoutPending() noexcept {
    if (queue_initialized_) {
      io_uring_queue_exit(&uring_);
      queue_initialized_ = false;
    }
    if (file_descriptor_ >= 0) {
      static_cast<void>(close(file_descriptor_));
      file_descriptor_ = -1;
    }
  }

  std::string source_name_;
  std::uint64_t declared_pack_size_ = 0;
  PacketRing& packet_ring_;
  DirectPacketIoConfig config_{};
  pid_t owner_pid_ = 0;
  int file_descriptor_ = -1;
  io_uring uring_{};
  bool queue_initialized_ = false;
  bool direct_supported_ = false;
  DirectIoAlignment alignment_{};
  DirectIoCapability capability_{};
  std::size_t ring_slot_count_ = 0;
  std::uint64_t next_operation_id_ = 1;
  std::map<std::uint64_t, PendingRead> pending_;
};

}  // namespace detail

DirectPackFile::DirectPackFile(
    std::string path,
    std::uint64_t declared_pack_size,
    PacketRing& ring,
    DirectPacketIoConfig config)
    : impl_(std::make_unique<detail::DirectPackFileImpl>(
          std::move(path), declared_pack_size, ring, config)) {}

DirectPackFile::~DirectPackFile() {
  if (impl_ != nullptr && !impl_->IsOwnerProcess()) {
    // The child did not inherit io_uring mappings due to ring_dontfork. Leak
    // this small facade copy in the child; normal fork users must exec/_exit.
    static_cast<void>(impl_.release());
  }
}

const std::string& DirectPackFile::source_name() const noexcept {
  return impl_->source_name();
}

std::uint64_t DirectPackFile::declared_pack_size() const noexcept {
  return impl_->declared_pack_size();
}

std::uint32_t DirectPackFile::queue_depth() const noexcept {
  return impl_->queue_depth();
}

std::size_t DirectPackFile::in_flight() const noexcept {
  return impl_->in_flight();
}

DirectIoAlignment DirectPackFile::direct_alignment() const noexcept {
  return impl_->direct_alignment();
}

DirectIoCapability DirectPackFile::direct_capability() const noexcept {
  return impl_->direct_capability();
}

bool DirectPackFile::direct_supported() const noexcept {
  return impl_->direct_supported();
}

std::size_t DirectPackFile::ring_slot_count() const noexcept {
  return impl_->ring_slot_count();
}

DirectPacketLoadResult DirectPackFile::Submit(
    const PacketReadRequest& request,
    const CancellationToken& cancellation) {
  return impl_->Submit(request, cancellation);
}

std::optional<DirectPacketLoadResult> DirectPackFile::Poll() {
  return impl_->Poll();
}

DirectPacketLoadResult DirectPackFile::Wait() { return impl_->Wait(); }

void DirectPackFile::InjectCompletionResultsForTest(
    std::vector<int> results) {
  std::lock_guard<std::mutex> lock(g_completion_fault_mutex);
  g_completion_faults.assign(results.begin(), results.end());
}

void DirectPackFile::ClearCompletionResultsForTest() noexcept {
  std::lock_guard<std::mutex> lock(g_completion_fault_mutex);
  g_completion_faults.clear();
}

void DirectPackFile::SuppressStatxDioalignForTestOnce() noexcept {
  g_suppress_statx_dioalign_once.store(true, std::memory_order_relaxed);
}

}  // namespace harpq::native_runtime
