#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

constexpr std::size_t kDefaultBytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kDefaultWarmup = 10;
constexpr std::size_t kDefaultIterations = 100;
constexpr std::size_t kMaximumBytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumWarmup = 1000;
constexpr std::size_t kMaximumIterations = 10000;
constexpr std::size_t kMaximumWarmupBytes = 128ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumTimedBytes = 1ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kVerificationSegmentBytes = 4096;
constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

struct Options {
  std::size_t bytes = kDefaultBytes;
  std::size_t warmup = kDefaultWarmup;
  std::size_t iterations = kDefaultIterations;
  bool json = false;
  bool help = false;
};

struct BenchmarkResult {
  std::string device_name;
  int compute_major = 0;
  int compute_minor = 0;
  std::size_t vram_bytes = 0;
  int async_engine_count = 0;
  int runtime_version = 0;
  int driver_api_version = 0;
  std::size_t payload_bytes = 0;
  std::size_t registered_bytes = 0;
  std::size_t page_size = 0;
  std::size_t warmup = 0;
  std::size_t iterations = 0;
  double elapsed_ms = 0.0;
  double bytes_per_second = 0.0;
  double decimal_gb_per_second = 0.0;
  double gib_per_second = 0.0;
  std::size_t verification_bytes = 0;
  std::uint64_t verification_checksum = 0;
};

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

std::size_t CheckedMultiply(std::size_t left,
                            std::size_t right,
                            const char* label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::invalid_argument(std::string(label) + " overflows size_t");
  }
  return left * right;
}

std::size_t RoundUp(std::size_t value,
                    std::size_t alignment,
                    const char* label) {
  if (alignment == 0) {
    throw std::runtime_error("host page size cannot be zero");
  }
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

std::size_t ParsePositive(const std::string& text,
                          const char* option,
                          std::size_t maximum) {
  if (text.empty()) {
    throw std::invalid_argument(std::string(option) + " requires a decimal integer");
  }
  std::uint64_t parsed = 0;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto result = std::from_chars(begin, end, parsed, 10);
  if (result.ec == std::errc::result_out_of_range ||
      parsed > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::invalid_argument(std::string(option) + " is too large");
  }
  if (result.ec != std::errc{} || result.ptr != end) {
    throw std::invalid_argument(std::string(option) + " must be a decimal integer");
  }
  if (parsed == 0) {
    throw std::invalid_argument(std::string(option) + " must be greater than zero");
  }
  if (parsed > maximum) {
    std::ostringstream message;
    message << option << " exceeds the safety limit of " << maximum;
    throw std::invalid_argument(message.str());
  }
  return static_cast<std::size_t>(parsed);
}

bool StartsWith(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  bool bytes_seen = false;
  bool warmup_seen = false;
  bool iterations_seen = false;
  bool json_seen = false;

  const auto take_value = [&](int& index,
                              const std::string& argument,
                              std::string_view option) -> std::string {
    const std::string equals_prefix = std::string(option) + '=';
    if (StartsWith(argument, equals_prefix)) {
      return argument.substr(equals_prefix.size());
    }
    if (argument == option) {
      if (index + 1 >= argc) {
        throw std::invalid_argument(std::string(option) + " requires a value");
      }
      ++index;
      return argv[index];
    }
    return {};
  };

  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      options.help = true;
      continue;
    }
    if (argument == "--json") {
      if (json_seen) {
        throw std::invalid_argument("--json may be specified only once");
      }
      json_seen = true;
      options.json = true;
      continue;
    }
    if (argument == "--bytes" || StartsWith(argument, "--bytes=")) {
      if (bytes_seen) {
        throw std::invalid_argument("--bytes may be specified only once");
      }
      bytes_seen = true;
      options.bytes = ParsePositive(
          take_value(index, argument, "--bytes"), "--bytes", kMaximumBytes);
      continue;
    }
    if (argument == "--warmup" || StartsWith(argument, "--warmup=")) {
      if (warmup_seen) {
        throw std::invalid_argument("--warmup may be specified only once");
      }
      warmup_seen = true;
      options.warmup = ParsePositive(
          take_value(index, argument, "--warmup"), "--warmup", kMaximumWarmup);
      continue;
    }
    if (argument == "--iterations" || StartsWith(argument, "--iterations=")) {
      if (iterations_seen) {
        throw std::invalid_argument("--iterations may be specified only once");
      }
      iterations_seen = true;
      options.iterations = ParsePositive(take_value(index, argument, "--iterations"),
                                         "--iterations", kMaximumIterations);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument);
  }

  const std::size_t warmup_bytes =
      CheckedMultiply(options.bytes, options.warmup, "warm-up transfer volume");
  const std::size_t timed_bytes =
      CheckedMultiply(options.bytes, options.iterations, "timed transfer volume");
  if (warmup_bytes > kMaximumWarmupBytes) {
    throw std::invalid_argument("warm-up transfer volume exceeds the 128 GiB safety limit");
  }
  if (timed_bytes > kMaximumTimedBytes) {
    throw std::invalid_argument("timed transfer volume exceeds the 1 TiB safety limit");
  }
  return options;
}

class RegisteredHostBuffer final {
 public:
  explicit RegisteredHostBuffer(std::size_t payload_bytes)
      : payload_bytes_(payload_bytes) {
    errno = 0;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
      throw std::runtime_error(
          std::string("sysconf(_SC_PAGESIZE) failed: ") + std::strerror(errno));
    }
    page_size_ = static_cast<std::size_t>(page_size);
    registered_bytes_ = RoundUp(payload_bytes_, page_size_, "registered allocation");

    void* allocation = nullptr;
    const int allocation_status =
        posix_memalign(&allocation, page_size_, registered_bytes_);
    if (allocation_status != 0) {
      throw std::runtime_error(
          std::string("posix_memalign failed: ") + std::strerror(allocation_status));
    }
    data_ = static_cast<std::uint8_t*>(allocation);

    const cudaError_t registration_status =
        cudaHostRegister(data_, registered_bytes_, cudaHostRegisterDefault);
    if (registration_status != cudaSuccess) {
      std::free(data_);
      data_ = nullptr;
      std::ostringstream message;
      message << "cudaHostRegister failed; pageable fallback is intentionally disabled: "
              << cudaGetErrorName(registration_status) << " ("
              << cudaGetErrorString(registration_status) << ')';
      throw std::runtime_error(message.str());
    }
    registered_ = true;
  }

  ~RegisteredHostBuffer() {
    if (registered_) {
      static_cast<void>(cudaHostUnregister(data_));
    }
    std::free(data_);
  }

  RegisteredHostBuffer(const RegisteredHostBuffer&) = delete;
  RegisteredHostBuffer& operator=(const RegisteredHostBuffer&) = delete;
  RegisteredHostBuffer(RegisteredHostBuffer&&) = delete;
  RegisteredHostBuffer& operator=(RegisteredHostBuffer&&) = delete;

  [[nodiscard]] std::uint8_t* data() noexcept { return data_; }
  [[nodiscard]] const std::uint8_t* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t payload_bytes() const noexcept { return payload_bytes_; }
  [[nodiscard]] std::size_t registered_bytes() const noexcept {
    return registered_bytes_;
  }
  [[nodiscard]] std::size_t page_size() const noexcept { return page_size_; }

 private:
  std::uint8_t* data_ = nullptr;
  std::size_t payload_bytes_ = 0;
  std::size_t registered_bytes_ = 0;
  std::size_t page_size_ = 0;
  bool registered_ = false;
};

template <typename T>
class DeviceBuffer final {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                         CheckedMultiply(count_, sizeof(T), "device allocation")),
              "cudaMalloc");
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      static_cast<void>(cudaFree(data_));
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

class Stream final {
 public:
  Stream() {
    CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags");
  }

  ~Stream() {
    if (stream_ != nullptr) {
      // Exception unwinding may begin while queued copies still reference the
      // host and device buffers. This object is destroyed before those
      // buffers, so a best-effort synchronization keeps their lifetime valid.
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

class Event final {
 public:
  Event() { CheckCuda(cudaEventCreate(&event_), "cudaEventCreate"); }

  ~Event() {
    if (event_ != nullptr) {
      static_cast<void>(cudaEventDestroy(event_));
    }
  }

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

std::uint8_t PatternByte(std::size_t index) noexcept {
  const std::uint64_t mixed =
      static_cast<std::uint64_t>(index) * 131ULL +
      (static_cast<std::uint64_t>(index) >> 8U) * 17ULL + 0x5AULL;
  return static_cast<std::uint8_t>(mixed & 0xFFU);
}

std::uint64_t UpdateFnv1a(std::uint64_t checksum,
                          const std::uint8_t* bytes,
                          std::size_t size) noexcept {
  for (std::size_t index = 0; index < size; ++index) {
    checksum ^= bytes[index];
    checksum *= kFnvPrime;
  }
  return checksum;
}

std::pair<std::size_t, std::uint64_t> VerifySample(
    const RegisteredHostBuffer& host,
    const DeviceBuffer<std::uint8_t>& device,
    cudaStream_t stream) {
  const std::size_t segment_bytes =
      std::min(host.payload_bytes(), kVerificationSegmentBytes);
  std::vector<std::size_t> offsets{0};
  if (host.payload_bytes() >= 3 * segment_bytes) {
    offsets.push_back((host.payload_bytes() - segment_bytes) / 2);
    offsets.push_back(host.payload_bytes() - segment_bytes);
  } else if (host.payload_bytes() >= 2 * segment_bytes) {
    offsets.push_back(host.payload_bytes() - segment_bytes);
  }
  std::sort(offsets.begin(), offsets.end());
  offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());

  std::vector<std::uint8_t> returned(segment_bytes);
  std::size_t verified_bytes = 0;
  std::uint64_t source_checksum = kFnvOffsetBasis;
  std::uint64_t device_checksum = kFnvOffsetBasis;
  for (const std::size_t offset : offsets) {
    CheckCuda(cudaMemcpyAsync(returned.data(), device.data() + offset, segment_bytes,
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync(verification D2H)");
    CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(verification)");
    if (!std::equal(returned.begin(), returned.end(), host.data() + offset)) {
      throw std::runtime_error("D2H sample verification found corrupted data");
    }
    source_checksum =
        UpdateFnv1a(source_checksum, host.data() + offset, segment_bytes);
    device_checksum = UpdateFnv1a(device_checksum, returned.data(), segment_bytes);
    verified_bytes += segment_bytes;
  }
  if (source_checksum != device_checksum) {
    throw std::runtime_error("D2H sample checksum does not match its host source");
  }
  return {verified_bytes, device_checksum};
}

std::string CudaVersionString(int encoded) {
  if (encoded <= 0) {
    return "unknown";
  }
  const int major = encoded / 1000;
  const int minor = (encoded % 1000) / 10;
  const int patch = encoded % 10;
  std::ostringstream value;
  value << major << '.' << minor;
  if (patch != 0) {
    value << '.' << patch;
  }
  return value.str();
}

std::string JsonString(std::string_view value) {
  std::ostringstream escaped;
  escaped << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        escaped << "\\\"";
        break;
      case '\\':
        escaped << "\\\\";
        break;
      case '\b':
        escaped << "\\b";
        break;
      case '\f':
        escaped << "\\f";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        if (character < 0x20U) {
          escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                  << static_cast<unsigned int>(character) << std::dec;
        } else {
          escaped << static_cast<char>(character);
        }
    }
  }
  escaped << '"';
  return escaped.str();
}

std::string HexChecksum(std::uint64_t checksum) {
  std::ostringstream value;
  value << "0x" << std::hex << std::setw(16) << std::setfill('0') << checksum;
  return value.str();
}

bool ProbeCuda(std::string* reason) noexcept {
  try {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
      if (reason != nullptr) {
        std::ostringstream message;
        message << cudaGetErrorName(status) << ": " << cudaGetErrorString(status);
        *reason = message.str();
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
  } catch (...) {
    if (reason != nullptr) {
      try {
        *reason = "CUDA device probe failed while formatting its diagnostic";
      } catch (...) {
      }
    }
    return false;
  }
}

BenchmarkResult RunBenchmark(const Options& options) {
  int device_index = 0;
  CheckCuda(cudaGetDevice(&device_index), "cudaGetDevice");
  cudaDeviceProp properties{};
  CheckCuda(cudaGetDeviceProperties(&properties, device_index),
            "cudaGetDeviceProperties");

  int runtime_version = 0;
  int driver_api_version = 0;
  CheckCuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  CheckCuda(cudaDriverGetVersion(&driver_api_version), "cudaDriverGetVersion");

  RegisteredHostBuffer host(options.bytes);
  for (std::size_t index = 0; index < host.payload_bytes(); ++index) {
    host.data()[index] = PatternByte(index);
  }
  DeviceBuffer<std::uint8_t> device(options.bytes);
  Stream stream;
  Event start;
  Event stop;

  for (std::size_t iteration = 0; iteration < options.warmup; ++iteration) {
    CheckCuda(cudaMemcpyAsync(device.data(), host.data(), options.bytes,
                              cudaMemcpyHostToDevice, stream.get()),
              "cudaMemcpyAsync(warm-up H2D)");
  }
  CheckCuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize(warm-up)");

  CheckCuda(cudaEventRecord(start.get(), stream.get()), "cudaEventRecord(start)");
  for (std::size_t iteration = 0; iteration < options.iterations; ++iteration) {
    CheckCuda(cudaMemcpyAsync(device.data(), host.data(), options.bytes,
                              cudaMemcpyHostToDevice, stream.get()),
              "cudaMemcpyAsync(timed H2D)");
  }
  CheckCuda(cudaEventRecord(stop.get(), stream.get()), "cudaEventRecord(stop)");
  CheckCuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize(stop)");

  float elapsed_ms_float = 0.0F;
  CheckCuda(cudaEventElapsedTime(&elapsed_ms_float, start.get(), stop.get()),
            "cudaEventElapsedTime");
  const double elapsed_ms = static_cast<double>(elapsed_ms_float);
  if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0) {
    throw std::runtime_error("CUDA events reported a non-positive elapsed time");
  }

  const auto verification = VerifySample(host, device, stream.get());
  const long double transferred_bytes =
      static_cast<long double>(options.bytes) *
      static_cast<long double>(options.iterations);
  const long double elapsed_seconds = static_cast<long double>(elapsed_ms) / 1000.0L;
  const double bytes_per_second =
      static_cast<double>(transferred_bytes / elapsed_seconds);
  if (!std::isfinite(bytes_per_second) || bytes_per_second <= 0.0) {
    throw std::runtime_error("computed H2D bandwidth is not finite and positive");
  }

  BenchmarkResult result;
  result.device_name = properties.name;
  result.compute_major = properties.major;
  result.compute_minor = properties.minor;
  result.vram_bytes = properties.totalGlobalMem;
  result.async_engine_count = properties.asyncEngineCount;
  result.runtime_version = runtime_version;
  result.driver_api_version = driver_api_version;
  result.payload_bytes = options.bytes;
  result.registered_bytes = host.registered_bytes();
  result.page_size = host.page_size();
  result.warmup = options.warmup;
  result.iterations = options.iterations;
  result.elapsed_ms = elapsed_ms;
  result.bytes_per_second = bytes_per_second;
  result.decimal_gb_per_second = bytes_per_second / 1.0e9;
  result.gib_per_second = bytes_per_second / 1073741824.0;
  result.verification_bytes = verification.first;
  result.verification_checksum = verification.second;
  return result;
}

void PrintUsage(std::ostream& output, const char* executable) {
  output << "Usage: " << executable
         << " [--bytes N] [--warmup N] [--iterations N] [--json]\n"
         << "  --bytes       H2D payload bytes (1.." << kMaximumBytes << ")\n"
         << "  --warmup      untimed copies (1.." << kMaximumWarmup << ")\n"
         << "  --iterations  timed copies (1.." << kMaximumIterations << ")\n"
         << "Defaults: --bytes " << kDefaultBytes << " --warmup " << kDefaultWarmup
         << " --iterations " << kDefaultIterations << '\n';
}

void PrintJsonStatus(const char* status, const std::string& reason) {
  std::cout << "{\"status\":" << JsonString(status)
            << ",\"reason\":" << JsonString(reason) << "}\n";
}

void PrintResult(const BenchmarkResult& result, bool json) {
  std::cout << std::setprecision(10);
  if (json) {
    std::cout
        << "{\"status\":\"ok\",\"scope\":\"pinned_h2d_microbenchmark_only\""
        << ",\"device\":" << JsonString(result.device_name)
        << ",\"compute_capability\":"
        << JsonString(std::to_string(result.compute_major) + "." +
                      std::to_string(result.compute_minor))
        << ",\"vram_bytes\":" << result.vram_bytes
        << ",\"async_engine_count\":" << result.async_engine_count
        << ",\"cuda_runtime_version\":" << result.runtime_version
        << ",\"cuda_runtime\":"
        << JsonString(CudaVersionString(result.runtime_version))
        << ",\"cuda_driver_api_version\":" << result.driver_api_version
        << ",\"cuda_driver_api\":"
        << JsonString(CudaVersionString(result.driver_api_version))
        << ",\"payload_bytes\":" << result.payload_bytes
        << ",\"registered_bytes\":" << result.registered_bytes
        << ",\"host_page_bytes\":" << result.page_size
        << ",\"warmup\":" << result.warmup
        << ",\"iterations\":" << result.iterations
        << ",\"elapsed_ms\":" << result.elapsed_ms
        << ",\"bandwidth_bytes_per_second\":" << result.bytes_per_second
        << ",\"bandwidth_decimal_gb_per_second\":"
        << result.decimal_gb_per_second
        << ",\"bandwidth_gib_per_second\":" << result.gib_per_second
        << ",\"verification_bytes\":" << result.verification_bytes
        << ",\"verification_checksum_fnv1a64\":"
        << JsonString(HexChecksum(result.verification_checksum)) << "}\n";
    return;
  }

  std::cout << "HARP-Q pinned H2D microbenchmark\n"
            << "device: " << result.device_name << " (CC " << result.compute_major
            << '.' << result.compute_minor << ")\n"
            << "VRAM bytes: " << result.vram_bytes << '\n'
            << "async engines: " << result.async_engine_count << '\n'
            << "CUDA runtime / driver API: "
            << CudaVersionString(result.runtime_version) << " / "
            << CudaVersionString(result.driver_api_version) << '\n'
            << "payload/registered bytes: " << result.payload_bytes << " / "
            << result.registered_bytes << '\n'
            << "warmup/iterations: " << result.warmup << " / " << result.iterations
            << '\n'
            << "elapsed ms: " << result.elapsed_ms << '\n'
            << "bandwidth: " << result.decimal_gb_per_second << " GB/s (decimal), "
            << result.gib_per_second << " GiB/s\n"
            << "verified sample bytes/checksum: " << result.verification_bytes << " / "
            << HexChecksum(result.verification_checksum) << '\n'
            << "scope: pinned H2D microbenchmark only; no model throughput claim\n";
}

bool WantsJson(int argc, char** argv) {
  for (int index = 1; index < argc; ++index) {
    if (std::string_view(argv[index]) == "--json") {
      return true;
    }
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  std::cout.imbue(std::locale::classic());
  std::cerr.imbue(std::locale::classic());
  const bool wants_json = WantsJson(argc, argv);

  Options options;
  try {
    options = ParseOptions(argc, argv);
  } catch (const std::exception& error) {
    if (wants_json) {
      PrintJsonStatus("error", error.what());
    } else {
      std::cerr << "argument error: " << error.what() << '\n';
      PrintUsage(std::cerr, argv[0]);
    }
    return 2;
  }

  if (options.help) {
    PrintUsage(std::cout, argv[0]);
    return 0;
  }

  std::string unavailable_reason;
  if (!ProbeCuda(&unavailable_reason)) {
    if (options.json) {
      PrintJsonStatus("skipped", unavailable_reason);
    } else {
      std::cout << "SKIP: CUDA H2D benchmark requires a working CUDA device: "
                << unavailable_reason << '\n';
    }
    return 77;
  }

  try {
    PrintResult(RunBenchmark(options), options.json);
    return 0;
  } catch (const std::exception& error) {
    if (options.json) {
      PrintJsonStatus("error", error.what());
    } else {
      std::cerr << "H2D benchmark failed: " << error.what() << '\n';
    }
    return 1;
  }
}
