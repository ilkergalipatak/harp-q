#include "harpq/cuda_direct_packet_io.hpp"
#include "harpq/native/stage_codec.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using namespace harpq::native_runtime;
using Clock = std::chrono::steady_clock;

void Require(bool value, const char* message) {
  if (!value) { throw std::runtime_error(message); }
}

std::size_t Number(const char* argument, std::size_t maximum) {
  const std::string text(argument);
  std::size_t result = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  Require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() &&
              result > 0 && result <= maximum, "invalid or unbounded benchmark argument");
  return result;
}

double Micros(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - begin).count();
}

struct Sample {
  DirectPacketReadStats stats;
  double critical_us = 0;
  double io_us = 0;
  double h2d_us = 0;
  double enqueue_host_us = 0;
};

template <class Source>
Sample ReadOne(Source& source, const PacketReadRequest& request,
               const std::vector<std::uint8_t>& expected, bool verify) {
  const auto begin = Clock::now();
  auto submitted = source.Submit(request);
  Require(submitted.status == DirectPacketLoadStatus::kSubmitted, "read not admitted");
  auto ready = source.Wait();
  Require(ready.status == DirectPacketLoadStatus::kReady && ready.ready.has_value() &&
              ready.operation_id == submitted.operation_id, "read completion mismatch");
  Require(ready.stats.logical_bytes == request.logical_bytes &&
              ready.stats.file_bytes_returned == request.physical_bytes &&
              ready.stats.used_ring_slot_buffer && !ready.stats.legacy_unpadded_tail,
          "unexpected direct read accounting");
  Sample sample;
  sample.stats = ready.stats;
  if constexpr (std::is_same_v<Source, CudaDirectPackFile>) {
    const auto enqueue_begin = Clock::now();
    auto transfer = std::move(*ready.ready).EnqueueH2d();
    sample.enqueue_host_us = Micros(enqueue_begin, Clock::now());
    Require(transfer.Wait() == CompletionState::kComplete, "H2D did not complete");
    const auto* observation = transfer.observation();
    Require(observation != nullptr, "completed transfer has no observation");
    sample.critical_us = observation->critical_path_us;
    sample.io_us = observation->io_service_us;
    sample.h2d_us = observation->h2d_service_us;
    if (verify) {
      std::vector<std::uint8_t> copy(expected.size());
      Require(cudaMemcpy(copy.data(), transfer.device_data(), copy.size(), cudaMemcpyDeviceToHost) ==
                  cudaSuccess, "D2H verification failed");
      Require(copy == expected, "GPU data differs from deterministic fixture");
    }
  } else {
    sample.critical_us = Micros(begin, Clock::now());
    sample.io_us = sample.critical_us;
    if (verify) {
      Require(std::equal(expected.begin(), expected.end(), ready.ready->view().data),
              "CPU data differs from deterministic fixture");
    }
  }
  return sample;
}

template <class Source>
void Run(Source& source, const char* mode, std::uint64_t pack_bytes, std::size_t bytes,
         std::size_t warmup, std::size_t iterations) {
  const auto capability = source.direct_capability();
  Require(capability.supported && capability.alignment.source == DirectIoAlignmentSource::kStatxDioalign,
          "authoritative direct-I/O capability unavailable");
  std::vector<std::uint8_t> expected(bytes);
  for (std::size_t index = 0; index < bytes; ++index) {
    expected[index] = static_cast<std::uint8_t>(index & 255U);
  }
  PacketReadRequest request{"benchmark", 0, bytes, bytes,
                            Crc32c(expected.data(), expected.size()), PacketRequirement::kRequired};
  const auto read = [&](std::size_t index, bool verify) {
    request.file_offset = (index % (pack_bytes / bytes)) * bytes;
    return ReadOne(source, request, expected, verify);
  };
  // Correctness checks are outside timing; every timed packet still validates CRC.
  static_cast<void>(read(0, true));
  for (std::size_t index = 0; index < warmup; ++index) {
    static_cast<void>(read(index, false));
  }
  std::vector<Sample> samples;
  samples.reserve(iterations);
  const auto begin = Clock::now();
  for (std::size_t index = 0; index < iterations; ++index) {
    samples.push_back(read(index, false));
  }
  const double elapsed_us = Micros(begin, Clock::now());
  static_cast<void>(read(iterations - 1, true));
  Require(std::isfinite(elapsed_us) && elapsed_us > 0, "invalid benchmark clock duration");

  std::size_t physical = 0, logical = 0, submitted = 0, submissions = 0, completions = 0;
  double io = 0, h2d = 0, enqueue = 0;
  std::vector<double> latency;
  latency.reserve(samples.size());
  for (const auto& sample : samples) {
    physical += sample.stats.file_bytes_returned;
    logical += sample.stats.logical_bytes;
    submitted += sample.stats.bytes_submitted;
    submissions += sample.stats.read_submissions;
    completions += sample.stats.read_completions;
    io += sample.io_us;
    h2d += sample.h2d_us;
    enqueue += sample.enqueue_host_us;
    latency.push_back(sample.critical_us);
  }
  Require(physical == bytes * iterations && logical == physical && submitted == physical &&
              submissions == iterations && completions == iterations, "unexpected retry or byte totals");
  std::sort(latency.begin(), latency.end());
  std::cout << std::setprecision(17)
            << "{\"status\":\"ok\",\"mode\":\"" << mode << "\",\"verified\":true,"
            << "\"acceptance_evidence\":false,\"payload_bytes\":" << bytes
            << ",\"pack_bytes\":" << pack_bytes << ",\"queue_depth\":1,\"ring_slots\":1"
            << ",\"warmup\":" << warmup << ",\"iterations\":" << iterations
            << ",\"logical_bytes\":" << logical << ",\"physical_io_bytes\":" << physical
            << ",\"bytes_submitted\":" << submitted << ",\"submissions\":" << submissions
            << ",\"completions\":" << completions << ",\"read_amplification\":1"
            << ",\"statx_memory_alignment\":" << capability.alignment.memory_alignment
            << ",\"statx_offset_alignment\":" << capability.alignment.offset_alignment
            << ",\"ring_io_alignment\":4096,\"elapsed_us\":" << elapsed_us
            << ",\"bandwidth_bytes_per_second\":" << physical * 1e6 / elapsed_us
            << ",\"critical_path_p50_us\":" << latency[(latency.size() - 1) / 2]
            << ",\"critical_path_p95_us\":" << latency[static_cast<std::size_t>(std::ceil(0.95 * latency.size())) - 1]
            << ",\"mean_io_service_us\":" << io / iterations
            << ",\"mean_h2d_service_us\":" << h2d / iterations
            << ",\"mean_enqueue_host_us\":" << enqueue / iterations
            << ",\"timing\":\"host wall includes CRC, dispatch, allocation/free; D2H outside timing\"}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Require(argc == 6, "usage: harpq_transfer_benchmark direct|bridge PACK BYTES WARMUP ITERATIONS");
    const std::string mode(argv[1]);
    Require(mode == "direct" || mode == "bridge", "mode must be direct or bridge");
    const auto bytes = Number(argv[3], 1048576);
    Require(bytes == 65536 || bytes == 262144 || bytes == 1048576, "unsupported payload size");
    const auto warmup = Number(argv[4], 1000);
    const auto iterations = Number(argv[5], 4096);
    Require(bytes * iterations <= 1073741824 && bytes * warmup <= 134217728, "transfer volume exceeds safety bound");
    Require(std::filesystem::is_regular_file(argv[2]), "pack must be a regular file");
    const auto pack_bytes = std::filesystem::file_size(argv[2]);
    Require(pack_bytes >= bytes && pack_bytes <= 1073741824 && pack_bytes % bytes == 0,
            "pack must be aligned and no larger than 1 GiB");
    if (mode == "bridge") {
      CudaPacketRing ring(PacketRingConfig{1, bytes, 4096});
      CudaDirectPackFile source(ring, argv[2], pack_bytes, DirectPacketIoConfig{1});
      Run(source, "bridge", pack_bytes, bytes, warmup, iterations);
    } else {
      PacketRing ring(PacketRingConfig{1, bytes, 4096});
      DirectPackFile source(argv[2], pack_bytes, ring, DirectPacketIoConfig{1});
      Run(source, "direct", pack_bytes, bytes, warmup, iterations);
    }
  } catch (const std::exception& error) {
    std::cerr << "transfer benchmark failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
