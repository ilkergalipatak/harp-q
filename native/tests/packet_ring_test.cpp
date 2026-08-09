#include "harpq/native/packet_ring.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace {

using harpq::native_runtime::CompletionState;
using harpq::native_runtime::FillingSlotLease;
using harpq::native_runtime::InFlightSlotLease;
using harpq::native_runtime::PacketRing;
using harpq::native_runtime::PacketRingConfig;
using harpq::native_runtime::PacketSlotMetadata;
using harpq::native_runtime::PacketSlotState;
using harpq::native_runtime::ReadySlotLease;
using harpq::native_runtime::TransferCompletion;

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

struct CompletionTracker {
  CompletionState poll_state = CompletionState::kPending;
  CompletionState wait_state = CompletionState::kComplete;
  std::size_t poll_calls = 0;
  std::size_t wait_calls = 0;
};

class FakeCompletion final : public TransferCompletion {
 public:
  explicit FakeCompletion(std::shared_ptr<CompletionTracker> tracker)
      : tracker_(std::move(tracker)) {}

  CompletionState Poll() noexcept override {
    ++tracker_->poll_calls;
    return tracker_->poll_state;
  }

  CompletionState WaitUntilSafe() noexcept override {
    ++tracker_->wait_calls;
    return tracker_->wait_state;
  }

 private:
  std::shared_ptr<CompletionTracker> tracker_;
};

PacketSlotMetadata Metadata(std::string packet_id, std::size_t logical_bytes = 4) {
  return PacketSlotMetadata{
      std::move(packet_id), logical_bytes, 4096, 0x12345678U};
}

void TestConfigurationAndAlignment() {
  ExpectThrows<std::invalid_argument>(
      [] { PacketRing ring(PacketRingConfig{0, 4096, 4096}); },
      "reject zero slots");
  ExpectThrows<std::invalid_argument>(
      [] { PacketRing ring(PacketRingConfig{1, 0, 4096}); },
      "reject zero capacity");
  ExpectThrows<std::invalid_argument>(
      [] { PacketRing ring(PacketRingConfig{1, 4096, 3000}); },
      "reject non-power-of-two I/O alignment");
  ExpectThrows<std::invalid_argument>(
      [] { PacketRing ring(PacketRingConfig{1, 5000, 4096}); },
      "reject capacity not divisible by I/O alignment");

  PacketRing ring(PacketRingConfig{3, 4096, 4096});
  const long page_result = sysconf(_SC_PAGESIZE);
  Check(page_result > 0, "host page size query");
  const auto page_size = static_cast<std::size_t>(page_result);
  Check(ring.io_alignment() == 4096, "I/O alignment remains format alignment");
  Check(
      ring.allocation_alignment() >= page_size &&
          ring.allocation_alignment() % page_size == 0,
      "allocation uses host page alignment");
  Check(
      ring.slot_stride() % ring.allocation_alignment() == 0,
      "slot stride uses allocation alignment");

  const auto registration = ring.registration_storage();
  Check(registration.data != nullptr, "registration storage pointer");
  Check(
      reinterpret_cast<std::uintptr_t>(registration.data) % registration.alignment == 0,
      "registration storage address alignment");
  Check(
      registration.bytes == ring.slot_count() * ring.slot_stride(),
      "registration storage covers every fixed slot");

  auto first = ring.TryAcquireFilling();
  auto second = ring.TryAcquireFilling();
  Check(first.has_value() && second.has_value(), "acquire aligned slots");
  if (first.has_value() && second.has_value()) {
    const auto first_bytes = first->bytes();
    const auto second_bytes = second->bytes();
    Check(
        reinterpret_cast<std::uintptr_t>(first_bytes.data) %
                ring.allocation_alignment() ==
            0,
        "first slot address alignment");
    Check(
        second_bytes.data - first_bytes.data ==
            static_cast<std::ptrdiff_t>(ring.slot_stride()),
        "fixed slot stride");
  }
}

void TestDeterministicAcquisitionAndGeneration() {
  PacketRing ring(PacketRingConfig{2, 4096, 4096});
  auto first = ring.TryAcquireFilling();
  auto second = ring.TryAcquireFilling();
  Check(first.has_value() && second.has_value(), "fill both slots");
  if (!first.has_value() || !second.has_value()) {
    return;
  }
  Check(first->slot_index() == 0, "first acquisition uses slot zero");
  Check(second->slot_index() == 1, "second acquisition uses slot one");
  Check(!ring.TryAcquireFilling().has_value(), "full ring reports no slot");

  const std::uint64_t old_generation = first->generation();
  first->Cancel();
  ExpectThrows<std::logic_error>(
      [&] { static_cast<void>(first->bytes()); },
      "cancelled filling lease is empty");
  auto replacement = ring.TryAcquireFilling();
  Check(replacement.has_value(), "cancelled slot is reusable");
  if (replacement.has_value()) {
    Check(replacement->slot_index() == 0, "round-robin returns freed slot");
    Check(
        replacement->generation() == old_generation + 1,
        "slot generation advances on reuse");
  }
}

void TestPublishReadyAndValidation() {
  PacketRing ring(PacketRingConfig{1, 4096, 4096});
  auto filling = ring.TryAcquireFilling();
  Check(filling.has_value(), "acquire filling lease");
  if (!filling.has_value()) {
    return;
  }
  auto bytes = filling->bytes();
  bytes.data[0] = 10;
  bytes.data[1] = 20;
  bytes.data[2] = 30;
  bytes.data[3] = 40;

  ExpectThrows<std::invalid_argument>(
      [&] {
        auto invalid = std::move(*filling).Publish(Metadata("", 4));
        static_cast<void>(invalid);
      },
      "reject empty ready packet ID");
  Check(filling->valid(), "failed publish retains filling ownership");
  ExpectThrows<std::invalid_argument>(
      [&] {
        auto invalid = std::move(*filling).Publish(
            PacketSlotMetadata{"packet", 4, 4000, 0});
        static_cast<void>(invalid);
      },
      "reject unaligned physical extent");

  auto ready = std::move(*filling).Publish(Metadata("base", 4));
  Check(!filling->valid(), "publish consumes filling lease");
  const auto view = ready.view();
  Check(view.packet_id == "base", "ready packet ID");
  Check(view.logical_bytes == 4 && view.physical_bytes == 4096, "ready extents");
  Check(view.data[0] == 10 && view.data[3] == 40, "ready bytes remain stable");
  Check(
      ring.Snapshot()[0].state == PacketSlotState::kReady,
      "published slot state");

  ready.Release();
  Check(!ready.valid(), "release empties ready lease");
  Check(
      ring.Snapshot()[0].state == PacketSlotState::kFree,
      "ready release returns slot to free");
}

void TestCompletionLifecycle() {
  PacketRing ring(PacketRingConfig{1, 4096, 4096});
  auto filling = ring.TryAcquireFilling();
  if (!filling.has_value()) {
    Check(false, "acquire slot for completion lifecycle");
    return;
  }
  auto ready = std::move(*filling).Publish(Metadata("residual"));

  ExpectThrows<std::invalid_argument>(
      [&] {
        auto invalid = std::move(ready).BeginH2d(nullptr);
        static_cast<void>(invalid);
      },
      "reject null transfer completion");
  Check(ready.valid(), "null completion does not consume ready lease");

  auto tracker = std::make_shared<CompletionTracker>();
  auto in_flight = std::move(ready).BeginH2d(
      std::make_unique<FakeCompletion>(tracker));
  Check(!ready.valid(), "begin H2D consumes ready lease");
  Check(
      ring.Snapshot()[0].state == PacketSlotState::kH2dInFlight,
      "H2D state is visible");
  Check(!ring.TryAcquireFilling().has_value(), "in-flight slot cannot be reused");
  Check(
      in_flight.PollAndRecycle() == CompletionState::kPending,
      "pending completion retains lease");
  Check(in_flight.valid(), "pending in-flight lease remains valid");

  tracker->poll_state = CompletionState::kComplete;
  Check(
      in_flight.PollAndRecycle() == CompletionState::kComplete,
      "completed transfer recycles slot");
  Check(!in_flight.valid(), "completed in-flight lease is empty");
  Check(tracker->wait_calls == 0, "successful poll does not block");
  Check(ring.TryAcquireFilling().has_value(), "completed slot is reusable");
}

void TestLeaseDestructorsAndFacadeLifetime() {
  PacketRing ring(PacketRingConfig{1, 4096, 4096});
  {
    auto filling = ring.TryAcquireFilling();
    Check(filling.has_value(), "acquire filling lease for destructor cancellation");
  }
  Check(
      ring.Snapshot()[0].state == PacketSlotState::kFree,
      "filling destructor cancels and frees slot");
  {
    auto filling = ring.TryAcquireFilling();
    if (filling.has_value()) {
      auto ready = std::move(*filling).Publish(Metadata("ready-destructor"));
      static_cast<void>(ready);
    }
  }
  Check(
      ring.Snapshot()[0].state == PacketSlotState::kFree,
      "ready destructor frees slot");

  std::optional<FillingSlotLease> retained_filling;
  {
    PacketRing transient(PacketRingConfig{1, 4096, 4096});
    auto filling = transient.TryAcquireFilling();
    if (filling.has_value()) {
      retained_filling.emplace(std::move(*filling));
    }
  }
  Check(retained_filling.has_value(), "filling lease outlives ring facade");
  if (retained_filling.has_value()) {
    const auto bytes = retained_filling->bytes();
    bytes.data[0] = 0x5A;
    Check(bytes.data[0] == 0x5A, "retained filling storage remains alive");
    retained_filling.reset();
  }

  std::optional<ReadySlotLease> retained_ready;
  {
    PacketRing transient(PacketRingConfig{1, 4096, 4096});
    auto filling = transient.TryAcquireFilling();
    if (filling.has_value()) {
      filling->bytes().data[0] = 0xA5;
      retained_ready.emplace(
          std::move(*filling).Publish(Metadata("retained-ready", 1)));
    }
  }
  Check(retained_ready.has_value(), "ready lease outlives ring facade");
  if (retained_ready.has_value()) {
    Check(
        retained_ready->view().data[0] == 0xA5,
        "retained ready storage remains alive");
    retained_ready.reset();
  }
}

void TestFailedCompletionWaitsBeforeRecycle() {
  PacketRing ring(PacketRingConfig{1, 4096, 4096});
  auto filling = ring.TryAcquireFilling();
  if (!filling.has_value()) {
    Check(false, "acquire slot for failed completion");
    return;
  }
  auto ready = std::move(*filling).Publish(Metadata("residual"));
  auto tracker = std::make_shared<CompletionTracker>();
  tracker->poll_state = CompletionState::kFailed;
  tracker->wait_state = CompletionState::kFailed;
  auto in_flight = std::move(ready).BeginH2d(
      std::make_unique<FakeCompletion>(tracker));

  Check(
      in_flight.PollAndRecycle() == CompletionState::kFailed,
      "failed event query remains observable");
  Check(tracker->wait_calls == 1, "failed query invokes safe wait fallback");
  Check(!in_flight.valid(), "failed transfer recycles only after safe wait");
  Check(ring.TryAcquireFilling().has_value(), "failed transfer slot is reusable");
}

void TestBlockingCompletionCannotRemainPending() {
  PacketRing ring(PacketRingConfig{1, 4096, 4096});
  auto filling = ring.TryAcquireFilling();
  if (!filling.has_value()) {
    Check(false, "acquire slot for invalid blocking completion");
    return;
  }
  auto ready = std::move(*filling).Publish(Metadata("residual"));
  auto tracker = std::make_shared<CompletionTracker>();
  tracker->wait_state = CompletionState::kPending;
  auto in_flight = std::move(ready).BeginH2d(
      std::make_unique<FakeCompletion>(tracker));
  Check(
      in_flight.WaitAndRecycle() == CompletionState::kFailed,
      "pending result from blocking wait becomes a visible protocol failure");
  Check(!in_flight.valid(), "safe blocking contract recycles invalid completion");
  Check(ring.TryAcquireFilling().has_value(), "invalid completion does not leak slot");
}

void TestLeaseKeepsStorageAliveThroughTeardown() {
  std::optional<InFlightSlotLease> retained;
  auto tracker = std::make_shared<CompletionTracker>();
  {
    PacketRing ring(PacketRingConfig{1, 4096, 4096});
    auto filling = ring.TryAcquireFilling();
    if (!filling.has_value()) {
      Check(false, "acquire slot for teardown");
      return;
    }
    auto ready = std::move(*filling).Publish(Metadata("base"));
    retained.emplace(std::move(ready).BeginH2d(
        std::make_unique<FakeCompletion>(tracker)));
  }
  Check(tracker->wait_calls == 0, "ring facade destruction does not free live storage");
  retained.reset();
  Check(tracker->wait_calls == 1, "in-flight destructor waits before storage teardown");
}

}  // namespace

int main() {
  TestConfigurationAndAlignment();
  TestDeterministicAcquisitionAndGeneration();
  TestPublishReadyAndValidation();
  TestCompletionLifecycle();
  TestLeaseDestructorsAndFacadeLifetime();
  TestFailedCompletionWaitsBeforeRecycle();
  TestBlockingCompletionCannotRemainPending();
  TestLeaseKeepsStorageAliveThroughTeardown();
  if (failures != 0) {
    std::cerr << failures << " packet ring test(s) failed\n";
    return 1;
  }
  std::cout << "packet ring tests passed\n";
  return 0;
}
