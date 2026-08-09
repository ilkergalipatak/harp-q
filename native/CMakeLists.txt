add_library(harpq_native STATIC
  src/stage_codec.cpp
)
add_library(HARPQ::native ALIAS harpq_native)

target_include_directories(harpq_native
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(harpq_native PUBLIC cxx_std_17)
set_target_properties(harpq_native PROPERTIES
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

if(MSVC)
  target_compile_options(harpq_native PRIVATE /W4 /permissive-)
else()
  target_compile_options(harpq_native PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(UNIX)
  find_package(Threads REQUIRED)
  add_library(harpq_packet_io STATIC
    src/packet_ring.cpp
    src/packet_io_posix.cpp
  )
  add_library(HARPQ::packet_io ALIAS harpq_packet_io)
  target_include_directories(harpq_packet_io
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  target_compile_features(harpq_packet_io PUBLIC cxx_std_17)
  target_link_libraries(harpq_packet_io
    PUBLIC HARPQ::native
    PRIVATE Threads::Threads
  )
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(harpq_packet_io PRIVATE _FILE_OFFSET_BITS=64)
  endif()
  set_target_properties(harpq_packet_io PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
  )
  target_compile_options(harpq_packet_io PRIVATE -Wall -Wextra -Wpedantic)
else()
  message(STATUS
    "HARP-Q packet I/O targets require POSIX pread and are disabled on this host")
endif()

if(BUILD_TESTING)
  add_executable(harpq_native_tests tests/stage_codec_test.cpp)
  target_link_libraries(harpq_native_tests PRIVATE HARPQ::native)
  target_include_directories(harpq_native_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_compile_features(harpq_native_tests PRIVATE cxx_std_17)
  add_test(NAME native.stage_codec COMMAND harpq_native_tests)

  if(UNIX)
    add_executable(harpq_packet_ring_test tests/packet_ring_test.cpp)
    target_link_libraries(harpq_packet_ring_test PRIVATE HARPQ::packet_io)
    target_compile_features(harpq_packet_ring_test PRIVATE cxx_std_17)
    add_test(NAME native.packet_ring COMMAND harpq_packet_ring_test)

    add_executable(harpq_packet_io_test tests/packet_io_test.cpp)
    target_link_libraries(harpq_packet_io_test PRIVATE HARPQ::packet_io)
    target_compile_features(harpq_packet_io_test PRIVATE cxx_std_17)
    add_test(NAME native.packet_io COMMAND harpq_packet_io_test)
  endif()
endif()

# CUDA is an optional peer of the dependency-free CPU oracle.  CPU-only builds
# still register an explicit skipped CTest entry so a missing CUDA test cannot
# be mistaken for a passing one.
if(HARPQ_HAS_CUDA)
  add_subdirectory(cuda)
elseif(BUILD_TESTING)
  add_executable(harpq_cuda_unavailable tests/cuda_unavailable.cpp)
  target_compile_features(harpq_cuda_unavailable PRIVATE cxx_std_17)
  foreach(cuda_test IN ITEMS
      native.cuda_oracle
      native.progressive_gemv
      native.cuda_packet_ring
      native.h2d_smoke
      native.gemv_benchmark_smoke)
    add_test(NAME ${cuda_test} COMMAND harpq_cuda_unavailable)
    set_tests_properties(${cuda_test} PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()
endif()
