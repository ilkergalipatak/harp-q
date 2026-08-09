find_package(CUDAToolkit REQUIRED)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.8)
  message(FATAL_ERROR "HARP-Q CUDA targets require CUDA toolkit 11.8 or newer")
endif()

if(NOT TARGET HARPQ::packet_io)
  message(FATAL_ERROR
    "HARP-Q CUDA packet-ring targets currently require POSIX packet I/O; "
    "configure with HARPQ_ENABLE_CUDA=OFF on unsupported hosts")
endif()

add_library(harpq_cuda_oracle STATIC
  ../src/cuda_oracle.cu
)
add_library(HARPQ::cuda_oracle ALIAS harpq_cuda_oracle)

target_include_directories(harpq_cuda_oracle
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(harpq_cuda_oracle PUBLIC cxx_std_17 cuda_std_17)
target_link_libraries(harpq_cuda_oracle
  PUBLIC
    HARPQ::native
    CUDA::cudart
    CUDA::cublas
)
set_target_properties(harpq_cuda_oracle PROPERTIES
  CXX_EXTENSIONS OFF
  CUDA_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

add_library(harpq_progressive_gemv STATIC
  ../src/progressive_gemv.cu
)
add_library(HARPQ::progressive_gemv ALIAS harpq_progressive_gemv)
target_include_directories(harpq_progressive_gemv
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(harpq_progressive_gemv PUBLIC cxx_std_17 cuda_std_17)
target_link_libraries(harpq_progressive_gemv
  PUBLIC
    HARPQ::native
    CUDA::cudart
)
set_target_properties(harpq_progressive_gemv PROPERTIES
  CXX_EXTENSIONS OFF
  CUDA_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

add_library(harpq_cuda_packet_ring STATIC
  ../src/cuda_packet_ring.cpp
)
add_library(HARPQ::cuda_packet_ring ALIAS harpq_cuda_packet_ring)
target_include_directories(harpq_cuda_packet_ring
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(harpq_cuda_packet_ring PUBLIC cxx_std_17)
target_link_libraries(harpq_cuda_packet_ring
  PUBLIC
    HARPQ::packet_io
    CUDA::cudart
)
set_target_properties(harpq_cuda_packet_ring PROPERTIES
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

add_executable(harpq_h2d_benchmark
  ../benchmarks/h2d_benchmark.cu
)
target_compile_features(harpq_h2d_benchmark PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(harpq_h2d_benchmark PRIVATE CUDA::cudart)
set_target_properties(harpq_h2d_benchmark PROPERTIES
  CXX_EXTENSIONS OFF
  CUDA_EXTENSIONS OFF
)

add_executable(harpq_gemv_benchmark
  ../benchmarks/gemv_benchmark.cu
)
target_compile_features(harpq_gemv_benchmark PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(harpq_gemv_benchmark
  PRIVATE
    HARPQ::progressive_gemv
    CUDA::cudart
)
set_target_properties(harpq_gemv_benchmark PROPERTIES
  CXX_EXTENSIONS OFF
  CUDA_EXTENSIONS OFF
)

if(BUILD_TESTING)
  add_executable(harpq_cuda_oracle_test
    ../tests/cuda_oracle_test.cu
  )
  target_include_directories(harpq_cuda_oracle_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../tests
  )
  target_compile_features(harpq_cuda_oracle_test PRIVATE cxx_std_17 cuda_std_17)
  target_link_libraries(harpq_cuda_oracle_test PRIVATE HARPQ::cuda_oracle)
  set_target_properties(harpq_cuda_oracle_test PROPERTIES
    CXX_EXTENSIONS OFF
    CUDA_EXTENSIONS OFF
  )
  add_test(NAME native.cuda_oracle COMMAND harpq_cuda_oracle_test)

  add_executable(harpq_progressive_gemv_test
    ../tests/progressive_gemv_test.cu
  )
  target_include_directories(harpq_progressive_gemv_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../tests
  )
  target_compile_features(
    harpq_progressive_gemv_test PRIVATE cxx_std_17 cuda_std_17)
  target_link_libraries(harpq_progressive_gemv_test
    PRIVATE
      HARPQ::progressive_gemv
      HARPQ::cuda_oracle
  )
  set_target_properties(harpq_progressive_gemv_test PROPERTIES
    CXX_EXTENSIONS OFF
    CUDA_EXTENSIONS OFF
  )
  add_test(NAME native.progressive_gemv COMMAND harpq_progressive_gemv_test)

  add_executable(harpq_cuda_packet_ring_test
    ../tests/cuda_packet_ring_test.cpp
  )
  target_compile_features(harpq_cuda_packet_ring_test PRIVATE cxx_std_17)
  target_link_libraries(harpq_cuda_packet_ring_test
    PRIVATE HARPQ::cuda_packet_ring)
  set_target_properties(harpq_cuda_packet_ring_test PROPERTIES
    CXX_EXTENSIONS OFF
  )
  add_test(NAME native.cuda_packet_ring COMMAND harpq_cuda_packet_ring_test)
  set_tests_properties(native.cuda_packet_ring PROPERTIES TIMEOUT 30)

  add_test(
    NAME native.h2d_smoke
    COMMAND harpq_h2d_benchmark
      --bytes 1048576
      --warmup 2
      --iterations 5
      --json
  )
  set_tests_properties(native.h2d_smoke PROPERTIES TIMEOUT 30)

  add_test(
    NAME native.gemv_benchmark_smoke
    COMMAND harpq_gemv_benchmark
      --rows 257
      --columns 513
      --tile 128
      --group 128
      --prefix 2
      --warmup 1
      --iterations 5
      --repeats 3
      --json
  )
  set_tests_properties(native.gemv_benchmark_smoke PROPERTIES TIMEOUT 30)

  if(HARPQ_ENABLE_CUDA STREQUAL "AUTO")
    set_tests_properties(
      native.cuda_oracle
      native.progressive_gemv
      native.cuda_packet_ring
      native.h2d_smoke
      native.gemv_benchmark_smoke
      PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()
