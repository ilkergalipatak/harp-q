find_package(CUDAToolkit REQUIRED)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.8)
  message(FATAL_ERROR "HARP-Q CUDA targets require CUDA toolkit 11.8 or newer")
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
  if(HARPQ_ENABLE_CUDA STREQUAL "AUTO")
    set_tests_properties(native.cuda_oracle PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()
