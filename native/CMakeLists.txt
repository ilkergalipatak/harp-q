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

if(BUILD_TESTING)
  add_executable(harpq_native_tests tests/stage_codec_test.cpp)
  target_link_libraries(harpq_native_tests PRIVATE HARPQ::native)
  target_include_directories(harpq_native_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_compile_features(harpq_native_tests PRIVATE cxx_std_17)
  add_test(NAME native.stage_codec COMMAND harpq_native_tests)
endif()

# CUDA is an optional peer of the dependency-free CPU oracle.  CPU-only builds
# still register an explicit skipped CTest entry so a missing CUDA test cannot
# be mistaken for a passing one.
if(HARPQ_HAS_CUDA)
  add_subdirectory(cuda)
elseif(BUILD_TESTING)
  add_executable(harpq_cuda_unavailable tests/cuda_unavailable.cpp)
  target_compile_features(harpq_cuda_unavailable PRIVATE cxx_std_17)
  add_test(NAME native.cuda_oracle COMMAND harpq_cuda_unavailable)
  set_tests_properties(native.cuda_oracle PROPERTIES SKIP_RETURN_CODE 77)
endif()
