cmake_minimum_required(VERSION 3.24)

project(HARPQ VERSION 0.1.0 LANGUAGES CXX)

set(HARPQ_ENABLE_CUDA "AUTO" CACHE STRING "Enable CUDA targets: AUTO, ON, or OFF")
set_property(CACHE HARPQ_ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${HARPQ_ENABLE_CUDA}" HARPQ_ENABLE_CUDA)
if(NOT HARPQ_ENABLE_CUDA MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "HARPQ_ENABLE_CUDA must be AUTO, ON, or OFF")
endif()

set(HARPQ_HAS_CUDA OFF)
set(HARPQ_CUDA_ARCHITECTURES_USER_DEFINED OFF)
if(DEFINED CMAKE_CUDA_ARCHITECTURES OR
   (DEFINED ENV{CUDAARCHS} AND NOT "$ENV{CUDAARCHS}" STREQUAL ""))
  set(HARPQ_CUDA_ARCHITECTURES_USER_DEFINED ON)
endif()
if(NOT HARPQ_ENABLE_CUDA STREQUAL "OFF")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.8)
      if(HARPQ_ENABLE_CUDA STREQUAL "ON")
        message(FATAL_ERROR
          "HARPQ_ENABLE_CUDA=ON requires CUDA toolkit 11.8 or newer; "
          "found ${CMAKE_CUDA_COMPILER_VERSION}")
      else()
        message(STATUS
          "CUDA toolkit ${CMAKE_CUDA_COMPILER_VERSION} is older than 11.8; "
          "continuing with CPU targets and explicit CUDA test skips")
      endif()
    else()
      if(NOT HARPQ_CUDA_ARCHITECTURES_USER_DEFINED)
        # enable_language(CUDA) seeds a compiler default in the cache. Replace
        # only that implicit value; never override an architecture supplied by
        # the caller.
        set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING
          "CUDA architectures for HARP-Q" FORCE)
      endif()
      set(HARPQ_HAS_CUDA ON)
    endif()
  elseif(HARPQ_ENABLE_CUDA STREQUAL "ON")
    message(FATAL_ERROR "HARPQ_ENABLE_CUDA=ON, but no CUDA compiler was found")
  endif()
endif()

message(STATUS "HARP-Q CUDA targets: ${HARPQ_HAS_CUDA} (requested ${HARPQ_ENABLE_CUDA})")

include(CTest)
add_subdirectory(native)
