cmake_minimum_required(VERSION 3.24)

project(HARPQ VERSION 0.1.0 LANGUAGES CXX)

set(HARPQ_ENABLE_CUDA "AUTO" CACHE STRING "Enable CUDA targets: AUTO, ON, or OFF")
set_property(CACHE HARPQ_ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${HARPQ_ENABLE_CUDA}" HARPQ_ENABLE_CUDA)
if(NOT HARPQ_ENABLE_CUDA MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "HARPQ_ENABLE_CUDA must be AUTO, ON, or OFF")
endif()

set(HARPQ_ENABLE_IO_URING "AUTO" CACHE STRING
  "Enable Linux io_uring + O_DIRECT targets: AUTO, ON, or OFF")
set_property(CACHE HARPQ_ENABLE_IO_URING PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${HARPQ_ENABLE_IO_URING}" HARPQ_ENABLE_IO_URING)
if(NOT HARPQ_ENABLE_IO_URING MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "HARPQ_ENABLE_IO_URING must be AUTO, ON, or OFF")
endif()

set(HARPQ_HAS_IO_URING OFF)
if(NOT HARPQ_ENABLE_IO_URING STREQUAL "OFF")
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(HARPQ_ENABLE_IO_URING STREQUAL "ON")
      message(FATAL_ERROR
        "HARPQ_ENABLE_IO_URING=ON requires Linux; "
        "this host is ${CMAKE_SYSTEM_NAME}")
    endif()
  else()
    find_path(HARPQ_LIBURING_INCLUDE_DIR NAMES liburing.h)
    find_library(HARPQ_LIBURING_LIBRARY NAMES uring)
    if(HARPQ_LIBURING_INCLUDE_DIR AND HARPQ_LIBURING_LIBRARY)
      include(CheckCXXSourceCompiles)
      set(_HARPQ_SAVED_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES}")
      set(_HARPQ_SAVED_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES}")
      set(CMAKE_REQUIRED_INCLUDES "${HARPQ_LIBURING_INCLUDE_DIR}")
      set(CMAKE_REQUIRED_LIBRARIES "${HARPQ_LIBURING_LIBRARY}")
      check_cxx_source_compiles([=[
        #include <liburing.h>
        int main() {
          io_uring ring{};
          const int dontfork_status = io_uring_ring_dontfork(&ring);
          io_uring_probe* probe = io_uring_get_probe_ring(&ring);
          if (probe != nullptr) {
            io_uring_free_probe(probe);
          }
          return dontfork_status;
        }
      ]=] HARPQ_LIBURING_REQUIRED_API)
      set(CMAKE_REQUIRED_INCLUDES "${_HARPQ_SAVED_REQUIRED_INCLUDES}")
      set(CMAKE_REQUIRED_LIBRARIES "${_HARPQ_SAVED_REQUIRED_LIBRARIES}")
      unset(_HARPQ_SAVED_REQUIRED_INCLUDES)
      unset(_HARPQ_SAVED_REQUIRED_LIBRARIES)

      if(HARPQ_LIBURING_REQUIRED_API)
        add_library(HARPQ::liburing UNKNOWN IMPORTED)
        set_target_properties(HARPQ::liburing PROPERTIES
          IMPORTED_LOCATION "${HARPQ_LIBURING_LIBRARY}"
          INTERFACE_INCLUDE_DIRECTORIES "${HARPQ_LIBURING_INCLUDE_DIR}"
        )
        set(HARPQ_HAS_IO_URING ON)
      elseif(HARPQ_ENABLE_IO_URING STREQUAL "ON")
        message(FATAL_ERROR
          "HARPQ_ENABLE_IO_URING=ON found liburing, but its headers/library "
          "cannot compile and link the required ring_dontfork/probe APIs")
      endif()
    elseif(HARPQ_ENABLE_IO_URING STREQUAL "ON")
      message(FATAL_ERROR
        "HARPQ_ENABLE_IO_URING=ON, but liburing headers and library were not found")
    endif()
    mark_as_advanced(
      HARPQ_LIBURING_INCLUDE_DIR
      HARPQ_LIBURING_LIBRARY
      HARPQ_LIBURING_REQUIRED_API)
  endif()
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
message(STATUS
  "HARP-Q io_uring targets: ${HARPQ_HAS_IO_URING} "
  "(requested ${HARPQ_ENABLE_IO_URING})")

include(CTest)
add_subdirectory(native)
