#include <iostream>

int main() {
  std::cout
      << "SKIP: io_uring + O_DIRECT targets were not built; configure with "
         "-DHARPQ_ENABLE_IO_URING=ON on a Linux host with liburing\n";
  return 77;
}
