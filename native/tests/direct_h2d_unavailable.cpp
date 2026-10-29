#include <iostream>

int main() {
  std::cout << "SKIP: direct-H2D bridge requires both CUDA and Linux liburing targets; "
               "configure HARPQ_ENABLE_CUDA=ON and HARPQ_ENABLE_IO_URING=ON\n";
  return 77;
}
