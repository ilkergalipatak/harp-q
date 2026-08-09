#include <iostream>

int main() {
  std::cout
      << "SKIP: CUDA oracle was not built; configure with "
         "-DHARPQ_ENABLE_CUDA=ON on an NVIDIA/CUDA host\n";
  return 77;
}
