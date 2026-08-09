PYTHON ?= python3
CMAKE ?= cmake
NATIVE_BUILD_DIR ?= build/native-cpu
NATIVE_CUDA ?= OFF
NATIVE_IO_URING ?= AUTO

.PHONY: test smoke demo golden-check native-configure native-build native-test check

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src $(PYTHON) -m harpq.cli doctor --json

demo:
	PYTHONPATH=src $(PYTHON) -m harpq.cli demo --output artifacts/demo

golden-check:
	PYTHONPATH=src $(PYTHON) scripts/generate_native_golden.py --check

native-configure:
	$(CMAKE) -S . -B $(NATIVE_BUILD_DIR) -DHARPQ_ENABLE_CUDA=$(NATIVE_CUDA) -DHARPQ_ENABLE_IO_URING=$(NATIVE_IO_URING) -DBUILD_TESTING=ON

native-build: native-configure
	$(CMAKE) --build $(NATIVE_BUILD_DIR) --parallel

native-test: golden-check native-build
	ctest --test-dir $(NATIVE_BUILD_DIR) --output-on-failure

check: test smoke native-test
