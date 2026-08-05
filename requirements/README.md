# Dependency locks

- `runtime.lock` is the exact NumPy version used by the Mac reference suite.
- `build.lock` is the exact setuptools/wheel pair used to validate the source distribution.

The test suite itself uses `unittest` from the Python standard library. Linux/CUDA dependencies
will be recorded in a separate lock only after the target driver/toolkit matrix is profiled; they
must not be guessed from the Mac environment.
