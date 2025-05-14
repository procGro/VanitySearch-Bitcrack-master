# Package Contents for VanitySearch-Bitcrack Ubuntu 22.04 CUDA 12 Build

This directory contains the following files needed to compile VanitySearch-Bitcrack for Ubuntu 22.04 with CUDA 12 support:

## Build Files
- `Makefile`: Makefile configured for Ubuntu 22.04 with CUDA 12 support
- `build.sh`: Script to automate the build process

## Test Files
- `test_hash160.sh`: Test script to verify the Hash160 search functionality

## Documentation
- `README.md`: Documentation with features and usage examples
- `COMPILE_ON_UBUNTU.txt`: Instructions for compiling on Ubuntu 22.04
- `PACKAGE_CONTENTS.md`: This file, describing package contents

## Directory Structure Required for Compilation

To successfully compile, you need to have the following directory structure:

```
Parent Directory
├── New-Ubuntu-Cuda12             # This directory with all build files
│   ├── Makefile
│   ├── build.sh
│   └── ...
├── Base58.cpp                    # Source files copied from main project
├── Base58.h
├── Bech32.cpp
├── Bech32.h
├── Int.cpp
├── Int.h
├── IntGroup.cpp
├── IntGroup.h
├── IntMod.cpp
├── main.cpp
├── Point.cpp
├── Point.h
├── Random.cpp
├── Random.h
├── SECP256K1.cpp
├── SECP256k1.h
├── Timer.cpp
├── Timer.h
├── Vanity.cpp
├── Vanity.h
├── Wildcard.cpp
├── Wildcard.h
├── GPU                           # GPU-related source files
│   ├── GPUEngine.cu
│   └── GPUGenerate.cpp
└── hash                          # Hash function implementations
    ├── ripemd160.cpp
    ├── ripemd160_sse.cpp
    ├── sha256.cpp
    ├── sha256_sse.cpp
    └── sha512.cpp
```

## Expected Output

After successful compilation, the build will produce a single executable file called `vanitysearch` in this directory. This binary will include all the new features:

1. Hash160 direct search mode
2. Multi-GPU support
3. Key range endpoint specification
4. Performance optimizations

## Features Added

1. **Hash160 Direct Search Mode**
   - Search for Hash160 values directly in hex format
   - 20-30% faster than searching for encoded addresses
   - Implemented in both CPU and GPU code

2. **Multi-GPU Support**
   - Support for comma-separated GPU IDs
   - Automatic division of key space among GPUs
   - Each GPU works on its own dedicated range

3. **End Key Range Specification**
   - Added the -end parameter to specify the ending private key
   - More precise control over search ranges

4. **Optimized Batch Modular Inverse**
   - Improved key generation algorithm
   - Better performance for multi-GPU setups

## Building the Binary

To build the binary on an Ubuntu 22.04 system with CUDA 12:

1. Create a directory called New-Ubuntu-Cuda12
2. Copy all files listed above to the directory
3. Make the build.sh and test_hash160.sh scripts executable with:
   ```
   chmod +x build.sh test_hash160.sh
   ```
4. Run the build script:
   ```
   ./build.sh
   ```
5. The compiled binary will be created as `vanitysearch` in the same directory 