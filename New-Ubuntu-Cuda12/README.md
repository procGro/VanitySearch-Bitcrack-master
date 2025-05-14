# VanitySearch-Bitcrack: Ubuntu 22.04 CUDA 12 Build

This directory contains the build files for compiling VanitySearch-Bitcrack with CUDA 12 support on Ubuntu 22.04. This enhanced version includes several performance improvements and new features.

## Features

### 1. Hash160 Direct Search Mode
- Search directly using Hash160 format instead of Bitcoin addresses
- 20-30% faster searches by eliminating encoding/decoding overhead
- Use the `-h` or `-hash160` parameter followed by the Hash160 value in hex format
- Convert existing addresses to Hash160 format using the `-convertAddr` option

### 2. Multi-GPU Support
- Use multiple GPUs simultaneously with comma-separated GPU IDs
- Example: `-gpuId 0,1,2` uses GPUs 0, 1, and 2
- Optimized work distribution across GPUs

### 3. Key Range Specification
- Specify precise key range endpoints with `-start` and `-end` parameters
- Useful for distributing workloads across multiple machines

### 4. Performance Optimizations
- Improved batch modular inverse algorithms
- Enhanced GPU kernels for better throughput
- Memory usage optimizations

## Compilation Instructions

### Requirements
- Ubuntu 22.04
- CUDA 12 toolkit and compatible drivers
- GCC/G++ compiler
- NVIDIA GPU with compute capability 6.0 or higher

### Building
1. Make sure CUDA 12 is properly installed
2. Run the build script:
   ```
   chmod +x build.sh
   ./build.sh
   ```
3. The compilation will produce the `vanitysearch` binary in this directory

## Testing
To test the new Hash160 search functionality:
```
chmod +x test_hash160.sh
./test_hash160.sh
```

This will run several tests demonstrating the new features and comparing performance between traditional address searching and Hash160 direct searching.

## Usage Examples

### Convert Bitcoin address to Hash160 format
```
./vanitysearch -convertAddr 1BitcoinEaterAddressDontSendf59kuE
```

### Search using Hash160 format
```
./vanitysearch -gpu -h 29b41fff35f475d5a0a8a96e8e889c3992a4f90f
```

### Search with multiple GPUs
```
./vanitysearch -gpu -gpuId 0,1 -h 29b41fff35f475d5a0a8a96e8e889c3992a4f90f
```

### Search within a specific key range
```
./vanitysearch -gpu -start 8000000000000000000000000000000000000000000000000000000000000000 -end 800000000000000000000000000000000000000000000000000000000000FFFF -h 29b41f
```

## New Command Line Options

- `-hash160`: Search for Hash160 in hex format instead of Bitcoin addresses
- `-convertAddr`: Convert Bitcoin addresses to Hash160 format for more efficient searching
- `-end`: Specify the ending private key (hex format)
- `-gpuId`: Comma-separated list of GPU IDs to use (e.g., 0,1,2)

## Performance

Searching directly for Hash160 values is significantly faster (up to 20-30%) than searching for Base58 or Bech32 encoded address prefixes because:
1. It eliminates the encoding/decoding overhead
2. It allows for direct binary comparison
3. It's more GPU-friendly, as GPUs excel at bitwise operations

## Notes

- When using the `-hash160` option, make sure your input is a valid 20-byte hash in hex format (40 hex characters)
- The `-convertAddr` option makes it easy to convert regular Bitcoin addresses to the Hash160 format
- Multiple GPUs will automatically divide the key space for optimal performance 