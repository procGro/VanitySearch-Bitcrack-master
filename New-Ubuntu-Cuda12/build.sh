#!/bin/bash

# Build script for VanitySearch-Bitcrack with Hash160 search mode
# For Ubuntu 22.04 with CUDA 12

echo "Building VanitySearch-Bitcrack with Hash160 search mode for Ubuntu 22.04 CUDA 12..."

# Make sure the obj directories exist
mkdir -p obj/GPU obj/hash

# Run make to build the project
make -j$(nproc)

# Check if the build was successful
if [ -f "vanitysearch" ]; then
    echo "Build completed successfully!"
    echo "The binary is: vanitysearch"
    # Make the binary executable
    chmod +x vanitysearch
else
    echo "Build failed!"
fi 