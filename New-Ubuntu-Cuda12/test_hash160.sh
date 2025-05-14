#!/bin/bash

# Test script for Hash160 search mode
# This script will run several tests to verify the Hash160 direct search mode works correctly

echo "Running Hash160 search mode tests..."

# Test 1: Convert address to Hash160 format
echo "Test 1: Converting Bitcoin address to Hash160 format"
./vanitysearch -convertAddr 1PoolkYVES7qeScJJQu5K9wHxPJW7zNXe1

# Expected output should show the Hash160 value: c7309dc6851b55cdbbd6f2372070304a688f0c6e

# Test 2: Run a quick search with Hash160 format (should be faster than address format)
# First with regular address
echo "Test 2a: Running search with regular address format (short run for comparison)"
time ./vanitysearch -t 1 -gpu -gpuId 0 -maxFound 1 -r 1 1PoolkY

# Then with Hash160 format
echo "Test 2b: Running search with Hash160 format (should be faster)"
HASH160="c7309dc6851b55cdbbd6f2372070304a688f0c"
time ./vanitysearch -t 1 -gpu -gpuId 0 -maxFound 1 -r 1 -h "$HASH160"

# Test 3: Test multi-GPU support with Hash160 format
echo "Test 3: Testing multi-GPU support with Hash160 format (if multiple GPUs available)"
# Get the number of available GPUs
GPU_COUNT=$(./vanitysearch -check | grep "GPU #" | wc -l)

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Multiple GPUs detected ($GPU_COUNT). Testing multi-GPU search."
    # Create a comma-separated list of GPU IDs
    GPU_IDS=$(seq -s "," 0 $(($GPU_COUNT-1)))
    time ./vanitysearch -t 1 -gpu -gpuId "$GPU_IDS" -maxFound 1 -r 1 -h "$HASH160"
else
    echo "Only one GPU detected. Skipping multi-GPU test."
fi

# Test 4: Test the -end parameter
echo "Test 4: Testing the -end parameter with Hash160 search"
START_KEY="1"
END_KEY="FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140"
time ./vanitysearch -t 1 -gpu -gpuId 0 -maxFound 1 -r 1 -start "$START_KEY" -end "$END_KEY" -h "$HASH160"

echo "Hash160 search mode tests completed."
echo "If all tests passed, the Hash160 search functionality is working correctly." 