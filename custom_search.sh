#!/bin/bash

# Save terminal settings
stty -g > /tmp/term_settings.txt

# Trap CTRL+C to ensure terminal settings are restored
trap 'echo "Restoring terminal settings..."; stty $(cat /tmp/term_settings.txt); echo "Terminal restored"; exit' INT TERM

# Custom parameters
START_KEY="400000000000000000"
END_KEY="7fffffffffffffffff"
TARGET_ADDR="1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"
TARGET_HASH160="f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8"
OUTPUT_FILE="search_results.txt"
GPU_ID=0
TIMEOUT=180  # 3 minutes timeout

# Function to run a search with the given method
run_search() {
    local method=$1
    local pattern_file=$2
    local search_type=$3
    
    echo "====================================================="
    echo "Starting $search_type search with $method"
    echo "Range: $START_KEY to $END_KEY"
    echo "Output: $OUTPUT_FILE"
    echo "====================================================="
    
    timeout ${TIMEOUT}s ./vanitysearch $method -i $pattern_file -gpuId $GPU_ID -o $OUTPUT_FILE -start $START_KEY -end $END_KEY -stop
    
    if [ $? -eq 124 ]; then
        echo "Search timed out after ${TIMEOUT} seconds."
    fi
    echo ""
}

# Method 1: Address search (most compatible)
echo "Method 1: Standard Bitcoin Address Search"
echo "$TARGET_ADDR" > address_pattern.txt
run_search "" address_pattern.txt "Bitcoin address"

# Method 2: Hash160 search
echo "Method 2: Hash160 Search"
echo -n "$TARGET_HASH160" > hash160_pattern.txt
run_search "-hash160" hash160_pattern.txt "Hash160"

# Restore terminal settings no matter what
stty $(cat /tmp/term_settings.txt)
echo "Terminal restored"
echo "Search results (if any) saved to $OUTPUT_FILE"
exit 0 