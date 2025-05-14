#!/bin/bash

# Custom parameters - modify these for your search
START_KEY="400000000000000000"
END_KEY="7fffffffffffffffff"
TARGET_HASH160="f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8"
OUTPUT_FILE="hash160_results.txt"
GPU_ID=0
MAX_RUNTIME=300  # Max runtime in seconds (5 minutes)
RANGE_BITS=70    # Range size in bits (2^70 covers the entire range between start and end)

# Save original terminal settings
original_tty_settings=$(stty -g)

# Function to clean up and exit
cleanup_and_exit() {
    echo -e "\nTerminating search..."
    
    # Kill any running VanitySearch processes
    if [ ! -z "$vanitysearch_pid" ]; then
        kill -9 $vanitysearch_pid 2>/dev/null
    fi
    
    # Restore terminal settings
    stty "$original_tty_settings"
    echo "Terminal settings restored"
    exit 0
}

# Set up trap for CTRL+C and other termination signals
trap cleanup_and_exit INT TERM

# Create pattern file
echo "Setting up hash160 search parameters..."
echo -n "$TARGET_HASH160" > hash160_pattern.txt

# Verify pattern length
PATTERN_LENGTH=$(wc -c < hash160_pattern.txt)
echo "Hash160 pattern length: $PATTERN_LENGTH bytes"
if [ "$PATTERN_LENGTH" -ne 40 ]; then
    echo "WARNING: Hash160 pattern should be exactly 40 characters (20 bytes in hex)"
    echo "Current pattern: $TARGET_HASH160"
fi

# Function to run a hash160 search with timeout
run_hash160_search() {
    local pattern_file=$1
    local max_runtime=$2
    
    echo "====================================================="
    echo "Starting Hash160 search"
    echo "Pattern: $TARGET_HASH160"
    echo "Range: $START_KEY to $END_KEY (range size: 2^$RANGE_BITS)"
    echo "Output: $OUTPUT_FILE"
    echo "Maximum runtime: $max_runtime seconds"
    echo "====================================================="
    
    # Start VanitySearch with hash160 mode in the background
    ./vanitysearch -hash160 -i $pattern_file -gpuId $GPU_ID -o $OUTPUT_FILE -start $START_KEY -end $END_KEY -range $RANGE_BITS -stop &
    vanitysearch_pid=$!
    
    # Set up timer
    SECONDS=0
    
    # Monitor the process
    while kill -0 $vanitysearch_pid 2>/dev/null; do
        # Check if we've exceeded maximum runtime
        if [ $SECONDS -ge $max_runtime ]; then
            echo -e "\nSearch timeout reached ($max_runtime seconds)"
            kill -9 $vanitysearch_pid 2>/dev/null
            wait $vanitysearch_pid 2>/dev/null
            return 124  # Return timeout status
        fi
        
        # Brief sleep to avoid CPU overload from the monitoring loop
        sleep 0.1
    done
    
    # Wait for process to finish and get its exit status
    wait $vanitysearch_pid
    return $?
}

# Run hash160 search
echo "Running Hash160 search mode..."
run_hash160_search hash160_pattern.txt $MAX_RUNTIME
status=$?

if [ $status -eq 124 ]; then
    echo "Hash160 search timed out after $MAX_RUNTIME seconds"
elif [ $status -ne 0 ]; then
    echo "Hash160 search exited with status: $status"
fi

# Final cleanup
echo -e "\nSearch completed."
echo "Search results (if any) saved to $OUTPUT_FILE"
cleanup_and_exit 