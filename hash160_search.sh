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

# Create a valid Bitcoin address pattern for the target hash160
echo "Setting up search parameters for Hash160: $TARGET_HASH160"

# Method 1: Convert the Hash160 to a Bitcoin address
# We'll create a standard P2PKH address (starting with 1)
# This is equivalent to running: ./vanitysearch -bitcoinAddress $TARGET_HASH160
echo "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU" > bitcoin_address.txt

# Function to run a search with timeout
run_search_with_timeout() {
    local method=$1
    local pattern_file=$2
    local search_type=$3
    local max_runtime=$4
    
    echo "====================================================="
    echo "Starting $search_type search"
    echo "Target: $(cat $pattern_file)"
    echo "Original Hash160: $TARGET_HASH160"
    echo "Range: $START_KEY to $END_KEY (range size: 2^$RANGE_BITS)"
    echo "Output: $OUTPUT_FILE"
    echo "Maximum runtime: $max_runtime seconds"
    echo "====================================================="
    
    # Start VanitySearch with the specified method in the background
    ./vanitysearch $method -i $pattern_file -gpuId $GPU_ID -o $OUTPUT_FILE -start $START_KEY -end $END_KEY -range $RANGE_BITS -stop &
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

# Run Bitcoin address search (since hash160 mode has issues)
echo "Running search for Bitcoin address corresponding to Hash160"
run_search_with_timeout "" bitcoin_address.txt "Bitcoin address (Hash160 equivalent)" $MAX_RUNTIME
status=$?

if [ $status -eq 124 ]; then
    echo "Search timed out after $MAX_RUNTIME seconds"
elif [ $status -ne 0 ]; then
    echo "Search exited with status: $status"
fi

# Final cleanup
echo -e "\nSearch completed."
echo "Any results were saved to $OUTPUT_FILE"
echo "Note: This search was for the Bitcoin address corresponding to Hash160: $TARGET_HASH160"
cleanup_and_exit 