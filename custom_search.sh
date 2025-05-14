#!/bin/bash

# Custom parameters - modify these for your search
START_KEY="400000000000000000"
END_KEY="7fffffffffffffffff"
TARGET_ADDR="1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"
TARGET_HASH160="f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8"
OUTPUT_FILE="search_results.txt"
GPU_ID=0
MAX_RUNTIME=180  # Max runtime in seconds

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

# Create pattern files
echo "Setting up search parameters..."
echo "$TARGET_ADDR" > address_pattern.txt
echo -n "$TARGET_HASH160" > hash160_pattern.txt

# Function to run a search with timeout
run_search_with_timeout() {
    local method=$1
    local pattern_file=$2
    local search_type=$3
    local max_runtime=$4
    
    echo "====================================================="
    echo "Starting $search_type search"
    echo "Range: $START_KEY to $END_KEY"
    echo "Output: $OUTPUT_FILE"
    echo "Maximum runtime: $max_runtime seconds"
    echo "====================================================="
    
    # Start VanitySearch in the background
    ./vanitysearch $method -i $pattern_file -gpuId $GPU_ID -o $OUTPUT_FILE -start $START_KEY -end $END_KEY -stop &
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

# Run address search first
echo "Method 1: Standard Bitcoin Address Search"
run_search_with_timeout "" address_pattern.txt "Bitcoin address" $MAX_RUNTIME
status=$?

if [ $status -eq 124 ]; then
    echo "Address search timed out after $MAX_RUNTIME seconds"
elif [ $status -ne 0 ]; then
    echo "Address search exited with status: $status"
fi

echo ""

# Run hash160 search second
echo "Method 2: Hash160 Search"
run_search_with_timeout "-hash160" hash160_pattern.txt "Hash160" $MAX_RUNTIME
status=$?

if [ $status -eq 124 ]; then
    echo "Hash160 search timed out after $MAX_RUNTIME seconds"
elif [ $status -ne 0 ]; then
    echo "Hash160 search exited with status: $status"
fi

# Final cleanup
echo -e "\nAll searches completed."
echo "Search results (if any) saved to $OUTPUT_FILE"
cleanup_and_exit 