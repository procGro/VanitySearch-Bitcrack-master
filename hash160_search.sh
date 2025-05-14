#!/bin/bash

# Save terminal settings
stty -g > /tmp/term_settings.txt

# Create a properly formatted patterns file with exactly 40 hex characters
echo "Creating properly formatted Hash160 pattern file..."
# Pattern with exactly 40 hex characters (no leading zeros) - this is critical!
echo -n "f6f5431d25bbf7b12e8add9af5e3475c44a0a5bc" > patterns.txt

# Verify the length is exactly 40 (no newlines or other characters)
echo "Pattern file length: $(wc -c < patterns.txt) bytes"

# Create a verification file to help debug the issue
xxd -p patterns.txt

# Run the search with hash160 mode - SINGLE GPU ONLY
echo "Running single GPU search with Hash160 mode..."
# Explicitly call hash160 mode with a single GPU
./vanitysearch -gpuId 0 -hash160 -i patterns.txt -o output.txt -start 400000000000000000 -end 7fffffffffffffffff

# Restore terminal settings no matter what
stty $(cat /tmp/term_settings.txt)
echo "Terminal restored" 