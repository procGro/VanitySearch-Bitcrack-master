#!/bin/bash

# Save terminal settings
stty -g > /tmp/term_settings.txt

# Trap CTRL+C to ensure terminal settings are restored
trap 'echo "Restoring terminal settings..."; stty $(cat /tmp/term_settings.txt); echo "Terminal restored"; exit' INT TERM

# Create a Bitcoin address pattern file instead of using hash160 mode
echo "Creating Bitcoin address pattern file..."

# Generate a valid Bitcoin address that corresponds to the hash160 we want to search for
# Format: 1<base58 encoding of hash160>
echo "1NWxZ8D5FpHLG7yadTjugE2mPwzP1W4Xrp" > bitcoin_address.txt

echo "Pattern file created:"
cat bitcoin_address.txt

# Run search with timeout of 60 seconds (adjust as needed)
echo "Running search with direct Bitcoin address mode..."
timeout 60s ./vanitysearch -gpuId 0 -i bitcoin_address.txt -o output.txt -start 400000000000000000 -end 7fffffffffffffffff -stop

# Check if vanitysearch was killed by timeout
if [ $? -eq 124 ]; then
    echo "Search timed out after 60 seconds."
fi

# Restore terminal settings no matter what
stty $(cat /tmp/term_settings.txt)
echo "Terminal restored"
exit 0 