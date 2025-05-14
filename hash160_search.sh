#!/bin/bash

# Save terminal settings
stty -g > /tmp/term_settings.txt

# Create a Bitcoin address pattern file instead of using hash160 mode
echo "Creating Bitcoin address pattern file..."

# Generate a valid Bitcoin address that corresponds to the hash160 we want to search for
# Format: 1<base58 encoding of hash160>
echo "1NWxZ8D5FpHLG7yadTjugE2mPwzP1W4Xrp" > bitcoin_address.txt

echo "Pattern file created:"
cat bitcoin_address.txt

# Try direct Bitcoin address search (no hash160 flag)
echo "Running search with direct Bitcoin address mode..."
./vanitysearch -gpuId 0 -i bitcoin_address.txt -o output.txt -start 400000000000000000 -end 7fffffffffffffffff

# Restore terminal settings no matter what
stty $(cat /tmp/term_settings.txt)
echo "Terminal restored" 