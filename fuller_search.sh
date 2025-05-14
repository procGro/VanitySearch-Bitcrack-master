#!/bin/bash
# Save terminal settings
stty -g > /tmp/term_settings.txt

# Try multiple approaches for searching Bitcoin addresses

# Approach 1: Standard Bitcoin address (P2PKH)
echo "Approach 1: Using standard Bitcoin address (P2PKH)"
echo "1NWxZ8D5FpHLG7yadTjugE2mPwzP1W4Xrp" > address1.txt
echo "Running with P2PKH address..."
./vanitysearch -gpuId 0 -i address1.txt -o output.txt -start 400000000000000000 -end 7fffffffffffffffff
echo ""

# Approach 2: BECH32 address
echo "Approach 2: Using BECH32 address"
echo "bc1qd6h6vp99qwstk3z668md42q0zc44vpwkk824zh" > address2.txt
echo "Running with BECH32 address..."
./vanitysearch -gpuId 0 -i address2.txt -o output.txt -start 400000000000000000 -end 7fffffffffffffffff
echo ""

# Approach 3: P2SH address
echo "Approach 3: Using P2SH address"
echo "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy" > address3.txt
echo "Running with P2SH address..."
./vanitysearch -gpuId 0 -i address3.txt -o output.txt -start 400000000000000000 -end 7fffffffffffffffff
echo ""

# Restore terminal settings
stty $(cat /tmp/term_settings.txt)
echo "Terminal restored"
