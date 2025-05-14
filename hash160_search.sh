#!/bin/bash

# Save terminal settings
stty -g > /tmp/term_settings.txt

# Get hash160 pattern from patterns.txt
HASH=$(cat patterns.txt | tr -d '\n\r ')
ALT_HASH=$(cat patterns_alt.txt | tr -d '\n\r ')

echo "Trying hash160 search with pattern: $HASH"
echo "Alternative pattern: $ALT_HASH"

# Try different argument orders and formats
echo "Attempt 1: Regular format"
./vanitysearch -gpuId 0 -hash160 -i patterns.txt -o output.txt -start 400000000000000000 -end 7fffffffffffffffff

echo "Attempt 2: Moving -hash160 to the end"
./vanitysearch -gpuId 0 -i patterns.txt -o output.txt -start 400000000000000000 -end 7fffffffffffffffff -hash160

echo "Attempt 3: Direct argument"
./vanitysearch -gpuId 0 -o output.txt -start 400000000000000000 -end 7fffffffffffffffff -hash160 $HASH

echo "Attempt 4: With range parameter"
./vanitysearch -gpuId 0 -hash160 -i patterns.txt -o output.txt -start 400000000000000000 -end 7fffffffffffffffff -range 70

echo "Attempt 5: Using alternative pattern file"
./vanitysearch -gpuId 0 -hash160 -i patterns_alt.txt -o output.txt -start 400000000000000000 -end 7fffffffffffffffff

echo "Attempt 6: Direct argument with alternative format"
./vanitysearch -gpuId 0 -o output.txt -start 400000000000000000 -end 7fffffffffffffffff -hash160 $ALT_HASH

# Additional attempt: Add bitcoin address prefix
echo "Attempt 7: Adding bitcoin address prefix"
./vanitysearch -gpuId 0 -o output.txt -start 400000000000000000 -end 7fffffffffffffffff $HASH

# Restore terminal settings no matter what
stty $(cat /tmp/term_settings.txt)
echo "Terminal restored" 