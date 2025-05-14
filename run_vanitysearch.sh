#!/bin/bash

# Save terminal settings
stty -g > /tmp/term_settings.txt

# Run vanitysearch with all command line arguments
# If -hash160 is in arguments, verify the pattern file
if [[ "$*" == *"-hash160"* ]] && [[ "$*" == *"-i"* ]]; then
  # Extract pattern file name
  pattern_file=$(echo "$*" | grep -o "\-i [^ ]*" | cut -d' ' -f2)
  if [ -f "$pattern_file" ]; then
    echo "Hash160 mode detected with pattern file: $pattern_file"
    # Check pattern length
    length=$(wc -c < "$pattern_file")
    echo "Pattern file length: $length bytes"
    if [ "$length" -ne 40 ]; then
      echo "WARNING: Pattern file should be exactly 40 bytes for hash160 mode"
      echo "Current content:"
      xxd -p "$pattern_file"
    fi
  fi
fi

echo "Running command: ./vanitysearch $@"
./vanitysearch "$@"

# Restore terminal settings no matter what
stty $(cat /tmp/term_settings.txt)
echo "Terminal restored" 