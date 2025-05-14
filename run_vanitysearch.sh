#!/bin/bash

# Save terminal settings
stty -g > /tmp/term_settings.txt

# Run vanitysearch with all command line arguments
./vanitysearch "$@"

# Restore terminal settings no matter what
stty $(cat /tmp/term_settings.txt)
echo "Terminal restored" 